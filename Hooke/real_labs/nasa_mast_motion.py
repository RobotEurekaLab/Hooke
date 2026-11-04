"""Measured anchored-rover pan inspection through two actual mounted cameras.

This is a dry geometric target observation with ideal authored stereo optics.
It does not estimate calibrated range, reproduce NASA hardware or drive wheels.
"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.nasa_mast_geometry import CAMERAS

SCENE = "nasa_planetary"


@lru_cache(maxsize=1)
def _binding():
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml, assets={r["file"]: Path(r["source"]).read_bytes() for r in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        manifest={
            k: manifest.get(k)
            for k in (
                "camera",
                "camera_names",
                "mounted_cameras",
                "specimens",
                "benches",
            )
        },
    )


def _require_binding(session):
    if not np.isfinite(float(session.data.time)):
        raise ValueError("Inspection requires finite simulation time")
    if session.data.model is not session.model:
        raise ValueError("NASA mast data must belong to its exact model")
    ref = _binding()
    if (
        session.manifest.get("id") != SCENE
        or session.controls != ref["controls"]
        or session.equipment != ref["equipment"]
        or any(session.manifest.get(k) != v for k, v in ref["manifest"].items())
    ):
        raise ValueError("NASA mast scene, control or observation metadata changed")
    if _model_signature(session.model) not in ref["signatures"]:
        raise ValueError("NASA mast geometry, collision or physics binding changed")
    if session.manifest.get("scene_sha256", ref["scene_sha256"]) != ref["scene_sha256"]:
        raise ValueError("NASA mast package differs from current source")
    return ref


def full_state(session):
    d = session.data
    return dict(
        time_s=float(d.time),
        **{
            name: getattr(d, name).tolist()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "eq_active",
            )
        },
    )


class MastInspection:
    """Source-bound pan motion; all relevant enabled distinct-weld pairs retained.

    The caller owns the session exclusively. Canonical geometry is checked before
    commands and at phase boundaries; concurrent external model writers are not
    supported. Each executed step checks masks, physics, held controls and direct
    distances, including pairs hidden by MuJoCo's parent-body contact filtering.
    """

    distance_cap = 0.003
    minimum_gap = 0.0002

    def __init__(self, session):
        self.ref = _require_binding(session)
        self.session, self.m, self.d = session, session.model, session.data
        m, d = self.m, self.d
        self.control = session.controls["rover", "mast_pan"]
        self.q = int(m.jnt_qposadr[self.control.joint])
        self.v = int(m.jnt_dofadr[self.control.joint])
        self.a = self.control.actuator
        self.interface = session.equipment["rover"]["mast_inspection"]
        self.cameras = [m.camera(name).id for name in CAMERAS]
        self.camera_sites = [
            m.site(session.equipment["rover"]["sites"][name]).id
            for name in ("stereo_left", "stereo_right")
        ]
        self.target_site = m.site(self.interface["target"]["site"]).id
        self.target_geom = m.geom(self.interface["target"]["geom"]).id
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        included = {
            int(g) for g in enabled if m.geom(int(g)).name.startswith("rover__mast_")
        }
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if (a in included or b in included)
                and m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            ],
            dtype=int,
        )
        self.held = {
            key: (c, float(d.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator != self.a
        }
        self.signature = _model_signature(m)
        self.flags = int(m.opt.disableflags), int(m.opt.enableflags)
        self.masks = m.geom_contype.copy(), m.geom_conaffinity.copy()
        self.eq, self.warnings = d.eq_active.copy(), d.warning.number.copy()
        self.initial_time, self.dt, self.home = (
            float(d.time),
            float(m.opt.timestep),
            float(d.qpos[self.q]),
        )
        self.report = dict(
            success=False,
            failure=None,
            phases=[],
            captures=[],
            telemetry=[],
            physics_steps=0,
            observer_entries=0,
            geometry_checks=0,
            completed_checks=0,
            direct_queries=0,
            retained_pair_count=len(self.pairs),
            pairs_pruned=0,
            minimum_checked_clearance_lower_bound_m=None,
            maximum_tracking_error_rad=0.0,
            held_control_maximum_error={},
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            time_discontinuity_detected=False,
            physics_events=[],
            scene_sha256=self.ref["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_state=full_state(session),
            model_signature=self.signature,
        )

    def owned(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("NASA mast session model or state was replaced")

    def distances(self):
        m, d = self.m, self.d
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0)
        gaps = np.full(len(self.pairs), self.distance_cap)
        near = np.flatnonzero(np.linalg.norm(separation, axis=1) < self.distance_cap)
        for i in near:
            a, b = self.pairs[i]
            gaps[i] = min(
                float(
                    mujoco.mj_geomDistance(
                        m, d, int(a), int(b), self.distance_cap, None
                    )
                ),
                self.distance_cap,
            )
        return gaps, len(near)

    def observe(self):
        self.owned()
        m, d, r = self.m, self.d, self.report
        r["observer_entries"] += 1
        if self.flags != (int(m.opt.disableflags), int(m.opt.enableflags)) or not all(
            np.array_equal(a, b)
            for a, b in zip(self.masks, (m.geom_contype, m.geom_conaffinity))
        ):
            raise RuntimeError("collision_configuration_changed")
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.ctrl)):
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(d.warning.number, self.warnings) or not np.array_equal(
            d.eq_active, self.eq
        ):
            raise RuntimeError("physics_warning_or_equality_change")
        if abs(float(d.time) - self.initial_time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (c, target) in self.held.items():
            error = abs(self.session._measured(c) - target)
            name = ".".join(key)
            r["held_control_maximum_error"][name] = max(
                error, r["held_control_maximum_error"].get(name, 0)
            )
            tolerance = (
                0.0001 if c.units == "m" else 0.002 if c.mode == "position" else 0.02
            )
            if d.ctrl[c.actuator] != target or error > tolerance:
                raise RuntimeError("inactive_control_changed_or_drifted:" + name)
        if (
            not self.control.lower - 0.00002
            <= d.qpos[self.q]
            <= self.control.upper + 0.00002
        ):
            raise RuntimeError("pan_joint_limit_violation")
        error = abs(float(d.qpos[self.q] - d.ctrl[self.a]))
        r["maximum_tracking_error_rad"] = max(error, r["maximum_tracking_error_rad"])
        if error > 0.02 or abs(d.qvel[self.v]) > 0.35:
            raise RuntimeError("pan_tracking_or_velocity_limit")
        gaps, queries = self.distances()
        r["direct_queries"] += queries
        r["geometry_checks"] += 1
        bad = np.flatnonzero(gaps < self.minimum_gap)
        if len(bad):
            a, b = self.pairs[bad[0]]
            raise RuntimeError(
                f"measured_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
            )
        for camera, site in zip(self.cameras, self.camera_sites):
            if not np.allclose(
                d.cam_xpos[camera], d.site_xpos[site], rtol=0, atol=1e-10
            ):
                raise RuntimeError("mounted_camera_pose_changed")
        minimum = float(np.min(gaps))
        previous = r["minimum_checked_clearance_lower_bound_m"]
        r["minimum_checked_clearance_lower_bound_m"] = (
            minimum if previous is None else min(previous, minimum)
        )
        r["completed_checks"] += 1
        if r["physics_steps"] % max(1, round(0.02 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    time_s=float(d.time),
                    pan_rad=float(d.qpos[self.q]),
                    clearance_lower_bound_m=minimum,
                    camera_positions_m=d.cam_xpos[self.cameras].tolist(),
                )
            )

    def phase(self, name, target, ramp=5.0, settle=0.6):
        self.owned()
        _require_binding(self.session)
        if (
            not math.isfinite(target)
            or not self.control.lower <= target <= self.control.upper
        ):
            raise ValueError("Pan target outside original limits")
        m, d, r = self.m, self.d, self.report
        start = float(d.ctrl[self.a])
        n, hold = max(1, round(ramp / self.dt)), round(settle / self.dt)
        record = dict(name=name, target_rad=target, success=False, executed_steps=0)
        r["phases"].append(record)
        before = {
            k: r[k] for k in ("observer_entries", "geometry_checks", "completed_checks")
        }
        try:
            for i in range(n + hold):
                self.owned()
                t = min(1, (i + 1) / n)
                d.ctrl[self.a] = start + (target - start) * t * t * t * (
                    10 - 15 * t + 6 * t * t
                )
                if (
                    not np.isfinite(d.qfrc_applied).all()
                    or not np.isfinite(d.xfrc_applied).all()
                ):
                    raise RuntimeError("nonfinite_external_force")
                before_time = float(d.time)
                before_warnings = _warning_state(d)
                reset_inputs = {
                    key: getattr(d, key).copy()
                    for key in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                r["physics_steps"] += 1
                record["executed_steps"] += 1
                mujoco.mj_forward(m, d)
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, reset_inputs
                )
                if event is not None:
                    r["physics_events"].append(
                        dict(executed_step=r["physics_steps"], **event)
                    )
                    r.update(combine_reset_reports(r["physics_events"]))
                    r["time_discontinuity_detected"] = any(
                        not e["time_continuous"] for e in r["physics_events"]
                    )
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                self.observe()
            if abs(d.qpos[self.q] - target) > 0.0001 or abs(d.qvel[self.v]) > 0.0002:
                raise RuntimeError("pan_endpoint_not_reached_or_settled")
            record["success"] = True
        finally:
            record.update({k: r[k] - v for k, v in before.items()})
            record["coverage_complete"] = (
                record["executed_steps"] > 0
                and record["executed_steps"] == record["completed_checks"]
            )
            record["state"] = full_state(self.session)

    def capture(self):
        self.owned()
        _require_binding(self.session)
        m, d = self.m, self.d
        target = d.site_xpos[self.target_site].copy()
        records = []
        for camera in self.cameras:
            position, rotation = (
                d.cam_xpos[camera].copy(),
                d.cam_xmat[camera].reshape(3, 3).copy(),
            )
            local = rotation.T @ (target - position)
            depth = -float(local[2])
            if depth <= 0:
                raise RuntimeError("target_behind_camera")
            scale = 350 / math.tan(math.radians(m.cam_fovy[camera]) / 2)
            pixel = [560 + scale * local[0] / depth, 350 - scale * local[1] / depth]
            hit = np.array([-1], dtype=np.int32)
            direction = target - position
            distance = float(
                mujoco.mj_ray(
                    m,
                    d,
                    position,
                    direction / np.linalg.norm(direction),
                    np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8),
                    1,
                    int(m.cam_bodyid[camera]),
                    hit,
                )
            )
            visible = bool(
                int(hit[0]) == self.target_geom
                and 0 < pixel[0] < 1120
                and 0 < pixel[1] < 700
            )
            if not visible:
                raise RuntimeError("stereo_target_not_visible")
            records.append(
                dict(
                    camera=m.camera(camera).name,
                    camera_body=m.body(int(m.cam_bodyid[camera])).name,
                    camera_position_m=position.tolist(),
                    camera_rotation_matrix=rotation.reshape(-1).tolist(),
                    fovy_deg=float(m.cam_fovy[camera]),
                    resolution=[1120, 700],
                    target_site=self.interface["target"]["site"],
                    target_geom=self.interface["target"]["geom"],
                    target_position_m=target.tolist(),
                    target_pixel=list(map(float, pixel)),
                    projected_target_diameter_px=float(
                        self.interface["target"]["diameter_m"] * scale / depth
                    ),
                    depth_m=depth,
                    visible=visible,
                    ray_distance_m=distance,
                    first_hit=m.geom(int(hit[0])).name,
                )
            )
        actual_baseline = float(
            np.linalg.norm(d.cam_xpos[self.cameras[0]] - d.cam_xpos[self.cameras[1]])
        )
        if abs(actual_baseline - self.interface["baseline_m"]) > 1e-9:
            raise RuntimeError("stereo_baseline_changed")
        if not np.allclose(
            d.cam_xmat[self.cameras[0]], d.cam_xmat[self.cameras[1]], atol=1e-10, rtol=0
        ):
            raise RuntimeError("stereo_cameras_not_parallel")
        centers = np.array([x["target_pixel"] for x in records])
        if np.linalg.norm(centers.mean(axis=0) - [560, 350]) > 0.1:
            raise RuntimeError("target_not_registered_to_stereo_midpoint")
        self.report["captures"].append(
            dict(
                label="registered_stereo_target",
                views=records,
                measured_baseline_m=actual_baseline,
                horizontal_disparity_px=float(centers[0, 0] - centers[1, 0]),
                state=full_state(self.session),
                scope="Actual body-mounted pinhole geometry and target visibility; no calibrated stereo depth estimate",
            )
        )

    def execute(self):
        if self.session.equipment["rover"].get("nasa_rover_task"):
            raise ValueError(
                "The anchored mast protocol is historical; use inspect_nasa_mast(session) for the grounded parked-base protocol"
            )
        self.owned()
        _require_binding(self.session)
        m, d, r = self.m, self.d, self.report
        if (
            not math.isfinite(float(d.time))
            or not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.ctrl, d.act))
            or r["physics_steps"]
            or abs(d.qpos[self.q]) > 0.0001
            or abs(d.ctrl[self.a]) > 0.0001
            or np.max(abs(d.qvel)) > 0.01
        ):
            raise ValueError("Begin a fresh finite resting pan-at-zero episode")
        gaps, _ = self.distances()
        if np.any(gaps < self.minimum_gap):
            raise ValueError(
                "Initial mast/cache configuration lacks required clearance"
            )
        anchor = d.xanchor[self.control.joint]
        vector = d.site_xpos[self.target_site] - anchor
        target = math.atan2(float(vector[1]), float(vector[0]))
        try:
            self.phase("settle_held_rover", 0, ramp=0.002, settle=0.25)
            self.phase("align_stereo_target", target)
            self.capture()
            self.phase("return_pan_home", self.home)
            _require_binding(self.session)
            r["success"] = True
        except (RuntimeError, ValueError) as exc:
            r["failure"] = str(exc)
            if (
                self.session.model is m
                and self.session.data is d
                and d.model is m
                and np.isfinite(d.qpos[self.q])
            ):
                d.ctrl[self.a] = np.clip(
                    d.qpos[self.q], self.control.lower, self.control.upper
                )
                r["hold_requested"] = True
        complete = (
            r["physics_steps"] > 0 and r["physics_steps"] == r["completed_checks"]
        )
        r.update(
            coverage_complete=complete,
            unchecked_steps=r["physics_steps"] - r["completed_checks"],
            whole_run_clearance_lower_bound_m=(
                r["minimum_checked_clearance_lower_bound_m"] if complete else None
            ),
            model_unchanged=_model_signature(m) == self.signature,
            final_state=full_state(self.session),
            return_error_rad=abs(float(d.qpos[self.q]) - self.home),
            limitations=[
                "Authored stereo geometry and target, not NASA/OEM hardware or calibrated optics",
                "Fixed rover, retained inert cache caps and stationary wheels/arm; no navigation, grasp or sample transfer",
                "Every completed physics step checked; finite sampling is not continuous collision certification",
                "Exclusive session ownership required; a failure requests measured-position hold, not instantaneous stopping",
            ],
        )
        return r


def inspect_nasa_mast(session):
    if session.equipment["rover"].get("nasa_rover_task"):
        from real_labs.nasa_rover_task import inspect_grounded_nasa_mast

        return inspect_grounded_nasa_mast(session)
    return MastInspection(session).execute()


def main():
    import argparse, json
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene(SCENE)
    session = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = inspect_nasa_mast(session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report.get("operation") == "parked_mast_inspection":
        detail_fields = (
            "geometry_checks",
            "minimum_direct_m",
            "max_parked_mast_drift_m",
        )
    else:
        detail_fields = (
            "completed_checks",
            "whole_run_clearance_lower_bound_m",
            "return_error_rad",
        )
    fields = ("success", "failure", "physics_steps", *detail_fields)
    print({key: report[key] for key in fields})
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
