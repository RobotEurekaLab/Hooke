"""Measured two-fiducial registration through an instrument-mounted camera.

An already retained inert slide moves under its actual objective. Observations
are geometric pinhole projections and physical clearances, not calibrated focus,
microscope image formation or a scientific experiment. CPU pixel rendering is
an offline replay of the recorded full state.
"""

from copy import copy
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.kyoto_instrument_geometry import CAMERA

SCENE = "kyoto_yokokawa_nanometrics"
EQUIPMENT = "tirf_workspace"


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
            for k in ("camera", "camera_names", "mounted_cameras", "specimens")
        },
    )


def _require_binding(session):
    if session.data.model is not session.model:
        raise ValueError("Kyoto data must belong to its exact model object")
    expected = _binding()
    if (
        session.manifest.get("id") != SCENE
        or session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
    ):
        raise ValueError("Kyoto authored scene/control/equipment binding changed")
    if any(session.manifest.get(k) != v for k, v in expected["manifest"].items()):
        raise ValueError("Kyoto authored camera/specimen binding changed")
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError(
            "Kyoto authored geometry, collision or physics binding changed"
        )
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("Kyoto package hash does not match current source")
    return expected


def full_state(session):
    d = session.data
    return dict(
        time_s=float(d.time),
        qpos=d.qpos.tolist(),
        qvel=d.qvel.tolist(),
        ctrl=d.ctrl.tolist(),
        act=d.act.tolist(),
        qacc_warmstart=d.qacc_warmstart.tolist(),
        qfrc_applied=d.qfrc_applied.tolist(),
        xfrc_applied=d.xfrc_applied.tolist(),
        eq_active=d.eq_active.tolist(),
    )


class SlideInspection:
    """A source-bound dry route with all relevant enabled pairs retained."""

    distance_cap = 0.003
    minimum_gap = 0.0001

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.m, self.d = session, session.model, session.data
        m, d = self.m, self.d
        if not np.isfinite(float(d.time)):
            raise ValueError("Inspection requires finite simulation time")
        self.item = session.equipment[EQUIPMENT]
        self.interface = self.item["inspection_interface"]
        self.controls = [
            session.controls[EQUIPMENT, key]
            for key in ("slide_x", "slide_y", "objective_focus")
        ]
        self.a = np.array([c.actuator for c in self.controls])
        self.q = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.v = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.camera = m.camera(CAMERA).id
        self.camera_body = int(m.cam_bodyid[self.camera])
        self.camera_site = m.site(self.item["sites"]["inspection_camera"]).id
        self.face = m.geom(self.interface["objective_face"]).id
        self.face_site = m.site(self.interface["objective_site"]).id
        self.glass = m.geom(self.interface["glass_geom"]).id
        self.signature = _model_signature(m)
        self.home = d.qpos[self.q].copy()
        self.time, self.dt = float(d.time), float(m.opt.timestep)
        self.warnings, self.eq = d.warning.number.copy(), d.eq_active.copy()
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.held = {
            key: (c, float(d.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator not in self.a
        }
        roots = {int(m.jnt_bodyid[c.joint]) for c in self.controls}
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in roots
                    or m.body_weldid[m.geom_bodyid[b]] in roots
                )
            ],
            dtype=int,
        )
        # Geometric visibility uses an independent clone, excluding transparent
        # glass and hidden proxy geometry from opaque rays only. Live geometry,
        # contact masks and the actual camera remain untouched.
        self.ray_model = copy(m)
        alpha = m.geom_rgba[:, 3].copy()
        for i, material in enumerate(m.geom_matid):
            if material >= 0:
                alpha[i] = min(alpha[i], m.mat_rgba[material, 3])
        self.ray_model.geom_group[:] = np.where(
            (alpha >= 0.95) & (m.geom_group < 3), 0, 1
        )
        self.ray_data = mujoco.MjData(self.ray_model)
        self.report = dict(
            success=False,
            failure=None,
            phases=[],
            captures=[],
            physics_steps=0,
            observer_entries=0,
            geometry_checks=0,
            completed_checks=0,
            direct_queries=0,
            retained_pair_count=len(self.pairs),
            pairs_pruned=0,
            minimum_checked_clearance_lower_bound_m=None,
            maximum_tracking_error_m=[0.0, 0.0, 0.0],
            held_control_maximum_error={},
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            time_discontinuity_detected=False,
            physics_events=[],
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_signature=self.signature,
            initial_state=full_state(session),
            telemetry=[],
            scope="Actual dry registration of two rigidly retained inert fiducials, measured lens-to-glass clearance, body-mounted pinhole observations and return; no calibrated optics or scientific process",
        )

    def owned(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Kyoto session model or data was replaced")

    def working_gap(self):
        return float(
            self.d.geom_xpos[self.glass, 2]
            - self.m.geom_size[self.glass, 2]
            - self.d.site_xpos[self.face_site, 2]
        )

    def observe(self):
        self.owned()
        m, d, r = self.m, self.d, self.report
        r["observer_entries"] += 1
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("collision_configuration_changed")
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.ctrl)):
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(d.warning.number, self.warnings):
            raise RuntimeError("physics_warning")
        if not np.array_equal(d.eq_active, self.eq):
            raise RuntimeError("equality_state_changed")
        if abs(d.time - self.time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (control, target) in self.held.items():
            error = abs(self.session._measured(control) - target)
            r["held_control_maximum_error"][".".join(key)] = max(
                error, r["held_control_maximum_error"].get(".".join(key), 0.0)
            )
            if d.ctrl[control.actuator] != target or error > 0.0001:
                raise RuntimeError("inactive_control_changed_or_drifted")
        if np.any(d.qpos[self.q] < self.lower - 0.00002) or np.any(
            d.qpos[self.q] > self.upper + 0.00002
        ):
            raise RuntimeError("joint_limit_violation")
        error = np.abs(d.qpos[self.q] - d.ctrl[self.a])
        r["maximum_tracking_error_m"] = np.maximum(
            error, r["maximum_tracking_error_m"]
        ).tolist()
        if np.any(error > 0.001):
            raise RuntimeError("trajectory_tracking_limit")
        if np.any(np.abs(d.qvel[self.v]) > 0.025):
            raise RuntimeError("instrument_velocity_limit")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        minimum, violation = self.distance_cap, None
        for a, b in self.pairs[np.linalg.norm(separation, axis=1) < self.distance_cap]:
            gap = float(
                mujoco.mj_geomDistance(m, d, int(a), int(b), self.distance_cap, None)
            )
            r["direct_queries"] += 1
            minimum = min(minimum, gap)
            if gap < self.minimum_gap:
                violation = (
                    f"rigid_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
                )
        r["geometry_checks"] += 1
        if violation:
            raise RuntimeError(violation)
        if not np.allclose(
            d.cam_xpos[self.camera], d.site_xpos[self.camera_site], rtol=0, atol=1e-10
        ):
            raise RuntimeError("mounted_camera_pose_changed")
        if self.working_gap() < 0.0009:
            raise RuntimeError("objective_to_slide_clearance_lost")
        previous = r["minimum_checked_clearance_lower_bound_m"]
        r["minimum_checked_clearance_lower_bound_m"] = (
            minimum if previous is None else min(previous, minimum)
        )
        # Complete means all invariants above, not merely entry or geometry.
        r["completed_checks"] += 1
        if r["physics_steps"] % max(1, round(0.02 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    time_s=float(d.time),
                    position_m=d.qpos[self.q].tolist(),
                    camera_position_m=d.cam_xpos[self.camera].tolist(),
                    working_clearance_m=self.working_gap(),
                    clearance_lower_bound_m=minimum,
                )
            )

    def phase(self, name, target, ramp, settle):
        self.owned()
        _require_binding(self.session)
        m, d, r = self.m, self.d, self.report
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Inspection target exceeds the original axis limits")
        if not np.isfinite(float(d.time)):
            raise ValueError("Phase requires finite simulation time before commands")
        start = d.ctrl[self.a].copy()
        held_axes = np.flatnonzero(np.abs(start - target) < 1e-12)
        n, hold = max(1, round(ramp / self.dt)), round(settle / self.dt)
        record = dict(
            name=name,
            target_m=target.tolist(),
            success=False,
            executed_steps=0,
            held_axis_indices=held_axes.tolist(),
            maximum_held_axis_error_m=0.0,
        )
        r["phases"].append(record)
        counts = [
            r[k] for k in ("observer_entries", "geometry_checks", "completed_checks")
        ]
        try:
            for i in range(n + hold):
                self.owned()
                t = min(1.0, (i + 1) / n)
                smooth = t * t * t * (10 - 15 * t + 6 * t * t)
                d.ctrl[self.a] = start + smooth * (target - start)
                if (
                    not np.isfinite(d.qfrc_applied).all()
                    or not np.isfinite(d.xfrc_applied).all()
                ):
                    raise RuntimeError("nonfinite_external_force")
                before_time, before_warnings = float(d.time), _warning_state(d)
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
                    record.update(combine_reset_reports([event]))
                    record["physics_event"] = event
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                # Phase-level hold is included in completion, even on the
                # first step that fails it; count nothing before this check.
                held_error = (
                    float(np.max(np.abs(d.qpos[self.q[held_axes]] - target[held_axes])))
                    if len(held_axes)
                    else 0.0
                )
                record["maximum_held_axis_error_m"] = max(
                    record["maximum_held_axis_error_m"], held_error
                )
                if held_error > 0.0001:
                    raise RuntimeError("held_axis_drift")
                self.observe()
            if np.any(np.abs(d.qpos[self.q] - target) > 0.00002):
                raise RuntimeError("endpoint_tracking_tolerance")
            if np.any(np.abs(d.qvel[self.v]) > 0.0001):
                raise RuntimeError("endpoint_not_settled")
            record["success"] = True
        finally:
            for key, before in zip(
                ("observer_entries", "geometry_checks", "completed_checks"), counts
            ):
                record[key] = r[key] - before
            record["coverage_complete"] = (
                record["executed_steps"] > 0
                and record["executed_steps"] == record["completed_checks"]
            )
            record["state"] = full_state(self.session)
            record["working_clearance_m"] = self.working_gap()

    def capture(self, label, site_name, geom_name, diameter):
        self.owned()
        _require_binding(self.session)
        m, d = self.m, self.d
        camera = d.cam_xpos[self.camera].copy()
        rotation = d.cam_xmat[self.camera].reshape(3, 3).copy()
        target = d.site_xpos[m.site(site_name).id].copy()
        local = rotation.T @ (target - camera)
        depth = -float(local[2])
        if depth <= 0:
            raise RuntimeError("target_behind_instrument_camera")
        width, height = 1120, 700
        scale = height / (2 * np.tan(np.deg2rad(m.cam_fovy[self.camera]) / 2))
        pixel = [
            float(width / 2 + scale * local[0] / depth),
            float(height / 2 - scale * local[1] / depth),
        ]
        rd = self.ray_data
        rd.qpos[:] = d.qpos
        mujoco.mj_forward(self.ray_model, rd)
        vector = target - camera
        hit = np.array([-1], dtype=np.int32)
        distance = float(
            mujoco.mj_ray(
                self.ray_model,
                rd,
                camera,
                vector / np.linalg.norm(vector),
                np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                1,
                self.camera_body,
                hit,
            )
        )
        gap = self.working_gap()
        record = dict(
            label=label,
            camera=CAMERA,
            camera_body=m.body(self.camera_body).name,
            camera_position_m=camera.tolist(),
            camera_rotation_matrix=rotation.reshape(-1).tolist(),
            fovy_deg=float(m.cam_fovy[self.camera]),
            resolution=[width, height],
            target_site=site_name,
            target_geom=geom_name,
            target_position_m=target.tolist(),
            target_pixel=pixel,
            target_diameter_px=float(diameter * scale / depth),
            depth_m=depth,
            axis_error_m=float(np.linalg.norm(local[:2])),
            physical_working_clearance_m=gap,
            ray_distance_m=distance,
            first_opaque_hit=m.geom(int(hit[0])).name if hit[0] >= 0 else None,
            visible=int(hit[0]) == m.geom(geom_name).id
            and 0 < pixel[0] < width
            and 0 < pixel[1] < height,
            state=full_state(self.session),
            capture_kind="Measured entrance-pupil geometric view; opaque ray skips transparent glass on a private clone; no refraction or microscope image-formation calibration",
        )
        if not record["visible"] or record["axis_error_m"] > 0.00002:
            raise RuntimeError("fiducial_not_registered_or_visible")
        if abs(gap - self.interface["inspection_working_clearance_m"]) > 0.00002:
            raise RuntimeError("authored_working_clearance_not_reached")
        self.report["captures"].append(record)

    def execute(self):
        self.owned()
        _require_binding(self.session)
        m, d, r = self.m, self.d, self.report
        if (
            r["physics_steps"]
            or not np.isfinite(float(d.time))
            or not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.ctrl))
            or np.max(np.abs(d.qpos[self.q])) > 0.0001
            or np.max(np.abs(d.ctrl[self.a])) > 0.0001
            or np.max(np.abs(d.qvel)) > 0.01
        ):
            raise ValueError("Begin a fresh, centered, resting inspection episode")
        try:
            self.phase("settle", (0, 0, 0), 0.002, 0.25)
            self.phase("register_secondary_fiducial", (-0.010, 0.006, 0), 1.5, 0.4)
            self.phase("approach_secondary_clearance", (-0.010, 0.006, 0.002), 1.0, 0.3)
            self.capture(
                "secondary_fiducial",
                self.interface["registration_site"],
                self.interface["registration_geom"],
                0.001,
            )
            self.phase("retract_secondary", (-0.010, 0.006, 0), 1.0, 0.3)
            self.phase("register_primary_fiducial", (0, 0, 0), 1.5, 0.4)
            self.phase("approach_primary_clearance", (0, 0, 0.002), 1.0, 0.3)
            self.capture(
                "primary_fiducial",
                self.item["sample_site"],
                self.interface["target_geom"],
                0.002,
            )
            self.phase("return_to_home", (0, 0, 0), 1.0, 0.3)
            _require_binding(self.session)
            r["success"] = True
        except (RuntimeError, ValueError) as exc:
            r["failure"] = str(exc)
            if (
                self.session.model is m
                and self.session.data is d
                and d.model is m
                and np.isfinite(d.qpos[self.q]).all()
            ):
                d.ctrl[self.a] = np.clip(d.qpos[self.q], self.lower, self.upper)
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
            return_error_m=np.abs(d.qpos[self.q] - self.home).tolist(),
            limitations=[
                "Rigid retained inert slide; no loading/grasping or biological procedure",
                "Authored pinhole observation and working clearance, not optical focus or image formation",
                "Every completed physics step checked; no continuous-time collision guarantee",
                "Source-altered environments reject before commands; concurrent writers unsupported",
                "Failure requests measured-position hold, not instantaneous stopping",
            ],
        )
        return r


def inspect_kyoto(session):
    return SlideInspection(session).execute()


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
    result = inspect_kyoto(session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        {
            k: result[k]
            for k in (
                "success",
                "failure",
                "physics_steps",
                "completed_checks",
                "direct_queries",
                "whole_run_clearance_lower_bound_m",
                "return_error_m",
            )
        }
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
