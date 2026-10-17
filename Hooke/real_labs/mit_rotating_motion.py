"""Measured dry target indexing through a body-mounted inspection camera.

Only the two original actuators are commanded. The target is fixed to the
rotor. Camera observations record measured pose, geometric visibility and
projection, not calibrated optics or fluid dynamics. Pixels are rendered
separately from the recorded state with the CPU snapshot workflow.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.mit_rotating_geometry import CAMERA

SCENE = "mit_paoc_rotating_fluids"
EQUIPMENT = "rotating_table"


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
    s = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=s.controls,
        equipment=s.equipment,
        manifest={
            k: manifest.get(k)
            for k in ("camera", "camera_names", "mounted_cameras", "specimens")
        },
    )


def _require_binding(session):
    if not np.isfinite(float(session.data.time)):
        raise ValueError("Inspection requires finite simulation time")
    if session.data.model is not session.model:
        raise ValueError("MIT inspection requires data owned by its exact model")
    b = _binding()
    if (
        session.manifest.get("id") != SCENE
        or session.controls != b["controls"]
        or session.equipment != b["equipment"]
    ):
        raise ValueError("MIT authored control/equipment binding changed")
    if any(session.manifest.get(k) != v for k, v in b["manifest"].items()):
        raise ValueError("MIT authored camera/specimen binding changed")
    if _model_signature(session.model) not in b["signatures"]:
        raise ValueError("MIT authored geometry, contact or physics binding changed")
    if session.manifest.get("scene_sha256", b["scene_sha256"]) != b["scene_sha256"]:
        raise ValueError("MIT package hash does not match current source")
    return b


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


class RotatingInspection:
    """Source-bound, fixed dry route; all nearby enabled pairs checked each step."""

    cap = 0.003
    clearance = 0.0003

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.m, self.d = session, session.model, session.data
        m, d = self.m, self.d
        self.controls = [
            session.controls[EQUIPMENT, key]
            for key in ("table_rotation", "camera_traverse")
        ]
        self.a = np.array([c.actuator for c in self.controls])
        self.q = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.v = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.camera = m.camera(CAMERA).id
        self.target = m.site(session.equipment[EQUIPMENT]["sample_site"]).id
        self.target_geom = m.geom(
            session.equipment[EQUIPMENT]["inspection_interface"]["target_geom"]
        ).id
        self.camera_body = int(m.cam_bodyid[self.camera])
        self.signature = _model_signature(m)
        self.held = {
            key: (c, float(d.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator not in self.a
        }
        self.home = d.qpos[self.q].copy()
        self.time = float(d.time)
        self.dt = float(m.opt.timestep)
        self.warnings = d.warning.number.copy()
        self.eq = d.eq_active.copy()
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        roots = {int(m.jnt_bodyid[c.joint]) for c in self.controls}
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        all_pairs = [
            (int(a), int(b))
            for i, a in enumerate(enabled)
            for b in enabled[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in roots
                or m.body_weldid[m.geom_bodyid[b]] in roots
            )
        ]
        # Exact source binding establishes two independent world-anchored axes.
        # Spheres enclose the whole hinge rotation and the whole slide interval,
        # not just the chosen route. Unknown moving bodies are never pruned.
        by_body = {int(m.jnt_bodyid[c.joint]): c for c in self.controls}

        def sphere(g):
            weld = int(m.body_weldid[m.geom_bodyid[g]])
            if weld == 0:
                return d.geom_xpos[g].copy(), float(m.geom_rbound[g])
            if weld not in by_body:
                return None
            c = by_body[weld]
            if m.body_weldid[m.body_parentid[weld]] != 0 or m.body_jntnum[weld] != 1:
                raise ValueError("Expected independent world-anchored inspection axes")
            if m.jnt_type[c.joint] == mujoco.mjtJoint.mjJNT_HINGE:
                center = d.xanchor[c.joint].copy()
                return center, float(
                    np.linalg.norm(d.geom_xpos[g] - center) + m.geom_rbound[g]
                )
            if m.jnt_type[c.joint] != mujoco.mjtJoint.mjJNT_SLIDE:
                raise ValueError("Unsupported inspection joint")
            midpoint = (c.lower + c.upper) / 2
            center = d.geom_xpos[g] + d.xaxis[c.joint] * (
                midpoint - d.qpos[m.jnt_qposadr[c.joint]]
            )
            return center, float(m.geom_rbound[g] + (c.upper - c.lower) / 2)

        spheres = [sphere(g) for g in range(m.ngeom)]
        self.pairs = np.array(
            [
                (a, b)
                for a, b in all_pairs
                if spheres[a] is None
                or spheres[b] is None
                or np.linalg.norm(spheres[a][0] - spheres[b][0])
                <= spheres[a][1] + spheres[b][1] + self.cap + 1e-9
            ],
            dtype=int,
        )
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
            original_pair_count=len(all_pairs),
            retained_pair_count=len(self.pairs),
            minimum_clearance_lower_bound_m=self.cap,
            maximum_tracking_error=[0.0, 0.0],
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
            scope="Measured dry rotor indexing and centered mounted-camera geometric observations; fixed target, no transfer or fluid process",
        )

    def owned(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("MIT session model or data was replaced")

    def observe(self):
        self.owned()
        m, d, r = self.m, self.d, self.report
        r["observer_entries"] += 1
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("contact_configuration_changed")
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.ctrl)):
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(d.warning.number, self.warnings):
            raise RuntimeError("physics_warning")
        if not np.array_equal(d.eq_active, self.eq):
            raise RuntimeError("equality_state_changed")
        if abs(d.time - self.time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (c, target) in self.held.items():
            error = abs(self.session._measured(c) - target)
            r["held_control_maximum_error"][".".join(key)] = max(
                error, r["held_control_maximum_error"].get(".".join(key), 0.0)
            )
            if d.ctrl[c.actuator] != target or error > (
                0.0005 if c.units == "m" else 0.003
            ):
                raise RuntimeError("inactive_control_changed_or_drifted")
        if np.any(d.qpos[self.q] < self.lower - 0.0001) or np.any(
            d.qpos[self.q] > self.upper + 0.0001
        ):
            raise RuntimeError("joint_limit_violation")
        error = np.abs(d.qpos[self.q] - d.ctrl[self.a])
        r["maximum_tracking_error"] = np.maximum(
            error, r["maximum_tracking_error"]
        ).tolist()
        if np.any(error > (0.25, 0.02)):
            raise RuntimeError("trajectory_tracking_limit")
        if np.any(np.abs(d.qvel[self.v]) > (1.0, 0.4)):
            raise RuntimeError("mechanism_velocity_limit")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        minimum = self.cap
        violation = None
        for a, b in self.pairs[np.linalg.norm(separation, axis=1) < self.cap]:
            gap = float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap, None))
            r["direct_queries"] += 1
            minimum = min(minimum, gap)
            if gap < self.clearance:
                violation = (
                    f"rigid_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
                )
        r["geometry_checks"] += 1
        if violation:
            raise RuntimeError(violation)
        # The camera is an actual child of the moving body, not a renamed fixed
        # view. Its compiled pose must continue matching the physical lens site.
        site = m.site(self.session.equipment[EQUIPMENT]["sites"]["camera_center"]).id
        if not np.allclose(
            d.cam_xpos[self.camera], d.site_xpos[site], rtol=0, atol=1e-10
        ):
            raise RuntimeError("mounted_camera_pose_changed")
        r["minimum_clearance_lower_bound_m"] = min(
            r["minimum_clearance_lower_bound_m"], minimum
        )
        r["completed_checks"] += 1
        if r["physics_steps"] % max(1, round(0.02 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    time_s=float(d.time),
                    position=d.qpos[self.q].tolist(),
                    camera_position_m=d.cam_xpos[self.camera].tolist(),
                    clearance_lower_bound_m=minimum,
                )
            )

    def phase(self, name, target, ramp, settle):
        self.owned()
        _require_binding(self.session)
        m, d, r = self.m, self.d, self.report
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Target outside retained inspection limits")
        start = d.ctrl[self.a].copy()
        n = max(1, round(ramp / self.dt))
        hold = round(settle / self.dt)
        leg = dict(
            name=name,
            target=target.tolist(),
            success=False,
            executed_steps=0,
            geometry_checks=0,
            completed_checks=0,
        )
        r["phases"].append(leg)
        g0, c0 = r["geometry_checks"], r["completed_checks"]
        try:
            for i in range(n + hold):
                self.owned()
                t = min(1.0, (i + 1) / n)
                smooth = t * t * t * (10 - 15 * t + 6 * t * t)
                d.ctrl[self.a] = start + smooth * (target - start)
                before_time = float(d.time)
                before_warnings = _warning_state(d)
                reset_inputs = {
                    key: getattr(d, key).copy()
                    for key in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                r["physics_steps"] += 1
                leg["executed_steps"] += 1
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
            if np.any(np.abs(d.qpos[self.q] - target) > (0.002, 0.0001)):
                raise RuntimeError("endpoint_tracking_tolerance")
            if np.any(np.abs(d.qvel[self.v]) > (0.003, 0.0005)):
                raise RuntimeError("endpoint_not_settled")
            leg["success"] = True
        finally:
            leg.update(
                geometry_checks=r["geometry_checks"] - g0,
                completed_checks=r["completed_checks"] - c0,
                state=full_state(self.session),
            )

    def capture(self, label):
        self.owned()
        _require_binding(self.session)
        m, d = self.m, self.d
        position = d.cam_xpos[self.camera].copy()
        rotation = d.cam_xmat[self.camera].reshape(3, 3).copy()
        target = d.site_xpos[self.target].copy()
        local = rotation.T @ (target - position)
        depth = -local[2]
        width, height = 1120, 700
        fovy = float(m.cam_fovy[self.camera])
        scale = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        pixels = [
            float(width / 2 + scale * local[0] / depth),
            float(height / 2 - scale * local[1] / depth),
        ]
        ray = (target - position) / np.linalg.norm(target - position)
        hit = np.array([-1], dtype=np.int32)
        distance = float(
            mujoco.mj_ray(m, d, position, ray, None, 1, self.camera_body, hit)
        )
        record = dict(
            label=label,
            camera=CAMERA,
            camera_body=m.body(self.camera_body).name,
            camera_position_m=position.tolist(),
            camera_rotation_matrix=rotation.reshape(-1).tolist(),
            fovy_deg=fovy,
            resolution=[width, height],
            target_position_m=target.tolist(),
            target_pixel=pixels,
            target_diameter_px=float(2 * 0.045 * scale / depth),
            depth_m=float(depth),
            axis_error_m=float(np.linalg.norm(local[:2])),
            ray_distance_m=distance,
            first_hit=m.geom(int(hit[0])).name if hit[0] >= 0 else None,
            visible=int(hit[0]) == self.target_geom
            and depth > 0
            and 0 < pixels[0] < width
            and 0 < pixels[1] < height,
            state=full_state(self.session),
            capture_kind="Geometric observation from the actual moving MuJoCo camera; CPU pixel snapshot is an offline replay of this recorded state",
        )
        if not record["visible"] or record["axis_error_m"] > 0.0005:
            raise RuntimeError("mounted_camera_target_not_centered_or_visible")
        self.report["captures"].append(record)

    def execute(self):
        self.owned()
        _require_binding(self.session)
        m, d, r = self.m, self.d, self.report
        if (
            r["physics_steps"]
            or not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or not np.isfinite(d.ctrl).all()
            or np.max(np.abs(d.qpos[self.q])) > 0.0001
            or np.max(np.abs(d.ctrl[self.a])) > 0.0001
            or np.max(np.abs(d.qvel)) > 0.01
        ):
            raise ValueError("Start a fresh centered resting episode")
        try:
            self.phase("settle", (0, 0), 0.002, 0.25)
            self.phase("align_first_target", (0, 0.22), 2.5, 0.6)
            self.capture("first_alignment")
            self.phase("recenter_camera", (0, 0), 2.5, 0.6)
            self.phase("index_half_turn", (np.pi, 0), 10.0, 1.0)
            self.phase("align_second_target", (np.pi, -0.22), 2.5, 0.6)
            self.capture("half_turn_alignment")
            self.phase("return_camera", (np.pi, 0), 2.5, 0.6)
            self.phase("return_rotor", (0, 0), 10.0, 1.0)
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
                r["minimum_clearance_lower_bound_m"] if complete else None
            ),
            model_unchanged=_model_signature(m) == self.signature,
            final_state=full_state(self.session),
            return_error=np.abs(d.qpos[self.q] - self.home).tolist(),
            limitations=[
                "Original bounded position drives, not continuous speed or fluid rotation",
                "Rigid fastened target; no loading or free sample motion",
                "Authored ideal bearing clearances, dimensions and camera optics",
                "Each executed step observed; no continuous-time clearance guarantee",
                "Modified source environments are rejected before commands; concurrent writers are unsupported",
                "Failure requests measured-position hold, not instantaneous stopping",
            ],
        )
        return r


def inspect_mit(session):
    return RotatingInspection(session).execute()


def main():
    import argparse, json
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene(SCENE)
    s = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = inspect_mit(s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        {
            k: report[k]
            for k in (
                "success",
                "failure",
                "physics_steps",
                "completed_checks",
                "direct_queries",
                "original_pair_count",
                "retained_pair_count",
                "whole_run_clearance_lower_bound_m",
                "return_error",
            )
        }
    )
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
