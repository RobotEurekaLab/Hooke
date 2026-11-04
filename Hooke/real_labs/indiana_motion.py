"""Measured dry carrier registration and geometric objective inspection.

The actuator-driven operation starts with an already loaded free inert carrier.
It does not implement optical autofocus, imaging physics or fluid procedures.
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


@lru_cache(maxsize=1)
def _authored_binding():
    """Bind the generated source and its portable mesh-path representation."""
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene("indiana_microfluidics")
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml,
        assets={item["file"]: Path(item["source"]).read_bytes() for item in resources},
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        camera=manifest["camera"],
        specimens=manifest["specimens"],
    )


def _require_binding(session):
    """Reject unsupported hardware before any control, hold or physics mutation."""
    if session.data.model is not session.model:
        raise ValueError("Session data must belong to its exact model object")
    expected = _authored_binding()
    if session.manifest.get("id") != "indiana_microfluidics":
        raise ValueError("Inspection requires its authored Indiana scene")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or session.manifest.get("camera") != expected["camera"]
        or session.manifest.get("specimens") != expected["specimens"]
    ):
        raise ValueError(
            "Authored control, equipment, camera or specimen binding changed"
        )
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError(
            "Authored geometry, collision, constraints or physics binding changed"
        )
    declared = session.manifest.get("scene_sha256")
    if declared is not None and declared != expected["scene_sha256"]:
        raise ValueError("Manifest scene hash does not match the authored model")
    return expected


class _Inspection:
    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session = session
        self.m, self.d = session.model, session.data
        m, d = self.m, self.d
        self.item = session.equipment["microfluidic_station"]
        self.interface = self.item.get("inspection_interface")
        if (
            not self.interface
            or self.interface.get("services_connected_to_chip") is not False
        ):
            raise ValueError(
                "Requires the authored disconnected dry inspection interface"
            )
        self.controls = [
            session.controls["microfluidic_station", key]
            for key in ("stage_x", "stage_y", "focus")
        ]
        if any(c.mode != "position" or c.units != "m" for c in self.controls):
            raise ValueError(
                "Registration requires three bounded linear position servos"
            )
        self.a = np.array([c.actuator for c in self.controls])
        self.q = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.v = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        if not np.allclose(
            d.xaxis[[c.joint for c in self.controls]], np.eye(3), rtol=0, atol=1e-12
        ):
            raise ValueError("Inspection requires the authored Cartesian joint frames")
        if not np.all(m.actuator_forcelimited[self.a]) or not np.allclose(
            m.actuator_forcerange[self.a],
            ((-0.2, 0.2), (-0.2, 0.2), (-120, 120)),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Retain the authored finite stage and focus force limits")
        self.carrier = m.body(self.item["process"]["sample_body"]).id
        cj = m.body_jntadr[self.carrier]
        if (
            m.body_jntnum[self.carrier] != 1
            or m.jnt_type[cj] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(
                "Inspection carrier must retain its six free degrees of freedom"
            )
        for i, kind in enumerate(m.eq_type):
            if kind in (
                mujoco.mjtEq.mjEQ_CONNECT,
                mujoco.mjtEq.mjEQ_WELD,
            ) and self.carrier in (m.eq_obj1id[i], m.eq_obj2id[i]):
                raise ValueError(
                    "Inspection carrier cannot be welded or connected by equality"
                )
        self.sample = m.geom("microfluidic_station__specimen__collision").id
        self.site = m.site(self.item["sample_site"]).id
        self.objective = m.site(self.interface["objective_site"]).id
        self.face = m.geom(self.interface["objective_face"]).id
        self.slide = m.geom(self.interface["slide_geom"]).id
        focus_body = m.jnt_bodyid[self.controls[2].joint]
        stage_body = m.jnt_bodyid[self.controls[1].joint]
        if (
            m.geom_bodyid[self.sample] != self.carrier
            or m.site_bodyid[self.objective] != focus_body
            or m.geom_bodyid[self.face] != focus_body
            or m.site_bodyid[self.site] != stage_body
            or m.geom_bodyid[self.slide] != stage_body
        ):
            raise ValueError(
                "Inspection datums and contact geometry must belong to their actual moving frames"
            )
        if not np.allclose(
            d.site_xpos[self.objective],
            d.geom_xpos[self.face] + (0, 0, m.geom_size[self.face, 1]),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Objective datum must be on the physical lens face")
        names = self.interface.get("required_contact_geoms", [])
        if len(names) < 40:
            raise ValueError("Missing authored guide collision interface")
        self.required = [m.geom(n).id for n in names]
        self.support = m.geom("microfluidic_station__carrier_bearing_pad").id
        self.required += [self.sample, self.support]
        self.pocket = [
            m.geom(f"microfluidic_station__carrier_guide_{axis}_{sign}").id
            for axis in (0, 1)
            for sign in (-1, 1)
        ]
        self.required += self.pocket
        if any(
            m.geom_contype[g] != 1 or m.geom_conaffinity[g] != 1 for g in self.required
        ):
            raise ValueError(
                "Authored guide, objective and carrier collision coverage must remain enabled"
            )
        for flag in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            if int(m.opt.disableflags) & int(flag):
                raise ValueError(
                    "Contacts, constraints, limits and actuation must remain enabled"
                )
        if m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
            raise ValueError("Contact overrides alter the inspection mechanics")
        self.dt = float(m.opt.timestep)
        if (
            not 0 < self.dt <= 0.002
            or not np.isfinite(float(d.time))
            or not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or np.max(np.abs(d.qpos[self.q])) > 0.0001
            or np.max(np.abs(d.qvel)) > 0.01
        ):
            raise ValueError(
                "Begin at the authored centered resting state with timestep at most2ms"
            )
        self.held = {
            k: (c, float(d.ctrl[c.actuator]))
            for k, c in session.controls.items()
            if c.actuator not in self.a
        }
        self.initial_eq = d.eq_active.copy()
        self.warning = d.warning.number.copy()
        self.time = float(d.time)
        self.signature = _model_signature(m)
        self.home = d.qpos[self.q].copy()
        self.relative_carrier = d.xpos[self.carrier] - d.site_xpos[self.site]
        self.initial_rotation = d.xmat[self.carrier].copy()
        roots = {m.jnt_bodyid[c.joint] for c in self.controls}
        self.bodies = {self.carrier}
        for body in range(m.nbody):
            ancestor = body
            while ancestor and ancestor not in roots:
                ancestor = m.body_parentid[ancestor]
            if ancestor in roots:
                self.bodies.add(body)
        enabled = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("microfluidic_station__")
            and m.geom_contype[g]
            and m.geom_conaffinity[g]
        ]
        self.filtered = []
        for i, a in enumerate(enabled):
            wa = m.body_weldid[m.geom_bodyid[a]]
            for b in enabled[i + 1 :]:
                wb = m.body_weldid[m.geom_bodyid[b]]
                if not wa or not wb or wa == wb:
                    continue
                if (
                    m.body_weldid[m.body_parentid[wa]] == wb
                    or m.body_weldid[m.body_parentid[wb]] == wa
                ):
                    self.filtered.append((a, b))
        self.allowed = {
            frozenset((self.sample, g)) for g in [self.support] + self.pocket
        }
        self.report = dict(
            success=False,
            failure=None,
            phases=[],
            physics_steps=0,
            observer_entries=0,
            geometry_checks=0,
            completed_checks=0,
            preflight_entries=0,
            preflight_geometry_checks=0,
            preflight_completed_checks=0,
            forbidden_contacts=[],
            parent_filtered_pair_count=len(self.filtered),
            parent_filtered_distance_queries=0,
            minimum_parent_filtered_distance_m=None,
            maximum_carrier_tracking_error_m=0.0,
            maximum_carrier_penetration_m=0.0,
            held_control_maximum_error={},
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            time_discontinuity_detected=False,
            physics_events=[],
            model_signature=self.signature,
            telemetry=[],
            telemetry_interval_s=0.020,
            authored_scene_sha256=self.binding["scene_sha256"],
            model_binding="Canonical source/portable model and control/equipment/camera/specimen metadata before mutation; exact model/data ownership each step; compiled-model signature before each phase and after execution. Altered environments require separate validation and are rejected.",
            interpretation="Measured registration of an already-loaded free inert carrier and an authored geometric objective-to-slide distance. No calibrated focus, optical response, pressure-connected services, fluid transfer or biological procedure.",
        )

    def owned(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise RuntimeError("session_model_or_data_replaced")

    def observe(self):
        m, d, r = self.m, self.d, self.report
        r["observer_entries"] += 1
        self.owned()
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(d.warning.number, self.warning):
            raise RuntimeError("physics_warning")
        if not np.array_equal(d.eq_active, self.initial_eq):
            raise RuntimeError("equality_state_changed")
        if abs(d.time - self.time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (c, target) in self.held.items():
            error = abs(self.session._measured(c) - target)
            r["held_control_maximum_error"][".".join(key)] = max(
                error, r["held_control_maximum_error"].get(".".join(key), 0.0)
            )
            if d.ctrl[c.actuator] != target or error > (
                0.002 if c.units == "m" else 0.01
            ):
                raise RuntimeError("inactive_control_changed_or_drifted")
        for c, address in zip(self.controls, self.q):
            if not c.lower - 0.0001 <= d.qpos[address] <= c.upper + 0.0001:
                raise RuntimeError("joint_limit_violation")
        error = float(
            np.linalg.norm(
                d.xpos[self.carrier] - d.site_xpos[self.site] - self.relative_carrier
            )
        )
        r["maximum_carrier_tracking_error_m"] = max(
            error, r["maximum_carrier_tracking_error_m"]
        )
        if error > 0.0005:
            raise RuntimeError("carrier_displaced_or_not_retained")
        rotation = d.xmat[self.carrier].reshape(3, 3).T @ self.initial_rotation.reshape(
            3, 3
        )
        if math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))) > 0.02:
            raise RuntimeError("carrier_tilt")
        for c in d.contact:
            if not ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & self.bodies):
                continue
            pair = frozenset((int(c.geom1), int(c.geom2)))
            if pair in self.allowed:
                r["maximum_carrier_penetration_m"] = max(
                    r["maximum_carrier_penetration_m"], -float(c.dist)
                )
            elif c.dist < -1e-5:
                r["forbidden_contacts"].append(
                    dict(
                        time_s=float(d.time),
                        distance_m=float(c.dist),
                        geoms=[m.geom(c.geom1).name, m.geom(c.geom2).name],
                        detection="native contact",
                    )
                )
        for a, b in self.filtered:
            distance = float(mujoco.mj_geomDistance(m, d, a, b, 0.001, None))
            r["parent_filtered_distance_queries"] += 1
            previous = r["minimum_parent_filtered_distance_m"]
            r["minimum_parent_filtered_distance_m"] = (
                distance if previous is None else min(previous, distance)
            )
            if distance < -1e-5:
                r["forbidden_contacts"].append(
                    dict(
                        time_s=float(d.time),
                        distance_m=distance,
                        geoms=[m.geom(a).name, m.geom(b).name],
                        detection="direct enabled parent-filtered distance",
                    )
                )
        r["geometry_checks"] += 1
        if r["forbidden_contacts"]:
            raise RuntimeError("unexpected_instrument_collision")
        if r["maximum_carrier_penetration_m"] > 0.0001:
            raise RuntimeError("carrier_contact_penetration")
        if np.max(np.abs(d.qvel[self.v])) > 0.04:
            raise RuntimeError("instrument_velocity_limit")
        if np.max(np.abs(d.qpos[self.q] - d.ctrl[self.a])) > 0.012:
            raise RuntimeError("trajectory_tracking_limit")
        gap = float(
            d.geom_xpos[self.slide, 2]
            - m.geom_size[self.slide, 2]
            - d.site_xpos[self.objective, 2]
        )
        if gap < 0.004:
            raise RuntimeError("objective_working_clearance")
        registration = float(
            np.linalg.norm(d.xpos[self.carrier, :2] - d.site_xpos[self.objective, :2])
        )
        if r["physics_steps"] % max(1, round(0.020 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    step=r["physics_steps"],
                    time_s=float(d.time),
                    measured_m=d.qpos[self.q].tolist(),
                    carrier_world_m=d.xpos[self.carrier].tolist(),
                    registration_error_m=registration,
                    objective_slide_gap_m=gap,
                )
            )
        r["completed_checks"] += 1
        return registration, gap

    def phase(self, name, target, ramp, settle, *, registration=False, gap=None):
        self.owned()
        m, d, r = self.m, self.d, self.report
        if _model_signature(m) != self.signature:
            raise RuntimeError("compiled_model_changed")
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or any(not c.lower <= v <= c.upper for c, v in zip(self.controls, target))
        ):
            raise RuntimeError("target_outside_retained_limits")
        if not np.isfinite(float(d.time)):
            raise ValueError("Phase requires finite simulation time before commands")
        start = d.ctrl[self.a].copy()
        n = max(1, round(ramp / self.dt))
        hold = round(settle / self.dt)
        record = dict(
            name=name,
            target_m=target.tolist(),
            executed_steps=0,
            geometry_checks=0,
            completed_checks=0,
            success=False,
        )
        r["phases"].append(record)
        g0, c0 = r["geometry_checks"], r["completed_checks"]
        for index in range(n + hold):
            t = min(1.0, (index + 1) / n)
            d.ctrl[self.a] = start + t * t * (3 - 2 * t) * (target - start)
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
            try:
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
                position_error, measured_gap = self.observe()
            finally:
                record["geometry_checks"] = r["geometry_checks"] - g0
                record["completed_checks"] = r["completed_checks"] - c0
        record.update(
            measured_m=d.qpos[self.q].tolist(),
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            registration_error_m=position_error,
            objective_slide_gap_m=measured_gap,
        )
        if np.max(np.abs(d.qpos[self.q] - target)) > 0.0001:
            raise RuntimeError("endpoint_tracking_tolerance")
        if registration and position_error > 0.0001:
            raise RuntimeError("carrier_registration_tolerance")
        if gap is not None and abs(measured_gap - gap) > 0.00005:
            raise RuntimeError("objective_distance_tolerance")
        record["success"] = True
        return record


def inspect_indiana(session):
    """Offset, register the measured free carrier, inspect and return by actuation."""
    run = _Inspection(session)
    m, d, r = run.m, run.d, run.report
    try:
        mujoco.mj_forward(m, d)
        try:
            run.observe()
        finally:
            for field in ("entries", "geometry_checks", "completed_checks"):
                source = "observer_entries" if field == "entries" else field
                r["preflight_" + field] = r[source]
                r[source] = 0
        run.phase("settle", run.home, 0.002, 0.5)
        target = run.home.copy()
        target[:2] = (0.020, -0.015)
        run.phase("offset_loaded_stage", target, 3.0, 1.0)
        target[:2] = (
            d.qpos[run.q[:2]] + d.site_xpos[run.objective, :2] - d.xpos[run.carrier, :2]
        )
        run.phase("register_measured_carrier", target, 3.0, 1.0, registration=True)
        desired_gap = 0.007
        current_gap = (
            d.geom_xpos[run.slide, 2]
            - m.geom_size[run.slide, 2]
            - d.site_xpos[run.objective, 2]
        )
        target[2] = d.qpos[run.q[2]] + current_gap - desired_gap
        run.phase(
            "set_geometric_working_distance",
            target,
            1.0,
            0.5,
            registration=True,
            gap=desired_gap,
        )
        run.phase(
            "inspect_registered_carrier",
            target,
            0.002,
            0.5,
            registration=True,
            gap=desired_gap,
        )
        r["registered_carrier_position_m"] = d.xpos[run.carrier].tolist()
        r["inspection_objective_face_m"] = d.site_xpos[run.objective].tolist()
        target[2] = 0
        run.phase("retract_objective", target, 1.0, 0.5, registration=True)
        run.phase("return_home", run.home, 1.0, 0.5)
        r["readiness"] = session.process_readiness("microfluidic_station")
        if not r["readiness"]["ready"]:
            raise RuntimeError("returned_carrier_readiness_failed")
        if _model_signature(m) != run.signature:
            raise RuntimeError("compiled_model_changed")
        r["success"] = True
    except RuntimeError as exc:
        r["failure"] = str(exc)
        for c, address in zip(run.controls, run.q):
            d.ctrl[c.actuator] = np.clip(d.qpos[address], c.lower, c.upper)
        r["hold_requested"] = True
    r.update(
        coverage_complete=r["physics_steps"] > 0
        and r["completed_checks"] == r["physics_steps"],
        geometry_coverage_complete=r["physics_steps"] > 0
        and r["geometry_checks"] == r["physics_steps"],
        unchecked_steps=r["physics_steps"] - r["completed_checks"],
        elapsed_simulation_s=float(d.time) - run.time,
        model_unchanged=_model_signature(m) == run.signature,
        final_qpos=d.qpos.tolist(),
        final_qvel=d.qvel.tolist(),
        final_ctrl=d.ctrl.tolist(),
        coverage_interpretation="Completed geometry observations and fully passed state/safety checks are separate; failed and early-aborted steps cannot claim complete safety coverage.",
    )
    return r


def main():
    import argparse, hashlib, json
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene("indiana_microfluidics")
    xml = ET.tostring(root, encoding="unicode")
    result = inspect_indiana(
        InstrumentSession(mujoco.MjModel.from_xml_string(xml), manifest)
    )
    result["compiled_source_xml_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
    _portable_resources(root)
    result["scene_sha256"] = hashlib.sha256(
        ET.tostring(root, encoding="unicode").encode()
    ).hexdigest()
    result["task_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k
            not in ("telemetry", "phases", "final_qpos", "final_qvel", "final_ctrl")
        }
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
