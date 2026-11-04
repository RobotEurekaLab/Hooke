"""Measured inert-carrier access and dry tool contact for Rochester surrogates.

This operation uses existing actuators and two freely supported rigid carriers.
It does not insert a sample robotically or implement a liquid/biological process.
"""

import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports


class _Run:
    """Local execution guard; no live state assignment or explicit reset."""

    def __init__(self, session):
        if not np.isfinite(float(session.data.time)):
            raise ValueError("Rochester operation requires finite simulation time")
        if session.data.model is not session.model:
            raise ValueError("Session data must belong to its exact model object")
        self.session = session
        self.model, self.data = session.model, session.data
        self.keys = (
            ("liquid_handler", "gantry_x"),
            ("liquid_handler", "gantry_y"),
            ("liquid_handler", "pipette_z"),
            ("sequencer", "flowcell_tray"),
        )
        self.controls = [session.controls[k] for k in self.keys]
        self.actuators = np.array([c.actuator for c in self.controls])
        self.qadr = np.array([self.model.jnt_qposadr[c.joint] for c in self.controls])
        self.dadr = np.array([self.model.jnt_dofadr[c.joint] for c in self.controls])
        if any(c.mode != "position" or c.units != "m" for c in self.controls):
            raise ValueError(
                "Rochester operation requires the four authored linear position controls"
            )
        self.interface = session.equipment["liquid_handler"].get(
            "dry_contact_interface"
        )
        if not self.interface or not session.equipment["sequencer"].get(
            "loading_interface"
        ):
            raise ValueError("Missing dry-contact or real loading-aperture interface")
        self.tool = self.model.site(self.interface["tool_site"]).id
        self.tips = {self.model.geom(n).id for n in self.interface["tip_geoms"]}
        pipette_body = self.model.jnt_bodyid[self.controls[2].joint]
        if (
            len(self.tips) != 8
            or self.model.site_bodyid[self.tool] != pipette_body
            or any(self.model.geom_bodyid[g] != pipette_body for g in self.tips)
        ):
            raise ValueError(
                "Dry-contact sites and eight tips must belong to the moving head"
            )
        zact = self.controls[2].actuator
        if not self.model.actuator_forcelimited[zact] or not np.allclose(
            self.model.actuator_forcerange[zact], (-5, 5), rtol=0, atol=1e-12
        ):
            raise ValueError(
                "The authored 5 N dry-contact axis force limits are required"
            )
        trayact = self.controls[3].actuator
        if not self.model.actuator_forcelimited[trayact] or not np.allclose(
            self.model.actuator_forcerange[trayact], (-20, 20), rtol=0, atol=1e-12
        ):
            raise ValueError("The authored 20 N tray force limits are required")
        for control, axis in zip(
            self.controls, ((1, 0, 0), (0, 1, 0), (0, 0, -1), (0, -1, 0))
        ):
            if not np.allclose(
                self.data.xaxis[control.joint], axis, rtol=0, atol=1e-10
            ):
                raise ValueError(
                    "Rochester tool and tray frames do not match the bound operation"
                )
        self.carriers = {
            key: self.model.body(session.equipment[key]["process"]["sample_body"]).id
            for key in ("liquid_handler", "sequencer")
        }
        self.carrier_geoms = {
            key: self.model.geom(key + "__specimen__collision").id
            for key in self.carriers
        }
        self.supports = {
            row["equipment"]: self.model.geom(row["support_geom"]).id
            for row in session.manifest["specimens"]
            if row["equipment"] in self.carriers
        }
        if set(self.supports) != set(self.carriers) or any(
            self.model.geom_bodyid[self.carrier_geoms[key]] != body
            for key, body in self.carriers.items()
        ):
            raise ValueError(
                "Carrier geometry and support metadata do not match the free bodies"
            )
        self.allowed_carrier_support = {
            frozenset((self.carrier_geoms[key], support))
            for key, support in self.supports.items()
        }
        self.allowed_carrier_support.update(
            frozenset(
                (
                    self.carrier_geoms["sequencer"],
                    self.model.geom(f"sequencer__flowcell_stop_1_{axis}_{sign}").id,
                )
            )
            for axis in (0, 1)
            for sign in (-1, 1)
        )
        essential = (
            list(self.tips)
            + list(self.carrier_geoms.values())
            + list(self.supports.values())
        )
        essential += [
            self.model.geom(f"liquid_handler__tip_mount_contact_{i}").id
            for i in range(1, 9)
        ]
        essential += [
            g
            for g in range(self.model.ngeom)
            if self.model.geom(g).name.startswith(
                (
                    "liquid_handler__y_rail_",
                    "liquid_handler__y_bearing_",
                    "liquid_handler__z_rail_",
                    "liquid_handler__z_bearing_",
                )
            )
        ]
        essential += [
            self.model.geom(n).id
            for n in (
                "liquid_handler__gantry_contact",
                "liquid_handler__carriage_contact",
                "liquid_handler__head_contact",
                "sequencer__flowcell_front",
                "sequencer__front_left",
                "sequencer__front_right",
                "sequencer__separator",
                "sequencer__upper_lintel",
                "sequencer__flowcell_guide_-1",
                "sequencer__flowcell_guide_1",
                "sequencer__flowcell_runner_-1",
                "sequencer__flowcell_runner_1",
            )
        ]
        essential += [
            self.model.geom(f"sequencer__flowcell_stop_1_{a}_{s}").id
            for a in (0, 1)
            for s in (-1, 1)
        ]
        if any(
            not (self.model.geom_contype[g] and self.model.geom_conaffinity[g])
            for g in essential
        ):
            raise ValueError("Instrument/carrier collision coverage is incomplete")
        for body in self.carriers.values():
            if (
                self.model.body_jntnum[body] != 1
                or self.model.jnt_type[self.model.body_jntadr[body]]
                != mujoco.mjtJoint.mjJNT_FREE
            ):
                raise ValueError(
                    "Both carriers must retain six free degrees of freedom"
                )
        for index, kind in enumerate(self.model.eq_type):
            if kind in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT) and (
                self.model.eq_obj1id[index] in self.carriers.values()
                or self.model.eq_obj2id[index] in self.carriers.values()
            ):
                raise ValueError("Carriers cannot be attached by equality constraints")
        self.initial_equalities = self.data.eq_active.copy()
        required = (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        )
        if any(int(self.model.opt.disableflags) & int(flag) for flag in required):
            raise ValueError(
                "Contacts, constraints, limits and actuation must remain enabled"
            )
        if self.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
            raise ValueError(
                "Contact overrides change the authored mechanical contract"
            )
        self.dt = float(self.model.opt.timestep)
        if not 0 < self.dt <= 0.002:
            raise ValueError("Rochester operation requires a timestep at most 2 ms")
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or np.max(np.abs(self.data.qpos[self.qadr])) > 0.0001
            or np.max(np.abs(self.data.qvel)) > 0.01
        ):
            raise ValueError(
                "Begin with the retracted, centered tool and closed tray at rest"
            )
        self.signature = _model_signature(self.model)
        self.initial_time = float(self.data.time)
        self.warning = self.data.warning.number.copy()
        self.held = {
            key: (c, float(self.data.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator not in self.actuators
        }
        self.home = self.data.qpos[self.qadr].copy()
        self.initial_carrier_pose = {
            key: self.data.xpos[body].copy() for key, body in self.carriers.items()
        }
        self.initial_carrier_rotation = {
            key: self.data.xmat[body].copy() for key, body in self.carriers.items()
        }
        self.roots = {self.model.jnt_bodyid[c.joint] for c in self.controls}
        self.relevant_bodies = set(self.carriers.values())
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor and ancestor not in self.roots:
                ancestor = self.model.body_parentid[ancestor]
            if ancestor in self.roots:
                self.relevant_bodies.add(body)
        # Native contact filtering omits adjacent moving bodies. Query these
        # enabled parent-child pairs directly; a successful contact list alone
        # cannot prove their guide interfaces are free from interpenetration.
        enabled = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom(g).name.startswith("liquid_handler__")
            and self.model.geom_contype[g]
            and self.model.geom_conaffinity[g]
        ]
        self.filtered_pairs = []
        for offset, a in enumerate(enabled):
            wa = self.model.body_weldid[self.model.geom_bodyid[a]]
            for b in enabled[offset + 1 :]:
                wb = self.model.body_weldid[self.model.geom_bodyid[b]]
                if wa == wb or not wa or not wb:
                    continue
                pa = self.model.body_weldid[self.model.body_parentid[wa]]
                pb = self.model.body_weldid[self.model.body_parentid[wb]]
                if pa == wb or pb == wa:
                    self.filtered_pairs.append((a, b))
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
            parent_filtered_pair_count=len(self.filtered_pairs),
            parent_filtered_distance_queries=0,
            minimum_parent_filtered_distance_m=None,
            peak_tip_normal_force_n=0.0,
            tip_contact_steps=0,
            maximum_carrier_support_penetration_m=0.0,
            maximum_tip_penetration_m=0.0,
            held_control_maximum_error={},
            carrier_maximum_tracking_error_m={},
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            time_discontinuity_detected=False,
            physics_events=[],
            model_signature=self.signature,
            model_binding="Full compiled-model signature checked before every phase and after execution; state, equality activation and held controls checked after every actual physics step.",
            telemetry=[],
            telemetry_interval_s=0.020,
            interpretation="Measured dry tool contact and open/access/close of an already loaded, freely supported inert carrier. No robotic sample insertion, liquid transport, sequencing or biological procedure.",
        )

    def observe(self, *, tip_contact=False):
        m, d, r = self.model, self.data, self.report
        r["observer_entries"] += 1
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("session_model_or_data_replaced")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(d.warning.number, self.warning):
            raise RuntimeError("physics_warning")
        if not np.array_equal(d.eq_active, self.initial_equalities):
            raise RuntimeError("equality_state_changed")
        if abs(float(d.time) - self.initial_time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (control, target) in self.held.items():
            label = ".".join(key)
            error = abs(self.session._measured(control) - target)
            r["held_control_maximum_error"][label] = max(
                error, r["held_control_maximum_error"].get(label, 0.0)
            )
            tolerance = 0.002 if control.units == "m" else 0.01
            if d.ctrl[control.actuator] != target or error > tolerance:
                raise RuntimeError("inactive_control_changed_or_drifted")
        for control, address in zip(self.controls, self.qadr):
            if not control.lower - 0.0001 <= d.qpos[address] <= control.upper + 0.0001:
                raise RuntimeError("joint_limit_violation")
        for key, body in self.carriers.items():
            expected = self.initial_carrier_pose[key].copy()
            if key == "sequencer":
                expected[1] -= d.qpos[self.qadr[3]]
            error = float(np.linalg.norm(d.xpos[body] - expected))
            r["carrier_maximum_tracking_error_m"][key] = max(
                error, r["carrier_maximum_tracking_error_m"].get(key, 0.0)
            )
            if error > 0.0015:
                raise RuntimeError("carrier_displaced_or_not_retained")
            rotation = d.xmat[body].reshape(3, 3).T @ self.initial_carrier_rotation[
                key
            ].reshape(3, 3)
            if math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))) > 0.02:
                raise RuntimeError("carrier_tilt")
        normal = 0.0
        contacting_tips = set()
        force = np.zeros(6)
        for index, c in enumerate(d.contact):
            if not (
                {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & self.relevant_bodies
            ):
                continue
            pair = frozenset((int(c.geom1), int(c.geom2)))
            tool_pair = self.carrier_geoms["liquid_handler"] in pair and bool(
                pair & self.tips
            )
            if pair in self.allowed_carrier_support:
                r["maximum_carrier_support_penetration_m"] = max(
                    r["maximum_carrier_support_penetration_m"], -float(c.dist)
                )
            elif tool_pair and tip_contact:
                mujoco.mj_contactForce(m, d, index, force)
                normal += max(0.0, float(force[0]))
                if force[0] > 0.0001:
                    contacting_tips.update(pair & self.tips)
                r["maximum_tip_penetration_m"] = max(
                    r["maximum_tip_penetration_m"], -float(c.dist)
                )
            elif c.dist < -1e-5:
                r["forbidden_contacts"].append(
                    dict(
                        time_s=float(d.time),
                        distance_m=float(c.dist),
                        geoms=[m.geom(c.geom1).name, m.geom(c.geom2).name],
                    )
                )
        for a, b in self.filtered_pairs:
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
                        detection="direct distance of enabled parent-filtered geometry",
                    )
                )
        r["geometry_checks"] += 1
        r["peak_tip_normal_force_n"] = max(r["peak_tip_normal_force_n"], normal)
        r["tip_contact_steps"] += int(normal > 0.005)
        if r["physics_steps"] % max(1, round(0.020 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    time_s=float(d.time),
                    step=r["physics_steps"],
                    measured_m=d.qpos[self.qadr].tolist(),
                    tip_normal_force_n=normal,
                    contacting_tips=len(contacting_tips),
                    carrier_positions_m={
                        k: d.xpos[b].tolist() for k, b in self.carriers.items()
                    },
                )
            )
        if r["forbidden_contacts"]:
            raise RuntimeError("unexpected_instrument_collision")
        if r["maximum_carrier_support_penetration_m"] > 0.0003:
            raise RuntimeError("carrier_support_penetration")
        if r["maximum_tip_penetration_m"] > 0.00015:
            raise RuntimeError("tip_penetration_limit")
        if normal > 0.2:
            raise RuntimeError("dry_contact_force_limit")
        if np.max(np.abs(d.qvel[self.dadr])) > 0.20:
            raise RuntimeError("instrument_velocity_limit")
        if np.max(np.abs(d.qpos[self.qadr] - d.ctrl[self.actuators])) > 0.04:
            raise RuntimeError("trajectory_tracking_limit")
        r["completed_checks"] += 1
        return normal, len(contacting_tips)

    def phase(
        self, name, target, ramp, settle, *, tip_contact=False, force_at_end=False
    ):
        m, d, r = self.model, self.data, self.report
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("session_model_or_data_replaced")
        if _model_signature(m) != self.signature:
            raise RuntimeError("compiled_model_changed")
        target = np.asarray(target, dtype=float)
        if any(not c.lower <= v <= c.upper for c, v in zip(self.controls, target)):
            raise RuntimeError("target_outside_retained_limits")
        if not np.isfinite(float(d.time)):
            raise ValueError("Phase requires finite simulation time before commands")
        start = d.ctrl[self.actuators].copy()
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
        g0 = r["geometry_checks"]
        c0 = r["completed_checks"]
        for i in range(n + hold):
            t = min(1.0, (i + 1) / n)
            blend = t * t * (3 - 2 * t)
            d.ctrl[self.actuators] = start + blend * (target - start)
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
                normal, count = self.observe(tip_contact=tip_contact)
            finally:
                record["geometry_checks"] = r["geometry_checks"] - g0
                record["completed_checks"] = r["completed_checks"] - c0
        record.update(
            measured_m=d.qpos[self.qadr].tolist(),
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            tip_normal_force_n=normal,
            contacting_tips=count,
        )
        if np.max(np.abs(d.qpos[self.qadr] - target)) > 0.0005:
            raise RuntimeError("endpoint_tracking_tolerance")
        if force_at_end and not 0.005 <= normal <= 0.15:
            raise RuntimeError("dry_contact_not_established")
        if force_at_end and count != len(self.tips):
            raise RuntimeError("incomplete_tool_array_contact")
        record["success"] = True
        return record


def run_rochester_access(session):
    """Dry touch/retract and sequencer carrier access, with honest failure hold."""
    run = _Run(session)
    m, d, r = run.model, run.data, run.report
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
        carrier = run.carrier_geoms["liquid_handler"]
        top = float(d.geom_xpos[carrier, 2] + m.geom_size[carrier, 2])
        center = d.xpos[run.carriers["liquid_handler"]].copy()
        tool = d.site_xpos[run.tool].copy()
        target = run.home.copy()
        target[:2] += center[:2] - tool[:2]
        r["dry_contact_target"] = dict(
            carrier_top_world_m=top,
            tool_home_world_m=tool.tolist(),
            tool_site=m.site(run.tool).name,
            normal_overtravel_command_m=0.00025,
            interpretation="Tool-array centre aligned to actual inert carrier centre; controlled dry contact, not a liquid-handling operation.",
        )
        run.phase("align_above_carrier", target, 2.0, 1.0)
        target[2] = tool[2] - top - 0.003
        run.phase("approach_3mm_standoff", target, 4.0, 1.0)
        target[2] = tool[2] - top + 0.00025
        run.phase("dry_touch", target, 1.0, 0.5, tip_contact=True, force_at_end=True)
        target[2] = 0
        run.phase("retract_tool", target, 4.0, 1.0, tip_contact=True)
        run.phase("return_tool_home", run.home, 2.0, 1.0)
        target = run.home.copy()
        target[3] = 0.25
        run.phase("open_loaded_tray", target, 4.0, 1.0)
        run.phase("inspect_loading_access", target, 0.002, 0.75)
        from real_labs.usability import camera_target_screening

        access = next(
            row
            for row in camera_target_screening(m, d, session.manifest)
            if row["equipment"] == "sequencer"
        )
        r["open_loading_camera_screening"] = access
        if "sequencer_loading" not in access["candidate_views"]:
            raise RuntimeError("loading_target_not_visible")
        run.phase("close_loaded_tray", run.home, 4.0, 1.0)
        r["closed_readiness"] = session.process_readiness("sequencer")
        if not r["closed_readiness"]["ready"]:
            raise RuntimeError("closed_carrier_readiness_failed")
        if _model_signature(m) != run.signature:
            raise RuntimeError("compiled_model_changed")
        r["success"] = True
    except RuntimeError as exc:
        r["failure"] = str(exc)
        for c, address in zip(run.controls, run.qadr):
            d.ctrl[c.actuator] = np.clip(d.qpos[address], c.lower, c.upper)
        r["hold_requested"] = True
    r.update(
        coverage_complete=r["physics_steps"] > 0
        and r["completed_checks"] == r["physics_steps"],
        geometry_coverage_complete=r["physics_steps"] > 0
        and r["geometry_checks"] == r["physics_steps"],
        unchecked_steps=r["physics_steps"] - r["completed_checks"],
        elapsed_simulation_s=float(d.time) - run.initial_time,
        model_unchanged=_model_signature(m) == run.signature,
        final_qpos=d.qpos.tolist(),
        final_qvel=d.qvel.tolist(),
        final_ctrl=d.ctrl.tolist(),
        coverage_interpretation="Geometry observations and fully passed safety checks are separate. Failed or early-aborted steps cannot establish completed coverage.",
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
    root, manifest = build_scene("rochester_genomics")
    xml = ET.tostring(root, encoding="unicode")
    session = InstrumentSession(mujoco.MjModel.from_xml_string(xml), manifest)
    result = run_rochester_access(session)
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
            if k not in ("telemetry", "phases", "final_qpos", "final_qvel")
        }
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
