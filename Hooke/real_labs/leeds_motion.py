"""Measured generic loaded sliding, without a calibrated tribology claim.

Only actuator commands advance the live state. Each actual physics step checks
the contact pair, load, motion and held axes. Failure requests a measured-position
hold without explicitly resetting; this is not a certified emergency stop or an unloading
procedure. A compliant contact position deliberately differs from its servo target.
"""

import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports


def run_loaded_slide(session):
    """Approach, load, traverse -10/+10/0 mm, then retract to the measured home.

    Requires the authored home state and enabled contacts. Reported force is the
    MuJoCo constraint response; measured sliding integrates contact-point relative
    velocity. Neither is an experimentally validated coefficient or wear prediction.
    """
    model, data = session.model, session.data
    operation = session.equipment.get("tribology", {}).get("contact_operation")
    if not operation or operation.get("kind") != "authored_generic_loaded_sliding":
        raise ValueError("Scene has no authored Leeds loaded-sliding operation")
    controls = [
        session.controls["tribology", key]
        for key in ("contact_approach", "specimen_slide")
    ]
    if any(c.mode != "position" or c.units != "m" for c in controls):
        raise ValueError("Loaded sliding requires two bounded linear position servos")
    qadr = np.array([model.jnt_qposadr[c.joint] for c in controls])
    dadr = np.array([model.jnt_dofadr[c.joint] for c in controls])
    actuators = np.array([c.actuator for c in controls])
    pair = frozenset(model.geom(operation[key]).id for key in ("counterface", "coupon"))
    if len(pair) != 2 or any(
        not (model.geom_contype[g] and model.geom_conaffinity[g]) for g in pair
    ):
        raise ValueError("Both counterface and coupon must retain enabled contacts")
    required = (
        mujoco.mjtDisableBit.mjDSBL_CONTACT,
        mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
        mujoco.mjtDisableBit.mjDSBL_LIMIT,
        mujoco.mjtDisableBit.mjDSBL_ACTUATION,
    )
    if any(int(model.opt.disableflags) & int(flag) for flag in required):
        raise ValueError(
            "Contact, constraints, limits and actuation must remain enabled"
        )
    if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
        raise ValueError("Contact override changes the authored loading contract")
    if not np.allclose(
        model.jnt_axis[[c.joint for c in controls]], ((0, 0, -1), (1, 0, 0)), atol=1e-12
    ):
        raise ValueError("Loaded-sliding joint directions differ from the contract")
    if (
        not np.allclose(model.actuator_gainprm[actuators, 0], (1000, 6000))
        or not np.allclose(model.actuator_forcerange[actuators], ((-10, 10), (-40, 40)))
        or not np.all(model.actuator_forcelimited[actuators])
    ):
        raise ValueError("Authored compliance or actuator force limits changed")
    roots = {int(model.jnt_bodyid[c.joint]) for c in controls}
    moving_bodies = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor and ancestor not in roots:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor in roots:
            moving_bodies.add(body)
    moving_geoms = {
        g for g in range(model.ngeom) if model.geom_bodyid[g] in moving_bodies
    }
    # Exterior contacts stay enabled. Cosmetic guide/fastener details are not
    # mistaken for contact proxies; every currently enabled moving geom is tested.
    essential = (
        "specimen_platen",
        "coupon",
        "flat_lever",
        "load_shaft",
        "lower_collar",
        "upper_collar",
        "counterface_stem",
        "counterface",
    ) + tuple(
        f"{key}_{i}"
        for i in range(3)
        for key in ("clip_post", "clip_bridge", "clip_pad")
    )
    if any(
        not (
            model.geom_contype[model.geom("tribology__leeds_" + name).id]
            and model.geom_conaffinity[model.geom("tribology__leeds_" + name).id]
        )
        for name in essential
    ):
        raise ValueError("Contact fixture exterior collision coverage is incomplete")
    bearings = session.equipment["tribology"].get("bearing_contact_geoms", ())
    if not bearings or any(
        not (
            model.geom_contype[model.geom(name).id]
            and model.geom_conaffinity[model.geom(name).id]
        )
        for name in bearings
    ):
        raise ValueError("Open bearing contact coverage is incomplete")
    for name in (
        "approach_guide_-0.28",
        "approach_guide_-0.22",
        "slide_rail_-0.126",
        "slide_rail_0.006",
    ):
        geom = model.geom("tribology__leeds_" + name).id
        if not (model.geom_contype[geom] and model.geom_conaffinity[geom]):
            raise ValueError("Bearing/rail support contacts must remain enabled")
    signature = _model_signature(model)
    held = {
        key: (c, float(data.ctrl[c.actuator]), session._measured(c))
        for key, c in session.controls.items()
        if c.actuator not in actuators
    }
    stationary_dofs = np.setdiff1d(np.arange(model.nv), dadr)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    initial_time = float(data.time)
    warnings = data.warning.number.copy()
    dt = float(model.opt.timestep)
    if not math.isfinite(dt) or not 0 < dt <= 0.004:
        raise ValueError("Loaded contact requires an authored timestep at most 4 ms")
    if (
        not math.isfinite(initial_time)
        or not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
        or np.max(np.abs(data.qpos[qadr])) > 0.0001
        or np.max(np.abs(data.qvel)) > 0.002
    ):
        raise ValueError("Start with the retracted, centered fixture at rest")
    door = session.controls["tribology", "access_door"]
    if session._measured(door) < 0.65:
        raise ValueError(
            "The authored inspection operation requires the open access door"
        )
    for key, (c, target, measured) in held.items():
        if c.mode != "position" or abs(target - measured) > 0.001:
            raise ValueError(f"Inactive control is not holding: {key}")
    phases = [
        ("approach_and_load", (0.024, 0), 2.0, 1.0, False),
        ("loaded_negative", (0.024, -0.010), 1.0, 0.25, True),
        ("loaded_positive", (0.024, 0.010), 2.0, 0.25, True),
        ("loaded_center", (0.024, 0), 1.0, 0.25, True),
        ("retract", (0, 0), 2.0, 0.5, False),
    ]
    for _, targets, *_ in phases:
        if any(not c.lower <= t <= c.upper for c, t in zip(controls, targets)):
            raise ValueError("Required motion lies outside the retained joint limits")
    result = dict(
        success=False,
        failure=None,
        phases=[],
        physics_steps=0,
        observer_entries=0,
        geometry_checks=0,
        completed_checks=0,
        contact_steps=0,
        preflight_entries=0,
        preflight_geometry_checks=0,
        preflight_completed_checks=0,
        normal_force_range_n=None,
        peak_normal_force_n=0.0,
        peak_tangential_force_n=0.0,
        tangential_impulse_magnitude_ns=0.0,
        force_on_coupon_peak_absolute_xyz_n=[0.0, 0.0, 0.0],
        telemetry=[],
        telemetry_interval_s=0.020,
        maximum_contact_penetration_m=0.0,
        loaded_relative_path_m=0.0,
        held_maximum_drift=0.0,
        forbidden_contacts=[],
        hold_requested=False,
        reset_performed=False,
        explicit_reset_performed=False,
        automatic_reset_detected=False,
        time_discontinuity_detected=False,
        physics_events=[],
        model_signature=signature,
        interpretation="Sampled MuJoCo normal constraint response and relative contact motion for authored rigid mechanics. No calibrated friction, wear, heating, gas or vacuum prediction.",
    )
    force = np.zeros(6)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    force_samples = []

    def observe(loaded):
        result["observer_entries"] += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("nonfinite_state")
        if not np.array_equal(data.warning.number, warnings):
            raise RuntimeError("physics_warning")
        expected_time = initial_time + result["physics_steps"] * dt
        if abs(float(data.time) - expected_time) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (c, command, measured) in held.items():
            drift = abs(session._measured(c) - measured)
            result["held_maximum_drift"] = max(result["held_maximum_drift"], drift)
            if data.ctrl[c.actuator] != command or drift > 0.001:
                raise RuntimeError("inactive_control_changed_or_drifted")
        if np.max(np.abs(data.qvel[stationary_dofs]), initial=0) > 0.01:
            raise RuntimeError("inactive_motion")
        if any(
            not c.lower - 0.0001 <= data.qpos[a] <= c.upper + 0.0001
            for c, a in zip(controls, qadr)
        ):
            raise RuntimeError("joint_limit_violation")
        normal, relative_speed = 0.0, 0.0
        tangential_world = np.zeros(3)
        coupon_force_world = np.zeros(3)
        coupon = model.geom(operation["coupon"]).id
        for index, contact in enumerate(data.contact):
            if contact.geom1 not in moving_geoms and contact.geom2 not in moving_geoms:
                continue
            if frozenset((int(contact.geom1), int(contact.geom2))) != pair:
                if contact.dist < -1e-5:
                    result["forbidden_contacts"].append(
                        dict(
                            time_s=float(data.time),
                            distance_m=float(contact.dist),
                            geoms=[
                                model.geom(contact.geom1).name,
                                model.geom(contact.geom2).name,
                            ],
                        )
                    )
                continue
            mujoco.mj_contactForce(model, data, index, force)
            normal += max(0.0, float(force[0]))
            frame = contact.frame.reshape(3, 3)
            sign = 1 if contact.geom2 == coupon else -1
            coupon_force_world += sign * (frame.T @ force[:3])
            tangential_world += sign * (frame[1:].T @ force[1:3])
            result["maximum_contact_penetration_m"] = max(
                result["maximum_contact_penetration_m"], -float(contact.dist)
            )
            velocities = []
            for geom in (contact.geom1, contact.geom2):
                mujoco.mj_jac(
                    model, data, jacp, jacr, contact.pos, int(model.geom_bodyid[geom])
                )
                velocities.append(jacp @ data.qvel)
            relative = velocities[1] - velocities[0]
            direction = contact.frame[:3]
            relative_speed = max(
                relative_speed,
                float(
                    np.linalg.norm(relative - np.dot(relative, direction) * direction)
                ),
            )
        result["geometry_checks"] += 1
        tangential = float(np.linalg.norm(tangential_world))
        result["peak_tangential_force_n"] = max(
            result["peak_tangential_force_n"], tangential
        )
        result["tangential_impulse_magnitude_ns"] += tangential * dt
        result["force_on_coupon_peak_absolute_xyz_n"] = np.maximum(
            result["force_on_coupon_peak_absolute_xyz_n"], np.abs(coupon_force_world)
        ).tolist()
        result["peak_normal_force_n"] = max(result["peak_normal_force_n"], normal)
        if result["physics_steps"] % max(1, round(0.020 / dt)) == 0:
            result["telemetry"].append(
                dict(
                    time_s=float(data.time),
                    step=result["physics_steps"],
                    normal_force_n=normal,
                    tangential_force_n=tangential,
                    force_on_coupon_world_n=coupon_force_world.tolist(),
                    relative_tangential_speed_m_per_s=relative_speed,
                    measured_position_m=data.qpos[qadr].tolist(),
                    commanded_position_m=data.ctrl[actuators].tolist(),
                )
            )
        if normal > 0.5:
            result["contact_steps"] += 1
            force_samples.append(normal)
            result["loaded_relative_path_m"] += relative_speed * dt
        if result["forbidden_contacts"]:
            raise RuntimeError("unexpected_fixture_collision")
        if normal > 7.5:
            raise RuntimeError("normal_load_limit")
        if result["maximum_contact_penetration_m"] > 0.0005:
            raise RuntimeError("contact_penetration_limit")
        if loaded and normal < 0.5:
            raise RuntimeError("loaded_contact_lost")
        # Normal-axis tracking error is intentional compliance while loaded.
        if abs(data.qpos[qadr[1]] - data.ctrl[actuators[1]]) > 0.002:
            raise RuntimeError("tangential_tracking_limit")
        if abs(data.qvel[dadr[0]]) > 0.08 or abs(data.qvel[dadr[1]]) > 0.08:
            raise RuntimeError("fixture_velocity_limit")
        result["completed_checks"] += 1
        return normal

    try:
        mujoco.mj_forward(model, data)
        # A preflight check is recorded separately from executed-step coverage.
        try:
            observe(False)
        finally:
            result["preflight_entries"] = result["observer_entries"]
            result["preflight_geometry_checks"] = result["geometry_checks"]
            result["preflight_completed_checks"] = result["completed_checks"]
            result["observer_entries"] = result["geometry_checks"] = result[
                "completed_checks"
            ] = 0
        for name, target, ramp, settle, loaded in phases:
            start = data.ctrl[actuators].copy()
            target = np.asarray(target)
            ramp_steps, settle_steps = round(ramp / dt), round(settle / dt)
            record = dict(
                name=name,
                target_m=target.tolist(),
                executed_steps=0,
                completed_checks=0,
                success=False,
            )
            result["phases"].append(record)
            before_checks = result["completed_checks"]
            for step in range(ramp_steps + settle_steps):
                t = min(1.0, (step + 1) / ramp_steps)
                blend = t * t * (3 - 2 * t)
                data.ctrl[actuators] = start + blend * (target - start)
                before_time, before_warnings = float(data.time), _warning_state(data)
                reset_inputs = {
                    key: getattr(data, key).copy()
                    for key in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(model, data)
                result["physics_steps"] += 1
                record["executed_steps"] += 1
                mujoco.mj_forward(model, data)
                try:
                    event, _ = _physics_event(
                        model, data, before_time, before_warnings, reset_inputs
                    )
                    if event is not None:
                        result["physics_events"].append(
                            dict(executed_step=result["physics_steps"], **event)
                        )
                        result.update(combine_reset_reports(result["physics_events"]))
                        result["time_discontinuity_detected"] = any(
                            not e["time_continuous"] for e in result["physics_events"]
                        )
                        record.update(combine_reset_reports([event]))
                        record["physics_event"] = event
                        raise RuntimeError(
                            "native_physics_warning_or_time_discontinuity"
                        )
                    normal = observe(loaded)
                finally:
                    record["completed_checks"] = (
                        result["completed_checks"] - before_checks
                    )
            record.update(
                measured_m=data.qpos[qadr].tolist(),
                normal_force_n=normal,
                qpos=data.qpos.tolist(),
                qvel=data.qvel.tolist(),
                ctrl=data.ctrl.tolist(),
                time_s=float(data.time),
            )
            if abs(data.qpos[qadr[1]] - target[1]) > 0.0001:
                raise RuntimeError("tangential_endpoint_tolerance")
            if name == "approach_and_load" and not 2.5 <= normal <= 5.5:
                raise RuntimeError("normal_load_not_established")
            if name == "retract" and (
                np.max(np.abs(data.qpos[qadr])) > 0.0001 or normal > 0.05
            ):
                raise RuntimeError("retraction_endpoint_tolerance")
            record["success"] = True
        if not 0.035 <= result["loaded_relative_path_m"] <= 0.045:
            raise RuntimeError("insufficient_or_excess_sliding")
        if _model_signature(model) != signature:
            raise RuntimeError("compiled_model_changed")
        result["success"] = True
    except RuntimeError as exc:
        result["failure"] = str(exc)
        for c, address in zip(controls, qadr):
            data.ctrl[c.actuator] = np.clip(data.qpos[address], c.lower, c.upper)
        result["hold_requested"] = True
    result.update(
        final_position_m=data.qpos[qadr].tolist(),
        final_velocity_m_per_s=data.qvel[dadr].tolist(),
        elapsed_simulation_s=float(data.time) - initial_time,
        coverage_complete=result["physics_steps"] > 0
        and result["completed_checks"] == result["physics_steps"],
        unchecked_steps=result["physics_steps"] - result["completed_checks"],
        geometry_coverage_complete=result["physics_steps"] > 0
        and result["geometry_checks"] == result["physics_steps"],
        coverage_interpretation="geometry_checks counts observed contact queries; completed_checks counts steps passing every safety invariant. A failing or early-aborted step is not a completed safety check.",
        model_unchanged=_model_signature(model) == signature,
        starting_joint_positions=initial_qpos.tolist(),
        starting_joint_velocities=initial_qvel.tolist(),
    )
    if force_samples:
        result["normal_force_range_n"] = [min(force_samples), max(force_samples)]
    return result


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene("leeds_environmental_tribology")
    xml = ET.tostring(root, encoding="unicode")
    session = InstrumentSession(mujoco.MjModel.from_xml_string(xml), manifest)
    report = run_loaded_slide(session)
    report["scene_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
    report["task_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
