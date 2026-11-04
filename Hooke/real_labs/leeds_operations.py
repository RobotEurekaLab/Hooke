"""Public command envelope and measured recovery for the Leeds contact fixture.

The complete joint ranges and the loaded-slide operation remain available.
Scalar commands cover dry inspection only; contact loading requires the measured
operation. Recovery unloads the current contact without repositioning live state.
"""

from copy import copy
import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding

SCENE = "leeds_environmental_tribology"
DRY_APPROACH_MAX_M = 0.012


class OperationRequired(ValueError):
    """A scalar command needs a declared operation or measured recovery."""

    def __init__(self, operation, reason):
        self.operation, self.reason = operation, reason
        super().__init__(f"{reason}; use the {operation} operation")


def _finite_state(session):
    d = session.data
    if not math.isfinite(float(d.time)) or not all(
        np.isfinite(getattr(d, key)).all()
        for key in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
    ):
        raise ValueError("Fixture state must be finite")
    if np.any(d.warning.number):
        raise ValueError("Fixture has unresolved physics warnings")
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
        raise ValueError("External forces are outside the authored operation contract")


def validate_command(session, equipment_id, control, target):
    """Reject unsafe scalar targets before any command or physics mutation.

    A dry normal move may start anywhere within the settled dry envelope. Door
    and specimen-slide commands require the normal axis at its measured home.
    Other scenes are unaffected. This function does not execute the command.
    """
    if session.manifest.get("id") != SCENE or equipment_id != "tribology":
        return
    require_binding(session, SCENE)
    target = float(target)
    if not math.isfinite(target):
        raise ValueError("Target must be finite")
    spec = session.controls[equipment_id, control]
    if not spec.lower <= target <= spec.upper:
        raise ValueError("Target outside retained control range")
    if control == "contact_approach" and target > DRY_APPROACH_MAX_M:
        raise OperationRequired(
            "loaded_slide", "Contact approach exceeds the dry envelope"
        )
    _finite_state(session)
    model, data = session.model, session.data
    normal = session.controls["tribology", "contact_approach"]
    measured = session._measured(normal)
    commanded = float(data.ctrl[normal.actuator])
    dry = (
        -1e-4 <= measured <= DRY_APPROACH_MAX_M + 1e-4
        and 0 <= commanded <= DRY_APPROACH_MAX_M
        and abs(measured - commanded) <= 1e-4
    )
    if not dry:
        raise OperationRequired(
            "retract", "Normal axis is loaded or outside the verified dry state"
        )
    if np.max(np.abs(data.qvel), initial=0) > 0.002:
        raise OperationRequired(
            "retract", "Fixture must settle before a scalar command"
        )
    for item in session.controls.values():
        if abs(session._measured(item) - data.ctrl[item.actuator]) > 0.001:
            raise OperationRequired(
                "retract", "A fixture axis is not holding its target"
            )
    # Recompute poses privately, so stale derived data cannot hide live contact.
    observed = copy(data)
    mujoco.mj_forward(model, observed)
    counterface = model.geom("tribology__leeds_counterface").id
    coupon = model.geom("tribology__leeds_coupon").id
    if mujoco.mj_geomDistance(model, observed, counterface, coupon, 0.1, None) < 0.006:
        raise OperationRequired(
            "retract", "Measured counterface clearance is outside the dry envelope"
        )
    if control != "contact_approach" and max(abs(measured), abs(commanded)) > 1e-4:
        raise OperationRequired(
            "retract", "Retract the normal axis before moving the door or specimen"
        )


def run_retract(session):
    """Ramp the measured normal position to home while holding the specimen.

    Recovery permits an initial contact indentation up to 1 mm and load up to
    15 N; during unloading it allows at most 0.5 N or 50 micrometres above
    the measured initial load or indentation.
    These recovery bounds do not authorize contact loading by scalar commands.
    No reset or direct assignment of live joint positions is performed.
    """
    require_binding(session, SCENE)
    _finite_state(session)
    model, data = session.model, session.data
    controls = [
        session.controls["tribology", key]
        for key in ("contact_approach", "specimen_slide")
    ]
    normal, slide = controls
    qadr = np.array([model.jnt_qposadr[c.joint] for c in controls])
    dadr = np.array([model.jnt_dofadr[c.joint] for c in controls])
    actuators = np.array([c.actuator for c in controls])
    if any(
        not c.lower - 1e-4 <= data.qpos[a] <= c.upper + 1e-4
        for c, a in zip(controls, qadr)
    ):
        raise ValueError("Recovery requires fixture positions inside retained limits")
    if np.max(np.abs(data.qvel), initial=0) > 0.08:
        raise ValueError("Fixture is moving too quickly for measured recovery")
    held = {
        key: (c, float(data.ctrl[c.actuator]), session._measured(c))
        for key, c in session.controls.items()
        if c.actuator not in actuators
    }
    if any(abs(target - measured) > 0.001 for c, target, measured in held.values()):
        raise ValueError("Inactive fixture axes must hold before recovery")
    operation = session.equipment["tribology"]["contact_operation"]
    pair = frozenset(model.geom(operation[key]).id for key in ("counterface", "coupon"))
    moving_welds = {int(model.body_weldid[model.jnt_bodyid[c.joint]]) for c in controls}
    moving_geoms = {
        g
        for g in range(model.ngeom)
        if int(model.body_weldid[model.geom_bodyid[g]]) in moving_welds
    }
    force = np.zeros(6)

    def contact_state(observed):
        load, penetration = 0.0, 0.0
        for index, contact in enumerate(observed.contact):
            geoms = frozenset((int(contact.geom1), int(contact.geom2)))
            if geoms == pair:
                mujoco.mj_contactForce(model, observed, index, force)
                load += max(0.0, float(force[0]))
                penetration = max(penetration, -float(contact.dist))
            elif geoms & moving_geoms and contact.dist < -1e-5:
                raise RuntimeError("unexpected_fixture_collision")
        return load, penetration

    # Preflight uses current measured contacts; no ctrl/time writes on rejection.
    observed = copy(data)
    mujoco.mj_forward(model, observed)
    initial_load, initial_penetration = contact_state(observed)
    if initial_load > 15 or initial_penetration > 0.001:
        raise ValueError("Measured contact exceeds the recovery envelope")
    initial_time = float(data.time)
    initial_position = data.qpos[qadr].copy()
    signature = _model_signature(model)
    dt = float(model.opt.timestep)
    result = dict(
        success=False,
        failure=None,
        physics_steps=0,
        completed_checks=0,
        physics_events=[],
        hold_requested=False,
        reset_performed=False,
        explicit_reset_performed=False,
        automatic_reset_detected=False,
        initial_position_m=initial_position.tolist(),
        initial_normal_force_n=initial_load,
        initial_penetration_m=initial_penetration,
        peak_normal_force_n=initial_load,
        maximum_contact_penetration_m=initial_penetration,
        telemetry=[],
    )
    start_normal = max(0.0, float(initial_position[0]))
    start_slide = float(initial_position[1])
    steps = round(2.5 / dt)
    ramp_steps = round(2.0 / dt)
    try:
        for step in range(steps):
            fraction = min(1.0, (step + 1) / ramp_steps)
            blend = fraction * fraction * (3 - 2 * fraction)
            data.ctrl[normal.actuator] = start_normal * (1 - blend)
            data.ctrl[slide.actuator] = start_slide
            before_time, before_warnings = float(data.time), _warning_state(data)
            inputs = {
                key: getattr(data, key).copy()
                for key in ("ctrl", "qfrc_applied", "xfrc_applied")
            }
            mujoco.mj_step(model, data)
            result["physics_steps"] += 1
            mujoco.mj_forward(model, data)
            event, _ = _physics_event(model, data, before_time, before_warnings, inputs)
            if event is not None:
                result["physics_events"].append(
                    dict(executed_step=result["physics_steps"], **event)
                )
                result.update(combine_reset_reports(result["physics_events"]))
                raise RuntimeError("native_physics_warning_or_time_discontinuity")
            _finite_state(session)
            load, penetration = contact_state(data)
            result["peak_normal_force_n"] = max(result["peak_normal_force_n"], load)
            result["maximum_contact_penetration_m"] = max(
                result["maximum_contact_penetration_m"], penetration
            )
            if load > initial_load + 0.5 or penetration > initial_penetration + 5e-5:
                raise RuntimeError("recovery_contact_limit")
            if np.max(np.abs(data.qvel[dadr])) > 0.08:
                raise RuntimeError("fixture_velocity_limit")
            if abs(data.qpos[qadr[1]] - start_slide) > 0.0002:
                raise RuntimeError("specimen_hold_drift")
            if any(
                data.ctrl[c.actuator] != target
                or abs(session._measured(c) - measured) > 0.001
                for c, target, measured in held.values()
            ):
                raise RuntimeError("inactive_control_changed_or_drifted")
            if any(
                not c.lower - 1e-4 <= data.qpos[a] <= c.upper + 1e-4
                for c, a in zip(controls, qadr)
            ):
                raise RuntimeError("joint_limit_violation")
            result["completed_checks"] += 1
            if step % max(1, round(0.02 / dt)) == 0:
                result["telemetry"].append(
                    dict(
                        time_s=float(data.time),
                        normal_force_n=load,
                        penetration_m=penetration,
                        measured_position_m=data.qpos[qadr].tolist(),
                    )
                )
        if (
            abs(data.qpos[qadr[0]]) > 1e-4
            or np.max(np.abs(data.qvel[dadr])) > 0.002
            or load > 0.05
        ):
            raise RuntimeError("retraction_endpoint_tolerance")
        if _model_signature(model) != signature:
            raise RuntimeError("compiled_model_changed")
        result["success"] = True
    except (RuntimeError, ValueError) as exc:
        result["failure"] = str(exc)
        for control, address in zip(controls, qadr):
            if math.isfinite(float(data.qpos[address])):
                data.ctrl[control.actuator] = np.clip(
                    data.qpos[address], control.lower, control.upper
                )
        result["hold_requested"] = True
    result.update(
        final_position_m=data.qpos[qadr].tolist(),
        final_velocity_m_per_s=data.qvel[dadr].tolist(),
        elapsed_simulation_s=float(data.time) - initial_time,
        coverage_complete=result["physics_steps"] > 0
        and result["physics_steps"] == result["completed_checks"],
        unchecked_steps=result["physics_steps"] - result["completed_checks"],
        model_unchanged=_model_signature(model) == signature,
    )
    return result
