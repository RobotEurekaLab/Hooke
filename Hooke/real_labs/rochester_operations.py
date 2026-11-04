"""Public access rules for Rochester's coupled plate-loading fixture.

The physical strokes are unchanged. Loading and retrieval use the measured
contact-only route; unrelated scalar targets cannot replace its lift, retract
and door-opening prerequisites. This module never resets or places live state.
"""

from copy import copy
import math

import mujoco
import numpy as np

from real_labs.rigid_access import require_binding
from real_labs.rochester_loading_motion import run_rochester_plate_loading

SCENE = "rochester_genomics"
OWNER = "liquid_handler"
LOADER_AXES = ("loader_x", "loader_y", "loader_z")


class OperationRequired(ValueError):
    """The requested motion needs the ordered loading operation."""

    def __init__(self, operation, reason):
        self.operation, self.reason = operation, reason
        super().__init__(f"{reason}; use the {operation} operation")


def _finite_state(session):
    data = session.data
    if not math.isfinite(float(data.time)) or not all(
        np.isfinite(getattr(data, key)).all()
        for key in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
    ):
        raise ValueError("Loading requires finite state")
    if np.any(data.warning.number):
        raise ValueError("Loading has unresolved physics warnings")
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("External forces are outside the loading contract")


def _require_home(session, keys, reason):
    for key in keys:
        spec = session.controls[OWNER, key]
        measured = session.data.qpos[session.model.jnt_qposadr[spec.joint]]
        if max(abs(measured), abs(session.data.ctrl[spec.actuator])) > 1e-4:
            raise OperationRequired("plate_loading", reason)


def validate_command(session, equipment_id, control, target):
    """Validate a scalar request without mutating controls, state or clock.

    X/Y translation requires the complete operation. Vertical inspection is
    available over its full stroke while X/Y stay parked. Door motion requires
    a parked fork and home pipette. These rules do not authorize arbitrary raw
    actuator writes; callers must use this preflight before ``session.command``.
    """
    if session.manifest.get("id") != SCENE or equipment_id != OWNER:
        return
    require_binding(session, SCENE)
    spec = session.controls[equipment_id, control]
    target = float(target)
    if not math.isfinite(target) or not spec.lower <= target <= spec.upper:
        raise ValueError("Target must be finite and inside the retained range")
    if control in ("loader_x", "loader_y"):
        raise OperationRequired(
            "plate_loading", "Horizontal fork travel requires ordered lift and access"
        )
    _finite_state(session)
    model, data = session.model, session.data
    if np.max(np.abs(data.qvel), initial=0) > 0.01:
        raise OperationRequired(
            "plate_loading", "Equipment must settle before inspection"
        )
    for item in session.controls.values():
        if (
            item.mode == "position"
            and abs(data.qpos[model.jnt_qposadr[item.joint]] - data.ctrl[item.actuator])
            > 0.001
        ):
            raise OperationRequired(
                "plate_loading", "An axis is not holding its target"
            )
    _require_home(
        session, LOADER_AXES[:2], "Park the horizontal fork before inspection"
    )
    if control != "loader_z":
        _require_home(session, ("loader_z",), "Lower the parked fork before inspection")
    if control == "front_door_hinge":
        _require_home(
            session,
            ("gantry_x", "gantry_y", "pipette_z"),
            "Return the pipette home before moving the door",
        )
    elif control != "loader_z":
        _require_home(
            session, ("front_door_hinge",), "Close access before pipette motion"
        )
    # Recompute derived poses privately; a stale live xpos cannot hide a plate
    # displaced into the tool or door sweep.
    observed = copy(data)
    mujoco.mj_forward(model, observed)
    interface = session.equipment[OWNER]["plate_loading_interface"]
    plate = model.body(interface["body"]).id
    supported = (
        min(
            np.linalg.norm(observed.xpos[plate] - np.asarray(interface[key]))
            for key in ("stage_world", "dock_world")
        )
        <= 0.001
    )
    if not supported or abs(observed.xquat[plate, 0]) < 0.999:
        raise OperationRequired(
            "plate_loading", "Plate is outside its supported placement"
        )


def command(session, equipment_id, control, target, *, duration=2.0):
    """Apply a scalar command only after the scene-specific preflight passes."""
    validate_command(session, equipment_id, control, target)
    return session.command(equipment_id, control, target, duration=duration)


def run_plate_loading(session, *, observe=None):
    """Run the complete contact-only load/retrieve route from authored home.

    An interrupted or externally disturbed state is rejected. This entry point
    does not silently reset it or claim to recover an arbitrary partial route.
    """
    require_binding(session, SCENE)
    _finite_state(session)
    return run_rochester_plate_loading(session, observe=observe)
