"""Allowlisted measured operations and command policies for laboratory clients.

Operation availability describes an authored workflow, not its readiness in the
current state. Each runner checks its own starting conditions and reports actual
physics results. InstrumentSession.command applies these policies; only its
private _command_unchecked entry point bypasses them for bounded diagnostics.
"""

from dataclasses import dataclass
from importlib import import_module
import math

import numpy as np

from real_labs.rigid_access import require_binding


@dataclass(frozen=True)
class _Operation:
    scene: str
    id: str
    equipment_id: str
    label: str
    description: str
    module: str
    function: str

    def metadata(self):
        return {
            key: getattr(self, key)
            for key in ("id", "equipment_id", "label", "description")
        }


_OPERATIONS = (
    _Operation(
        "manchester_collaborative_robotics",
        "empty_tool_inspection",
        "collaborative_arm",
        "Inspect block and return",
        "Plan a collision-checked, downward-facing empty-tool standoff above "
        "the retained block and return to the measured starting pose. "
        "The fixed fingers do not grasp or transfer the block.",
        "real_labs.manchester_operations",
        "run_inspection",
    ),
    _Operation(
        "nasa_planetary",
        "arm_clearance_yaw",
        "rover",
        "Lift and rotate rover arm",
        "From home, lift clear of the pickup fixture before rotating the arm "
        "to pi/2. The sample remains on its fixture and the arm stays lifted. "
        "Requires observed actuator forces to check the drive limits.",
        "real_labs.nasa_operations",
        "run_arm_clearance_yaw",
    ),
    _Operation(
        "nasa_planetary",
        "rover_traverse_and_stage",
        "rover",
        "Traverse and stage rover sample",
        "Run the authored six-wheel, paired-gripper sample route from home. "
        "Requires observed actuator forces to check the drive limits. "
        "This is an inert sample-transfer operation.",
        "real_labs.nasa_operations",
        "run_traverse_and_stage",
    ),
    _Operation(
        "rochester_genomics",
        "plate_loading",
        "liquid_handler",
        "Load and retrieve plate",
        "Run the ordered lift, load, close, retrieve and park cycle from home. "
        "The free plate must start on its staging supports; no liquid is processed.",
        "real_labs.rochester_operations",
        "run_plate_loading",
    ),
    _Operation(
        "leeds_environmental_tribology",
        "loaded_slide",
        "tribology",
        "Run loaded sliding",
        "Load, slide and retract the contact fixture from its centered home "
        "with the access door open. Forces are simulated, not calibrated measurements.",
        "real_labs.leeds_motion",
        "run_loaded_slide",
    ),
    _Operation(
        "leeds_environmental_tribology",
        "retract",
        "tribology",
        "Retract contact fixture",
        "Unload the measured contact while holding the specimen position. "
        "Recovery requires the fixture to remain within its declared limits.",
        "real_labs.leeds_operations",
        "run_retract",
    ),
)
_GUARDS = {
    "manchester_collaborative_robotics": "real_labs.manchester_operations",
    "nasa_planetary": "real_labs.nasa_operations",
    "rochester_genomics": "real_labs.rochester_operations",
    "leeds_environmental_tribology": "real_labs.leeds_operations",
}


def available_operations(session):
    """Return descriptions, without stepping physics or claiming readiness."""
    return [item.metadata() for item in _scene_operations(session)]


def _scene_operations(session):
    from real_labs.inspection import recipe_for

    scene = session.manifest.get("id")
    operations = tuple(item for item in _OPERATIONS if item.scene == scene)
    recipe = recipe_for(scene)
    if recipe is not None:
        operations += (_Operation(
            scene, "dry_inspection", recipe["devices"][0]["equipment"],
            "Inspect mechanisms and return",
            "Execute the authored dry access and positioning sequence, observe "
            "actual rigid specimen poses and return each mechanism to its "
            "measured starting position. No scientific process is simulated.",
            "real_labs.inspection", "run_inspection",
        ),)
    return operations


def control_policy(session):
    """Describe UI restrictions without changing the model's physical ranges."""
    scene = session.manifest.get("id")
    if scene == "manchester_collaborative_robotics":
        return [
            dict(
                equipment_id=owner,
                control=control,
                mode="operation_only",
                operation="empty_tool_inspection",
            )
            for owner, control in session.controls
            if owner == "collaborative_arm"
        ]
    if scene == "nasa_planetary":
        from real_labs.nasa_operations import operation_for_control

        return [
            dict(
                equipment_id=owner,
                control=control,
                mode="operation_only",
                operation=operation_for_control(control),
            )
            for owner, control in session.controls
            if owner == "rover" and operation_for_control(control) is not None
        ]
    if scene == "rochester_genomics":
        return [
            dict(
                equipment_id="liquid_handler",
                control=axis,
                mode="operation_only",
                operation="plate_loading",
            )
            for axis in ("loader_x", "loader_y")
        ]
    if scene == "leeds_environmental_tribology":
        from real_labs.leeds_operations import DRY_APPROACH_MAX_M

        return [
            dict(
                equipment_id="tribology",
                control="contact_approach",
                mode="bounded",
                operation="loaded_slide",
                lower=0.0,
                upper=DRY_APPROACH_MAX_M,
            )
        ]
    return []


def _operation(session, operation_id):
    scene = session.manifest.get("id")
    if isinstance(operation_id, str):
        for item in _scene_operations(session):
            if (item.scene, item.id) == (scene, operation_id):
                return item
    raise ValueError(f"Unknown operation for scene {scene}: {operation_id!r}")


def validate_operation(session, operation_id):
    """Read-only allowlist, model and state checks before accepting a request.

    Workflow-specific home/contact conditions are checked by the selected runner
    before motion; this preflight does not replace those checks.
    """
    _require_recovered(session)
    operation = _operation(session, operation_id)
    require_binding(session, operation.scene)
    data = session.data
    if not math.isfinite(float(data.time)) or any(
        not np.isfinite(getattr(data, field)).all()
        for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
    ):
        raise ValueError("Operation requires finite simulation state")
    if np.any(data.warning.number):
        raise ValueError("Operation requires resolved physics warnings")
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("External forces are outside the authored operation contract")
    return operation.metadata()


def run_operation(session, operation_id):
    """Execute an authored workflow and retain its measured evidence unchanged.

    Revalidate immediately before dispatch, even if a client preflighted earlier.
    Exceptions and measured failures block subsequent motion until an explicit
    successful reset; no implicit reset or recovery motion occurs.
    """
    metadata = validate_operation(session, operation_id)
    operation = _operation(session, operation_id)
    runner = getattr(import_module(operation.module), operation.function)
    try:
        report = runner(session)
        if not isinstance(report, dict):
            raise RuntimeError("Operation returned no explicit boolean success result")
        success = report.get("success", report.get("passed"))
        if not isinstance(success, bool):
            raise RuntimeError("Operation returned no explicit boolean success result")
    except Exception as error:
        session.operation_failure = {
            "operation": operation_id,
            "error": f"{type(error).__name__}: {error}",
        }
        raise
    if not success:
        session.operation_failure = {
            "operation": operation_id,
            "error": str(report.get("failure") or report.get("error") or "Operation reported failure"),
        }
    return dict(report, success=success, operation=metadata)


def _require_recovered(session):
    if getattr(session, "operation_failure", None):
        raise RuntimeError("An operation failed; explicitly reset the scene before further motion")


def validate_command(session, equipment_id, control, target):
    """Apply declared scene policies in addition to normal bounded controls."""
    _require_recovered(session)
    retired = session.equipment.get(equipment_id, {}).get("retired_controls", {})
    if control in retired:
        raise KeyError(retired[control])
    key = equipment_id, control
    spec = session.mechanisms.get(key) or session.controls.get(key)
    if spec is None:
        raise KeyError(key)
    target = float(target)
    if not math.isfinite(target) or not spec.lower <= target <= spec.upper:
        raise ValueError("Target must be finite and inside the retained control range")
    module = _GUARDS.get(session.manifest.get("id"))
    if module is not None:
        import_module(module).validate_command(session, equipment_id, control, target)


def command(session, equipment_id, control, target, *, duration=2.0):
    """Run a scalar or coupled command after its nonmutating policy checks."""
    duration = float(duration)
    if not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("Duration must be in (0, 30] seconds")
    validate_command(session, equipment_id, control, target)
    return session.command(equipment_id, control, target, duration=duration)
