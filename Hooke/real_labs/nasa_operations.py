"""Ordered access to the grounded rover's coupled controls.

The source control ranges and diagnostic targets remain unchanged. A wheel,
steering joint or gripper finger is not an independent navigation/grasp command;
the authored rover task coordinates these actuators and checks actual contacts.
"""

import math

import numpy as np

from real_labs.nasa_rover_task import NasaRoverTask, _require_binding

SCENE = "nasa_planetary"
OWNER = "rover"
CLEARANCE_YAW = "arm_clearance_yaw"
TRAVERSE_AND_STAGE = "rover_traverse_and_stage"


class OperationRequired(ValueError):
    """A scalar target needs the corresponding ordered rover operation."""

    def __init__(self, operation, reason):
        self.operation, self.reason = operation, reason
        super().__init__(f"{reason}; use the {operation} operation")


def operation_for_control(control):
    """Return a policy for coupled controls, without changing their ranges."""
    if control == "arm_base_yaw":
        return CLEARANCE_YAW
    if control.startswith(("wheel_", "steer_wheel_", "arm_gripper_")):
        return TRAVERSE_AND_STAGE
    return None


def validate_command(session, equipment_id, control, target):
    """Reject an unsafe independent command before changing state or time."""
    if session.manifest.get("id") != SCENE or equipment_id != OWNER:
        return
    _require_binding(session)
    spec = session.controls[equipment_id, control]
    target = float(target)
    if not math.isfinite(target) or not spec.lower <= target <= spec.upper:
        raise ValueError("Target must be finite and inside the retained range")
    operation = operation_for_control(control)
    if operation is not None:
        reason = (
            "Arm rotation requires lifting clear of the pickup fixture"
            if operation == CLEARANCE_YAW
            else "Wheel, steering and gripper motion requires coordinated control"
        )
        raise OperationRequired(operation, reason)


def run_arm_clearance_yaw(session):
    """Lift away from the pickup fixture, then reach the original yaw target.

    This finite inspection starts at authored home, keeps the gripper open, and
    leaves the arm lifted at pi/2. It uses the existing task controller and all
    of its contact, direct-geometry, effort and tracking limits. No reset or
    full traversal/sample-transfer completion is implied.
    """
    task = NasaRoverTask(session, record=False)
    task.operation = CLEARANCE_YAW
    sample_start = task.d.xpos[task.sample].copy()
    task.phase = "settle"
    task.wait(2)
    task.checkpoint("settled")
    orientation = task.d.site_xmat[task.tcp].reshape(3, 3).copy()
    clearance = np.array([-0.08, 0.0, 0.065])
    task.phase = "lift_clearance"
    joints, _ = task.ik(task.d.site_xpos[task.tcp] + clearance, orientation)
    task.arm(joints)
    task.checkpoint("lift_clearance_complete")
    task.phase = "yaw_after_clearance"
    joints = task.target.copy()
    joints[0] = math.pi / 2
    task.arm(joints)
    task.checkpoint("yaw_complete")
    sample_displacement = float(np.linalg.norm(task.d.xpos[task.sample] - sample_start))
    if sample_displacement > 0.002:
        raise RuntimeError("Arm clearance displaced the sample from its pickup fixture")
    _require_binding(session)
    task.completed = True
    return dict(
        task.report(),
        target_yaw_rad=math.pi / 2,
        measured_yaw_rad=float(task.d.qpos[task.armq[0]]),
        clearance_vector_m=clearance.tolist(),
        sample_displacement_m=sample_displacement,
        raw_control_results_replaced=False,
        full_traversal_completed=False,
        scope=(
            "Authored-home lift and pi/2 arm rotation with unchanged source "
            "guards; the sample stays on its pickup fixture. The arm remains "
            "lifted. No full sample-transfer or scientific-process claim."
        ),
    )


def run_traverse_and_stage(session):
    """Run the existing six-wheel/paired-gripper source task from its home."""
    task = NasaRoverTask(session, record=False)
    task.operation = TRAVERSE_AND_STAGE
    return task.perform()
