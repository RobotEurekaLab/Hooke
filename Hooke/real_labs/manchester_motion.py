"""Measured empty-hand inspection of Manchester's retained tabletop block.

The Gen3 is a licensed substitute, not the photographed robot. Its fixed hand
has exterior contacts but no finger actuation. This operation checks positioning
and return; it does not claim a grasp, image measurement or physical experiment.
"""

from copy import copy
from dataclasses import replace
import math

import mujoco
import numpy as np

from real_labs.motion import JointMotion, PlanningError


# Joint 7 is an unneeded tool-axis roll. Keep its existing servo holding its
# measured start; shared execution checks it like every other inactive control.
CONTROLS = tuple(f"joint_{index}_offset" for index in range(1, 7))


def _skew(vector):
    x, y, z = vector
    return np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))


def _direction_error(actual, desired):
    return math.acos(float(np.clip(np.dot(actual, desired), -1.0, 1.0)))


def plan_inspection(motion, tool_site, target, direction, *, seed=7, attempts=24):
    """Fit tool position/direction privately, then check the whole articulated route.

    Tool roll remains free. Finite IK and motion-search failure is reported as a
    budgeted failure, not global unreachability. No live qpos/ctrl is assigned.
    """
    target = np.asarray(target, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (
        target.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1.0)
    ):
        raise ValueError("Inspection needs a finite position and unit tool direction")
    if not 1 <= attempts <= 100:
        raise ValueError("IK attempts must be between 1 and 100")
    motion._refresh_checker()
    motion._require_stationary_world()
    contacts = motion.configuration_contacts(motion.measured())
    if contacts:
        raise PlanningError(
            f"Start violates full-hand inspection clearance: {contacts[0]}"
        )
    site = motion.model.site(tool_site).id
    if motion.model.site_bodyid[site] not in motion.moving_bodies:
        raise ValueError("Tool site must belong to the selected arm")
    random = np.random.default_rng(seed)
    jacp = np.zeros((3, motion.model.nv))
    jacr = np.zeros_like(jacp)
    for attempt in range(attempts):
        q = (
            motion.measured()
            if attempt == 0
            else random.uniform(motion.lower, motion.upper)
        )
        for _ in range(400):
            motion._pose(q)
            axis = motion.data.site_xmat[site].reshape(3, 3)[:, 2]
            position_error = target - motion.data.site_xpos[site]
            direction_error = direction - axis
            if (
                np.linalg.norm(position_error) <= 0.00025
                and np.linalg.norm(direction_error) <= 0.001
            ):
                break
            mujoco.mj_jacSite(motion.model, motion.data, jacp, jacr, site)
            jacobian = np.vstack(
                (jacp[:, motion.dadr], -0.2 * _skew(axis) @ jacr[:, motion.dadr])
            )
            error = np.r_[position_error, 0.2 * direction_error]
            increment = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 4e-6 * np.eye(6), error
            )
            increment *= min(1.0, 0.15 / max(float(np.max(np.abs(increment))), 1e-12))
            q = np.clip(q + increment, motion.lower, motion.upper)
        motion._pose(q)
        actual_axis = motion.data.site_xmat[site].reshape(3, 3)[:, 2]
        if (
            np.linalg.norm(target - motion.data.site_xpos[site]) > 0.00025
            or _direction_error(actual_axis, direction) > 0.001
        ):
            continue
        try:
            path = motion.plan_joint_target(q, seed=seed + attempt)
        except PlanningError:
            continue
        return replace(
            path,
            target_site=tool_site,
            target_position_m=tuple(map(float, target)),
            description="Full-hand collision-checked downward inspection and standoff",
        )
    raise PlanningError(
        "No position/direction goal and collision-clear route found within budget"
    )


class _MeasuredClearance:
    """Check actual complete live configurations in a private collision model."""

    def __init__(self, motion):
        self.model = copy(motion.model)
        self.data = mujoco.MjData(self.model)
        self.moving_geoms = motion.moving_geoms
        self.clearance = motion.clearance
        self.observer_entries = 0
        # Count completed geometry checks, not merely observer invocations.
        self.samples = 0
        self.minimum_observed_m = None
        self.violations = []
        self.held = {
            f"{key[0]}.{key[1]}": (
                control,
                float(motion.session.data.ctrl[control.actuator]),
            )
            for key, control in motion.session.controls.items()
            if control.actuator not in motion.actuators
        }
        self.held_drift = {key: 0.0 for key in self.held}

    def __call__(self, model, data):
        self.observer_entries += 1
        # Copy full state, including inactive mechanism and mocap motion. This
        # check does not substitute planned positions for measured positions.
        self.data.qpos[:] = data.qpos
        self.data.qvel[:] = data.qvel
        self.data.mocap_pos[:] = data.mocap_pos
        self.data.mocap_quat[:] = data.mocap_quat
        self.data.eq_active[:] = data.eq_active
        mujoco.mj_forward(self.model, self.data)
        for key, (control, target) in self.held.items():
            if data.ctrl[control.actuator] != target:
                raise RuntimeError("inactive_inspection_command_changed")
            measured = (
                data.qpos[model.jnt_qposadr[control.joint]]
                if control.mode == "position"
                else data.qvel[model.jnt_dofadr[control.joint]]
            )
            self.held_drift[key] = max(
                self.held_drift[key], abs(float(measured) - target)
            )
            tolerance = 0.0002 if control.units == "m" else 0.003
            if self.held_drift[key] > tolerance:
                raise RuntimeError("inactive_inspection_control_drift")
        contacts = [
            c
            for c in self.data.contact
            if c.geom1 in self.moving_geoms or c.geom2 in self.moving_geoms
        ]
        if contacts:
            minimum = min(float(c.dist) for c in contacts)
            self.minimum_observed_m = (
                minimum
                if self.minimum_observed_m is None
                else min(self.minimum_observed_m, minimum)
            )
        for contact in contacts:
            if contact.dist < self.clearance:
                self.violations.append(
                    dict(
                        time_s=float(data.time),
                        distance_m=float(contact.dist),
                        geoms=[
                            model.geom(contact.geom1).name,
                            model.geom(contact.geom2).name,
                        ],
                    )
                )
        self.samples += 1
        if self.violations:
            raise RuntimeError("measured_inspection_clearance_violation")

    def result(self, *, executed_steps):
        """Keep partial observations separate from coverage of the whole leg."""
        coverage_complete = executed_steps > 0 and self.samples == executed_steps
        checked_bound = self.clearance if self.samples and not self.violations else None
        return dict(
            executed_steps=executed_steps,
            observer_entries=self.observer_entries,
            checked_steps=self.samples,
            unchecked_steps=executed_steps - self.samples,
            coverage_complete=coverage_complete,
            # Retain the original field as an alias for completed checks.
            samples=self.samples,
            required_m=self.clearance,
            sampled_clearance_lower_bound_m=(
                checked_bound if coverage_complete else None
            ),
            checked_steps_clearance_lower_bound_m=checked_bound,
            minimum_contact_within_query_margin_m=self.minimum_observed_m,
            inactive_control_maximum_drift=self.held_drift,
            violations=self.violations,
            interpretation="Bounds describe completed geometry checks. A whole-leg sampled bound requires a completed check for every executed step and no violation. Unchecked steps are unverified; these are neither continuous-time guarantees nor exact minimum distances.",
        )


def inspect_manchester(session):
    """Execute a downward-facing empty-tool standoff and return without resetting."""
    equipment = session.equipment.get("collaborative_arm", {})
    operation = equipment.get("inspection_operation")
    if not operation:
        raise ValueError("Scene has no Manchester inspection operation")
    motion = JointMotion(session, "collaborative_arm", CONTROLS, clearance_m=0.015)
    model, data = session.model, session.data
    tool_name = operation["tool_site"]
    site = model.site(tool_name).id
    hand_geoms = np.flatnonzero(model.geom_bodyid == model.site_bodyid[site])
    if not len(hand_geoms) or not np.all(
        (model.geom_contype[hand_geoms] & 1) | (model.geom_conaffinity[hand_geoms] & 1)
    ):
        raise PlanningError(
            "Every fixed-hand exterior must retain scene collision coverage"
        )
    target = data.site_xpos[model.site(operation["standoff_site"]).id].copy()
    body = model.body(equipment["body"]).id
    direction = data.xmat[body].reshape(3, 3) @ np.asarray(
        operation["tool_axis_in_equipment_frame"]
    )
    initial = motion.measured()
    initial_position = data.site_xpos[site].copy()
    initial_direction = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
    object_site = model.site(operation["object_site"]).id
    object_start = data.site_xpos[object_site].copy()
    plan = plan_inspection(motion, tool_name, target, direction)

    def execute(leg, desired_direction):
        observer = _MeasuredClearance(motion)
        result = motion.execute(leg, position_tolerance_m=0.002, step_observer=observer)
        result["measured_clearance"] = observer.result(
            executed_steps=result["physics_steps"]
        )
        result["final_full_qpos"] = data.qpos.tolist()
        result["final_tool_direction_world"] = (
            data.site_xmat[site].reshape(3, 3)[:, 2].tolist()
        )
        result["tool_direction_error_rad"] = _direction_error(
            np.asarray(result["final_tool_direction_world"]), desired_direction
        )
        if (
            result["status"] == "succeeded"
            and result["tool_direction_error_rad"] > 0.01
        ):
            result.update(
                status="failed",
                failure="tool_direction_not_reached",
                hold_requested=motion._request_hold(),
            )
        return result

    outward = execute(plan, direction)
    returning = None
    if outward["status"] == "succeeded":
        try:
            return_plan = replace(
                motion.plan_joint_target(initial),
                target_site=tool_name,
                target_position_m=tuple(map(float, initial_position)),
                description="Return the same empty-tool site to its measured initial pose",
            )
            returning = execute(return_plan, initial_direction)
        except PlanningError as error:
            returning = dict(
                status="failed",
                failure=f"return_planning: {error}",
                hold_requested=motion._request_hold(),
                reset_performed=False,
            )
    returned = returning is not None and returning["status"] == "succeeded"
    return dict(
        status="succeeded" if returned else "failed",
        task="Manchester empty-tool standoff inspection and return",
        target_position_m=target.tolist(),
        target_direction_world=direction.tolist(),
        tool_site=tool_name,
        object_site=operation["object_site"],
        initial_joint_positions=initial.tolist(),
        initial_tool_position_m=initial_position.tolist(),
        outward=outward,
        returning=returning,
        returned_to_start=returned,
        final_tool_return_error_m=float(
            np.linalg.norm(data.site_xpos[site] - initial_position)
        ),
        object_displacement_m=float(
            np.linalg.norm(data.site_xpos[object_site] - object_start)
        ),
        fixed_tool_roll_requested=False,
        grasp_attempted=False,
        object_transfer_validated=False,
        scientific_process_validated=False,
        native_runtime_validated=False,
    )


def main():
    """Run the CPU demonstration explicitly and write its measured report."""
    import argparse
    import json
    from pathlib import Path
    import xml.etree.ElementTree as ET

    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report path under an ignored artifact directory",
    )
    args = parser.parse_args()
    root, manifest = build_scene("manchester_collaborative_robotics")
    session = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = inspect_manchester(session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
