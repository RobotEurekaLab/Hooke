"""Measured dry alignment of an open calibration slot around a fixed wire.

This operates the authored Birmingham inspection aid, not an EDM controller.
Planning uses a private collision model; execution steps the live actuators.
An explicit session.reset() starts a new episode. No operation here resets or
teleports state, and a failed movement only requests a measured position hold.
"""

from dataclasses import replace
from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError


@lru_cache(maxsize=1)
def _authored_contract():
    """Compile only the authored mechanism once, without a room or live state.

    Using the geometry builder avoids a second hand-maintained list of bore
    dimensions. This immutable reference is compared before any task command.
    """
    from real_labs.birmingham_fixture import add_fixture
    from real_labs.equipment import _Builder

    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="wire_reference")
    add_fixture(_Builder(root, "wire_reference"), base)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _require_authored_interface(session, equipment):
    """Reject missing, disabled, reshaped or rebound mechanical interfaces."""
    actual, expected = session.model, _authored_contract()

    def local_name(name):
        return equipment + name.removeprefix("wire_reference")

    def close(first, second):
        return np.allclose(first, second, rtol=0, atol=1e-10)

    def same_rotation(first, second):
        return abs(abs(float(np.dot(first, second))) - 1) < 1e-10

    body_ids = set()
    for i in range(1, expected.nbody):
        name = local_name(expected.body(i).name)
        body = actual.body(name).id
        body_ids.add(body)
        if i == 1:
            if actual.body_weldid[body] != 0:
                raise ValueError("Wire fixture base must remain fixed to the world")
            continue
        parent = local_name(expected.body(expected.body_parentid[i]).name)
        if (
            actual.body(actual.body_parentid[body]).name != parent
            or not close(actual.body_pos[body], expected.body_pos[i])
            or not same_rotation(actual.body_quat[body], expected.body_quat[i])
        ):
            raise ValueError(f"Authored wire fixture body binding changed: {name}")
    for i in range(expected.ngeom):
        if not expected.geom_contype[i]:
            continue
        name = local_name(expected.geom(i).name)
        geom = actual.geom(name).id
        body = local_name(expected.body(expected.geom_bodyid[i]).name)
        if (
            actual.body(actual.geom_bodyid[geom]).name != body
            or actual.geom_type[geom] != expected.geom_type[i]
            or actual.geom_contype[geom] != expected.geom_contype[i]
            or actual.geom_conaffinity[geom] != expected.geom_conaffinity[i]
            or not close(actual.geom_pos[geom], expected.geom_pos[i])
            or not close(actual.geom_size[geom], expected.geom_size[i])
            or not same_rotation(actual.geom_quat[geom], expected.geom_quat[i])
        ):
            raise ValueError(f"Authored wire fixture collider changed: {name}")
    for i in range(expected.njnt):
        expected_name = expected.joint(i).name
        joint = actual.joint(local_name(expected_name)).id
        semantic = expected_name.split("__joint_", 1)[1]
        control = session.controls[equipment, semantic]
        if (
            control.joint != joint
            or actual.actuator_trnid[control.actuator, 0] != joint
            or actual.jnt_type[joint] != expected.jnt_type[i]
            or actual.jnt_limited[joint] != expected.jnt_limited[i]
            or actual.body(actual.jnt_bodyid[joint]).name
            != local_name(expected.body(expected.jnt_bodyid[i]).name)
            or not close(actual.jnt_axis[joint], expected.jnt_axis[i])
            or not close(actual.jnt_pos[joint], expected.jnt_pos[i])
            or not close(actual.jnt_range[joint], expected.jnt_range[i])
        ):
            raise ValueError("Authored wire fixture joint binding or limits changed")
    for signature in actual.exclude_signature:
        if (int(signature) >> 16) in body_ids or (int(signature) & 65535) in body_ids:
            raise ValueError("Body collision exclusions affect the wire fixture")
    for i in range(expected.nsite):
        site = actual.site(local_name(expected.site(i).name)).id
        if actual.body(actual.site_bodyid[site]).name != local_name(
            expected.body(expected.site_bodyid[i]).name
        ) or not close(actual.site_pos[site], expected.site_pos[i]):
            raise ValueError("Authored wire fixture datum binding changed")
    expected_site = equipment + "__site_calibration_slot"
    if session.equipment[equipment]["sites"].get("calibration_slot") != expected_site:
        raise ValueError("Wire alignment must use the authored calibration-slot datum")


class WireAlignment:
    """Finite-sampled stationary-world planning with measured wire clearance."""

    def __init__(self, session, equipment="primary_apparatus"):
        self.session = session
        self.equipment = equipment
        item = session.equipment[equipment]
        if not item.get("wire_alignment"):
            raise ValueError("Equipment has no authored wire-alignment interface")
        _require_authored_interface(session, equipment)
        self.motion = JointMotion(
            session, equipment, ("fixture_x", "fixture_y"), clearance_m=0.00015
        )
        self.motion.resolution[:] = 0.0005
        self.site_name = item["sites"]["calibration_slot"]
        self.site = session.model.site(self.site_name).id
        self.wire = session.model.geom(equipment + "__wire_fixture_wire").id
        prefix = equipment + "__wire_fixture_"
        self.fixed_checks = np.array(
            [
                i
                for i in range(session.model.ngeom)
                if session.model.geom_contype[i]
                and (
                    session.model.geom(i).name
                    in {
                        prefix + "wire",
                        prefix + "lower_arm",
                        prefix + "upper_fork_-1",
                        prefix + "upper_fork_1",
                        prefix + "upper_wire_termination",
                    }
                    or session.model.geom(i).name.startswith(
                        (prefix + "lower_guide_proxy_", prefix + "upper_guide_proxy_")
                    )
                )
            ]
        )
        self.moving_checks = np.array(sorted(self.motion.moving_geoms))
        self.minimum_wire_gap = float("inf")
        self.minimum_guide_gap = float("inf")
        self.observed_steps = 0
        self.checked_steps = 0

    def _observe(self, model, data):
        # Include every enabled moving geom, even pairs MuJoCo parent filtering
        # could omit. Bounding spheres only discard provably distant pairs.
        self.observed_steps += 1
        fixed, moving = self.fixed_checks, self.moving_checks
        distance = np.linalg.norm(
            data.geom_xpos[fixed, None] - data.geom_xpos[moving], axis=2
        )
        bound = model.geom_rbound[fixed, None] + model.geom_rbound[moving] + 0.01
        wire_checked = False
        for row, column in zip(*np.where(distance <= bound)):
            first, second = int(fixed[row]), int(moving[column])
            gap = mujoco.mj_geomDistance(model, data, first, second, 0.01, None)
            # MuJoCo may return a distant-pair sentinel, or an uncapped analytic
            # distance, depending on shape. Uniform capping makes this a valid
            # lower bound even for broad-phase pairs omitted beyond 10 mm.
            gap = min(float(gap), 0.01)
            if first == self.wire:
                wire_checked = True
                self.minimum_wire_gap = min(self.minimum_wire_gap, float(gap))
            else:
                self.minimum_guide_gap = min(self.minimum_guide_gap, float(gap))
            if gap < 0.00010:
                raise RuntimeError("measured_wire_or_guide_clearance_lost")
        if not wire_checked:
            raise RuntimeError("wire_distance_not_observed")
        self.checked_steps += 1

    def move(self, target, *, step_observer=None):
        """Plan and execute one requested XY goal; blocked goals never advance time."""
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable")
        motion, session = self.motion, self.session
        self.minimum_wire_gap = float("inf")
        self.minimum_guide_gap = float("inf")
        self.observed_steps = 0
        self.checked_steps = 0
        try:
            plan = motion.plan_joint_target(target)
        except PlanningError as error:
            held = motion._request_hold()
            return dict(
                status="blocked",
                failure=str(error),
                hold_requested=held,
                reset_performed=False,
                physics_steps=0,
                wire_clearance_observed_steps=0,
                wire_clearance_fully_checked_steps=0,
                whole_leg_clearance_observed=False,
                minimum_measured_wire_gap_m=None,
                minimum_observed_guide_gap_m=None,
                final_joint_positions=motion.measured().tolist(),
            )
        # Preserve the requested target site explicitly and derive its position
        # from the compiled joint Jacobian rather than guessed room coordinates.
        jacobian = np.zeros((3, session.model.nv))
        mujoco.mj_jacSite(session.model, session.data, jacobian, None, self.site)
        goal = session.data.site_xpos[self.site] + jacobian[:, motion.dadr] @ (
            np.asarray(target, dtype=float) - motion.measured()
        )
        plan = replace(
            plan,
            target_site=self.site_name,
            target_position_m=tuple(goal),
            description="Dry wire/open-slot calibration fixture translation",
        )

        def observe(model, data):
            self._observe(model, data)
            if step_observer is not None:
                step_observer(model, data)

        result = motion.execute(
            plan, position_tolerance_m=0.0002, step_observer=observe
        )
        result["minimum_measured_wire_gap_m"] = (
            self.minimum_wire_gap if np.isfinite(self.minimum_wire_gap) else None
        )
        result["minimum_wire_clearance_lower_bound_m"] = result[
            "minimum_measured_wire_gap_m"
        ]
        result["wire_clearance_observed_steps"] = self.observed_steps
        result["wire_clearance_fully_checked_steps"] = self.checked_steps
        result["minimum_observed_guide_gap_m"] = (
            self.minimum_guide_gap if np.isfinite(self.minimum_guide_gap) else None
        )
        result["whole_leg_clearance_observed"] = bool(
            result["status"] == "succeeded"
            and result["physics_steps"] == self.checked_steps
            and self.checked_steps > 0
            and np.isfinite(self.minimum_wire_gap)
            and np.isfinite(self.minimum_guide_gap)
        )
        result["clearance_evidence_scope"] = (
            "Clearances are lower bounds capped at 10 mm, including legacy gap "
            "fields; nearer pairs use direct distances. A core failure may occur "
            "before the observer, and partial observations do not certify a whole leg"
        )
        result["clearance_reporting_cap_m"] = 0.01
        return result

    def run(self, *, step_observer=None):
        """Loading approach, XY alignment, entry, retraction and measured return.

        Start must be the retracted authored home within 0.2 mm. This explicit
        precondition avoids inventing a recovery route from a jammed fixture.
        """
        start = self.motion.measured().copy()
        if np.max(np.abs(start)) > 0.0002:
            raise PlanningError("Round trip requires the measured retracted home")
        phases = (
            ("loading_approach", [-0.06, -0.04]),
            ("lateral_alignment", [0, -0.04]),
            ("wire_entry", [0, 0.08]),
            ("wire_retraction", [0, -0.04]),
            ("return_home", start.tolist()),
        )
        reports = []
        aligned = None
        for phase, target in phases:
            report = self.move(target, step_observer=step_observer)
            reports.append(dict(phase=phase, **report))
            if report["status"] != "succeeded":
                break
            if phase == "wire_entry":
                data = self.session.data
                aligned = {
                    "datum_site": self.site_name,
                    "slot_datum_world_m": data.site_xpos[self.site].tolist(),
                    "simulation_time_s": float(data.time),
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist(),
                    "ctrl": data.ctrl.tolist(),
                    "model_signature": report["model_signature"],
                    "wire_axis_world_xy_m": data.geom_xpos[self.wire, :2].tolist(),
                    "lateral_alignment_error_m": float(
                        np.linalg.norm(
                            data.site_xpos[self.site, :2]
                            - data.geom_xpos[self.wire, :2]
                        )
                    ),
                }
        success = len(reports) == len(phases) and all(
            r["status"] == "succeeded" for r in reports
        )
        return dict(
            status="succeeded" if success else "failed",
            **combine_reset_reports(reports),
            phases=reports,
            alignment=aligned,
            measured_return_error_m=float(
                np.linalg.norm(self.motion.measured() - start)
            ),
            scope="Rigid supported wire and retained pre-slotted calibration coupon",
            limitations=[
                "Generic 0.20 mm wire and authored 8 mm calibration slot, not an OEM kerf",
                "Finite 0.5 mm planning samples and every-physics-step observations",
                "Arbitrary XY goals can be blocked; no collision-free full workspace claim",
                "No electrical discharge, wire tension, dielectric flow or material removal",
            ],
        )
