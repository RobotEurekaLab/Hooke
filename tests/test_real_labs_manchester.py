"""Full-hand obstacle retention, supported assembly and measured inspection."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.manchester_motion import CONTROLS, _MeasuredClearance, inspect_manchester
from real_labs.motion import JointMotion, PlanningError, _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.university_robotics_wave1 import GEN3, HOME


def session_with_obstacle(position=None):
    root, metadata = build_scene("manchester_collaborative_robotics")
    if position is not None:
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="retained_test_obstacle",
            type="sphere",
            size=".008",
            pos=" ".join(map(str, position)),
        )
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), metadata
    )


class ManchesterInspectionTests(unittest.TestCase):
    def test_source_ranges_ids_and_hand_contacts_are_preserved(self):
        session = session_with_obstacle()
        source = ET.parse(GEN3 / "gen3.xml").getroot()
        for index, joint in enumerate(source.findall(".//worldbody//joint")):
            control = session.controls["collaborative_arm", f"joint_{index+1}_offset"]
            expected = (
                np.array(list(map(float, joint.get("range").split()))) - HOME[index]
                if joint.get("range")
                else np.array([-np.pi, np.pi])
            )
            np.testing.assert_allclose(
                [control.lower, control.upper], expected, atol=1e-9
            )
        hand = session.model.body(
            "collaborative_arm__original_fixed_three_finger_tool"
        ).id
        geoms = np.flatnonzero(session.model.geom_bodyid == hand)
        self.assertEqual(len(geoms), 13)
        np.testing.assert_array_equal(session.model.geom_contype[geoms], np.ones(13))
        np.testing.assert_array_equal(
            session.model.geom_conaffinity[geoms], np.ones(13)
        )
        self.assertEqual(session.model.body_jntnum[hand], 0)

    def test_clamp_pads_and_service_endpoints_have_continuous_support(self):
        session = session_with_obstacle()
        model, data = session.model, session.data

        def distance(first, second):
            return mujoco.mj_geomDistance(
                model, data, model.geom(first).id, model.geom(second).id, 1, None
            )

        prefix = "collaborative_arm__"
        for side, jaw in (("left", "box_46"), ("right", "box_47")):
            pad = prefix + "retaining_pad_" + side
            self.assertLessEqual(distance(pad, prefix + jaw), 1e-8)
            self.assertLessEqual(distance(pad, prefix + "box_45"), 1e-8)
        pairs = (
            ("gen3_collision_13", "service_base_socket"),
            ("service_base_socket", "service_base_coupling"),
            ("service_base_coupling", "service_base_strain_relief"),
            ("service_base_strain_relief", "service_cable_0_0"),
            ("service_cable_2_11", "service_parked_strain_relief"),
            ("service_parked_strain_relief", "service_parked_coupling"),
            ("service_parked_coupling", "service_parking_plate"),
            ("service_parking_plate", "box_2"),
        )
        for first, second in pairs:
            self.assertLessEqual(
                distance(prefix + first, prefix + second), 1e-7, (first, second)
            )
        cable = [
            prefix + f"service_cable_{curve}_{segment}"
            for curve in range(3)
            for segment in range(12)
        ]
        for first, second in zip(cable, cable[1:]):
            self.assertLessEqual(distance(first, second), 1e-8)

    def test_unsafe_hand_table_pose_remains_blocked_without_moving_live_state(self):
        session = session_with_obstacle()
        motion = JointMotion(session, "collaborative_arm", CONTROLS)
        target = motion.measured()
        target[1] = 0.85820061
        contacts = motion.configuration_contacts(target)
        self.assertTrue(
            any(
                "robot_table_top" in row["geoms"]
                and any("capsule_34" in name for name in row["geoms"])
                for row in contacts
            )
        )
        before = session.data.qpos.copy(), session.data.ctrl.copy()
        with self.assertRaisesRegex(PlanningError, "goal violates collision clearance"):
            motion.plan_joint_target(target)
        np.testing.assert_array_equal(session.data.qpos, before[0])
        np.testing.assert_array_equal(session.data.ctrl, before[1])
        self.assertEqual(session.data.time, 0)

    def test_hand_only_obstacle_is_not_ignored_and_disabled_hand_is_rejected(self):
        original = session_with_obstacle()
        fingertip = original.data.geom_xpos[
            original.model.geom("collaborative_arm__sphere_33").id
        ]
        session = session_with_obstacle(fingertip)
        before = _model_signature(session.model)
        with self.assertRaisesRegex(PlanningError, "Start violates full-hand"):
            inspect_manchester(session)
        self.assertEqual(before, _model_signature(session.model))
        self.assertEqual(session.data.time, 0)
        self.assertEqual(session.model.geom("retained_test_obstacle").contype, 1)
        geom = original.model.geom("collaborative_arm__capsule_34").id
        original.model.geom_contype[geom] = original.model.geom_conaffinity[geom] = 0
        with self.assertRaisesRegex(PlanningError, "Every fixed-hand exterior"):
            inspect_manchester(original)
        self.assertEqual(original.data.time, 0)

    def test_live_clearance_observer_uses_measured_hand_not_planned_state(self):
        original = session_with_obstacle()
        fingertip = original.data.geom_xpos[
            original.model.geom("collaborative_arm__sphere_33").id
        ].copy()
        fingertip[
            2
        ] += 0.030  # 10 mm surface gap: no penetration, but below required 15 mm.
        session = session_with_obstacle(fingertip)
        motion = JointMotion(session, "collaborative_arm", CONTROLS)
        observer = _MeasuredClearance(motion)
        mujoco.mj_step(session.model, session.data)
        mujoco.mj_forward(session.model, session.data)
        with self.assertRaisesRegex(
            RuntimeError, "measured_inspection_clearance_violation"
        ):
            observer(session.model, session.data)
        self.assertEqual(observer.samples, 1)
        self.assertTrue(observer.violations)
        self.assertTrue(
            any("retained_test_obstacle" in row["geoms"] for row in observer.violations)
        )

    def test_measured_downward_inspection_and_return_preserves_static_world_and_roll(
        self,
    ):
        session = session_with_obstacle()
        model = session.model
        signature = _model_signature(model)
        inactive = session.controls["collaborative_arm", "joint_7_offset"]
        initial_command = float(session.data.ctrl[inactive.actuator])
        report = inspect_manchester(session)
        self.assertEqual(
            report["status"], "succeeded", report["outward"].get("failure")
        )
        np.testing.assert_allclose(report["target_position_m"], [-0.15, -0.35, 1.02])
        np.testing.assert_allclose(report["target_direction_world"], [0, 0, -1])
        self.assertLess(report["outward"]["cartesian_error_m"], 0.002)
        self.assertLess(report["final_tool_return_error_m"], 0.002)
        self.assertEqual(report["object_displacement_m"], 0)
        self.assertFalse(report["grasp_attempted"])
        for leg in (report["outward"], report["returning"]):
            self.assertEqual(leg["status"], "succeeded", leg.get("failure"))
            self.assertLess(leg["tool_direction_error_rad"], 0.01)
            clearance = leg["measured_clearance"]
            self.assertEqual(clearance["samples"], leg["physics_steps"])
            self.assertEqual(clearance["executed_steps"], leg["physics_steps"])
            self.assertEqual(clearance["observer_entries"], leg["physics_steps"])
            self.assertEqual(clearance["checked_steps"], leg["physics_steps"])
            self.assertEqual(clearance["unchecked_steps"], 0)
            self.assertTrue(clearance["coverage_complete"])
            self.assertGreater(clearance["samples"], 1000)
            self.assertEqual(clearance["sampled_clearance_lower_bound_m"], 0.015)
            self.assertFalse(clearance["violations"])
            self.assertLessEqual(
                clearance["inactive_control_maximum_drift"][
                    "collaborative_arm.joint_7_offset"
                ],
                0.003,
            )
            self.assertFalse(leg["collisions"])
            self.assertFalse(leg["held_control_drift"])
            self.assertFalse(leg["reset_performed"])
            self.assertEqual(leg["maximum_stationary_geometry_displacement_m"], 0)
        self.assertEqual(session.data.ctrl[inactive.actuator], initial_command)
        self.assertEqual(signature, _model_signature(model))

    def test_tracking_abort_does_not_certify_unobserved_final_step(self):
        session = session_with_obstacle()
        model = session.model
        signature = _model_signature(model)
        joint = model.joint("collaborative_arm__joint_6_offset").id
        session.data.qfrc_applied[model.jnt_dofadr[joint]] = 100
        result = inspect_manchester(session)
        outward = result["outward"]
        clearance = outward["measured_clearance"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(outward["failure"], "trajectory_tracking_limit")
        self.assertEqual(outward["physics_steps"], 10)
        self.assertEqual(clearance["executed_steps"], 10)
        self.assertEqual(clearance["observer_entries"], 9)
        self.assertEqual(clearance["checked_steps"], 9)
        self.assertEqual(clearance["unchecked_steps"], 1)
        self.assertFalse(clearance["coverage_complete"])
        self.assertIsNone(clearance["sampled_clearance_lower_bound_m"])
        self.assertEqual(clearance["checked_steps_clearance_lower_bound_m"], 0.015)
        self.assertTrue(outward["hold_requested"])
        self.assertFalse(outward["reset_performed"])
        self.assertIsNone(result["returning"])
        self.assertAlmostEqual(session.data.time, 10 * model.opt.timestep)
        np.testing.assert_array_equal(session.data.qpos, outward["final_full_qpos"])
        self.assertEqual(signature, _model_signature(model))

    def test_inactive_control_abort_before_geometry_query_has_zero_coverage(self):
        session = session_with_obstacle()
        model = session.model
        signature = _model_signature(model)
        motion = JointMotion(session, "collaborative_arm", CONTROLS)
        empty = _MeasuredClearance(motion).result(executed_steps=0)
        self.assertFalse(empty["coverage_complete"])
        self.assertIsNone(empty["sampled_clearance_lower_bound_m"])
        self.assertIsNone(empty["checked_steps_clearance_lower_bound_m"])
        joint = model.joint("collaborative_arm__joint_7_offset").id
        session.data.qfrc_applied[model.jnt_dofadr[joint]] = 100
        result = inspect_manchester(session)
        outward = result["outward"]
        clearance = outward["measured_clearance"]
        self.assertEqual(outward["failure"], "inactive_inspection_control_drift")
        self.assertEqual(outward["physics_steps"], 1)
        self.assertEqual(clearance["executed_steps"], 1)
        self.assertEqual(clearance["observer_entries"], 1)
        self.assertEqual(clearance["checked_steps"], 0)
        self.assertEqual(clearance["unchecked_steps"], 1)
        self.assertFalse(clearance["coverage_complete"])
        self.assertIsNone(clearance["sampled_clearance_lower_bound_m"])
        self.assertIsNone(clearance["checked_steps_clearance_lower_bound_m"])
        self.assertTrue(outward["hold_requested"])
        self.assertFalse(outward["reset_performed"])
        self.assertIsNone(result["returning"])
        np.testing.assert_array_equal(session.data.qpos, outward["final_full_qpos"])
        self.assertEqual(signature, _model_signature(model))

    def test_settling_failure_can_still_have_complete_clearance_coverage(self):
        session = session_with_obstacle()
        model = session.model
        joint = model.joint("collaborative_arm__joint_6_offset").id
        session.data.qfrc_applied[model.jnt_dofadr[joint]] = 30
        motion = JointMotion(session, "collaborative_arm", CONTROLS)
        observer = _MeasuredClearance(motion)
        report = motion.execute(
            motion.plan_joint_target(motion.measured()), step_observer=observer
        )
        coverage = observer.result(executed_steps=report["physics_steps"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"], "waypoint_did_not_settle")
        self.assertGreater(report["physics_steps"], 1000)
        self.assertEqual(coverage["checked_steps"], report["physics_steps"])
        self.assertEqual(coverage["observer_entries"], report["physics_steps"])
        self.assertTrue(coverage["coverage_complete"])
        self.assertEqual(coverage["sampled_clearance_lower_bound_m"], 0.015)
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])


if __name__ == "__main__":
    unittest.main()
