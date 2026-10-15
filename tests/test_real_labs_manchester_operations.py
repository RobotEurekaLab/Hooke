"""Unsafe scalar requests reject before mutation; the checked route remains usable."""

from copy import copy, deepcopy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs import operations
from real_labs.manchester_operations import OPERATION, OWNER, SCENE, OperationRequired
from real_labs.runtime import InstrumentSession


class ManchesterPublicOperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_unsafe_extreme_and_all_raw_arm_controls_reject_without_mutation(self):
        session = self.session()
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(session.model, kind))
        mujoco.mj_getState(session.model, session.data, before, kind)
        requests = [("joint_2_offset", 1.3062)] + [
            (control, .08) for owner, control in session.controls if owner == OWNER
        ]
        for control, target in requests:
            for invoke in (
                lambda: session.command(OWNER, control, target),
                lambda: operations.command(session, OWNER, control, target),
            ):
                with self.assertRaises(OperationRequired) as caught:
                    invoke()
                self.assertEqual(caught.exception.operation, OPERATION)
                after = np.empty_like(before)
                mujoco.mj_getState(session.model, session.data, after, kind)
                np.testing.assert_array_equal(after, before)
        self.assertIsNone(session.operation_failure)

    def test_ui_exposes_the_route_and_retains_original_physical_ranges(self):
        session = self.session()
        self.assertEqual([x["id"] for x in operations.available_operations(session)], [OPERATION])
        policy = operations.control_policy(session)
        self.assertEqual({(x["equipment_id"], x["control"]) for x in policy}, set(session.controls))
        self.assertTrue(all(x["mode"] == "operation_only" and x["operation"] == OPERATION for x in policy))
        self.assertGreater(session.controls[OWNER, "joint_2_offset"].upper, 1.3062)
        self.assertEqual(session.data.time, 0)

    def test_failure_evidence_is_retained_and_blocks_more_motion(self):
        session = self.session()
        report = {"status": "failed", "failure": "measured_collision", "outward": {"physics_steps": 7}}
        with patch("real_labs.manchester_operations.inspect_manchester", return_value=report):
            result = operations.run_operation(session, OPERATION)
        self.assertIs(result["success"], False)
        self.assertEqual(result["outward"], report["outward"])
        self.assertEqual(result["failure"], "measured_collision")
        self.assertEqual(session.operation_failure["operation"], OPERATION)
        with self.assertRaisesRegex(RuntimeError, "explicitly reset"):
            operations.run_operation(session, OPERATION)
        session.reset()
        self.assertIsNone(session.operation_failure)

    def test_actual_public_route_checks_full_hand_and_returns(self):
        session = self.session()
        result = operations.run_operation(session, OPERATION)
        self.assertIs(result["success"], True, result)
        self.assertTrue(result["returned_to_start"])
        self.assertLess(result["final_tool_return_error_m"], .002)
        self.assertEqual(result["object_displacement_m"], 0)
        for leg in (result["outward"], result["returning"]):
            measured = leg["measured_clearance"]
            self.assertTrue(measured["coverage_complete"])
            self.assertEqual(measured["checked_steps"], leg["physics_steps"])
            self.assertEqual(measured["violations"], [])
            self.assertGreaterEqual(measured["sampled_clearance_lower_bound_m"], .015)
        self.assertFalse(result["grasp_attempted"])
        self.assertGreater(session.data.time, 0)


if __name__ == "__main__":
    unittest.main()
