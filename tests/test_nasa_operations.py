"""Unsafe scalar requests cannot replace the rover's ordered physical task."""

from copy import copy, deepcopy
import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs import operations
from real_labs.nasa_operations import (
    CLEARANCE_YAW,
    TRAVERSE_AND_STAGE,
    OperationRequired,
    operation_for_control,
    run_arm_clearance_yaw,
    validate_command,
)
from real_labs.runtime import InstrumentSession
from real_labs.nasa_rover_task import NasaRoverTask, _observed_actuator_peaks


class NasaOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("nasa_planetary")
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    @staticmethod
    def snapshot(session):
        return float(session.data.time), {
            name: getattr(session.data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart", "xpos")
        }

    def assert_unchanged(self, session, snapshot):
        stamp, state = snapshot
        self.assertEqual(session.data.time, stamp)
        for name, value in state.items():
            np.testing.assert_array_equal(getattr(session.data, name), value)

    def test_coupled_commands_reject_before_motion_and_retain_ranges(self):
        session = self.session()
        before = self.snapshot(session)
        ranges = session.model.actuator_ctrlrange.copy()
        blocked = 0
        for (owner, control), spec in session.controls.items():
            operation = operation_for_control(control) if owner == "rover" else None
            if operation is None:
                continue
            with self.subTest(control=control):
                with self.assertRaises(OperationRequired) as failure:
                    validate_command(session, owner, control, (spec.lower + spec.upper) / 2)
                self.assertEqual(failure.exception.operation, operation)
                self.assert_unchanged(session, before)
                blocked += 1
        self.assertEqual(blocked, 15)
        np.testing.assert_array_equal(session.model.actuator_ctrlrange, ranges)
        self.assertEqual(operation_for_control("arm_base_yaw"), CLEARANCE_YAW)
        self.assertEqual(operation_for_control("wheel_right_2"), TRAVERSE_AND_STAGE)

    def test_bad_targets_and_modified_binding_reject_without_mutation(self):
        session = self.session()
        for target in (math.nan, math.inf, -math.inf, 100):
            with self.subTest(target=target):
                before = self.snapshot(session)
                with self.assertRaises(ValueError):
                    validate_command(session, "rover", "arm_base_yaw", target)
                self.assert_unchanged(session, before)
        session.model.body_mass[session.model.body("rover").id] *= 2
        before = self.snapshot(session)
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_command(session, "rover", "mast_pan", 0)
        self.assert_unchanged(session, before)

    def test_unrelated_instruments_and_scalar_mast_are_preserved(self):
        session = self.session()
        before = self.snapshot(session)
        validate_command(session, "xrf_analyzer", "sample_tray_joint", 0)
        validate_command(session, "rover", "mast_pan", 0)
        self.assert_unchanged(session, before)

    def test_registry_exposes_operations_and_public_command_cannot_bypass_guard(self):
        session = self.session()
        before = self.snapshot(session)
        self.assertEqual(
            {entry["id"] for entry in operations.available_operations(session)},
            {CLEARANCE_YAW, TRAVERSE_AND_STAGE},
        )
        self.assertEqual(len(operations.control_policy(session)), 15)
        with self.assertRaises(OperationRequired):
            session.command("rover", "arm_base_yaw", math.pi / 2)
        self.assert_unchanged(session, before)

    def test_missing_observed_effort_rejects_before_motion_or_forward(self):
        session = self.session()
        for channel in ({}, {"actuator_force": False}, {"actuator_force": 1}):
            with self.subTest(channel=channel):
                session.physics_observations = channel
                before = self.snapshot(session)
                with self.assertRaisesRegex(ValueError, "observed actuator forces"):
                    run_arm_clearance_yaw(session)
                self.assert_unchanged(session, before)

    def test_substep_effort_peaks_keep_original_limits(self):
        session = self.session()
        task = NasaRoverTask(session, record=False)
        peaks = np.zeros(session.model.nu)
        peaks[task.wheela[0]] = 12.1
        session.physics_observations["actuator_force_peak_abs"] = peaks.tolist()
        np.testing.assert_array_equal(_observed_actuator_peaks(session), peaks)
        with self.assertRaisesRegex(RuntimeError, "Drive force limit exceeded"):
            task.step()
        self.assertEqual(task.steps, 1)
        self.assertEqual(task.completed_checks, 0)

    def test_missing_or_invalid_effort_does_not_advance_active_task(self):
        session = self.session()
        task = NasaRoverTask(session, record=False)
        for observations in (
            {"actuator_force": False},
            {"actuator_force": True, "actuator_force_peak_abs": [0]},
            {"actuator_force": True, "actuator_force_peak_abs": [math.nan] * session.model.nu},
            {"actuator_force": True, "actuator_force_peak_abs": [-1] * session.model.nu},
        ):
            session.physics_observations = observations
            before = self.snapshot(session)
            with self.assertRaises(ValueError):
                task.step()
            self.assert_unchanged(session, before)

    def test_clearance_yaw_reaches_original_target_with_original_guards(self):
        session = self.session()
        model_before = session.model.actuator_ctrlrange.copy()
        report = run_arm_clearance_yaw(session)
        self.assertTrue(report["success"])
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["raw_control_results_replaced"])
        self.assertFalse(report["full_traversal_completed"])
        self.assertTrue(report["no_live_pose_writes"])
        self.assertTrue(report["no_resets"])
        self.assertEqual(report["physics_steps"], 14000)
        self.assertAlmostEqual(report["simulated_seconds"], 7.0, places=7)
        self.assertLess(abs(report["measured_yaw_rad"] - math.pi / 2), 0.012)
        self.assertGreaterEqual(report["minimum_direct_m"], -0.00005)
        self.assertLessEqual(report["sample_displacement_m"], 0.002)
        self.assertEqual([phase["name"] for phase in report["phases"]], [
            "settled", "lift_clearance_complete", "yaw_complete"
        ])
        np.testing.assert_array_equal(session.model.actuator_ctrlrange, model_before)


if __name__ == "__main__":
    unittest.main()
