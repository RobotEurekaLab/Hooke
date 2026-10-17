"""Real native resets and uncertain clocks must reach the public task report."""

import unittest
import xml.etree.ElementTree as ET

import mujoco

from real_labs.builder import build_scene
from real_labs.chicago_enclosure_motion import EnclosureInspection
from real_labs.runtime import InstrumentSession
from real_labs.whittle_probe_motion import ReferenceInspection


class TaskResetReportingTests(unittest.TestCase):
    def run_failure(self, scene, task_class, *, mode):
        root, manifest = build_scene(scene)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        if mode == "unknown_clock":
            session.data.time = 1e30
        else:
            session.data.time = 0 if mode == "repeated_warning" else 1
            session.data.qfrc_applied[0] = 1e25
            if mode == "repeated_warning":
                session.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
        messages = []
        previous = mujoco.get_mju_user_warning()
        try:
            mujoco.set_mju_user_warning(messages.append)
            result = task_class(session).run()
        finally:
            mujoco.set_mju_user_warning(previous)
        if mode == "native_reset":
            self.assertTrue(any("QACC" in message for message in messages))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["executed_steps"], 1)
        self.assertEqual(result["checked_steps"], 0)
        self.assertFalse(result["coverage_complete"])
        self.assertFalse(result["explicit_reset_performed"])
        self.assertTrue(result["phases"][0]["hold_requested"])
        expected = None if mode == "unknown_clock" else True
        self.assertIs(result["reset_performed"], expected)
        self.assertIs(result["phases"][0]["reset_performed"], expected)
        self.assertEqual(result["automatic_reset_detected"], expected is True)
        if "state_reset" in result:
            self.assertIs(result["state_reset"], expected)
        clearance = result.get(
            "whole_route_clearance_lower_bound_m",
            result.get("whole_run_clearance_lower_bound_m"),
        )
        self.assertIsNone(clearance)
        if expected is True:
            self.assertAlmostEqual(session.data.time, model.opt.timestep)
            self.assertEqual(session.data.qfrc_applied[0], 0)

    def test_chicago_public_failure_retains_actual_reset_evidence(self):
        for mode in ("native_reset", "repeated_warning", "unknown_clock"):
            with self.subTest(mode=mode):
                self.run_failure(
                    "chicago_mineralphysics_laser_bay", EnclosureInspection, mode=mode
                )

    def test_whittle_public_failure_retains_actual_reset_evidence(self):
        for mode in ("native_reset", "repeated_warning", "unknown_clock"):
            with self.subTest(mode=mode):
                self.run_failure(
                    "cambridge_whittle_low_speed_turbine",
                    ReferenceInspection,
                    mode=mode,
                )


if __name__ == "__main__":
    unittest.main()
