"""Execution identity is scoped, immutable, and separate from success."""

from copy import copy, deepcopy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from backends.closed_loop import PhysXTaskAdapter
from real_labs.builder import build_scene
from real_labs.nasa_rover_task import NasaRoverTask
from real_labs.runtime import InstrumentSession


def native_adapter(session):
    """Exercise real observation handling with a synthetic native reset payload."""
    adapter = PhysXTaskAdapter.__new__(PhysXTaskAdapter)
    adapter.task = session
    adapter.max_fk_position_error = adapter.max_fk_rotation_error = 0.0
    zeros = [0.0] * session.model.nu
    adapter.apply_state({
        "qpos": session.data.qpos.tolist(),
        "qvel": session.data.qvel.tolist(),
        "time": 0.0,
        "contacts": [],
        "actuator_effort": {
            "available": True,
            "quantity": "applied_explicit_actuator_input",
            "native_input_verified": True,
            "last": zeros, "mean": zeros, "peak_abs": zeros, "signed_peak": zeros,
            "sample_count": 0, "interval_s": 0.0,
        },
    })
    return adapter


class NasaExecutionProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("nasa_planetary")
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_source_partial_execution_is_not_whole_scene_qualification(self):
        task = NasaRoverTask(self.session(), record=False)
        task.step()
        report = task.report()
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(report["completed_checks"], 1)
        self.assertFalse(report["success"])
        self.assertEqual(report["execution_provenance"], {
            "backend": "mujoco", "physics_engine": "MuJoCo",
            "actuator_force_provenance": "MuJoCo actuator force",
        })
        self.assertNotIn("native_isaac_tested", report)
        self.assertIn("named operation only", report["qualification_scope"])

    def test_native_failure_and_context_exit_preserve_captured_identity(self):
        session = self.session()
        adapter = native_adapter(session)
        with adapter:
            task = NasaRoverTask(session, record=False)
            report = task.report("Synthetic metadata test failure")
        self.assertFalse(session.physics_observations["actuator_force"])
        self.assertEqual(session.physics_observations["backend"], "isaac")
        self.assertFalse(report["success"])
        self.assertEqual(report["physics_steps"], 0)
        self.assertEqual(task.report("failure")["execution_provenance"], report["execution_provenance"])
        self.assertEqual(report["execution_provenance"]["physics_engine"], "PhysX")
        self.assertEqual(report["execution_provenance"]["actuator_force_provenance"],
                         "applied_explicit_actuator_input")
        self.assertNotIn("native_isaac_tested", report)

    def test_failed_source_reset_retains_unavailable_native_provenance(self):
        session = self.session()
        with native_adapter(session):
            pass
        with patch.object(mujoco, "mj_forward", side_effect=RuntimeError("failed reset")):
            with self.assertRaisesRegex(RuntimeError, "failed reset"):
                session.reset()
        self.assertFalse(session.physics_observations["actuator_force"])
        self.assertEqual(session.physics_observations["backend"], "isaac")
        self.assertEqual(session.physics_observations["physics_engine"], "PhysX")
        session.reset()
        self.assertTrue(session.physics_observations["actuator_force"])
        self.assertEqual(session.physics_observations["backend"], "mujoco")
        self.assertEqual(session.physics_observations["physics_engine"], "MuJoCo")

    def test_changed_backend_rejects_before_control_or_physics(self):
        session = self.session()
        task = NasaRoverTask(session, record=False)
        initial = session.data.ctrl.copy()
        session.physics_observations["backend"] = "different_backend"
        with patch.object(mujoco, "mj_step") as step:
            with self.assertRaisesRegex(RuntimeError, "provenance changed"):
                task.step()
        step.assert_not_called()
        np.testing.assert_array_equal(session.data.ctrl, initial)
        self.assertEqual(session.data.time, 0.0)
        self.assertEqual(task.report("backend changed")["execution_provenance"]["backend"], "mujoco")

    def test_missing_identity_does_not_infer_engine_from_force_quantity(self):
        session = self.session()
        del session.physics_observations["backend"]
        del session.physics_observations["physics_engine"]
        task = NasaRoverTask(session, record=False)
        report = task.report()
        self.assertEqual(report["execution_provenance"]["backend"], "unknown")
        self.assertEqual(report["execution_provenance"]["physics_engine"], "unknown")

    def test_unavailable_native_channel_keeps_identity_without_force_claim(self):
        session = self.session()
        adapter = native_adapter(session)
        adapter._native_effort = {"available": False}
        adapter._apply_effort()
        self.assertEqual(session.physics_observations["backend"], "isaac")
        self.assertEqual(session.physics_observations["physics_engine"], "PhysX")
        self.assertFalse(session.physics_observations["actuator_force"])
        self.assertTrue(np.isnan(session.data.actuator_force).all())


if __name__ == "__main__":
    unittest.main()
