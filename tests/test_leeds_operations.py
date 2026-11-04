"""Public dry-command guard and physical unloading of existing Leeds contact."""

from copy import copy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.leeds_motion import run_loaded_slide
from real_labs.leeds_operations import OperationRequired, run_retract, validate_command
from real_labs.runtime import InstrumentSession


class LeedsOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("leeds_environmental_tribology")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), self.manifest)

    def command(self, session, control, target):
        validate_command(session, "tribology", control, target)
        return session.command("tribology", control, target)

    def assert_no_mutation(self, session, action, exception=ValueError):
        before = {
            k: getattr(session.data, k).copy()
            for k in ("qpos", "qvel", "ctrl", "qacc", "qfrc_applied", "xfrc_applied")
        }
        time = session.data.time
        with self.assertRaises(exception):
            action()
        self.assertEqual(session.data.time, time)
        for key, values in before.items():
            np.testing.assert_array_equal(getattr(session.data, key), values)

    def loaded_state(self, session):
        # Represent an already loaded legacy/failed operation using real servos.
        # No live qpos writes and no policy bypass in the production command API.
        normal = session.controls["tribology", "contact_approach"]
        slide = session.controls["tribology", "specimen_slide"]
        session.data.ctrl[slide.actuator] = 0.005
        session.advance(1000)
        for step in range(1000):
            t = (step + 1) / 1000
            session.data.ctrl[normal.actuator] = 0.024 * t * t * (3 - 2 * t)
            session.advance(1)
        session.advance(500)
        mujoco.mj_forward(session.model, session.data)
        self.assertGreater(session._measured(normal), 0.02)

    def test_dry_envelope_return_and_unchanged_deep_rejection(self):
        session = self.session()
        self.assert_no_mutation(
            session,
            lambda: validate_command(session, "tribology", "contact_approach", 0.02125),
            OperationRequired,
        )
        for value in (float("nan"), float("inf"), -0.001):
            self.assert_no_mutation(
                session,
                lambda: validate_command(
                    session, "tribology", "contact_approach", value
                ),
            )
        self.assertAlmostEqual(
            self.command(session, "contact_approach", 0.012)["measured"],
            0.012,
            places=6,
        )
        self.assert_no_mutation(
            session,
            lambda: validate_command(session, "tribology", "specimen_slide", 0.012),
            OperationRequired,
        )
        self.assertAlmostEqual(
            self.command(session, "contact_approach", 0)["measured"], 0, places=6
        )
        for axis, target in (
            ("specimen_slide", -0.012),
            ("specimen_slide", 0.012),
            ("specimen_slide", 0),
            ("access_door", 0),
            ("access_door", 1.2),
        ):
            self.assertAlmostEqual(
                self.command(session, axis, target)["measured"], target, places=5
            )
            self.assertFalse(any(c.dist < -1e-5 for c in session.data.contact))

    def test_model_contract_and_applied_force_reject_before_mutation(self):
        for change in ("contact", "force", "external"):
            session = self.session()
            if change == "contact":
                session.model.geom_contype[
                    session.model.geom("tribology__leeds_coupon").id
                ] = 0
            elif change == "force":
                session.model.actuator_forcerange[
                    session.controls["tribology", "contact_approach"].actuator
                ] *= 2
            else:
                session.data.qfrc_applied[:] = 1
            self.assert_no_mutation(
                session,
                lambda: validate_command(
                    session, "tribology", "contact_approach", 0.012
                ),
            )
            self.assert_no_mutation(session, lambda: run_retract(session))

    def test_loaded_contact_recovery_preserves_slide_and_clock(self):
        session = self.session()
        self.loaded_state(session)
        for control, target in (
            ("contact_approach", 0),
            ("specimen_slide", 0),
            ("access_door", 0),
        ):
            self.assert_no_mutation(
                session,
                lambda: validate_command(session, "tribology", control, target),
                OperationRequired,
            )
        start = float(session.data.time)
        report = run_retract(session)
        self.assertTrue(report["success"], report)
        self.assertGreater(report["initial_normal_force_n"], 3)
        self.assertGreater(report["initial_penetration_m"], 0)
        self.assertEqual(report["physics_steps"], 1250)
        self.assertEqual(report["completed_checks"], 1250)
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["reset_performed"])
        self.assertTrue(report["model_unchanged"])
        self.assertAlmostEqual(session.data.time - start, 2.5)
        np.testing.assert_allclose(report["final_position_m"], (0, 0.005), atol=1e-5)
        self.assertAlmostEqual(
            self.command(session, "specimen_slide", 0)["measured"], 0, places=6
        )

    def test_loaded_slide_retains_full_operation_and_limits(self):
        session = self.session()
        self.assertEqual(session.controls["tribology", "contact_approach"].upper, 0.025)
        report = run_loaded_slide(session)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["physics_steps"], 5125)
        self.assertGreater(report["loaded_relative_path_m"], 0.035)
        self.assertFalse(report["reset_performed"])
        validate_command(session, "tribology", "contact_approach", 0.012)

    def test_recovery_reports_interrupted_physics_without_success(self):
        session = self.session()
        self.loaded_state(session)
        native_step = mujoco.mj_step

        def disrupted_step(model, data):
            native_step(model, data)
            data.time += 0.01

        with patch(
            "real_labs.leeds_operations.mujoco.mj_step", side_effect=disrupted_step
        ):
            report = run_retract(session)
        self.assertFalse(report["success"])
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(report["completed_checks"], 0)
        self.assertEqual(report["unchecked_steps"], 1)
        self.assertTrue(report["hold_requested"])
        self.assertIsNone(report["reset_performed"])
        self.assertFalse(report["explicit_reset_performed"])
        self.assertEqual(
            report["failure"], "native_physics_warning_or_time_discontinuity"
        )


if __name__ == "__main__":
    unittest.main()
