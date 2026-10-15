"""Shared dispatch retains source binding and rejects unsafe client commands."""

from copy import copy, deepcopy
import json
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs import operations
from real_labs.runtime import InstrumentSession
from test_real_labs_runtime import MANIFEST, MODEL


class RealLabOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authored = {}
        for name in ("rochester_genomics", "leeds_environmental_tribology"):
            root, manifest = build_scene(name)
            cls.authored[name] = (
                mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
                manifest,
            )

    def session(self, scene="leeds_environmental_tribology"):
        model, manifest = self.authored[scene]
        return InstrumentSession(copy(model), deepcopy(manifest))

    @staticmethod
    def snapshot(session):
        data = session.data
        return float(data.time), {
            key: getattr(data, key).copy()
            for key in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "geom_xpos",
                "xpos",
                "xquat",
            )
        }

    def assert_unchanged(self, session, before):
        time, values = before
        self.assertEqual(float(session.data.time), time)
        for key, value in values.items():
            np.testing.assert_array_equal(getattr(session.data, key), value)

    def reject_without_mutation(self, session, call):
        before = self.snapshot(session)
        with self.assertRaises(ValueError):
            call()
        self.assert_unchanged(session, before)

    def test_metadata_and_preflight_do_not_step_or_mutate_state(self):
        for scene, expected in (
            ("rochester_genomics", {"plate_loading"}),
            ("leeds_environmental_tribology", {"loaded_slide", "retract"}),
        ):
            session = self.session(scene)
            before = self.snapshot(session)
            entries = operations.available_operations(session)
            self.assertEqual({entry["id"] for entry in entries}, expected)
            for entry in entries:
                self.assertEqual(
                    operations.validate_operation(session, entry["id"]), entry
                )
                self.assertIn(entry["equipment_id"], session.equipment)
            json.dumps(entries)
            self.assert_unchanged(session, before)
        entries[0]["label"] = "Edited by caller"
        self.assertNotEqual(
            operations.available_operations(session)[0]["label"], "Edited by caller"
        )

    def test_ui_policy_retains_full_physical_strokes(self):
        session = self.session()
        policy = operations.control_policy(session)[0]
        self.assertEqual((policy["lower"], policy["upper"]), (0.0, 0.012))
        self.assertEqual(session.controls["tribology", "contact_approach"].upper, 0.025)
        session = self.session("rochester_genomics")
        policies = operations.control_policy(session)
        self.assertEqual({p["control"] for p in policies}, {"loader_x", "loader_y"})
        self.assertTrue(all(p["mode"] == "operation_only" for p in policies))
        self.assertEqual(session.controls["liquid_handler", "loader_x"].lower, -0.70)
        self.assertEqual(session.controls["liquid_handler", "loader_y"].lower, -0.82)

    def test_unknown_and_cross_scene_operations_reject_without_dispatch(self):
        session = self.session()
        with patch("real_labs.leeds_motion.run_loaded_slide") as runner:
            for key in ("plate_loading", "missing", "../loaded_slide", None, []):
                self.reject_without_mutation(
                    session, lambda: operations.run_operation(session, key)
                )
        runner.assert_not_called()

    def test_binding_is_rechecked_after_successful_preflight(self):
        for change in ("contact", "drive", "manifest"):
            session = self.session()
            operations.validate_operation(session, "loaded_slide")
            if change == "contact":
                session.model.geom_contype[
                    session.model.geom("tribology__leeds_coupon").id
                ] = 0
            elif change == "drive":
                session.model.actuator_forcerange[
                    session.controls["tribology", "contact_approach"].actuator
                ] *= 2
            else:
                session.equipment["tribology"]["contact_operation"]["kind"] = "changed"
            with patch("real_labs.leeds_motion.run_loaded_slide") as runner:
                self.reject_without_mutation(
                    session, lambda: operations.run_operation(session, "loaded_slide")
                )
            runner.assert_not_called()

    def test_external_forces_and_nonfinite_state_reject_without_dispatch(self):
        for change in ("force", "nonfinite", "warning"):
            session = self.session()
            if change == "force":
                session.data.qfrc_applied[:] = 1
            elif change == "nonfinite":
                session.data.qpos[0] = np.nan
            else:
                session.data.warning.number[0] = 1
            self.reject_without_mutation(
                session, lambda: operations.validate_operation(session, "retract")
            )

    def test_known_unsafe_scalar_requests_never_reach_runtime(self):
        for scene, owner, control, target in (
            ("rochester_genomics", "liquid_handler", "loader_x", -0.35),
            ("rochester_genomics", "liquid_handler", "loader_y", -0.41),
            ("leeds_environmental_tribology", "tribology", "contact_approach", 0.02125),
        ):
            session = self.session(scene)
            with patch.object(session, "command") as raw:
                self.reject_without_mutation(
                    session,
                    lambda: operations.command(session, owner, control, target),
                )
            raw.assert_not_called()

    def test_direct_session_command_enforces_declared_policy(self):
        for scene, owner, control, target in (
            ("rochester_genomics", "liquid_handler", "loader_x", -0.35),
            ("leeds_environmental_tribology", "tribology", "contact_approach", 0.02125),
        ):
            session = self.session(scene)
            self.reject_without_mutation(
                session, lambda: session.command(owner, control, target)
            )

    def test_unregistered_scene_retains_bounded_runtime_controls(self):
        session = InstrumentSession(
            mujoco.MjModel.from_xml_string(MODEL), deepcopy(MANIFEST)
        )
        self.assertEqual(operations.available_operations(session), [])
        self.assertEqual(operations.control_policy(session), [])
        for target, duration in ((float("nan"), 2), (0.3, 2), (0.1, 0)):
            self.reject_without_mutation(
                session,
                lambda: operations.command(
                    session, "reader", "door", target, duration=duration
                ),
            )
        before = self.snapshot(session)
        for command in (session.command, lambda *args: operations.command(session, *args)):
            with self.assertRaises(KeyError) as caught:
                command("reader", "missing", 0)
            self.assertEqual(caught.exception.args, (("reader", "missing"),))
            self.assert_unchanged(session, before)
        report = operations.command(session, "reader", "door", 0.1)
        self.assertGreater(report["measured"], 0.09)

    def test_actual_loaded_slide_dispatch_returns_measured_success(self):
        session = self.session()
        result = operations.run_operation(session, "loaded_slide")
        self.assertIs(result["success"], True, result)
        self.assertEqual(result["operation"]["id"], "loaded_slide")
        self.assertEqual(result["physics_steps"], 5125)
        self.assertGreater(result["loaded_relative_path_m"], 0.035)
        self.assertFalse(result["reset_performed"])
        self.assertAlmostEqual(float(session.data.time), 10.25)

    def test_result_adapter_preserves_failures_and_original_evidence(self):
        session = self.session("rochester_genomics")
        for passed in (True, False):
            original = {
                "passed": passed,
                "steps": 7,
                "failure": None if passed else "contact",
            }
            with patch(
                "real_labs.rochester_operations.run_plate_loading",
                return_value=original,
            ):
                result = operations.run_operation(session, "plate_loading")
            self.assertIs(result["success"], passed)
            self.assertEqual(result["steps"], 7)
            self.assertEqual(result["failure"], original["failure"])
            self.assertNotIn("success", original)
        session.reset()
        with patch(
            "real_labs.rochester_operations.run_plate_loading",
            return_value={"passed": "yes"},
        ):
            with self.assertRaisesRegex(RuntimeError, "boolean"):
                operations.run_operation(session, "plate_loading")

    def test_failed_operation_blocks_all_motion_until_explicit_reset(self):
        session = self.session("rochester_genomics")

        def fail_after_motion(current):
            spec = current.controls["liquid_handler", "loader_z"]
            current.data.ctrl[spec.actuator] = 0.01
            current.advance(10)
            raise RuntimeError("Observed contact failure")

        with patch("real_labs.rochester_operations.run_plate_loading", side_effect=fail_after_motion):
            with self.assertRaisesRegex(RuntimeError, "Observed contact failure"):
                operations.run_operation(session, "plate_loading")
        self.assertGreater(session.data.time, 0)
        self.assertEqual(session.operation_failure["operation"], "plate_loading")
        before = self.snapshot(session)
        for action in (
            lambda: session.command("sequencer", "flowcell_tray", 0),
            lambda: operations.run_operation(session, "plate_loading"),
        ):
            with self.assertRaisesRegex(RuntimeError, "explicitly reset"):
                action()
            self.assert_unchanged(session, before)

        from webui.real_labs_api import _state

        state = _state(session)
        self.assertTrue(state["invalid_physics_state"])
        self.assertIn("Explicitly reset", state["state_error"])
        self.assertGreater(state["time_s"], 0)
        session.reset()
        self.assertIsNone(session.operation_failure)
        operations.validate_operation(session, "plate_loading")
        self.assertNotIn("invalid_physics_state", _state(session))

    def test_measured_failure_latches_but_preflight_rejection_does_not(self):
        session = self.session("rochester_genomics")
        with self.assertRaises(ValueError):
            operations.run_operation(session, "missing")
        self.assertIsNone(session.operation_failure)
        with patch("real_labs.rochester_operations.run_plate_loading", return_value={"passed": False, "failure": "plate_not_supported"}):
            report = operations.run_operation(session, "plate_loading")
        self.assertFalse(report["success"])
        self.assertEqual(session.operation_failure["error"], "plate_not_supported")
        with patch("mujoco.mj_forward", side_effect=RuntimeError("Reset failed")):
            with self.assertRaisesRegex(RuntimeError, "Reset failed"):
                session.reset()
        self.assertEqual(session.operation_failure["error"], "plate_not_supported")
        session.reset()
        self.assertIsNone(session.operation_failure)


if __name__ == "__main__":
    unittest.main()
