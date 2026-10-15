"""Actual native-reset evidence for three bounded custom inspection executors."""

from copy import copy, deepcopy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import (
    copenhagen_saw_motion as saw,
    mit_rotating_motion as mit,
    nasa_mast_motion as nasa,
    nasa_rover_task as rover,
)
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession

CASES = (
    (mit, mit.RotatingInspection),
    (saw, saw.SawRegistration),
)


def state(session):
    data = session.data
    return (
        np.asarray(data.time).tobytes(),
        *(
            getattr(data, name).tobytes()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "eq_active",
            )
        ),
    )


class CustomExecutorResetReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = {}
        for module, _ in CASES:
            root, manifest = build_scene(module.SCENE)
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            cls.templates[module.SCENE] = model, manifest

    def fresh(self, module):
        model, manifest = self.templates[module.SCENE]
        return InstrumentSession(copy(model), deepcopy(manifest))

    @staticmethod
    def execute_task(task):
        return task.run() if isinstance(task, saw.SawRegistration) else task.execute()

    @staticmethod
    def force_axis(task):
        if isinstance(task, saw.SawRegistration):
            return task.dadr[0]
        return task.v[0]

    def assert_failed_step(self, report, expected_reset):
        self.assertFalse(report.get("success", report.get("status") == "succeeded"))
        self.assertEqual(report.get("physics_steps", report.get("executed_steps")), 1)
        self.assertEqual(report["completed_checks"], 0)
        self.assertIs(report["reset_performed"], expected_reset)
        if "state_reset" in report:
            self.assertIs(report["state_reset"], expected_reset)
        self.assertFalse(report["explicit_reset_performed"])
        self.assertTrue(
            report.get("hold_requested", report.get("failure_hold_requested"))
        )
        self.assertIsNone(
            report.get(
                "whole_run_clearance_lower_bound_m",
                report.get("whole_route_unintended_clearance_lower_bound_m"),
            )
        )
        self.assertEqual(len(report["physics_events"]), 1)

    def test_actual_native_reset_at_nonzero_zero_and_repeated_warning_clock(self):
        for module, constructor in CASES:
            for initial_time, warning_count in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
                with self.subTest(
                    scene=module.SCENE, time=initial_time, history=warning_count
                ):
                    session = self.fresh(module)
                    session.data.time = initial_time
                    task = constructor(session)
                    session.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = (
                        warning_count
                    )
                    session.data.qfrc_applied[self.force_axis(task)] = 1e25
                    report = self.execute_task(task)
                    self.assert_failed_step(report, True)
                    self.assertTrue(report["automatic_reset_detected"])
                    self.assertAlmostEqual(
                        session.data.time, session.model.opt.timestep
                    )
                    self.assertFalse(np.any(session.data.qfrc_applied))
                    self.assertIn(
                        "qfrc_applied",
                        report["physics_events"][0]["reset_cleared_inputs"],
                    )

    def test_ambiguous_large_clock_is_not_claimed_as_no_reset(self):
        for module, constructor in CASES:
            with self.subTest(scene=module.SCENE):
                session = self.fresh(module)
                session.data.time = 1e30
                report = self.execute_task(constructor(session))
                self.assert_failed_step(report, None)
                self.assertFalse(report["automatic_reset_detected"])
                self.assertTrue(report["time_discontinuity_detected"])

    def test_stale_warning_without_distinguishing_native_reset_stays_unknown(self):
        for module, constructor in CASES:
            with self.subTest(scene=module.SCENE):
                session = self.fresh(module)
                session.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
                report = self.execute_task(constructor(session))
                self.assert_failed_step(report, None)
                self.assertFalse(report["automatic_reset_detected"])
                self.assertEqual(
                    report["physics_events"][0]["reset_cleared_inputs"], []
                )

    def test_nonfinite_clock_rejects_entry_without_mutating_state(self):
        for module, constructor in CASES:
            for clock in (float("nan"), float("inf"), -float("inf")):
                for replace_after_constructor in (False, True):
                    with self.subTest(
                        scene=module.SCENE, clock=clock, later=replace_after_constructor
                    ):
                        session = self.fresh(module)
                        task = (
                            constructor(session) if replace_after_constructor else None
                        )
                        session.data.time = clock
                        before = state(session)
                        with self.assertRaises(ValueError):
                            self.execute_task(task) if task else constructor(session)
                        self.assertEqual(state(session), before)

    def test_normal_short_phase_has_full_coverage_and_no_reset_events(self):
        for module, constructor in CASES:
            with self.subTest(scene=module.SCENE):
                session = self.fresh(module)
                task = constructor(session)
                if module is saw:
                    task._phase("short_hold", task.measured(), 0.02)
                    self.assertEqual(task.executed, task.checked)
                    self.assertGreater(task.checked, 0)
                    self.assertEqual(task.physics_events, [])
                else:
                    target = (0.0, 0.0)
                    task.phase("short_hold", target, ramp=0.02, settle=0.02)
                    self.assertEqual(
                        task.report["physics_steps"], task.report["completed_checks"]
                    )
                    self.assertGreater(task.report["completed_checks"], 0)
                    self.assertFalse(task.report["reset_performed"])
                    self.assertEqual(task.report["physics_events"], [])


class GroundedNasaResetReportingTests(unittest.TestCase):
    """Keep reset evidence coverage on the public grounded-mast API.

    This API rejects external forces before stepping. Native reset tests inject
    an unstable force at the real mj_step boundary and run the actual integrator;
    no reset-report function or MuJoCo warning/reset result is mocked.
    """

    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(nasa.SCENE)
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def fresh(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def assert_failed_step(self, report, expected_reset):
        self.assertFalse(report["success"])
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(report["completed_checks"], 0)
        self.assertEqual(report["unchecked_steps"], 1)
        self.assertFalse(report["coverage_complete"])
        self.assertIs(report["reset_performed"], expected_reset)
        self.assertIs(
            report["no_resets"],
            None if expected_reset is None else not expected_reset,
        )
        self.assertFalse(report["explicit_reset_performed"])
        self.assertEqual(len(report["physics_events"]), 1)

    def test_public_mast_detects_actual_native_resets_with_repeated_warning(self):
        native_step = mujoco.mj_step
        for initial_time, warning_count in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
            with self.subTest(time=initial_time, history=warning_count):
                session = self.fresh()
                control = session.controls["rover", "mast_pan"]
                qpos = int(session.model.jnt_qposadr[control.joint])
                dof = int(session.model.jnt_dofadr[control.joint])
                # A valid nonzero parked pan provides a reset-cleared command
                # even when native reset restores an identical warning count.
                session.data.qpos[qpos] = 0.01
                session.data.time = initial_time
                mujoco.mj_forward(session.model, session.data)
                session.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = (
                    warning_count
                )

                def unstable_step(model, data):
                    self.assertAlmostEqual(data.ctrl[control.actuator], 0.01)
                    data.qfrc_applied[dof] = 1e25
                    native_step(model, data)

                with patch.object(
                    rover.mujoco, "mj_step", side_effect=unstable_step
                ) as step:
                    report = nasa.inspect_nasa_mast(session)
                step.assert_called_once()
                self.assert_failed_step(report, True)
                self.assertTrue(report["automatic_reset_detected"])
                self.assertAlmostEqual(session.data.time, session.model.opt.timestep)
                self.assertFalse(np.any(session.data.qfrc_applied))
                self.assertIn("ctrl", report["physics_events"][0]["reset_cleared_inputs"])

    def test_public_mast_large_clock_preserves_unknown_reset(self):
        session = self.fresh()
        session.data.time = 1e30
        report = nasa.inspect_nasa_mast(session)
        self.assert_failed_step(report, None)
        self.assertFalse(report["automatic_reset_detected"])
        self.assertTrue(report["time_discontinuity_detected"])

    def test_public_mast_stale_warning_preserves_unknown_reset(self):
        session = self.fresh()
        session.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
        report = nasa.inspect_nasa_mast(session)
        self.assert_failed_step(report, None)
        self.assertFalse(report["automatic_reset_detected"])
        self.assertEqual(report["physics_events"][0]["reset_cleared_inputs"], [])

    def test_nonfinite_clock_rejected_before_constructor_or_step_mutation(self):
        for clock in (float("nan"), float("inf"), -float("inf")):
            for after_constructor in (False, True):
                with self.subTest(clock=clock, after_constructor=after_constructor):
                    session = self.fresh()
                    task = (
                        rover.NasaRoverTask(session, record=False, require_start=False)
                        if after_constructor
                        else None
                    )
                    session.data.time = clock
                    before = state(session)
                    with self.assertRaisesRegex(ValueError, "finite simulation time"):
                        if task is None:
                            nasa.inspect_nasa_mast(session)
                        else:
                            task.step()
                    self.assertEqual(state(session), before)

    def test_grounded_short_hold_checks_each_step_without_reset(self):
        session = self.fresh()
        task = rover.NasaRoverTask(session, record=False, require_start=False)
        task.wait(0.04)
        report = task.report()
        self.assertEqual(
            report["physics_steps"], round(0.04 / session.model.opt.timestep)
        )
        self.assertEqual(report["completed_checks"], report["physics_steps"])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["unchecked_steps"], 0)
        self.assertEqual(report["physics_events"], [])
        self.assertFalse(report["reset_performed"])
        self.assertFalse(report["automatic_reset_detected"])
        self.assertFalse(report["time_discontinuity_detected"])
        self.assertTrue(report["no_resets"])
        # Successful step checks are distinct from completing a full route.
        self.assertFalse(report["success"])


if __name__ == "__main__":
    unittest.main()
