"""Actual native-reset and clock failures in UVA and Stanford device tasks."""

from copy import copy, deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.stanford_gait_motion import GaitInspection
from real_labs.uva_motion import ChamberAccess


CASES = (
    (
        "uva_deposition",
        ChamberAccess,
        "open_inspect_return",
        ("deposition_chamber", "chamber_lid"),
    ),
    (
        "stanford_biomechatronics_gait",
        GaitInspection,
        "execute",
        ("gait_test_rig", "drive_left"),
    ),
)


def state(session):
    d = session.data
    return (
        np.asarray(d.time).tobytes(),
        *(
            getattr(d, key).tobytes()
            for key in (
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


def short(task):
    if isinstance(task, ChamberAccess):
        task._phase("settle_loaded_coupon", task.home, 0.02, 0.02)
    else:
        task.phase("short_hold", task.d.qpos[task.q].copy(), (2,), 0.02, 0.02)


class DeviceAccessResetReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = {}
        for scene, *_ in CASES:
            root, manifest = build_scene(scene)
            cls.templates[scene] = (
                mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
                manifest,
            )

    def fresh(self, scene):
        model, manifest = self.templates[scene]
        return InstrumentSession(copy(model), deepcopy(manifest))

    def assert_event(self, report, reset):
        self.assertFalse(report.get("success", report.get("status") == "succeeded"))
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(
            report.get("completed_checks", report.get("fully_checked_steps")), 0
        )
        self.assertFalse(report["coverage_complete"])
        self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
        self.assertTrue(report["hold_requested"])
        self.assertIs(report["reset_performed"], reset)
        self.assertFalse(report["explicit_reset_performed"])
        self.assertEqual(len(report["physics_events"]), 1)
        phase = report.get("phases", report.get("legs"))[0]
        self.assertIs(phase["reset_performed"], reset)
        self.assertFalse(phase["coverage_complete"])

    def test_actual_native_reset_at_zero_and_nonzero_with_existing_warnings(self):
        for scene, constructor, method, key in CASES:
            for clock, count in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
                with self.subTest(scene=scene, clock=clock, warning=count):
                    s = self.fresh(scene)
                    s.data.time = clock
                    s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = count
                    s.data.qfrc_applied[s.model.jnt_dofadr[s.controls[key].joint]] = (
                        1e25
                    )
                    report = getattr(constructor(s), method)()
                    self.assert_event(report, True)
                    self.assertTrue(report["automatic_reset_detected"])
                    self.assertAlmostEqual(s.data.time, s.model.opt.timestep)
                    self.assertFalse(np.any(s.data.qfrc_applied))
                    self.assertIn(
                        "qfrc_applied",
                        report["physics_events"][0]["reset_cleared_inputs"],
                    )

    def test_unresolvable_clock_and_ambiguous_existing_warning_stop_as_unknown(self):
        for scene, constructor, method, _ in CASES:
            for ambiguous in ("large_clock", "existing_bad_warning"):
                with self.subTest(scene=scene, ambiguous=ambiguous):
                    s = self.fresh(scene)
                    if ambiguous == "large_clock":
                        s.data.time = 1e30
                    else:
                        s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
                    report = getattr(constructor(s), method)()
                    self.assert_event(report, None)
                    self.assertFalse(report["automatic_reset_detected"])

    def test_nonfinite_clock_rejected_before_entry_or_cached_phase_mutation(self):
        for scene, constructor, method, _ in CASES:
            for clock in (float("nan"), float("inf"), -float("inf")):
                for cached in (False, True):
                    with self.subTest(scene=scene, clock=clock, cached=cached):
                        s = self.fresh(scene)
                        task = constructor(s) if cached else None
                        s.data.time = clock
                        before = state(s)
                        with self.assertRaises(ValueError):
                            short(task) if cached else getattr(constructor(s), method)()
                        self.assertEqual(state(s), before)

    def test_normal_short_operation_has_complete_checks_and_no_reset(self):
        for scene, constructor, _, _ in CASES:
            with self.subTest(scene=scene):
                s = self.fresh(scene)
                task = constructor(s)
                short(task)
                r = task.report
                self.assertEqual(r["physics_steps"], round(0.04 / s.model.opt.timestep))
                self.assertEqual(
                    r["physics_steps"],
                    r.get("completed_checks", r.get("fully_checked_steps")),
                )
                self.assertFalse(r["reset_performed"])
                self.assertFalse(r["explicit_reset_performed"])
                self.assertEqual(r["physics_events"], [])


if __name__ == "__main__":
    unittest.main()
