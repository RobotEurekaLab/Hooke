"""Actual native failures must not pass geometric/contact inspection tasks."""

from copy import copy, deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import berkeley_tank_motion as berkeley
from real_labs import kyoto_instrument_motion as kyoto
from real_labs import leeds_motion as leeds
from real_labs import lmu_snom_motion as lmu
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession

CASES = (
    (
        "berkeley_richmond_towing_tank",
        berkeley.inspect_berkeley_tank,
        ("towing_facility", "model_immersion"),
    ),
    (
        "kyoto_yokokawa_nanometrics",
        kyoto.inspect_kyoto,
        ("tirf_workspace", "objective_focus"),
    ),
    (
        "leeds_environmental_tribology",
        leeds.run_loaded_slide,
        ("tribology", "contact_approach"),
    ),
    ("lmu_hybrid_snom", lmu.inspect_lmu, ("snom_table", "fine_approach_z")),
)


def state(session):
    d = session.data
    return (
        np.asarray(d.time).tobytes(),
        *(
            getattr(d, name).tobytes()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "eq_active",
                "mocap_pos",
                "mocap_quat",
            )
        ),
    )


class CustomStageResetReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = {}
        for name, *_ in CASES:
            root, manifest = build_scene(name)
            cls.templates[name] = (
                mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
                manifest,
            )

    def fresh(self, name):
        model, manifest = self.templates[name]
        return InstrumentSession(copy(model), deepcopy(manifest))

    def check_failure(self, report, expected_reset):
        self.assertNotEqual(report.get("status"), "succeeded")
        self.assertFalse(report.get("success", report.get("passed", False)))
        self.assertEqual(report.get("physics_steps", report.get("executed_steps")), 1)
        self.assertEqual(report.get("completed_checks", report.get("checked_steps")), 0)
        self.assertFalse(report["coverage_complete"])
        self.assertIs(report["reset_performed"], expected_reset)
        self.assertFalse(report["explicit_reset_performed"])
        self.assertTrue(
            report.get(
                "hold_requested", report.get("bounded_measured_hold_commanded", False)
            )
        )
        self.assertEqual(len(report["physics_events"]), 1)
        self.assertIsNone(report.get("whole_route_clearance_lower_bound_m"))
        self.assertIsNone(report.get("whole_run_clearance_lower_bound_m"))

    def test_actual_badqacc_reset_is_failed_even_at_zero_with_repeated_history(self):
        for name, run, key in CASES:
            for start, prior in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
                with self.subTest(scene=name, initial_time=start, prior_warning=prior):
                    s = self.fresh(name)
                    s.data.time = start
                    s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = prior
                    dof = s.model.jnt_dofadr[s.controls[key].joint]
                    s.data.qfrc_applied[dof] = 1e25
                    report = run(s)
                    self.check_failure(report, True)
                    self.assertTrue(report["automatic_reset_detected"])
                    self.assertAlmostEqual(s.data.time, s.model.opt.timestep)
                    self.assertFalse(np.any(s.data.qfrc_applied))

    def test_ambiguous_clock_and_unchanged_bad_warning_never_claim_success(self):
        for name, run, _ in CASES:
            for kind in ("large_clock", "stale_warning"):
                with self.subTest(scene=name, kind=kind):
                    s = self.fresh(name)
                    if kind == "large_clock":
                        s.data.time = 1e30
                    else:
                        s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
                    report = run(s)
                    self.check_failure(report, None)
                    self.assertFalse(report["automatic_reset_detected"])
                    self.assertEqual(
                        report["physics_events"][0]["reset_cleared_inputs"], []
                    )

    def test_nonfinite_clock_rejects_before_commands_or_hold(self):
        for name, run, _ in CASES:
            for time in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(scene=name, time=time):
                    s = self.fresh(name)
                    s.data.time = time
                    before = state(s)
                    with self.assertRaises(ValueError):
                        run(s)
                    self.assertEqual(state(s), before)

    def test_cached_phase_nonfinite_clock_rejects_unchanged(self):
        for name, constructor in (
            ("berkeley_richmond_towing_tank", berkeley.TankTransit),
            ("kyoto_yokokawa_nanometrics", kyoto.SlideInspection),
        ):
            with self.subTest(scene=name):
                s = self.fresh(name)
                task = constructor(s)
                s.data.time = float("nan")
                before = state(s)
                with self.assertRaises(ValueError):
                    if name.startswith("berkeley"):
                        task._phase("invalid", np.zeros(2), 0.02)
                    else:
                        task.phase("invalid", np.zeros(3), 0.02, 0.02)
                self.assertEqual(state(s), before)


if __name__ == "__main__":
    unittest.main()
