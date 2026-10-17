"""Real native-reset evidence in ETH, Rochester and Indiana carrier operations."""

from copy import copy, deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import (
    eth_air_motion as eth,
    indiana_motion as ind,
    rochester_motion as roc,
)
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession

CASES = (
    (
        "eth_air_quality",
        eth.FilterAccess,
        eth.inspect_eth_filter,
        ("aerosol_sampler", "filter_drawer"),
    ),
    (
        "rochester_genomics",
        roc._Run,
        roc.run_rochester_access,
        ("liquid_handler", "gantry_x"),
    ),
    (
        "indiana_microfluidics",
        ind._Inspection,
        ind.inspect_indiana,
        ("microfluidic_station", "stage_x"),
    ),
)


def state(s):
    return (
        np.asarray(s.data.time).tobytes(),
        *(
            getattr(s.data, n).tobytes()
            for n in (
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


def short(task, seconds=0.02):
    task.phase("short_hold", task.home, seconds, seconds)


class CustomCarrierResetReportingTests(unittest.TestCase):
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
        m, mf = self.templates[scene]
        return InstrumentSession(copy(m), deepcopy(mf))

    def check_failure(self, report, reset):
        self.assertFalse(report["success"])
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(report.get("completed_checks", report.get("checked_steps")), 0)
        self.assertIs(report["reset_performed"], reset)
        self.assertFalse(report["explicit_reset_performed"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["unchecked_steps"], 1)
        self.assertIsNone(report.get("whole_run_clearance_lower_bound_m"))
        self.assertEqual(len(report["physics_events"]), 1)
        self.assertIs(report["phases"][0]["reset_performed"], reset)

    def test_public_actual_native_reset_with_zero_repeated_and_nonzero_history(self):
        for scene, _, run, key in CASES:
            for time, count in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
                with self.subTest(scene=scene, time=time, prior=count):
                    s = self.fresh(scene)
                    s.data.time = time
                    s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = count
                    s.data.qfrc_applied[s.model.jnt_dofadr[s.controls[key].joint]] = (
                        1e25
                    )
                    report = run(s)
                    self.check_failure(report, True)
                    self.assertTrue(report["automatic_reset_detected"])
                    self.assertAlmostEqual(s.data.time, s.model.opt.timestep)
                    self.assertFalse(np.any(s.data.qfrc_applied))
                    self.assertIn(
                        "qfrc_applied",
                        report["physics_events"][0]["reset_cleared_inputs"],
                    )

    def test_large_clock_and_unchanged_bad_warning_remain_unknown(self):
        for scene, _, run, _ in CASES:
            for kind in ("large_clock", "stale_warning"):
                with self.subTest(scene=scene, kind=kind):
                    s = self.fresh(scene)
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

    def test_nonfinite_entry_or_cached_phase_time_rejects_without_mutation(self):
        for scene, constructor, run, _ in CASES:
            for time in (float("nan"), float("inf"), -float("inf")):
                for cached in (False, True):
                    with self.subTest(scene=scene, time=time, cached=cached):
                        s = self.fresh(scene)
                        task = constructor(s) if cached else None
                        s.data.time = time
                        before = state(s)
                        with self.assertRaises(ValueError):
                            short(task) if cached else run(s)
                        self.assertEqual(state(s), before)

    def test_normal_short_phases_retain_full_coverage_without_reset(self):
        for scene, constructor, _, _ in CASES:
            with self.subTest(scene=scene):
                s = self.fresh(scene)
                task = constructor(s)
                short(task)
                r = task.report
                self.assertEqual(r["physics_steps"], 20)
                self.assertEqual(r.get("completed_checks", r.get("checked_steps")), 20)
                self.assertEqual(r["physics_events"], [])
                self.assertFalse(r["reset_performed"])
                self.assertFalse(r["explicit_reset_performed"])


if __name__ == "__main__":
    unittest.main()
