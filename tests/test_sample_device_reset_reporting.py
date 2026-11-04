"""Native-reset failures must not certify sample-handling inspection steps."""

from copy import copy, deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.mini_centrifuge_fixture import create_inspection_session
from real_labs.mini_centrifuge_motion import MiniInspection
from real_labs.runtime import InstrumentSession
from real_labs.uw_isolab_motion import IsolabInspection


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
    if isinstance(task, IsolabInspection):
        return task.leg("short_hold", settle_s=0.02)
    task._phase("short_hold", 0.02, parked=True)
    return task.phases[-1]


class SampleDeviceResetReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, manifest = build_scene("uw_isolab")
        cls.templates = {
            "uw": InstrumentSession(
                mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
                manifest,
            ),
            "mini": create_inspection_session("rochester_genomics"),
        }

    def fresh(self, name):
        s = self.templates[name]
        return InstrumentSession(copy(s.model), deepcopy(s.manifest))

    def task(self, name, s):
        return (IsolabInspection if name == "uw" else MiniInspection)(s)

    def assert_failed_event(self, r, reset):
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["executed_steps"], 1)
        self.assertEqual(r.get("checked_steps", r.get("completed_checks")), 0)
        self.assertIs(r["reset_performed"], reset)
        self.assertFalse(r["explicit_reset_performed"])
        self.assertEqual(len(r["physics_events"]), 1)
        p = r["phases"][0]
        self.assertIs(p["reset_performed"], reset)
        self.assertFalse(p.get("success", p.get("status") == "succeeded"))
        if "whole_leg_clearance_lower_bound_m" in p:
            self.assertIsNone(p["whole_leg_clearance_lower_bound_m"])
            self.assertTrue(p["hold_requested"])
        else:
            self.assertFalse(r["coverage_complete"])
            self.assertTrue(r["hold_requested"])

    def test_public_run_detects_actual_native_resets_including_repeated_warning(self):
        for name in self.templates:
            for clock, count in ((0.7, 0), (0.0, 0), (0.0, 1), (0.0, 7)):
                with self.subTest(name=name, clock=clock, count=count):
                    s = self.fresh(name)
                    s.data.time = clock
                    s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = count
                    task = self.task(name, s)
                    joint = task.controls[0].joint if name == "uw" else task.lid.joint
                    s.data.qfrc_applied[s.model.jnt_dofadr[joint]] = 1e25
                    r = task.run()
                    self.assert_failed_event(r, True)
                    self.assertTrue(r["automatic_reset_detected"])
                    self.assertAlmostEqual(s.data.time, s.model.opt.timestep)
                    self.assertIn(
                        "qfrc_applied", r["physics_events"][0]["reset_cleared_inputs"]
                    )

    def test_frozen_clock_and_ambiguous_warning_do_not_report_success(self):
        for name in self.templates:
            for kind in ("large_clock", "stale_warning"):
                with self.subTest(name=name, kind=kind):
                    s = self.fresh(name)
                    if kind == "large_clock":
                        s.data.time = 1e30
                    else:
                        s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
                    r = self.task(name, s).run()
                    self.assert_failed_event(r, None)
                    self.assertFalse(r["automatic_reset_detected"])

    def test_nonfinite_clock_rejects_entry_and_cached_operation_without_mutation(self):
        for name in self.templates:
            for clock in (float("nan"), float("inf"), -float("inf")):
                for cached in (False, True):
                    with self.subTest(name=name, clock=clock, cached=cached):
                        s = self.fresh(name)
                        task = self.task(name, s) if cached else None
                        s.data.time = clock
                        before = state(s)
                        with self.assertRaises(ValueError):
                            short(task) if cached else self.task(name, s).run()
                        self.assertEqual(state(s), before)

    def test_short_normal_holds_have_complete_checks_and_no_reset(self):
        for name in self.templates:
            with self.subTest(name=name):
                s = self.fresh(name)
                r = short(self.task(name, s))
                steps = r.get("physics_steps", r.get("executed_steps"))
                checked = r.get("checked_steps", r.get("completed_checks"))
                self.assertEqual(steps, round(0.02 / s.model.opt.timestep))
                self.assertEqual(steps, checked)
                self.assertTrue(r.get("success", r.get("status") == "succeeded"))
                self.assertFalse(r["reset_performed"])
                self.assertFalse(r["explicit_reset_performed"])


if __name__ == "__main__":
    unittest.main()
