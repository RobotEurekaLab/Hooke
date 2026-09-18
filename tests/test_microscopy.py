"""Contact, image feedback, conservative injection and LAN command boundaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from archetypes.task_catalog import CATALOG
from backends.assessment import EpisodeAssessment
from microscopy.scene import ORIGIN, RADIUS
from microscopy.tasks import (CalibrationPush as Push, CalibrationPickPlace as PickPlace,
                              CalibrationInjection as Injection)
from microscopy.worker import apply_request
from webui.microscopy_api import bp, WorkstationBusy
from webui.robot_registry import ROBOTS, robot_options_for


class MicroscopeExperiments(unittest.TestCase):
    def task(self, cls=Push, seed=0):
        task = cls.Expert(cls.load())
        task.reset(seed)
        return task

    def test_all_three_experts_complete_with_actual_contact_and_volume_evidence(self):
        for cls in (Push, PickPlace, Injection):
            with self.subTest(operation=cls.operation):
                task = self.task(cls)
                task.execute()
                self.assertTrue(task.check(), task.microscopy_report())
                self.assertFalse(np.any(task.data.warning.number))
                self.assertEqual(task.model.neq, 0, "No grasp weld or sample attachment")
                self.assertLess(task.data.time, task.time_limit)
                if cls is PickPlace:
                    self.assertGreater(task.mechanics.bilateral_contact_s, .05)
                    self.assertGreater(task.mechanics.lift_m, .0008)
                if cls is Injection:
                    self.assertAlmostEqual(task.mechanics.ledger.state("well").volume_m3 * 1e12, 100, places=6)

    def test_no_action_and_positive_pressure_away_from_chamber_cannot_succeed(self):
        task = self.task(Injection)
        initial = task.data.ctrl.copy()
        task.data.userdata[0] = 12000
        task.wait(.25)
        np.testing.assert_array_equal(task.data.ctrl, initial)
        self.assertEqual(task.mechanics.ledger.state("well").volume_m3, 0)
        self.assertGreater(task.mechanics.blocked_injection_s, .24)
        self.assertFalse(task.check())
        self.assertFalse(task.microscopy_checks()["delivered_100nl"])

    def test_partial_gripper_contact_is_not_a_bilateral_grasp(self):
        task = self.task(PickPlace)
        pad = next(iter(task.mechanics.pads))
        with patch.object(task.mechanics, "contacts", return_value={pad}):
            task.wait(.02)
        self.assertFalse(task.mechanics.grasped)
        self.assertEqual(task.mechanics.bilateral_contact_s, 0)
        self.assertEqual(task.mechanics.lift_m, 0)

    def test_image_localization_and_autofocus_follow_sensor_image(self):
        task = self.task()
        blurred = task.microscope.sharpness(task.microscope_image(annotate=False))
        task.autofocus()
        self.assertTrue(task.focused)
        self.assertGreater(task.microscope.sharpness(task.microscope_image(annotate=False)), blurred * 1.5)
        for sample, index in task.mechanics.samples.items():
            np.testing.assert_allclose(task.locate(sample), task.data.xpos[index, :2] - ORIGIN[:2], atol=45e-6)
        task.autofocus()
        self.assertEqual(len(task.focus_scan), 13, "A new scan must not reuse old image scores")

    def test_stage_motion_changes_fixed_optical_view(self):
        task = self.task()
        before = np.asarray(task.microscope_image())
        task.command({"stage_x": .001}, .3)
        state = task.mechanics.image_state(task.data)
        self.assertAlmostEqual(state["stage_m"][0], .001, delta=1e-5)
        self.assertFalse(np.array_equal(before, np.asarray(task.microscope_image())))
        self.assertAlmostEqual(state["samples"]["bead_push"][0],
                               task.data.xpos[task.mechanics.samples["bead_push"], 0] - ORIGIN[0])

    def test_bad_commands_do_not_partly_modify_or_step_the_experiment(self):
        task = self.task()
        initial = task.data.ctrl.copy()
        for commands in ({"probe_x": .001, "unknown": 0}, {"probe_x": float("nan")}, {"jaw_a": .1}):
            with self.subTest(commands=commands), self.assertRaises(ValueError):
                apply_request(task, {"command": "step", "actuators": commands, "pressure_pa": 12000})
            np.testing.assert_array_equal(task.data.ctrl, initial)
            self.assertEqual(task.data.time, 0)
            self.assertEqual(task.data.userdata[0], 0)

    def test_reset_and_report_do_not_fabricate_progress_or_reveal_sample_poses(self):
        task = self.task()
        expected = task.data.qpos.copy()
        report = json.dumps(task.microscopy_report(), sort_keys=True)
        for _ in range(3):
            self.assertFalse(task.check())
            self.assertEqual(report, json.dumps(task.microscopy_report(), sort_keys=True))
        task.wait(.1)
        task.reset(0)
        np.testing.assert_array_equal(task.data.qpos, expected)
        self.assertNotIn("samples", task.public_state())
        self.assertEqual(task.mechanics.push_contact_s, 0)
        self.assertAlmostEqual(task.model.body("bead_push").mass[0], 2500 * 4 / 3 * np.pi * RADIUS**3)

    def test_catalog_and_assessment_use_functional_workstation(self):
        self.assertTrue(ROBOTS["micro_workstation"].controller_available)
        self.assertEqual(robot_options_for("micro_workstation"), ["micro_workstation"])
        for name in ("push", "pick_place", "injection"):
            entry = CATALOG["microscopy_" + name]
            self.assertEqual(entry.robot, "micro_workstation")
            self.assertEqual(entry.completion_rule, "microscopy_experiment")
        task = self.task()
        observer = EpisodeAssessment(task, "microscopy_push")
        task.wait(.01)
        observer.update()
        report = observer.report()
        self.assertEqual(report["scope"], "microscopy_experiment")
        self.assertFalse(report["success"])


class MicroscopeRoutes(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_invalid_commands_do_not_start_a_gpu_worker(self):
        with patch("webui.microscopy_api.workstation.command") as command:
            for body in (None, [], {"command": "shell"}):
                self.assertEqual(self.client.post("/api/microscopy/control", json=body).status_code, 400)
            command.assert_not_called()

    def test_busy_and_validation_errors_have_distinct_status(self):
        for error, status in ((WorkstationBusy("busy"), 409), (ValueError("bad axis"), 400)):
            with patch("webui.microscopy_api.workstation.command", side_effect=error):
                response = self.client.post("/api/microscopy/control", json={"command": "step"})
                self.assertEqual(response.status_code, status)

    def test_missing_media_and_unknown_paths_have_no_fabricated_image(self):
        with tempfile.TemporaryDirectory() as directory, patch("webui.microscopy_api.LIVE", Path(directory)):
            self.assertEqual(self.client.get("/api/microscopy/state").get_json(), {"phase": "offline"})
            self.assertEqual(self.client.get("/api/microscopy/image/overview").status_code, 404)
            self.assertEqual(self.client.get("/api/microscopy/image/result.json").status_code, 404)
            self.assertEqual(self.client.get("/api/microscopy/recording/unknown").status_code, 404)


if __name__ == "__main__":
    unittest.main()
