"""Outdoor mobility, honest completion and gallery path boundaries."""

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
from surface.profiles import MISSIONS
from surface.evidence import control_passed, expert_passed
from surface.scene import BIN_CENTER
from surface.tasks import LunarSampling, MartianSampling
from webui.surface_api import bp


class SurfacePhysics(unittest.TestCase):
    def make_task(self, world="lunar"):
        cls = LunarSampling if world == "lunar" else MartianSampling
        task = cls(cls.load())
        task.reset(0)
        return task

    def test_wheels_move_rover_in_both_local_gravities_without_pose_writes(self):
        for world in MISSIONS:
            with self.subTest(world=world):
                task = self.make_task(world)
                np.testing.assert_allclose(
                    task.model.opt.gravity,
                    [0, 0, -MISSIONS[world].environment.gravity_m_s2],
                )
                initial = task.data.xpos[task.progress.rover].copy()
                task.drive.enabled = True
                task.drive.parked = False
                task.drive.target = np.asarray(MISSIONS[world].collection_stop)
                for _ in range(round(2 / task.dt)):
                    task.step_and_log({})
                self.assertGreater(
                    task.data.xpos[task.progress.rover, 0] - initial[0], 0.6
                )
                self.assertLess(abs(task.data.xpos[task.progress.rover, 1]), 0.04)
                self.assertFalse(task.check())
                self.assertTrue(np.isfinite(task.data.qpos).all())
                self.assertFalse(np.any(task.data.warning.number))

    def test_no_action_keeps_reset_controls_and_cannot_complete(self):
        task = self.make_task()
        controls = task.data.ctrl.copy()
        for _ in range(500):
            task.step_and_log({})
        np.testing.assert_array_equal(task.data.ctrl, controls)
        self.assertFalse(task.check())
        self.assertFalse(task.progress.visited_collection_site)
        self.assertFalse(task.progress.lifted_in_grasp)
        self.assertLess(task.progress.distance_m, 0.02)

    def test_initial_sample_in_bin_does_not_substitute_for_collection(self):
        task = self.make_task()
        joint = task.model.joint("field_sample_free")
        qa = int(joint.qposadr[0])
        task.data.qpos[qa : qa + 3] = (
            task.data.xpos[task.progress.rover] + BIN_CENTER + [0, 0, 0.05]
        )
        mujoco.mj_forward(task.model, task.data)
        for _ in range(700):
            task.step_and_log({})
        self.assertTrue(task.progress.inside_bin(task.data))
        self.assertFalse(task.progress.collected)
        self.assertFalse(task.check())

    def test_reading_success_does_not_advance_progress(self):
        task = self.make_task()
        before = json.dumps(task.mission_report(), sort_keys=True)
        for _ in range(5):
            self.assertFalse(task.check())
        self.assertEqual(before, json.dumps(task.mission_report(), sort_keys=True))

    def test_environment_and_catalogue_describe_outdoor_rover(self):
        for mission in MISSIONS.values():
            entry = CATALOG[mission.task_name]
            self.assertEqual(entry.robot, "surface_rover")
            self.assertEqual(entry.completion_rule, "surface_sampling")
            report = mission.environment_report()
            self.assertNotIn("workstation", report)
            self.assertEqual(report["support"], "wheel_ground_contact_on_rigid_terrain")
            self.assertIn("dust_transport", report["disabled"])


class SurfaceRoutes(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_fresh_deployment_has_no_fabricated_media_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("webui.surface_api.MEDIA", root), patch(
                "webui.surface_api.SUMMARY", root / "missing.json"
            ):
                data = self.client.get("/api/surface-missions").get_json()
                self.assertEqual(len(data["missions"]), 2)
                for mission in data["missions"]:
                    self.assertEqual(mission["qualification"], [])
                    self.assertFalse(mission["media"]["isaac"]["overview"]["image"])
                self.assertEqual(
                    self.client.get(
                        "/api/surface-missions/lunar/isaac/overview/image"
                    ).status_code,
                    404,
                )
        with self.client.get("/surface-missions") as response:
            self.assertEqual(response.status_code, 200)

    def test_media_only_reads_registered_names_and_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lunar-mujoco-follow.png").write_bytes(b"fixture-image")
            (root / "lunar-mujoco-follow.webm").write_bytes(b"fixture-video")
            with patch("webui.surface_api.MEDIA", root):
                response = self.client.get(
                    "/api/surface-missions/lunar/mujoco/follow/image"
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertEqual(response.data, b"fixture-image")
                response.close()
                with self.client.get(
                    "/api/surface-missions/lunar/mujoco/follow/webm"
                ) as video:
                    self.assertEqual(video.status_code, 200)
                    self.assertEqual(video.mimetype, "video/webm")
                for url in (
                    "/api/surface-missions/orbital/mujoco/follow/image",
                    "/api/surface-missions/lunar/unknown/follow/image",
                    "/api/surface-missions/lunar/mujoco/../../AGENTS.md",
                    "/api/surface-missions/lunar/mujoco/follow/model",
                    "/api/surface-missions/lunar/mujoco/scene/image",
                ):
                    self.assertEqual(self.client.get(url).status_code, 404)


class SurfaceEvidence(unittest.TestCase):
    def success(self):
        return dict(
            mode="expert",
            status="TASK_SUCCEEDED",
            source_success=True,
            within_declared_time_limit=True,
            simulation_s=10.0,
            surface_mission=dict(checks=dict(collected=True, returned=True)),
        )

    def test_success_label_cannot_replace_physical_checks_or_time_limit(self):
        result = self.success()
        self.assertTrue(expert_passed(result))
        result["surface_mission"]["checks"]["returned"] = False
        self.assertFalse(expert_passed(result))
        result = self.success()
        result["within_declared_time_limit"] = False
        self.assertFalse(expert_passed(result))
        result = self.success()
        result["mode"] = "preview"
        self.assertFalse(expert_passed(result))

    def test_idle_control_must_hold_actual_reset_commands_for_matched_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            np.savez(root / "source/model.npz", reset_ctrl=np.array([0.0, 1.0]))
            commands = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
            np.savez(root / "trajectory.npz", control=commands)
            control = dict(
                mode="no_action",
                status="CONTROL_COMPLETE",
                source_success=False,
                simulation_s=10.0,
            )
            self.assertTrue(control_passed(self.success(), control, root))
            control["simulation_s"] = 9.0
            self.assertFalse(control_passed(self.success(), control, root))
            control["simulation_s"] = 10.0
            commands[1, 0] = 0.1
            np.savez(root / "trajectory.npz", control=commands)
            self.assertFalse(control_passed(self.success(), control, root))


if __name__ == "__main__":
    unittest.main()
