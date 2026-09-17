"""Humanoid cooperation requires real actuator control and contact evidence."""

import sys
import io
import json
from contextlib import redirect_stderr, redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from archetypes.task_catalog import CATALOG
from backends.run import main as run_main, run
from surface.evidence import control_passed
from surface.cooperation import make_cooperative_task
from surface.humanoid import HOME, HumanoidNavigation
from surface.profiles import MISSIONS
from surface.recurrent import RecurrentPolicy
from surface.replay_render import render_replay
from webui.surface_api import bp


class HumanoidPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_type = make_cooperative_task("lunar")
        cls.task = cls.task_type(cls.task_type.load())

    def setUp(self):
        self.task.reset(0)

    def test_free_base_has_no_weld_or_automatic_idle_commands(self):
        task = self.task
        root = task.model.joint("/g1:floating_base_joint")
        self.assertEqual(root.type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(task.model.body("/g1:pelvis").parentid[0], 0)
        self.assertFalse(np.any(task.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
        controls = task.data.ctrl.copy()
        with patch("surface.humanoid.RecurrentPolicy", side_effect=AssertionError):
            for _ in range(100):
                task.step_and_log({})
        np.testing.assert_array_equal(task.data.ctrl, controls)
        self.assertIsNone(task.gait.policy)
        self.assertFalse(task.check())
        self.assertFalse(task.team.button_pressed)

    def test_hand_kinematics_only_writes_actuator_targets(self):
        task = self.task
        qpos = task.data.qpos.copy()
        qvel = task.data.qvel.copy()
        controls = task.data.ctrl.copy()
        task.finger.target = task.data.site_xpos[task.finger.site] + [0, 0.03, 0]
        task.finger.command_joints(task.data)
        np.testing.assert_array_equal(task.data.qpos, qpos)
        np.testing.assert_array_equal(task.data.qvel, qvel)
        changed = np.flatnonzero(task.data.ctrl != controls)
        self.assertGreater(len(changed), 0)
        self.assertTrue(set(changed) <= set(task.finger.actuators))

    def test_authorization_needs_location_contact_stroke_and_duration(self):
        task = self.task
        team, data = task.team, task.data
        data.qpos[team.button_qa] = 0.005
        with patch("surface.cooperation.touching", return_value=True):
            for _ in range(5):
                data.time += 0.02
                team.update(data)
            self.assertFalse(team.button_pressed)
            self.assertEqual(team.button_contact_s, 0)
            data.xpos[team.body, :2] = (
                np.asarray(MISSIONS["lunar"].collection_stop) + HOME
            )
            data.qpos[team.button_qa] = 0.003
            data.time += 0.1
            team.update(data)
            self.assertFalse(team.button_pressed)
            data.qpos[team.button_qa] = 0.005
            data.time += 0.04
            team.update(data)
            self.assertFalse(team.button_pressed)
        with patch("surface.cooperation.touching", return_value=False):
            data.time += 0.02
            team.update(data)
            self.assertEqual(team.button_contact_s, 0)
        with patch("surface.cooperation.touching", return_value=True):
            data.time += 0.051
            team.update(data)
        self.assertTrue(team.button_pressed)
        self.assertFalse(task.check())

    def test_gait_rejects_zero_or_invalid_gravity(self):
        for gravity in (0, -1, np.inf, np.nan):
            with self.subTest(gravity=gravity):
                with self.assertRaisesRegex(ValueError, "positive finite gravity"):
                    HumanoidNavigation(self.task.model, gravity)

    def test_original_index_collision_is_valid_contact_without_touchpad(self):
        task, data = self.task, self.task.data
        pad = task.model.geom("/g1:index_touchpad").id
        original = next(
            geom
            for geom in range(task.model.ngeom)
            if geom != pad
            and task.model.geom_bodyid[geom]
            == task.model.body("/g1:left_hand_index_1_link").id
            and task.model.geom_contype[geom] != 0
        )
        self.assertIn(original, task.team.hand_geoms)
        data.ncon = 0
        contact = mujoco.MjContact()
        contact.geom[:] = [original, task.model.geom("team_button_cap").id]
        self.assertEqual(mujoco.mj_addContact(task.model, data, contact), 0)
        task.team.visited = True
        data.qpos[task.team.button_qa] = 0.005
        for _ in range(3):
            data.time += 0.02
            task.team.update(data)
        self.assertTrue(task.team.button_pressed)
        self.assertFalse(task.check())

    def test_extended_duration_is_limited_to_cooperative_tasks(self):
        for world, mission in MISSIONS.items():
            self.assertEqual(CATALOG[mission.task_name].max_sim_seconds, 120)
            entry = CATALOG[f"space_{world}_humanoid_rover"]
            self.assertEqual(entry.max_sim_seconds, 180)
            self.assertEqual(entry.robot, "g1_rover_team")

    def test_active_cooperation_aborts_on_the_same_upright_failure_as_success_check(
        self,
    ):
        task = self.task
        task.team.max_tilt = 0.6
        task.gait.enabled = True
        with patch.object(task.gait, "command_joints"):
            with self.assertRaisesRegex(RuntimeError, "upright posture"):
                task.step_and_log({})
        task.gait.enabled = False
        task.step_and_log({})
        self.assertFalse(task.check())

    def test_compiled_palm_is_contact_surface_and_wrist_mesh_is_excluded(self):
        model, hand = self.task.model, self.task.team.hand_geoms
        for name, included in (
            ("left_hand_palm_link", True),
            ("left_wrist_yaw_link", False),
        ):
            mesh = model.mesh("/g1:" + name).id
            geoms = set(
                np.flatnonzero(
                    (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                    & (model.geom_dataid == mesh)
                )
            )
            self.assertTrue(geoms)
            self.assertEqual(geoms <= hand, included)
            if not included:
                self.assertTrue(geoms.isdisjoint(hand))

    def test_upright_history_failure_is_irreversible_without_reset(self):
        task = self.task
        self.assertIsNone(task.irreversible_control_failure())
        task.team.min_height = 0.54
        self.assertTrue(task.irreversible_control_failure()["irreversible"])
        self.assertFalse(task.mission_checks()["humanoid_remained_upright"])
        task.reset(0)
        self.assertIsNone(task.irreversible_control_failure())

    def test_physical_idle_control_stops_at_irreversible_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = SimpleNamespace(
                task="space_lunar_humanoid_rover",
                backend="mujoco",
                mode="no_action",
                seed=0,
                seconds=10.0,
                max_sim_seconds=180.0,
                gpu=7,
                no_render=True,
                output=folder,
                physics_options=None,
                science_model=None,
            )
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result["status"], "CONTROL_REJECTED_EARLY")
            self.assertLess(result["simulation_s"], 10)
            self.assertEqual(result["requested_control_horizon_s"], 10)
            self.assertFalse(result["source_success"])
            expert = dict(
                mode="expert",
                status="TASK_SUCCEEDED",
                source_success=True,
                within_declared_time_limit=True,
                simulation_s=10.0,
                surface_mission={"checks": {"humanoid_remained_upright": True}},
            )
            self.assertTrue(control_passed(expert, result, folder))
            result["control_rejection"]["irreversible"] = False
            self.assertFalse(control_passed(expert, result, folder))
            result["control_rejection"]["irreversible"] = True
            result["requested_control_horizon_s"] = 9.0
            self.assertFalse(control_passed(expert, result, folder))
            result["requested_control_horizon_s"] = 10.0
            result["control_rejection"]["max_tilt_rad"] = 0.1
            result["control_rejection"]["min_pelvis_height_m"] = 0.785
            self.assertFalse(control_passed(expert, result, folder))
            self.assertFalse(control_passed({}, result, folder))


class GaitBoundary(unittest.TestCase):
    def test_missing_or_malformed_policy_cannot_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            with self.assertRaisesRegex(FileNotFoundError, "gait_assets"):
                RecurrentPolicy(path)
            np.savez(path, arbitrary=np.zeros(47))
            with self.assertRaisesRegex(ValueError, "architecture"):
                RecurrentPolicy(path)

    def test_invalid_observation_does_not_mutate_recurrent_state(self):
        # Exercise the input boundary independently of downloaded weights.
        policy = RecurrentPolicy.__new__(RecurrentPolicy)
        policy.h = np.arange(64, dtype=np.float32)
        policy.c = -policy.h.copy()
        h, c = policy.h.copy(), policy.c.copy()
        for observation in (np.zeros(46), np.zeros((1, 47)), np.full(47, np.nan)):
            with self.assertRaisesRegex(ValueError, "47 finite"):
                policy(observation)
            np.testing.assert_array_equal(policy.h, h)
            np.testing.assert_array_equal(policy.c, c)


class TaskDuration(unittest.TestCase):
    def test_cli_defaults_to_the_selected_task_limit(self):
        for task, limit in (
            ("space_lunar_humanoid_rover", 180),
            ("space_martian_surface_sampling", 120),
        ):
            argv = ["run", "--task", task, "--backend", "mujoco", "--output", "/unused"]
            with (
                self.subTest(task=task),
                patch.object(sys, "argv", argv),
                patch(
                    "backends.run.run", return_value={"status": "TASK_SUCCEEDED"}
                ) as execute,
                self.assertRaises(SystemExit) as exited,
            ):
                run_main()
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(execute.call_args.args[0].max_sim_seconds, limit)

    def test_cli_cannot_extend_existing_solo_task_limit(self):
        argv = [
            "run",
            "--task",
            "space_lunar_surface_sampling",
            "--backend",
            "mujoco",
            "--output",
            "/unused",
            "--max-sim-seconds",
            "180",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("backends.run.run") as execute,
            patch("backends.run.faulthandler.enable"),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exited,
        ):
            run_main()
        self.assertEqual(exited.exception.code, 2)
        execute.assert_not_called()


class ReplayEvidence(unittest.TestCase):
    def test_failed_native_episode_cannot_create_replay_media(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original = folder / "result.json"
            original.write_text(json.dumps({"backend": "isaac", "status": "ERROR"}))
            recorded = original.read_bytes()
            output = folder / "replay"
            with patch("surface.replay_render.IsaacWorker") as worker:
                with self.assertRaisesRegex(ValueError, "qualified native"):
                    render_replay(folder, output, 6)
                worker.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(original.read_bytes(), recorded)

    def test_replay_cannot_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = dict(
                backend="isaac",
                mode="expert",
                status="TASK_SUCCEEDED",
                source_success=True,
                within_declared_time_limit=True,
                surface_mission={"checks": {"collection": True}},
            )
            (folder / "result.json").write_text(json.dumps(result))
            output = folder / "replay"
            output.mkdir()
            sentinel = output / "existing-record.txt"
            sentinel.write_text("preserve")
            with patch("surface.replay_render.IsaacWorker") as worker:
                with self.assertRaises(FileExistsError):
                    render_replay(folder, output, 6)
                worker.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve")


class TeamRoutes(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_fresh_clone_does_not_claim_team_weights_media_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("webui.surface_api.MEDIA", root),
                patch("webui.surface_api.SUMMARY", root / "solo.json"),
                patch("webui.surface_api.TEAM_SUMMARY", root / "team.json"),
                patch("webui.surface_api.GAIT_POLICY", root / "policy.npz"),
            ):
                data = self.client.get("/api/surface-missions").get_json()
                for mission in data["missions"]:
                    team = mission["team"]
                    self.assertFalse(team["policy_available"])
                    self.assertEqual(team["qualification"], [])
                    self.assertFalse(team["media"]["isaac"]["follow"]["image"])
                self.assertEqual(
                    self.client.get(
                        "/api/surface-missions/team/lunar/isaac/follow/image"
                    ).status_code,
                    404,
                )

    def test_team_media_cannot_resolve_solo_files_or_unregistered_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lunar-mujoco-follow.png").write_bytes(b"solo")
            with patch("webui.surface_api.MEDIA", root):
                url = "/api/surface-missions/team/lunar/mujoco/follow/image"
                self.assertEqual(self.client.get(url).status_code, 404)
                (root / "team-lunar-mujoco-follow.png").write_bytes(b"team")
                with self.client.get(url) as response:
                    self.assertEqual(response.data, b"team")
                    self.assertEqual(response.mimetype, "image/png")
                for bad in (
                    "orbital/mujoco/follow/image",
                    "lunar/unknown/follow/image",
                    "lunar/mujoco/follow/model",
                    "lunar/mujoco/../../AGENTS.md",
                ):
                    self.assertEqual(
                        self.client.get(
                            "/api/surface-missions/team/" + bad
                        ).status_code,
                        404,
                    )


if __name__ == "__main__":
    unittest.main()
