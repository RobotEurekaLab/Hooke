"""Failed seeds stay out of datasets and render contexts close on every exit."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from archetypes import demo_export
from archetypes.policy_client import CameraRenderer


class ScriptedExpert:
    dt = 0.05
    task = "export_contract"
    task_info = {"prefix": "export a successful episode"}
    arm = SimpleNamespace(act_span=range(6), gripper_id=6)
    data = SimpleNamespace(ctrl=np.zeros(7))

    def reset(self, seed):
        self.seed = seed

    def step_and_log(self, info):
        pass

    def execute(self):
        self.step_and_log({})
        if self.seed == 0:
            raise AssertionError("unreachable seed")

    def check(self):
        return True


class DemoExport(unittest.TestCase):
    def test_camera_closes_all_contexts_and_can_close_twice(self):
        camera = CameraRenderer()
        renderers = [Mock(), Mock()]
        camera._cache = dict(enumerate(renderers))
        camera.close()
        camera.close()
        self.assertFalse(camera._cache)
        for renderer in renderers:
            renderer.close.assert_called_once_with()

    def test_camera_closes_when_context_body_raises(self):
        camera = CameraRenderer()
        renderer = Mock()
        camera._cache[0] = renderer
        with self.assertRaisesRegex(RuntimeError, "failed render"):
            with camera:
                raise RuntimeError("failed render")
        renderer.close.assert_called_once_with()
        self.assertFalse(camera._cache)

    def test_failed_seed_is_visible_skipped_and_next_episode_exports(self):
        camera = CameraRenderer()
        renderer = Mock()
        camera._cache[0] = renderer
        observation = {
            "state": np.zeros(7, dtype=np.float32),
            "base_rgb": np.zeros((3, 4, 4), dtype=np.uint8),
            "wrist_rgb": np.zeros((3, 4, 4), dtype=np.uint8),
        }
        entry = SimpleNamespace(name="export_contract", make_expert=ScriptedExpert)
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            demo_export, "CameraRenderer", return_value=camera
        ), patch.object(
            camera, "get_observation", return_value=observation
        ), redirect_stdout(
            log
        ):
            output = Path(directory)
            stats = demo_export.export_demos(entry, [0, 1], out_dir=output)
            self.assertEqual((stats.attempted, stats.exported), (2, 1))
            self.assertFalse((output / "episode_00000.npz").exists())
            with np.load(output / "episode_00001.npz", allow_pickle=False) as data:
                self.assertEqual(data["action"].shape, (1, 7))
                self.assertEqual(data["base_rgb"].shape, (1, 3, 4, 4))
        self.assertIn("seed 0", log.getvalue())
        self.assertIn("AssertionError", log.getvalue())
        renderer.close.assert_called_once_with()

    def test_setup_exception_propagates_and_releases_render_contexts(self):
        camera = CameraRenderer()
        renderer = Mock()
        camera._cache[0] = renderer
        entry = SimpleNamespace(
            name="export_contract",
            make_expert=Mock(side_effect=RuntimeError("bad setup")),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            demo_export, "CameraRenderer", return_value=camera
        ), self.assertRaisesRegex(RuntimeError, "bad setup"):
            demo_export.export_demos(entry, [0], out_dir=Path(directory))
        renderer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
