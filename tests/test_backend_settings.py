"""Preflight rejects unsupported topology and configuration remains explicit."""

import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.capabilities import preflight
from backends.config import isaac_gpu
from backends.render_settings import RenderSettings
from backends.worker_client import IsaacWorker


class BackendSettings(unittest.TestCase):
    def test_ball_rejected_before_native_start_and_free_body_allowed(self):
        for joint, blockers in [
            ('<joint type="ball"/>', ["ball_joints"]),
            ("<freejoint/>", []),
        ]:
            xml = (
                "<mujoco><worldbody><body>"
                + joint
                + '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
            )
            model = mujoco.MjModel.from_xml_string(xml)
            r = preflight(model, xml)
            self.assertEqual(r["blockers"], blockers)
            self.assertEqual(r["can_attempt_native"], not blockers)
            self.assertFalse(r["parity_qualified"])

    def test_shared_frame_interval_and_invalid_dimensions(self):
        self.assertEqual(RenderSettings().step_interval(0.002), 25)
        self.assertEqual(
            RenderSettings(frames_per_second=10.0).step_interval(0.002), 50
        )
        for width in (0, 3.5, True, 4096):
            with self.assertRaises(ValueError):
                RenderSettings(width=width)

    def test_gpu_environment_precedence_and_invalid_gpu(self):
        with mock.patch.dict(os.environ, {"HOOKE_ISAAC_GPU": "7"}):
            self.assertEqual(isaac_gpu(), 7)
        with mock.patch.dict(os.environ, {"HOOKE_ISAAC_GPU": "-1"}):
            with self.assertRaises(ValueError):
                isaac_gpu()

    def test_failed_gpu_query_releases_lock_without_starting_worker(self):
        captured = []
        close = IsaacWorker.close

        def record_close(worker):
            captured.append(worker)
            close(worker)

        with tempfile.TemporaryDirectory() as output, mock.patch(
            "backends.worker_client.fcntl.flock"
        ), mock.patch(
            "backends.worker_client.subprocess.check_output",
            side_effect=OSError("probe failed"),
        ), mock.patch.object(
            IsaacWorker, "close", record_close
        ):
            with self.assertRaisesRegex(OSError, "probe failed"):
                IsaacWorker(Path(output), 6)
        self.assertEqual(len(captured), 1)
        self.assertIsNone(captured[0].gpu_lock)
        self.assertIsNone(captured[0].directory)
        self.assertIsNone(captured[0].process)


if __name__ == "__main__":
    unittest.main()
