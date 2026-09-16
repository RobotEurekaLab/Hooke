"""Scientific programme routes must select the correct boundary and runner."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
import webui.backend_api as api


class ScienceJobs(unittest.TestCase):
    def test_model_task_mismatch_is_rejected_before_process_launch(self):
        app = Flask(__name__)
        app.register_blueprint(api.bp)
        with patch.object(api.subprocess, "Popen") as popen:
            response = app.test_client().post(
                "/api/backends/jobs",
                json={
                    "task": "pipette_transfer",
                    "backend": "isaac",
                    "mode": "experiment",
                    "science_model": "thermal",
                },
            )
        self.assertEqual(response.status_code, 400)
        popen.assert_not_called()

    def test_thermal_route_has_explicit_seconds_model_and_sampling(self):
        app = Flask(__name__)
        app.register_blueprint(api.bp)
        process = MagicMock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory, patch.object(
            api, "JOBS", Path(directory)
        ), patch.object(api, "_jobs", {}), patch.object(
            api.subprocess, "Popen", return_value=process
        ) as popen, patch.object(
            api.threading, "Thread"
        ), patch.object(
            api, "GPULease"
        ):
            response = app.test_client().post(
                "/api/backends/jobs",
                json={
                    "task": "thermal_mixer",
                    "backend": "mujoco",
                    "mode": "experiment",
                    "science_model": "thermal",
                    "seconds": 2,
                },
            )
            self.assertEqual(response.status_code, 202)
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--seconds") + 1], "90.0")
            self.assertEqual(command[command.index("--mode") + 1], "no_action")
            self.assertEqual(command[command.index("--science-model") + 1], "thermal")
            self.assertEqual(popen.call_args.kwargs["env"]["HOOKE_RENDER_FPS"], "1")
            for job in api._jobs.values():
                job["log"].close()


if __name__ == "__main__":
    unittest.main()
