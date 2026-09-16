"""Shared kernel leases must prevent cross-backend and CLI/web collisions."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
import backends.gpu_lease as leases
import backends.source_renderer as source_renderer
import webui.backend_api as api


class GPULeases(unittest.TestCase):
    def test_source_render_rejects_before_driver_query_or_context_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            leases, "ROOT", Path(directory)
        ), patch.object(source_renderer.subprocess, "check_output") as query, patch(
            "mujoco.Renderer"
        ) as renderer, leases.GPULease(
            6
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU 6 already"):
                with source_renderer.mujoco_renderer(object(), 6):
                    self.fail("Busy renderer should not be entered")
            query.assert_not_called()
            renderer.assert_not_called()

    def test_other_process_is_rejected_then_can_acquire_after_close(self):
        source = str(Path(leases.__file__).resolve().parents[1])
        code = (
            "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "import backends.gpu_lease as m;m.ROOT=Path(sys.argv[2]);"
            "lease=m.GPULease(6);lease.close()"
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            leases, "ROOT", Path(directory)
        ):
            lease = leases.GPULease(6)
            try:
                rejected = subprocess.run(
                    [sys.executable, "-c", code, source, directory],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("GPU 6 already has a Hooke job", rejected.stderr)
            finally:
                lease.close()
            lease.close()
            accepted = subprocess.run(
                [sys.executable, "-c", code, source, directory],
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_command_line_lease_blocks_both_web_backends_before_launch(self):
        app = Flask(__name__)
        app.register_blueprint(api.bp)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            leases, "ROOT", Path(directory)
        ), patch.object(api, "isaac_gpu", return_value=6), patch.object(
            api, "_jobs", {}
        ), patch.object(
            api.subprocess, "Popen"
        ) as popen, leases.GPULease(
            6
        ):
            for backend in ("mujoco", "isaac"):
                response = app.test_client().post(
                    "/api/backends/jobs",
                    json={"task": "pipette_transfer", "backend": backend},
                )
                self.assertEqual(response.status_code, 409)
                self.assertIn("GPU 6", response.json["error"])
            popen.assert_not_called()

    def test_running_source_job_rejects_native_web_start(self):
        app = Flask(__name__)
        app.register_blueprint(api.bp)
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(
            api, "_jobs", {"source": {"backend": "mujoco", "process": process}}
        ), patch.object(api.subprocess, "Popen") as popen:
            response = app.test_client().post(
                "/api/backends/jobs",
                json={"task": "pipette_transfer", "backend": "isaac"},
            )
        self.assertEqual(response.status_code, 409)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
