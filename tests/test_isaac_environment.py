"""Deployment failures remain diagnosable and cannot destroy a live source session."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends import config
from backends.doctor import diagnose
from backends.environment import inspect_installation, isaac_environment, IsaacConfigurationError
from backends.worker_client import IsaacWorker
from webui import backend_api, microscopy_api


class IsaacDeployment(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.install = self.root / "another-account/isaacsim"
        for name in ("python.sh", "setup_python_env.sh", "kit/python/bin/python3", "kit/libcarb.so"):
            path = self.install / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
            path.chmod(0o700)
        self.local = self.root / "isaac_local.json"
        self.local.write_text(json.dumps(dict(isaac_path=str(self.install), gpu=4)))
        self.stack.enter_context(patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "config")}, clear=True))
        self.stack.enter_context(patch.object(config, "LOCAL_CONFIG", self.local))

    def test_accessible_installation_outside_service_home_is_allowed(self):
        report = isaac_environment()
        self.assertTrue(report["installation_ready"])
        self.assertEqual(report["configuration_scope"], "server_process")
        self.assertEqual(report["configuration_source"], str(self.local))
        self.assertEqual(report["gpu"], 4)
        self.assertFalse(report["native_runtime_checked"])

    def test_new_account_defaults_to_first_gpu_and_setup_preserves_selection(self):
        self.local.unlink()
        self.assertEqual(config.isaac_gpu(), 0)
        destination = config.save_account_settings(self.install)
        self.assertEqual(json.loads(destination.read_text())["gpu"], 0)
        config.save_account_settings(self.install, 2)
        config.save_account_settings(self.install)
        self.assertEqual(config.isaac_gpu(), 2)

    def test_account_settings_survive_checkout_relocation_and_environment_wins(self):
        destination = config.save_account_settings(self.install, 2)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
        self.local.unlink()
        self.assertEqual(config.isaac_gpu(), 2)
        self.assertEqual(config.isaac_installation(), self.install)
        self.local.write_text("invalid lower-priority configuration")
        self.assertEqual(config.isaac_gpu(), 2)
        with patch.dict(os.environ, {"HOOKE_ISAAC_PATH": str(self.install), "HOOKE_ISAAC_GPU": "7"}):
            destination.write_text("invalid ignored configuration")
            self.assertTrue(isaac_environment()["installation_ready"])
            self.assertEqual(config.isaac_gpu(), 7)

    def test_missing_installation_and_incomplete_bundle_are_distinct(self):
        missing = inspect_installation(self.root / "absent")
        self.assertEqual(missing["error_code"], "installation_missing")
        (self.install / "kit/python/bin/python3").unlink()
        incomplete = isaac_environment()
        self.assertEqual(incomplete["error_code"], "installation_incomplete")
        self.assertEqual(incomplete["blocked_path"], str(self.install / "kit/python/bin/python3"))

    def test_denied_parent_is_not_reported_as_missing_installation(self):
        denied = self.install.parent
        access = os.access

        def readable(path, mode, **kwargs):
            return False if Path(path) == denied else access(path, mode, **kwargs)

        with patch("backends.environment.os.access", side_effect=readable):
            report = isaac_environment()
        self.assertEqual(report["error_code"], "permission_denied")
        self.assertEqual(report["blocked_path"], str(denied))
        self.assertIn("权限", "\n".join(report["remediation"]))

    def test_real_launcher_permission_is_checked_before_gpu_or_subprocess(self):
        (self.install / "python.sh").chmod(0o600)
        with patch("backends.worker_client.GPULease") as lease, patch(
            "backends.worker_client.subprocess.Popen"
        ) as process:
            with self.assertRaises(IsaacConfigurationError) as error:
                IsaacWorker(self.root / "worker", 4)
        self.assertEqual(error.exception.report["error_code"], "not_executable")
        self.assertIn("权限", "\n".join(error.exception.report["remediation"]))
        lease.assert_not_called()
        process.assert_not_called()

    def test_malformed_configuration_still_produces_doctor_report(self):
        self.local.write_text("{broken")
        with patch("backends.doctor.subprocess.check_output", return_value="6, uuid, GPU, driver, 0, 24000"):
            report = diagnose(6)
        self.assertFalse(report["environment_files_ready"])
        self.assertEqual(report["isaac_environment"]["error_code"], "configuration_invalid")
        self.assertFalse(report["native_runtime_started"])

    def test_noninteger_and_negative_gpus_are_rejected(self):
        for value in (True, 1.5, -1, "oops"):
            with self.subTest(value=value):
                self.local.write_text(json.dumps(dict(isaac_path=str(self.install), gpu=value)))
                report = isaac_environment()
                self.assertFalse(report["installation_ready"])
                self.assertEqual(report["error_code"], "configuration_invalid")

    def test_explicit_doctor_gpu_overrides_invalid_saved_gpu(self):
        self.local.write_text(json.dumps(dict(isaac_path=str(self.install), gpu="broken")))
        with patch("backends.doctor.subprocess.check_output", return_value="2, uuid, GPU, driver, 0, 24000"):
            report = diagnose(2)
        self.assertEqual(report["gpu"], 2)
        self.assertTrue(report["checks"]["isaac_installation_access"])

    def test_container_uid_without_passwd_entry_is_still_diagnosable(self):
        with patch("backends.environment.pwd.getpwuid", side_effect=KeyError):
            report = isaac_environment()
        self.assertTrue(report["installation_ready"])
        self.assertEqual(report["service_user"], str(os.geteuid()))

    def test_environment_endpoint_is_read_only_and_never_launches_native(self):
        app = Flask(__name__)
        app.register_blueprint(backend_api.bp)
        with patch("subprocess.Popen") as process:
            response = app.test_client().get("/api/backends/environment")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertTrue(response.json["isaac"]["installation_ready"])
        self.assertEqual(app.test_client().post("/api/backends/environment", json={"isaac_path": "/tmp"}).status_code, 405)
        process.assert_not_called()

    def test_bad_installation_returns_structured_503_without_creating_job(self):
        app = Flask(__name__)
        app.register_blueprint(backend_api.bp)
        (self.install / "python.sh").unlink()
        with patch.object(backend_api, "_jobs", {}), patch.object(backend_api, "JOBS", self.root / "jobs"), patch.object(
            backend_api, "GPULease"
        ) as lease, patch("subprocess.Popen") as process:
            response = app.test_client().post("/api/backends/jobs", json=dict(task="pipette_transfer", backend="isaac"))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["environment"]["error_code"], "installation_incomplete")
        self.assertFalse((self.root / "jobs").exists())
        lease.assert_not_called()
        process.assert_not_called()

    def test_failed_native_switch_preserves_running_mujoco_session(self):
        app = Flask(__name__)
        app.register_blueprint(microscopy_api.bp)
        source = MagicMock()
        source.poll.return_value = None
        workstation = microscopy_api.Workstation()
        workstation.process, workstation.backend = source, "mujoco"
        (self.install / "python.sh").unlink()
        with patch.object(microscopy_api, "workstation", workstation), patch.object(
            workstation, "close"
        ) as close, patch("subprocess.Popen") as process:
            response = app.test_client().post("/api/microscopy/control", json=dict(command="reset", backend="isaac"))
        self.assertEqual(response.status_code, 503)
        self.assertIs(workstation.process, source)
        self.assertEqual(workstation.backend, "mujoco")
        close.assert_not_called()
        process.assert_not_called()

    def test_microscope_worker_uses_same_configured_gpu_as_native_worker(self):
        workstation = microscopy_api.Workstation()
        with patch.object(microscopy_api, "MEDIA", self.root / "media"), patch.object(
            workstation, "receive", return_value={"ready": True}
        ), patch("subprocess.Popen") as process:
            workstation.start("isaac")
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("--gpu") + 1], "4")
        workstation.log.close()

    def test_configure_cli_saves_valid_bundle_for_a_new_deployment(self):
        source = Path(config.__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-m", "backends.config", "--isaac-path", str(self.install), "--gpu", "2"],
            cwd=source, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["gpu"], 2)
        self.assertEqual(report["configuration_source"], str(config.account_config()))

    def test_explicit_setup_repairs_invalid_account_configuration(self):
        destination = config.account_config()
        destination.parent.mkdir(parents=True)
        destination.write_text("broken")
        self.local.write_text("invalid lower-priority configuration")
        config.save_account_settings(self.install, 2)
        self.assertTrue(isaac_environment()["installation_ready"])
        self.assertEqual(config.isaac_gpu(), 2)

    def test_launcher_setup_from_another_directory_uses_saved_gpu(self):
        script = Path(config.__file__).resolve().parents[2] / "scripts/start_hooke_backends.sh"
        environment = dict(os.environ, HOOKE_SOURCE_PYTHON=sys.executable)
        result = subprocess.run(
            ["/bin/bash", str(script), "--configure-isaac", "--isaac-path", str(self.install), "--gpu", "2"],
            cwd=self.root, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["gpu"], 2)


if __name__ == "__main__":
    unittest.main()
