"""Installation discovery respects account boundaries and explicit configuration."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends import config, discovery
from backends.environment import isaac_environment
from webui import backend_api

DEFAULT_CANDIDATES = discovery.default_candidates


class IsaacDiscovery(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.local = self.root / "local.json"
        self.stack.enter_context(patch.dict(os.environ, {"HOME": str(self.root / "home"),
                                                      "XDG_CONFIG_HOME": str(self.root / "config")}, clear=True))
        self.stack.enter_context(patch.object(config, "LOCAL_CONFIG", self.local))
        self.install = self.bundle(self.root / "owner/humanoid/isaacsim")
        self.stack.enter_context(patch.object(discovery, "default_candidates", return_value=[self.install]))

    def bundle(self, root):
        for name in ("python.sh", "setup_python_env.sh", "kit/python/bin/python3", "kit/libcarb.so"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
            path.chmod(0o700)
        return root

    def cli(self, *args):
        output = io.StringIO()
        with patch.object(sys, "argv", ["backends.config", *map(str, args)]), redirect_stdout(output):
            with self.assertRaises(SystemExit) as result:
                config.main()
        return result.exception.code, output.getvalue()

    def test_unique_unconfigured_installation_is_used_without_writes_or_execution(self):
        with patch("subprocess.Popen") as process:
            report = isaac_environment()
        self.assertTrue(report["installation_ready"])
        self.assertEqual(report["configuration_source"], "auto-discovery")
        self.assertEqual(report["isaac_path"], str(self.install))
        self.assertFalse(report["native_runtime_checked"])
        self.assertFalse(config.account_config().exists())
        process.assert_not_called()

    def test_explicit_configuration_wins_and_does_not_scan(self):
        self.local.write_text(json.dumps({"isaac_path": str(self.install)}))
        with patch.object(discovery, "discover_installations") as scan:
            report = isaac_environment()
        self.assertTrue(report["installation_ready"])
        self.assertEqual(report["configuration_source"], str(self.local))
        scan.assert_not_called()

    def test_broken_explicit_path_is_reported_instead_of_silently_replaced(self):
        with patch.dict(os.environ, {"HOOKE_ISAAC_PATH": str(self.root / "missing")}):
            report = isaac_environment()
        self.assertEqual(report["error_code"], "installation_missing")
        self.assertEqual(report["configuration_source"], "HOOKE_ISAAC_PATH")

    def test_multiple_installations_return_structured_503_without_starting_job(self):
        other = self.bundle(self.root / "other/isaacsim")
        app = Flask(__name__)
        app.register_blueprint(backend_api.bp)
        with patch.object(discovery, "default_candidates", return_value=[self.install, other]), patch(
            "subprocess.Popen"
        ) as process, patch.object(backend_api, "JOBS", self.root / "jobs"):
            response = app.test_client().post("/api/backends/jobs", json={"task": "pipette_transfer", "backend": "isaac"})
        self.assertEqual(response.status_code, 503)
        environment = response.json["environment"]
        self.assertEqual(environment["error_code"], "installation_ambiguous")
        self.assertEqual(set(environment["discovery"]["available_installations"]), {str(self.install), str(other)})
        self.assertFalse((self.root / "jobs").exists())
        process.assert_not_called()

    def test_explicit_other_account_search_root_finds_accessible_bundle(self):
        with patch.object(discovery, "default_candidates", return_value=[]):
            report = discovery.discover_installations([self.root / "owner"])
        self.assertEqual(discovery.select_installation(report), self.install)
        self.assertFalse(config.account_config().exists())

    def test_permissions_are_reported_and_never_changed(self):
        (self.install / "python.sh").chmod(0o600)
        report = discovery.discover_installations()
        self.assertEqual(report["available_installations"], [])
        self.assertEqual(report["candidates"][0]["error_code"], "not_executable")
        with self.assertRaises(discovery.IsaacDiscoveryError):
            discovery.select_installation(report)
        self.assertEqual((self.install / "python.sh").stat().st_mode & 0o777, 0o600)
        environment = isaac_environment()
        self.assertEqual(environment["error_code"], "installation_not_found")
        self.assertEqual(environment["discovery"]["candidates"][0]["error_code"], "not_executable")

    def test_partial_bundle_is_excluded_from_automatic_selection(self):
        partial = self.bundle(self.root / "partial")
        (partial / "kit/libcarb.so").unlink()
        with patch.object(discovery, "default_candidates", return_value=[partial, self.install]):
            report = discovery.discover_installations()
        self.assertEqual(discovery.select_installation(report), self.install)
        self.assertEqual(report["candidates"][0]["error_code"], "installation_incomplete")

    def test_default_search_covers_account_and_legacy_layouts_without_other_homes(self):
        home = self.root / "home"
        legacy = home / ".local/share/ov/pkg/isaac-sim-4.5.0"
        legacy.mkdir(parents=True)
        candidates = DEFAULT_CANDIDATES()
        self.assertIn(home / "isaacsim", candidates)
        self.assertIn(home / "humanoid/isaacsim", candidates)
        self.assertIn(legacy, candidates)
        self.assertNotIn(self.install, candidates)

    def test_symlink_alias_does_not_create_false_ambiguity(self):
        alias = self.root / "alias"
        alias.symlink_to(self.install, target_is_directory=True)
        with patch.object(discovery, "default_candidates", return_value=[self.install, alias]):
            report = discovery.discover_installations()
        self.assertEqual(discovery.select_installation(report), self.install)

    def test_bounded_incomplete_scan_cannot_choose_an_installation(self):
        for name in ("a", "b", "c"):
            (self.root / "scan" / name).mkdir(parents=True)
        with patch.object(discovery, "MAX_SEARCH_DIRECTORIES", 2):
            report = discovery.discover_installations([self.root / "scan"])
        self.assertTrue(report["search_truncated"])
        self.assertIsNone(report["selection"])
        with self.assertRaises(discovery.IsaacDiscoveryError) as error:
            discovery.select_installation(report)
        self.assertEqual(error.exception.code, "discovery_incomplete")

    def test_read_only_cli_does_not_save_account_configuration(self):
        code, output = self.cli("--discover", "--search-root", self.install.parent)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["selection"], str(self.install))
        self.assertFalse(config.account_config().exists())

    def test_auto_cli_saves_private_account_settings_and_selected_gpu(self):
        code, output = self.cli("--auto", "--search-root", self.install.parent, "--gpu", 2)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output)["installation_ready"])
        destination = config.account_config()
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
        self.assertEqual(config.isaac_gpu(), 2)
        self.assertEqual(config.isaac_installation(), self.install)

    def test_ambiguous_auto_cli_preserves_existing_configuration(self):
        destination = config.save_account_settings(self.install, 2)
        previous = destination.read_bytes()
        other = self.bundle(self.root / "other/isaacsim")
        with patch.object(discovery, "default_candidates", return_value=[self.install, other]), patch("sys.stderr", io.StringIO()):
            code, output = self.cli("--auto")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["error_code"], "installation_ambiguous")
        self.assertEqual(destination.read_bytes(), previous)

    def test_missing_search_root_produces_diagnostics_without_creating_directory(self):
        missing = self.root / "missing"
        with patch.object(discovery, "default_candidates", return_value=[]):
            report = discovery.discover_installations([missing])
        self.assertEqual(report["available_installations"], [])
        self.assertEqual(report["candidates"][0]["error_code"], "installation_missing")
        self.assertFalse(missing.exists())

    def test_missing_default_installation_preserves_all_candidate_diagnostics(self):
        missing = self.root / "missing"
        with patch.object(discovery, "default_candidates", return_value=[missing]):
            environment = isaac_environment()
        self.assertEqual(environment["error_code"], "installation_not_found")
        self.assertEqual(environment["discovery"]["candidates"][0]["blocked_path"], str(missing))


if __name__ == "__main__":
    unittest.main()
