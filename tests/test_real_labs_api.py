"""Read-only previews and bounded CPU mechanical API; no renderer or service."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

from flask import Flask
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.runtime import InstrumentSession
from webui import real_labs_api
from test_real_labs_runtime import MANIFEST, MODEL


class RealLabApiTests(unittest.TestCase):
    def setUp(self):
        self.definitions = {
            key: {"id": key, "title": key.upper()} for key in ("alpha", "beta", "gamma")
        }
        self.catalogue = patch.object(
            real_labs_api, "scenes", return_value=self.definitions
        )
        self.catalogue.start()
        self.addCleanup(self.catalogue.stop)
        self.builder = patch.object(
            real_labs_api, "_build_session", side_effect=self.build
        )
        self.mock_build = self.builder.start()
        self.addCleanup(self.builder.stop)
        self.app = Flask(__name__)
        self.app.register_blueprint(real_labs_api.bp)
        self.client = self.app.test_client()

    @staticmethod
    def build(identifier):
        manifest = deepcopy(MANIFEST)
        manifest["id"] = identifier
        return InstrumentSession(mujoco.MjModel.from_xml_string(MODEL), manifest)

    def test_catalogue_and_page_do_not_start_physics(self):
        self.assertEqual(len(self.client.get("/api/real-labs").get_json()["scenes"]), 3)
        with self.client.get("/real-labs") as response:
            page = response.get_data(as_text=True)
        self.assertIn("Recorded render", page)
        self.assertIn("Image does not update with controls", page)
        self.assertIn("CPU physics", page)
        self.mock_build.assert_not_called()

    def test_command_observes_physics_and_reset_restores_it(self):
        response = self.client.post(
            "/api/real-labs/alpha/command",
            json={
                "equipment": "reader",
                "control": "door",
                "value": 0.1,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.get_json()["result"]["measured"], 0.09)
        state = self.client.get("/api/real-labs/alpha/state").get_json()
        self.assertGreater(state["time_s"], 1.9)
        self.assertEqual(state["render_mode"], "recorded_preview")
        reset = self.client.post("/api/real-labs/alpha/reset", json={}).get_json()
        self.assertEqual(reset["time_s"], 0)
        self.assertEqual(reset["equipment"][0]["controls"]["door"]["measured"], 0)
        self.assertEqual(self.mock_build.call_count, 1)

    def test_invalid_inputs_have_clean_client_errors(self):
        base = {"equipment": "reader", "control": "door", "value": 0.1}
        invalid = [
            None,
            [],
            {},
            dict(base, value=1),
            dict(base, value=True),
            dict(base, value="0.1"),
            dict(base, control="absent"),
            dict(base, duration=6),
            dict(base, duration=0),
            dict(base, extra=1),
            dict(base, equipment=[]),
            dict(base, value=float("nan")),
        ]
        for body in invalid:
            with self.subTest(body=body):
                response = self.client.post("/api/real-labs/alpha/command", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_session_lru_retains_two_and_rebuilds_evicted_scene(self):
        for scene in ("alpha", "beta", "alpha", "gamma"):
            self.assertEqual(
                self.client.get(f"/api/real-labs/{scene}/state").status_code, 200
            )
        cache = self.app.extensions["real_labs_sessions"]
        self.assertEqual(list(cache.items), ["alpha", "gamma"])
        self.client.get("/api/real-labs/beta/state")
        self.assertEqual(self.mock_build.call_count, 4)
        self.assertEqual(len(cache.items), 2)

    def test_unknown_scenes_and_unlisted_files_never_read_or_build(self):
        for path in (
            "/api/real-labs/absent/state",
            "/api/real-labs/alpha/preview/scene.xml",
            "/api/real-labs/absent/preview/overview.png",
        ):
            self.assertEqual(self.client.get(path).status_code, 404)
        self.mock_build.assert_not_called()

    def test_only_registered_preview_can_be_served_and_symlinks_cannot_escape(self):
        temporary_root = real_labs_api.ROOT / "temp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            base = Path(directory)
            (base / "alpha").mkdir()
            image = base / "alpha/overview.png"
            image.write_bytes(b"test preview")
            with patch.dict("os.environ", {"HOOKE_REAL_LABS_OUTPUT": str(base)}):
                with self.client.get(
                    "/api/real-labs/alpha/preview/overview.png"
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_data(), b"test preview")
                (base / "alpha/workstation.png").symlink_to(Path(__file__).resolve())
                self.assertEqual(
                    self.client.get(
                        "/api/real-labs/alpha/preview/workstation.png"
                    ).status_code,
                    404,
                )
            self.mock_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
