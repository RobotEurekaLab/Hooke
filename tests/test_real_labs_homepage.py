"""Homepage laboratory previews use recorded files without invoking GPU renderers."""

import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from archetypes.task_catalog import CATALOG, CatalogEntry


class RealLabHomepageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The existing generic robot module loads its plugin relative to cwd.
        with contextlib.chdir(ROOT / "Hooke"):
            from webui import server
        cls.server = server

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "temp")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        (self.output / "eth_air_quality").mkdir()
        environment = patch.dict(
            "os.environ", {"HOOKE_REAL_LABS_OUTPUT": str(self.output)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.client = self.server.app.test_client()
        self.renderers = []
        for function in ("render_scene", "render_robot_preview", "compose_scene"):
            stub = patch.object(
                self.server, function, side_effect=AssertionError("Attempted rendering")
            )
            self.renderers.append(stub.start())
            self.addCleanup(stub.stop)

    def test_recorded_preview_never_loads_task_or_renderer(self):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), (50, 80, 100)).save(buffer, format="PNG")
        preview = buffer.getvalue()
        (self.output / "eth_air_quality/overview.png").write_bytes(preview)
        with patch.object(
            CatalogEntry, "load_classes", side_effect=AssertionError("Loaded task")
        ):
            response = self.client.post(
                "/api/scene", json={"task": "real_lab_eth_air_quality", "seed": 17}
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(base64.b64decode(body["image_png_base64"]), preview)
        self.assertEqual(body["render_mode"], "recorded_preview")
        self.assertFalse(body["task_info"]["seed_applied"])
        self.assertEqual(
            body["task_info"]["interaction_url"], "/real-labs?scene=eth_air_quality"
        )
        self.assertTrue(all(not renderer.called for renderer in self.renderers))

    def test_missing_preview_returns_controls_link_and_actionable_setup_message(self):
        response = self.client.post(
            "/api/scene", json={"task": "real_lab_eth_air_quality"}
        )
        self.assertEqual(response.status_code, 404)
        body = response.get_json()
        self.assertIn("HOOKE_REAL_LABS_OUTPUT", body["error"])
        self.assertEqual(body["interaction_url"], "/real-labs?scene=eth_air_quality")
        self.assertTrue(all(not renderer.called for renderer in self.renderers))

    def test_recorded_preview_response_has_a_size_bound(self):
        with (self.output / "eth_air_quality/overview.png").open("wb") as output:
            output.truncate(16 * 1024 * 1024 + 1)
        response = self.client.post(
            "/api/scene", json={"task": "real_lab_eth_air_quality"}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("16 MiB", response.get_json()["error"])
        self.assertTrue(all(not renderer.called for renderer in self.renderers))

    def test_homepage_catalogue_links_all_labs_to_their_dedicated_controls(self):
        with patch.object(
            CatalogEntry, "load_classes", side_effect=AssertionError("Loaded task")
        ):
            response = self.client.get("/api/catalog")
        labs = [
            item
            for item in response.get_json()["tasks"]
            if item["category"] == "real_labs"
        ]
        batch = json.loads(
            (
                ROOT / "Hooke/university/batches/qs2027_100_labs_20261003.json"
            ).read_text()
        )
        expected = set(batch["baseline_scene_ids"] + batch["scene_ids"])
        self.assertEqual(
            {lab["name"].removeprefix("real_lab_") for lab in labs}, expected
        )
        for lab in labs:
            self.assertEqual(
                lab["navigation"]["url"],
                "/real-labs?scene=" + lab["name"].removeprefix("real_lab_"),
            )
            self.assertEqual(lab["robot_options"], [lab["robot"]])

    def test_catalogue_adapter_loads_real_scene_and_does_not_claim_completion(self):
        task_class, expert_class = CATALOG["real_lab_eth_air_quality"].load_classes()
        task = task_class(task_class.load())
        self.assertGreater(task.model.ngeom, 0)
        self.assertGreater(task.model.nu, 0)
        state = task.reset(seed=17)
        self.assertEqual(state["interaction_url"], "/real-labs?scene=eth_air_quality")
        self.assertEqual(state["camera_mapping"]["image"], "overview")
        self.assertFalse(task.check())
        with self.assertRaisesRegex(NotImplementedError, "no autonomous"):
            expert_class.execute(task)


if __name__ == "__main__":
    unittest.main()
