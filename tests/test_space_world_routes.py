"""Read-only world pages expose scope and reject unregistered asset paths."""

from pathlib import Path
import sys
import unittest

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from webui.backend_api import bp


class WorldRoutes(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        self.client = app.test_client()

    def test_reference_conditions_and_disabled_mechanisms_remain_public(self):
        response = self.client.get("/api/space-worlds")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(
            {w["name"] for w in data["worlds"]}, {"orbital", "lunar", "martian"}
        )
        for world in data["worlds"]:
            self.assertFalse(world["scientific_process_validated"])
            self.assertIn("gas_dynamics", world["disabled"])
            self.assertIn("not_simulated", world["protected_module"]["status"])
        self.assertIn(
            "Universe", self.client.get("/space-worlds").get_data(as_text=True)
        )

    def test_unknown_world_backend_and_view_never_read_assets(self):
        for url in (
            "/api/space-worlds/unknown/image/isaac/overview",
            "/api/space-worlds/orbital/image/unknown/overview",
            "/api/space-worlds/orbital/image/isaac/unknown",
        ):
            self.assertEqual(self.client.get(url).status_code, 404)

    def test_new_catalogue_entries_are_display_scenes_without_fake_success(self):
        tasks = self.client.get("/api/backends/catalog").get_json()["tasks"]
        space = [t for t in tasks if t["category"] == "space" and t["name"].endswith("workstation")]
        self.assertEqual(len(space), 6)
        self.assertTrue(
            all(t["display_only"] and not t["check_constant_true"] for t in space)
        )
        experiments = [t for t in tasks if t["completion_rule"] == "space_experiment"]
        self.assertEqual(len(experiments), 9)
        self.assertTrue(all(not t["display_only"] for t in experiments))

    def test_external_assets_expose_attribution_and_only_registered_images(self):
        data = self.client.get("/api/space-assets").get_json()
        self.assertEqual(len(data["assets"]["assets"]), 4)
        self.assertFalse(data["scientific_process_validated"])
        for row in data["assets"]["assets"]:
            self.assertIn("academic_research_use", row)
            self.assertTrue(row["credit"])
        for url in (
            "/api/space-assets/unknown/image/isaac/overview",
            "/api/space-assets/orbital/image/unknown/overview",
            "/api/space-assets/orbital/image/isaac/unknown",
        ):
            self.assertEqual(self.client.get(url).status_code, 404)


if __name__ == "__main__":
    unittest.main()
