"""Every assigned school needs a sourced scene, not just another catalogue row."""

import json
from pathlib import Path
import unittest

from archetypes.task_catalog import CATALOG
from real_labs.catalog import scenes
from real_labs.inspection import recipe_for
from university.batches import load_batch
from university.catalog import affiliations, catalogue, research_candidates


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "Hooke/university/batches/qs2027_school_coverage_20261024.json"


class UniversitySchoolCoverageTests(unittest.TestCase):
    def test_every_in_scope_school_has_a_scene(self):
        directory = catalogue()
        missing = [
            row["id"]
            for row in directory["universities"]
            if row["in_scope"] and row["scene_count"] == 0
        ]
        self.assertEqual(missing, [])
        self.assertEqual(directory["coverage"]["scope_institutions_without_scenes"], 0)
        self.assertEqual(
            directory["coverage"]["scope_institutions_with_scenes"],
            directory["coverage"]["scope_institutions"],
        )

    def test_new_batch_closes_each_original_school_gap_once(self):
        batch = load_batch(BATCH)
        bindings = affiliations()
        baseline_schools = {
            bindings[sid]["institution_id"] for sid in batch["baseline_scene_ids"]
        }
        scoped = {
            row["id"] for row in catalogue()["universities"] if row["in_scope"]
        }
        required = set(batch["required_institution_ids"])
        added_schools = [
            bindings[sid]["institution_id"] for sid in batch["scene_ids"]
        ]
        self.assertEqual(required, scoped - baseline_schools)
        self.assertEqual(set(added_schools), required)
        self.assertEqual(len(added_schools), len(required))

    def test_each_added_scene_has_reviewed_spatial_evidence_and_an_operation(self):
        batch = load_batch(BATCH)
        definitions, bindings = scenes(), affiliations()
        candidates = {
            row["existing_scene_id"]: row
            for row in research_candidates()
            if row.get("existing_scene_id") in batch["scene_ids"]
        }
        for sid in batch["scene_ids"]:
            with self.subTest(scene=sid):
                definition, candidate = definitions[sid], candidates[sid]
                self.assertEqual(
                    candidate["institution_id"], bindings[sid]["institution_id"]
                )
                self.assertEqual(candidate["verification_level"], "image_reviewed")
                self.assertTrue(candidate["physical_site_verified"])
                self.assertIn(candidate["layout_evidence"], ("room", "workstation"))
                self.assertTrue(candidate["reviewed_visuals"])
                self.assertTrue(candidate["rights_status"])
                for visual in candidate["reviewed_visuals"]:
                    self.assertRegex(visual["sha256"], r"^[0-9a-f]{64}$")
                    self.assertTrue(visual["url"].startswith("https://"))
                self.assertIn(candidate["official_url"], definition["source_urls"])
                self.assertTrue(definition["observed_cues"])
                self.assertTrue(definition["limitations"])
                self.assertIn("real_lab_" + sid, CATALOG)
                recipe = recipe_for(sid)
                self.assertIsNotNone(recipe)
                self.assertEqual(recipe["scene"], sid)
                self.assertTrue(recipe["limits"])
                self.assertGreaterEqual(
                    sum(len(device["controls"]) for device in recipe["devices"]), 2
                )

    def test_toronto_selection_preserves_the_amnl_exclusion(self):
        batch = load_batch(BATCH)
        bindings, definitions = affiliations(), scenes()
        toronto = [
            sid for sid in batch["scene_ids"]
            if bindings[sid]["institution_id"] == "toronto"
        ]
        self.assertEqual(len(toronto), 1)
        source_identity = json.dumps(
            {
                "binding": bindings[toronto[0]],
                "sources": definitions[toronto[0]]["source_urls"],
            }
        ).casefold()
        self.assertNotIn("amnl", source_identity)
        self.assertNotIn("yu sun", source_identity)


if __name__ == "__main__":
    unittest.main()
