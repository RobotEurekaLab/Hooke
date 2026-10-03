"""The second university batch adds labs with actual controlled mechanisms."""

import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.runtime import InstrumentSession
from university.catalog import affiliations, research_candidates


class UniversityExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = json.loads(
            (ROOT / "Hooke/university/batches/qs2027_labs_20261003.json").read_text()
        )

    def test_ten_additional_distinct_labs_belong_to_ranked_institutions(self):
        ids = self.batch["scene_ids"]
        self.assertGreaterEqual(len(set(ids)), 10)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & set(self.batch["baseline_scene_ids"]))
        bindings = affiliations()
        identities = set()
        for identifier in ids:
            row = bindings[identifier]
            identities.add((row["institution_id"], row["lab_id"]))
            self.assertTrue(row["institution"]["in_scope"])
            self.assertEqual(row["institution"]["ranking_edition"], 2027)
            self.assertLessEqual(row["institution"]["rank"], 100)
        self.assertEqual(len(identities), len(ids))

    def test_built_scenes_retain_the_actual_reviewed_lab_sources(self):
        candidates = {row["id"]: row for row in research_candidates()}
        for identifier in self.batch["scene_ids"]:
            with self.subTest(scene=identifier):
                definition = scene(identifier)
                candidate = candidates[identifier]
                self.assertEqual(candidate["existing_scene_id"], identifier)
                self.assertIn(candidate["official_url"], definition["source_urls"])
                self.assertEqual(candidate["verification_level"], "image_reviewed")
                self.assertIn(candidate["layout_evidence"], ("room", "workstation"))
                self.assertTrue(definition["observed_cues"])
                self.assertTrue(definition["layout_rationale"])
                self.assertTrue(definition["reference_notes"])
                self.assertFalse(candidate["dimensions_verified"])

    def test_joint_targets_are_reached_in_both_directions_and_reset(self):
        for identifier in self.batch["scene_ids"]:
            with self.subTest(scene=identifier):
                root, manifest = build_scene(identifier)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                session = InstrumentSession(model, manifest)
                original = session.data.qpos.copy()
                self.assertGreaterEqual(len(session.controls), 2)
                for (equipment, control), spec in session.controls.items():
                    for fraction in (0.15, 0.85, 0.25):
                        target = spec.lower + fraction * (spec.upper - spec.lower)
                        result = session.command(equipment, control, target, duration=2)
                        self.assertLessEqual(
                            abs(result["error"]),
                            max(1e-6, (spec.upper - spec.lower) * 0.025),
                            (identifier, equipment, control, result),
                        )
                    session.reset()
                    np.testing.assert_allclose(session.data.qpos, original, atol=1e-12)

    def test_floor_instruments_have_real_collision_support_at_floor_height(self):
        for identifier in self.batch["scene_ids"]:
            root, manifest = build_scene(identifier)
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for equipment in manifest["equipment"]:
                if equipment.get("support", {}).get("type") != "floor":
                    continue
                with self.subTest(scene=identifier, equipment=equipment["id"]):
                    parent = model.body(equipment["id"]).id
                    bodies = {parent}
                    for body in range(parent + 1, model.nbody):
                        if model.body_parentid[body] in bodies:
                            bodies.add(body)
                    bottoms = []
                    for geom in range(model.ngeom):
                        if model.geom_bodyid[geom] not in bodies or not (
                            model.geom_contype[geom] or model.geom_conaffinity[geom]
                        ):
                            continue
                        local = model.geom_aabb[geom]
                        rotation = data.geom_xmat[geom].reshape(3, 3)
                        center = data.geom_xpos[geom] + rotation @ local[:3]
                        extent = np.abs(rotation) @ local[3:]
                        bottoms.append(center[2] - extent[2])
                    self.assertTrue(bottoms, "Floor mount has no collision geometry")
                    self.assertAlmostEqual(min(bottoms), 0, delta=0.005)


if __name__ == "__main__":
    unittest.main()
