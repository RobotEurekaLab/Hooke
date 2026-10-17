"""New-lab counts and screenshot reviews cannot silently accept stale work."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from university import batches, review
from real_labs.export import current_scene_digest, export_scene
from real_labs.render_version import render_source_digest


ROOT = Path(__file__).resolve().parents[1]


class UniversityBatchReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "temp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def batch(self, mutation=None):
        data = dict(
            id="test",
            baseline_scene_ids=["old"],
            baseline_lab_identities=[["school", "old_lab"]],
            scene_ids=["new"],
            required_new_labs=100,
            ranking_edition=2027,
        )
        if mutation:
            mutation(data)
        path = self.directory / "batch.json"
        path.write_text(json.dumps(data))
        return path

    def bindings(self):
        return {
            name: dict(
                institution_id="school",
                lab_id=name + "_lab",
                institution=dict(in_scope=True, ranking_edition=2027),
            )
            for name in ("old", "new")
        }

    def test_rejects_old_scenes_recounted_as_new(self):
        path = self.batch(lambda data: data["scene_ids"].append("old"))
        with self.assertRaisesRegex(ValueError, "baseline scene"):
            batches.load_batch(path)

    def test_another_view_of_the_same_lab_is_not_an_additional_lab(self):
        bindings = self.bindings()
        bindings["new"]["lab_id"] = "old_lab"
        with patch.object(
            batches, "scenes", return_value={"old": {}, "new": {}}
        ), patch.object(batches, "affiliations", return_value=bindings):
            with self.assertRaisesRegex(ValueError, "counted more than once"):
                batches.load_batch(self.batch())

    def test_out_of_scope_institution_is_rejected(self):
        bindings = deepcopy(self.bindings())
        bindings["new"]["institution"]["in_scope"] = False
        with patch.object(
            batches, "scenes", return_value={"old": {}, "new": {}}
        ), patch.object(batches, "affiliations", return_value=bindings):
            with self.assertRaisesRegex(ValueError, "ranked scope"):
                batches.load_batch(self.batch())

    def test_partial_batch_is_reported_incomplete_without_inventing_exports(self):
        with patch.object(
            batches, "scenes", return_value={"old": {}, "new": {}}
        ), patch.object(
            batches, "affiliations", return_value=self.bindings()
        ), patch.object(
            batches, "research_candidates", return_value=[]
        ):
            result = batches.progress(self.batch(), self.directory)
        self.assertEqual(result["implemented_new_labs"], 1)
        self.assertEqual(result["counts"]["ready"], 0)
        self.assertEqual(result["remaining_new_labs"], 100)
        self.assertFalse(result["ready_for_user_review"])

    def test_current_digest_matches_portable_export_and_source_changes_are_rejected(
        self,
    ):
        identifier = "kaist_usdl_magnetic_probe"
        package = self.directory / identifier
        export_scene(identifier, package)
        manifest = json.loads((package / "manifest.json").read_text())
        self.assertEqual(current_scene_digest(identifier), manifest["scene_sha256"])
        # Minimal placeholders: freshness fails before image/format validation.
        (package / "render.json").write_text(
            json.dumps(dict(scene_sha256=manifest["scene_sha256"]))
        )
        for name in ("laboratory.blend", "laboratory.glb", "laboratory.usdc"):
            (package / name).write_bytes(b"test fixture")
        with patch.object(
            review, "current_scene_digest", return_value="changed-source"
        ):
            with self.assertRaisesRegex(ValueError, "source changed after export"):
                review.collect_records(self.directory, [identifier])
        with self.assertRaisesRegex(ValueError, "Renderer changed"):
            review.collect_records(self.directory, [identifier])

    def test_contact_sheets_paginate_eleven_scenes_with_equal_aspect_tiles(self):
        image_path = self.directory / "tile.png"
        Image.new("RGB", (160, 100), "navy").save(image_path)
        records = [
            dict(
                institution=dict(name="Example University", rank_display="1"),
                lab_name=f"Laboratory {i}",
                images={
                    view: dict(file="tile.png") for view in ("overview", "workstation")
                },
            )
            for i in range(11)
        ]
        pages = review.contact_sheets(self.directory, records)
        self.assertEqual(
            [page["file"] for page in pages],
            [
                "overview_summary_01.jpg",
                "overview_summary_02.jpg",
                "workstation_summary_01.jpg",
                "workstation_summary_02.jpg",
            ],
        )
        for page in pages:
            with Image.open(self.directory / page["file"]) as image:
                self.assertEqual(image.size, (1800, 3310 if page["page"] == 1 else 750))

    def test_partial_render_cannot_reuse_old_main_view_images(self):
        identifier = "kaist_usdl_magnetic_probe"
        package = self.directory / identifier
        export_scene(identifier, package)
        manifest = json.loads((package / "manifest.json").read_text())
        for name in ("laboratory.blend", "laboratory.glb", "laboratory.usdc"):
            (package / name).write_bytes(b"visual asset test fixture")
        # Nonblank images from a previous complete run remain on disk. A later
        # --views workstation run must not attribute them to its new metadata.
        image = Image.new("RGB", (160, 100), "navy")
        image.paste("white", (40, 25, 120, 75))
        for view in review.VIEWS:
            image.save(package / (view + ".png"))
        metadata = dict(
            views=list(review.VIEWS),
            dimensions=[160, 100],
            scene_sha256=manifest["scene_sha256"],
            renderer_source_sha256=render_source_digest(),
        )
        path = package / "render.json"
        path.write_text(json.dumps(metadata))
        self.assertEqual(len(review.collect_records(self.directory, [identifier])), 1)

        metadata["views"] = ["workstation"]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Incomplete render views"):
            review.collect_records(self.directory, [identifier])


if __name__ == "__main__":
    unittest.main()
