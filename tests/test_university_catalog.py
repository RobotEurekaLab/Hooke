"""Institution provenance stays complete without promoting leads to scenes."""

import contextlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from archetypes.task_catalog import CATALOG, CatalogEntry
from real_labs import catalog as scene_catalog
from university import catalog


BATCH = json.loads(
    (ROOT / "Hooke/university/batches/qs2027_100_labs_20261003.json").read_text()
)
NEW_SCENES = set(BATCH["scene_ids"])
EXPECTED = set(BATCH["baseline_scene_ids"]) | NEW_SCENES
EXPECTED_SCHOOLS = {
    row["institution_id"]
    for sid, row in catalog.affiliations().items()
    if sid in EXPECTED and row["institution"]["in_scope"]
}


class UniversityCatalogTests(unittest.TestCase):
    def test_every_existing_scene_keeps_its_binding_and_runnable_task(self):
        directory = catalog.catalogue()
        definitions = scene_catalog.scenes()
        labs = directory["labs"] + directory["other_institutions"]
        self.assertEqual(set(definitions), EXPECTED)
        self.assertEqual(len(labs), len(definitions))
        self.assertEqual({row["scene_id"] for row in labs}, set(definitions))
        for row in labs:
            with self.subTest(scene=row["scene_id"]):
                self.assertIn(row["task_name"], CATALOG)
                self.assertEqual(row["task_name"], "real_lab_" + row["scene_id"])
                url = urlsplit(row["url"])
                self.assertEqual(url.path, "/real-labs")
                self.assertEqual(parse_qs(url.query), {"scene": [row["scene_id"]]})
                self.assertEqual(
                    row["layout_fidelity"],
                    definitions[row["scene_id"]]["layout_fidelity"],
                )

    def test_nasa_is_preserved_outside_the_university_picker(self):
        directory = catalog.catalogue()
        self.assertEqual(len(directory["labs"]), len(EXPECTED) - 1)
        self.assertEqual(
            [row["scene_id"] for row in directory["other_institutions"]],
            ["nasa_planetary"],
        )
        universities = {row["id"] for row in directory["universities"]}
        self.assertNotIn("nasa_glenn", universities)
        self.assertTrue(
            all(row["institution_id"] in universities for row in directory["labs"])
        )
        nasa = directory["other_institutions"][0]["institution"]
        self.assertEqual(nasa["kind"], "research_institute")
        self.assertFalse(nasa["in_scope"])

    def test_qs_2027_scope_includes_ties_and_all_102_published_institutions(self):
        directory = catalog.catalogue()
        scoped = [row for row in directory["universities"] if row["in_scope"]]
        self.assertEqual(directory["ranking"]["edition"], 2027)
        self.assertEqual(directory["ranking"]["institution_count"], 102)
        self.assertEqual(len(scoped), 102)
        self.assertTrue(
            all(type(row["rank"]) is int and 1 <= row["rank"] <= 100 for row in scoped)
        )
        self.assertEqual(
            {row["id"] for row in scoped if row["rank"] == 100},
            {"purdue", "university_college_dublin"},
        )
        self.assertEqual(directory["coverage"]["scope_institutions"], len(scoped))

    def test_existing_labs_outside_research_scope_are_retained(self):
        directory = catalog.catalogue()
        in_scope = [row for row in directory["labs"] if row["institution"]["in_scope"]]
        outside = [
            row for row in directory["labs"] if not row["institution"]["in_scope"]
        ]
        self.assertEqual(len(in_scope), 15 + len(NEW_SCENES))
        self.assertEqual(
            {row["institution_id"] for row in outside},
            {"waterloo", "rochester", "virginia", "indiana_bloomington"},
        )
        self.assertEqual(
            directory["coverage"]["scope_institutions_with_scenes"],
            len(EXPECTED_SCHOOLS),
        )
        self.assertEqual(directory["coverage"]["university_scenes"], len(EXPECTED) - 1)
        self.assertEqual(directory["coverage"]["other_institution_scenes"], 1)

    def test_missing_and_stale_scene_classifications_are_rejected(self):
        original = catalog.affiliations()
        missing = deepcopy(original)
        missing.pop("uw_isolab")
        stale = deepcopy(original)
        stale["removed_scene"] = stale.pop("uw_isolab")
        for bindings, expected in (
            (missing, "missing=.*uw_isolab"),
            (stale, "stale=.*removed_scene"),
        ):
            with self.subTest(expected=expected), patch.object(
                catalog, "affiliations", return_value=bindings
            ):
                with self.assertRaisesRegex(ValueError, expected):
                    catalog.catalogue()

    def test_scene_bindings_cannot_reference_unknown_institutions(self):
        read = catalog._read

        def altered(name):
            data = read(name)
            if name == "scene_index.json":
                data["uw_isolab"]["institution_id"] = "unknown_university"
            return data

        with patch.object(catalog, "_read", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "Unknown institution.*uw_isolab"):
                catalog.affiliations()

    def test_one_laboratory_can_have_multiple_distinct_scene_bindings(self):
        definitions = scene_catalog.scenes()
        definitions["uw_isolab_second_room"] = dict(
            definitions["uw_isolab"],
            id="uw_isolab_second_room",
            title="IsoLab second room",
        )
        read = catalog._read

        def altered(name):
            data = read(name)
            if name == "scene_index.json":
                data["uw_isolab_second_room"] = dict(data["uw_isolab"])
            return data

        with patch.object(catalog, "_read", side_effect=altered), patch.object(
            scene_catalog, "scenes", return_value=definitions
        ):
            directory = catalog.catalogue()
        rooms = [
            row
            for row in directory["labs"]
            if row["institution_id"] == "washington" and row["lab_id"] == "isolab"
        ]
        self.assertEqual(
            {row["scene_id"] for row in rooms}, {"uw_isolab", "uw_isolab_second_room"}
        )
        self.assertEqual({row["lab_id"] for row in rooms}, {"isolab"})
        self.assertEqual(len({row["url"] for row in rooms}), 2)
        self.assertEqual(
            next(
                row["scene_count"]
                for row in directory["universities"]
                if row["id"] == "washington"
            ),
            1
            + sum(
                row["institution_id"] == "washington"
                for row in catalog.affiliations().values()
            ),
        )
        self.assertEqual(
            directory["coverage"]["scope_institutions_with_scenes"],
            len(EXPECTED_SCHOOLS),
        )

    def test_shared_laboratory_identity_cannot_conflict_between_scenes(self):
        read = catalog._read
        for field, replacement in (
            ("subject_id", "robotics"),
            ("lab_name", "Different Lab"),
        ):

            def altered(name):
                data = read(name)
                if name == "scene_index.json":
                    data["uw_isolab_second_room"] = dict(
                        data["uw_isolab"], **{field: replacement}
                    )
                return data

            with self.subTest(field=field), patch.object(
                catalog, "_read", side_effect=altered
            ):
                with self.assertRaisesRegex(
                    ValueError, "Conflicting laboratory identity"
                ):
                    catalog.affiliations()

    def test_candidates_have_registered_institutions_and_do_not_create_scenes(self):
        candidates = catalog.research_candidates()
        institutions = catalog.institutions()
        self.assertEqual(len({row["id"] for row in candidates}), len(candidates))
        self.assertTrue(
            all(row["institution_id"] in institutions for row in candidates)
        )
        before = catalog.catalogue()
        lead = {"id": "washington_test_lead", "institution_id": "washington"}
        with patch.object(
            catalog, "research_candidates", return_value=candidates + [lead]
        ):
            after = catalog.catalogue()
        self.assertEqual(after["labs"], before["labs"])
        self.assertEqual(after["other_institutions"], before["other_institutions"])
        self.assertEqual(after["coverage"]["research_candidates"], len(candidates) + 1)
        old_count = next(
            row["candidate_count"]
            for row in before["universities"]
            if row["id"] == "washington"
        )
        new_count = next(
            row["candidate_count"]
            for row in after["universities"]
            if row["id"] == "washington"
        )
        self.assertEqual(new_count, old_count + 1)

    def test_unknown_and_duplicate_candidate_records_are_rejected(self):
        read = catalog._read
        valid = {"id": "washington_test_lead", "institution_id": "washington"}
        invalid_rows = [
            [dict(valid, institution_id="unknown_university")],
            [valid, valid],
        ]
        for rows in invalid_rows:

            def altered(name):
                return (
                    {"candidates": rows}
                    if name == "research_candidates.json"
                    else read(name)
                )

            with self.subTest(rows=rows), patch.object(
                catalog, "_read", side_effect=altered
            ):
                with self.assertRaisesRegex(
                    ValueError, "Duplicate candidate or unknown institution"
                ):
                    catalog.research_candidates()


class UniversityApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The generic robot module resolves an existing plugin relative to cwd.
        with contextlib.chdir(ROOT / "Hooke"):
            from webui import server
        cls.server = server

    def test_both_catalogue_endpoints_share_metadata_without_loading_simulation(self):
        from webui import real_labs_api

        with contextlib.ExitStack() as stack:
            blocked = []
            for target, name in (
                (CatalogEntry, "load_classes"),
                (self.server, "render_scene"),
                (self.server, "render_robot_preview"),
                (self.server, "compose_scene"),
                (real_labs_api, "_build_session"),
            ):
                blocked.append(
                    stack.enter_context(
                        patch.object(target, name, side_effect=AssertionError(name))
                    )
                )
            client = self.server.app.test_client()
            main = client.get("/api/catalog")
            universities = client.get("/api/university")
        self.assertEqual(main.status_code, 200)
        self.assertEqual(universities.status_code, 200)
        self.assertEqual(main.get_json()["university"], universities.get_json())
        self.assertTrue(all(not function.called for function in blocked))
        tasks = {row["name"] for row in main.get_json()["tasks"]}
        self.assertTrue(
            all(row["task_name"] in tasks for row in universities.get_json()["labs"])
        )
        self.assertIn("real_lab_nasa_planetary", tasks)


if __name__ == "__main__":
    unittest.main()
