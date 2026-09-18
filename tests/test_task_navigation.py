"""The homepage exposes every experiment and opens its actual controls."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from archetypes.task_catalog import CATALOG, CatalogEntry
from microscopy import OPERATIONS
from webui.task_navigation import task_navigation


class TaskNavigation(unittest.TestCase):
    def test_all_experiments_have_distinct_visible_groups_without_loading_engines(self):
        entries = [e for e in CATALOG.values() if e.category in ("microscopy", "space")]
        with patch.object(CatalogEntry, "load_classes", side_effect=AssertionError("Loaded engine")):
            groups = Counter(task_navigation(e)["category"] for e in entries)
        self.assertEqual(groups, {"microscopy": 5, "lunar": 7, "martian": 7, "orbital": 5})
        self.assertTrue(all(task_navigation(e)["url"] for e in entries))
        self.assertTrue(all(task_navigation(e)["task_label"] != e.name for e in entries))
        self.assertEqual({e.category for e in entries}, {"microscopy", "space"})

    def test_microscopy_links_select_each_registered_operation(self):
        operations = set()
        for entry in CATALOG.values():
            if entry.category != "microscopy":
                continue
            url = urlsplit(task_navigation(entry)["url"])
            self.assertEqual(url.path, "/microscopy")
            operation = parse_qs(url.query)["experiment"][0]
            self.assertEqual(entry.name, "microscopy_" + operation)
            operations.add(operation)
        self.assertEqual(operations, set(OPERATIONS))

    def test_surface_links_preserve_world_and_team_scenario(self):
        for world in ("lunar", "martian"):
            for operation in ("surface_sampling", "humanoid_rover"):
                with self.subTest(world=world, operation=operation):
                    url = urlsplit(task_navigation(CATALOG[f"space_{world}_{operation}"])["url"])
                    self.assertEqual(url.path, "/surface-missions")
                    parameters = parse_qs(url.query)
                    self.assertEqual(parameters["world"], [world])
                    self.assertEqual(parameters.get("scenario"), ["team"] if operation == "humanoid_rover" else None)

    def test_space_operations_and_display_scenes_open_exact_catalogue_task(self):
        for entry in CATALOG.values():
            if entry.category != "space" or entry.completion_rule == "surface_sampling":
                continue
            navigation = task_navigation(entry)
            url = urlsplit(navigation["url"])
            self.assertEqual(url.path, "/backends")
            self.assertEqual(parse_qs(url.query), {"task": [entry.name]})
            self.assertEqual("display only" in navigation["task_label"], entry.completion_rule is None)

    def test_existing_lab_tasks_keep_their_categories_and_preview_workflow(self):
        for entry in CATALOG.values():
            if entry.category in ("microscopy", "space"):
                continue
            navigation = task_navigation(entry)
            self.assertEqual(navigation["category"], entry.category)
            self.assertEqual(navigation["task_label"], entry.name)
            self.assertIsNone(navigation["url"])

    def test_unregistered_world_or_operation_keeps_safe_generic_presentation(self):
        base = CATALOG["space_orbital_sample_transfer"]
        for name in ("future", "space_venus_experiment"):
            navigation = task_navigation(replace(base, name=name))
            self.assertEqual(navigation["category"], "space")
            self.assertIsNone(navigation["url"])
        navigation = task_navigation(replace(base, name="microscopy_future", category="microscopy"))
        self.assertEqual(navigation["category"], "microscopy")
        self.assertIsNone(navigation["url"])


if __name__ == "__main__":
    unittest.main()
