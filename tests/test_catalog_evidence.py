"""Named evidence packages preserve legacy inline source records."""

from copy import deepcopy
import unittest

from real_labs.catalog import _add_detail_evidence, scene


class CatalogEvidenceTests(unittest.TestCase):
    def test_legacy_list_and_new_package_are_preserved_without_input_mutation(self):
        legacy = [{"source_url": "https://example.edu/reference", "observed": "Frame"}]
        addition = {"observed_details": ["Supported secondary platform"]}
        expected = deepcopy((legacy, addition))
        definition = {"reference_detail_evidence": legacy}
        _add_detail_evidence(definition, "source_context_details", addition)
        self.assertEqual(definition["reference_detail_evidence"],
                         {"legacy": expected[0], "source_context_details": expected[1]})
        definition["reference_detail_evidence"]["legacy"][0]["observed"] = "Modified copy"
        definition["reference_detail_evidence"]["source_context_details"]["observed_details"].append("Copy")
        self.assertEqual((legacy, addition), expected)

    def test_texas_retains_original_and_new_reference_evidence(self):
        detail = scene("texas_ferguson_structural_test_hall")["reference_detail_evidence"]
        self.assertEqual(detail["legacy"][0]["source_url"],
                         "https://fsel.engr.utexas.edu/images/featured/facilities3.jpg")
        self.assertIn("Parked mobile work platforms", detail["source_context_details"]["observed_details"])


if __name__ == "__main__":
    unittest.main()
