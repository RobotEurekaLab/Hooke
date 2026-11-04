"""Named contact activation is explicit, validated, and never a surface inflation."""

import math
import unittest

from backends.contact_offsets import resolve_contact_offset_overrides, validate_offset_pair
from backends.physics_options import merge_native_physics_options, validate_physics_options


class ContactOffsetTests(unittest.TestCase):
    def test_optional_map_is_normalized_without_mutating_caller(self):
        offsets = {"seat": .0005}
        result = validate_physics_options({"contact_offset_overrides_m": offsets})
        offsets["seat"] = .5
        self.assertEqual(result, {"contact_offset_overrides_m": {"seat": .0005}})
        self.assertEqual(validate_physics_options({}), {})

    def test_invalid_map_or_names_reject(self):
        for value in (None, [], "seat", {1: .001}, {"": .001}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_physics_options({"contact_offset_overrides_m": value})

    def test_nonfinite_nonpositive_or_nonnumeric_offsets_reject(self):
        for value in (math.nan, math.inf, -math.inf, 0, -.001, True, "0.001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_physics_options({"contact_offset_overrides_m": {"seat": value}})

    def test_only_named_active_geometry_is_selected(self):
        self.assertEqual(resolve_contact_offset_overrides(
            {"seat": .0005}, ["floor", "sample", "seat"], {0, 1, 2}, [0, 0, 0]), {2: .0005})
        self.assertEqual(resolve_contact_offset_overrides({}, ["floor"], {0}, [0]), {})

    def test_unknown_ambiguous_or_inactive_geometry_rejects(self):
        for names, colliders in ((["floor"], {0}), (["seat", "seat"], {0, 1}), (["seat"], set())):
            with self.subTest(names=names, colliders=colliders), self.assertRaises(ValueError):
                resolve_contact_offset_overrides({"seat": .0005}, names, colliders, [0] * len(names))

    def test_source_margin_cannot_be_reduced(self):
        self.assertEqual(resolve_contact_offset_overrides(
            {"seat": .0005}, ["seat"], {0}, [.001]), {0: .001})

    def test_rest_surface_remains_inside_contact_skin(self):
        self.assertEqual(validate_offset_pair(.0005, 0), (.0005, 0))
        for contact, rest in ((0, 0), (.0005, .0005), (.0005, .001), (math.inf, 0), (.0005, math.nan)):
            with self.subTest(contact=contact, rest=rest), self.assertRaises(ValueError):
                validate_offset_pair(contact, rest)

    def test_explicit_override_map_replaces_archived_map(self):
        self.assertEqual(merge_native_physics_options(
            {"native_physics_options": {"contact_offset_m": .00005,
                                        "contact_offset_overrides_m": {"seat": .0005}}},
            {"contact_offset_overrides_m": {}}),
            {"contact_offset_m": .00005, "contact_offset_overrides_m": {}})


if __name__ == "__main__":
    unittest.main()
