"""Rendering defaults preserve material roles and keep lights outside fixtures."""

import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.presentation import (
    fixture_emitter_bounds,
    inspection_softbox_visibility,
    primitive_face_smoothing,
    surface_profile,
)


class PresentationDefaultsTests(unittest.TestCase):
    def test_synthetic_light_visibility_requires_explicit_consistent_opt_in(self):
        self.assertIs(inspection_softbox_visibility({}), True)
        self.assertIs(inspection_softbox_visibility({"equipment": [{}]}), True)
        self.assertIs(inspection_softbox_visibility({"equipment": [
            {"inspection_softbox_visible_in_glass": False}, {}
        ]}), False)
        for values in (("false",), (0,), (True, False)):
            with self.assertRaises(ValueError):
                inspection_softbox_visibility({"equipment": [
                    {"inspection_softbox_visible_in_glass": value} for value in values
                ]})

    def test_photo_material_roles_do_not_reinterpret_legacy_glass(self):
        acrylic = surface_profile("fixture__mat_clear_acrylic", "pane", (1, 1, 1))
        self.assertEqual(acrylic.transmission, 1)
        self.assertEqual(acrylic.ior, 1.49)
        self.assertEqual(acrylic.metallic, 0)
        water = surface_profile("tank__mat_turbid_water", "water", (.1, .2, .1))
        self.assertLess(water.transmission, .1)
        self.assertGreater(water.color[1], water.color[0])
        self.assertIsNone(surface_profile("tank__mat_glass", "pane", (1, 1, 1)))

    def test_clear_water_requires_an_explicit_role_and_unscaled_reflection(self):
        profile = surface_profile("tank__mat_clear_water.001", "tank_volume", (1, 1, 1))
        self.assertEqual(profile.ior, 1.333)
        self.assertEqual(profile.transmission, 1.0)
        self.assertEqual(profile.specular, 0.5)
        self.assertEqual(profile.color, (1, 1, 1, 1))
        self.assertEqual(profile.metallic, 0)
        for name in ("tank__mat_glass", "tank__mat_water", "runtime_surface"):
            self.assertIsNone(surface_profile(name, "tank_water", (1, 1, 1)))
        metal = surface_profile("fixture__mat_metal", "fixture", (0.5, 0.5, 0.5))
        self.assertIsNone(metal.ior)
        self.assertIsNone(metal.transmission)
        self.assertIsNone(metal.color)

    def test_explicit_metal_paint_and_elastomer_roles_stay_distinct(self):
        profiles = {
            role: surface_profile(
                "fixture__mat_" + role, "fixture_rail_1", (0.6, 0.6, 0.6)
            )
            for role in ("metal", "bright", "copper", "blue", "rubber")
        }
        self.assertEqual(profiles["bright"].metallic, 1)
        self.assertEqual(profiles["copper"].metallic, 1)
        self.assertEqual(profiles["blue"].metallic, 0)
        self.assertGreater(profiles["rubber"].roughness, profiles["bright"].roughness)
        self.assertNotEqual(profiles["metal"], profiles["bright"])

    def test_unknown_glass_and_sample_materials_are_not_reinterpreted(self):
        for role in ("glass", "sample", "plant", "custom_unknown"):
            self.assertIsNone(
                surface_profile(
                    "fixture__mat_" + role, "fixture_steel_1", (0.5, 0.5, 0.5)
                )
            )
        self.assertIsNone(
            surface_profile("runtime_surface", "painted_rail_1", (0.02, 0.15, 0.7))
        )
        self.assertIsNone(
            surface_profile("runtime_surface", "blue_steel_cover", (0.02, 0.15, 0.7))
        )

    def test_explicit_unpainted_architecture_hardware_gets_a_finish(self):
        self.assertEqual(
            surface_profile(
                "runtime_surface", "bench_steel_top", (0.6, 0.65, 0.7)
            ).metallic,
            1,
        )
        self.assertEqual(
            surface_profile(
                "runtime_surface", "bench_rubber_bellows", (0.02, 0.025, 0.03)
            ).name,
            "rubber",
        )

    def test_emitter_remains_below_thick_translated_fixture(self):
        corners = list(itertools.product((2, 4), (-3, -2), (2.5, 2.9)))
        position, footprint = fixture_emitter_bounds(corners)
        self.assertEqual(position[:2], (3, -2.5))
        self.assertAlmostEqual(position[2], 2.485)
        self.assertEqual(footprint, (1.84, 0.92))
        self.assertLess(position[2], min(point[2] for point in corners))

    def test_tilted_fixture_emitter_uses_lowest_world_corner(self):
        corners = [
            (x, y, 3 + 0.25 * x + z)
            for x, y, z in itertools.product((-1, 1), (-0.2, 0.2), (-0.03, 0.03))
        ]
        position, _ = fixture_emitter_bounds(corners)
        self.assertLess(position[2], min(point[2] for point in corners))
        for invalid in ([], [(0, 1)], [(0, 0, float("nan"))]):
            with self.assertRaises(ValueError):
                fixture_emitter_bounds(invalid)

    def test_analytic_normals_preserve_caps_and_authored_mesh_normals(self):
        self.assertTrue(primitive_face_smoothing(5, 4))
        self.assertFalse(primitive_face_smoothing(5, 48))
        self.assertTrue(primitive_face_smoothing(2, 3))
        self.assertTrue(primitive_face_smoothing(3, 4))
        self.assertTrue(primitive_face_smoothing(4, 4))
        self.assertIsNone(primitive_face_smoothing(6, 4))
        self.assertIsNone(primitive_face_smoothing(7, 3))


if __name__ == "__main__":
    unittest.main()
