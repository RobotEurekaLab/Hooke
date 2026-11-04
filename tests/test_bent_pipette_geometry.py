"""Bent glass preserves physical arc length and separates shaft/contact axes."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.pipette_geometry import PipetteGeometry, for_workstation
from microscopy.pipettes import NEEDLE_LENGTH_M, pipette_back


class PipetteGeometryTests(unittest.TestCase):
    def test_bent_tip_is_horizontal_with_oem_shaft_angles(self):
        with patch.dict(os.environ, HOOKE_MICROSCOPY_MANIPULATOR="sutter-mp285-reference"):
            for tool, angle, sign in (("injector", 45., 1.), ("holder", 35., -1.)):
                geometry = for_workstation(tool, suspended=True)
                np.testing.assert_allclose(geometry.tip_back, [sign, 0., 0.])
                self.assertAlmostEqual(np.rad2deg(np.arcsin(geometry.shaft_back[2])), angle)
                distances = np.linspace(0., NEEDLE_LENGTH_M, 12001)
                points = geometry.centreline(distances)
                self.assertAlmostEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).sum(),
                                       NEEDLE_LENGTH_M, places=9)
                # A centred finite difference straddles the curvature jump at
                # either arc boundary. Bound its first-order error separately.
                derivative = np.gradient(points, distances, axis=0)
                np.testing.assert_allclose(derivative, geometry.tangents(distances), atol=2e-4)
                np.testing.assert_array_equal(points[0], [0., 0., 0.])
                self.assertLess(points[-1, 2], NEEDLE_LENGTH_M*geometry.shaft_back[2])

    def test_adherent_needle_stays_straight_and_profiles_do_not_leak(self):
        with patch.dict(os.environ, HOOKE_MICROSCOPY_MANIPULATOR="sutter-mp285-reference"):
            suspended = for_workstation("injector", suspended=True)
            adherent = for_workstation("injector")
        self.assertEqual(adherent.shaft_back, adherent.tip_back)
        self.assertEqual(adherent.bend_end_m, 0.)
        self.assertNotEqual(suspended.tip_back, adherent.tip_back)
        with patch.dict(os.environ, HOOKE_MICROSCOPY_MANIPULATOR="auto"):
            original = for_workstation("injector", suspended=True)
        np.testing.assert_array_equal(original.tip_back, pipette_back("injector"))

    def test_invalid_directions_and_degenerate_bends_are_rejected(self):
        for arguments in (((2., 0., 0.), (1., 0., 0.)),
                          ((0., 0., 1.), (1., 0., 0.)),
                          ((0., 0., 1.), (1., 0., 0.), .001, .001),
                          ((1., 0., 0.), (1., 0., 0.), .001, .002),
                          ((-1., 0., 0.), (1., 0., 0.), .001, .002)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PipetteGeometry(*arguments)

    def test_oblique_bend_has_continuous_ring_frames(self):
        from microscopy.cell_scene import hollow_pipette
        from microscopy.pipettes import INJECTION_PROFILE

        with patch.dict(os.environ, HOOKE_MICROSCOPY_MANIPULATOR="sutter-mp285-reference"):
            geometry = for_workstation("injector", suspended=True, azimuth_deg=45.)
        asset = ET.Element("asset")
        hollow_pipette(asset, geometry=geometry)
        vertices = np.fromstring(asset.find("mesh").get("vertex"), sep=" ").reshape(-1, 48, 3)
        distances = np.unique(np.r_[[section[0] for section in INJECTION_PROFILE.sections],
            np.linspace(geometry.bend_start_m, geometry.bend_end_m, 17)])[::-1]
        radial = vertices[:len(distances), 0]-geometry.centreline(distances)
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        turns = np.sum(radial[1:]*radial[:-1], axis=1)
        self.assertGreater(float(turns.min()), np.cos(np.deg2rad(4)))

    def test_unmounted_tool_combinations_fail_before_building(self):
        from microscopy.scene import build_xml

        with patch.dict(os.environ, HOOKE_MICROSCOPY_MANIPULATOR="sutter-mp285-reference"):
            for tools in (("probe", "gripper", "injector"), ("probe", "holder")):
                with self.subTest(tools=tools), self.assertRaisesRegex(ValueError, "opposed"):
                    build_xml(tools)


if __name__ == "__main__":
    unittest.main()
