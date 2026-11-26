"""Intracellular contrast follows recorded 3D depth, not screen overlap."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from microscopy.capillary_optics import composite_capillary
from microscopy.cell_appearance import cell_occupancy
from microscopy.cell_optics import CellMicroscope
from microscopy.optical_cache import PlaneBlurCache
from microscopy.optics import OpticalCalibration


class CapillaryDepthTests(unittest.TestCase):
    def test_screen_overlap_above_or_below_cell_is_not_intracellular(self):
        centre = np.array([2., -3., 7.]) * 1e-6
        radii = np.array([18., 14., 4.]) * 1e-6
        offsets = np.array([[0., 0., 0.], [17., 0., 0.], [0., 0., 5.],
                            [0., 0., -5.], [20., 0., 0.], [0., 15., 0.]]) * 1e-6
        occupied = cell_occupancy(centre + offsets, centre, radii)
        self.assertTrue(np.all(occupied[:2] > .99))
        self.assertTrue(np.all(occupied[2:] < .01))
        shift = np.array([11., 8., -6.]) * 1e-6
        np.testing.assert_allclose(
            cell_occupancy(centre + offsets + shift, centre + shift, radii), occupied)

    def test_removed_dimple_volume_stays_outside_while_its_shoulders_remain(self):
        radii = np.full(3, 18e-6)
        points = np.array([[17.2, 0., 0.], [15.5, 0., 0.],
                           [17.2, 4., 0.], [17.2, -4., 0.]]) * 1e-6
        original = cell_occupancy(points, np.zeros(3), radii)
        deformed = cell_occupancy(points, np.zeros(3), radii,
                                  membrane_contacts=((18e-6, 0., 0., 1.8e-6),),
                                  contact_width_fraction=.14)
        self.assertTrue(np.all(original > .9))
        self.assertLess(deformed[0], .01)
        self.assertTrue(np.all(deformed[1:] > .9))

    def test_visibility_receives_actual_tilted_tool_height_not_focus_height(self):
        calibration = OpticalCalibration(field_width_m=64e-6, pixels=64,
                                         defocus_scale_m=1e-6)
        image = Image.new("RGB", (64, 64), (200, 200, 200))
        layer = Image.new("RGBA", image.size)
        layer.putpixel((38, 32), (50, 50, 50, 255))
        layer.putpixel((44, 32), (50, 50, 50, 255))
        seen = []

        def visibility(points):
            seen.append(np.array(points, copy=True))
            return np.ones(len(points))

        args = (image, layer, np.array([0., 0., 3e-6]),
                np.array([.6, 0., .8]), 7e-6, calibration, PlaneBlurCache())
        original = composite_capillary(*args)
        transparent = composite_capillary(*args, visibility=visibility)
        points = np.concatenate(seen)
        points = points[np.argsort(points[:, 0])]
        np.testing.assert_allclose(points, np.array([[6., 0., 11.], [12., 0., 19.]]) * 1e-6,
                                   rtol=0, atol=1e-18)
        np.testing.assert_array_equal(original, transparent)

    def state(self):
        return dict(centres_m=[[0., 0., 0.]], cell_radii_m=[18e-6] * 3,
                    nuclear_radii_m=[6e-6] * 3, optical_reference_height_m=0.,
                    focus_m=0., tip_m=[14e-6, 0., 0.], needle_back=[1., 0., 0.],
                    delivered_pl=0., punctured=True, withdrawn=False, time_s=0.)

    @staticmethod
    def crop(scope, image, x0, x1, halfheight=2.):
        left, top = scope.calibration.pixel(np.array([x0, halfheight]) * 1e-6)
        right, bottom = scope.calibration.pixel(np.array([x1, -halfheight]) * 1e-6)
        return np.asarray(image, dtype=float)[round(top):round(bottom), round(left):round(right)]

    def test_render_retains_cell_texture_inside_and_exterior_contrast_outside(self):
        scope, state = CellMicroscope(), self.state()
        before = deepcopy(state)
        overlay = scope.render(dict(state, pipette_depth_mode="overlay"), annotate=False)
        volume = scope.render(dict(state, pipette_depth_mode="cell_volume"), annotate=False)
        cell = scope.render(dict(state, tip_m=[100e-6, 0., 0.]), annotate=False)
        background = self.crop(scope, cell, 14.5, 16.5, 1.)
        old_contrast = np.abs(self.crop(scope, overlay, 14.5, 16.5, 1.) - background).mean()
        new_contrast = np.abs(self.crop(scope, volume, 14.5, 16.5, 1.) - background).mean()
        self.assertGreater(old_contrast, 3.)
        self.assertGreater(new_contrast, .15 * old_contrast)
        self.assertLess(new_contrast, .8 * old_contrast)
        np.testing.assert_array_equal(self.crop(scope, overlay, 24., 42.),
                                      self.crop(scope, volume, 24., 42.))
        self.assertEqual(state, before)

    def test_prepuncture_and_geometrically_exterior_tools_keep_original_image(self):
        scope, state = CellMicroscope(), self.state()
        cases = (
            dict(state, punctured=False),
            dict(state, tip_m=[14e-6, 0., 25e-6]),
            dict(state, tip_m=[14e-6, 0., -25e-6]),
            dict(state, tip_m=[17.2e-6, 0., 0.], membrane_contacts=[
                dict(offset_m=[18e-6, 0., 0.], indentation_m=3e-6)]),
        )
        for index, current in enumerate(cases):
            with self.subTest(case=index):
                old = scope.render(dict(current, pipette_depth_mode="overlay"), annotate=False)
                new = scope.render(dict(current, pipette_depth_mode="cell_volume"), annotate=False)
                np.testing.assert_array_equal(old, new)


if __name__ == "__main__":
    unittest.main()
