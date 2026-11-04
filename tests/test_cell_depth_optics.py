"""Axial motion changes each optical plane without changing cell mechanics."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))
from microscopy.tasks import make_task


class CellDepthOptics(unittest.TestCase):
    def setUp(self):
        self.task = make_task("suction_injection")
        self.task.reset(0)
        self.optics = self.task.microscope
        self.state = self.task.mechanics.image_state(self.task.data)
        self.state["focus_m"] = 0.
        self.state["centres_m"] = [[0., 0., self.state["optical_reference_height_m"]]]

    def image(self, state):
        return self.optics.render(state, annotate=False)

    def test_cell_axial_displacement_defocuses_and_focus_motion_restores_image(self):
        original = self.image(self.state)
        shifted = deepcopy(self.state)
        # A 36 um cell still intersects the focal slab after a 2 um shift.
        # Move the entire specimen beyond that slab to test defocus.
        shifted["centres_m"][0][2] += 40e-6
        blurred = self.image(shifted)
        self.assertGreater(self.optics.sharpness(original), 2*self.optics.sharpness(blurred))
        shifted["focus_m"] = 40e-6
        np.testing.assert_array_equal(self.image(shifted), original)
        self.assertEqual(self.task.data.time, 0.)
        self.assertEqual(self.task.mechanics.ledger.state("cell").volume_m3, 0.)

    def test_separate_cell_planes_have_different_focus_in_the_same_image(self):
        h = self.state["optical_reference_height_m"]
        self.state["centres_m"] = [[-25e-6, 0., h], [25e-6, 0., h+3e-6]]
        first = self.image(self.state)
        self.state["focus_m"] = 3e-6
        second = self.image(self.state)
        left, right = (204,324,324,444), (444,324,564,444)
        self.assertGreater(self.optics.sharpness(first.crop(left)), 1.5*self.optics.sharpness(second.crop(left)))
        self.assertGreater(self.optics.sharpness(second.crop(right)), 1.5*self.optics.sharpness(first.crop(right)))

    def test_tip_plane_has_its_own_defocus(self):
        self.state["centres_m"] = []
        self.state["tip_m"] = [-20e-6, -20e-6, self.state["optical_reference_height_m"]]
        x, y = self.optics.calibration.pixel(self.state["tip_m"][:2])
        tip_region = (int(x)-24, int(y)-24, int(x)+24, int(y)+24)
        original = self.image(self.state)
        self.state["tip_m"][2] += 6e-6
        blurred = self.image(self.state)
        # Measure the distal tip: the inclined proximal shaft occupies other
        # depths and should not become sharp merely because its tip is focused.
        self.assertGreater(self.optics.sharpness(original.crop(tip_region)),
                           2*self.optics.sharpness(blurred.crop(tip_region)))
        self.state["focus_m"] = 6e-6
        np.testing.assert_array_equal(self.image(self.state), original)

    def test_focal_slab_tracks_the_actual_cell_section(self):
        from microscopy.cell_appearance import cell_fields

        radii = np.array([18., 18., 18.])*1e-6
        nuclear = radii*.4
        # At z=9 um, the sphere has radius sqrt(18^2-9^2)=15.59 um.
        x = np.array([[0., 14., 17.]])*1e-6
        y = np.zeros_like(x)
        _, _, interior, nucleus = cell_fields(
            x, y, radii, nuclear, index=0, axial_slab=(9e-6, 2e-6))
        self.assertGreater(interior[0, 1], 0.)
        self.assertEqual(interior[0, 2], 0.)
        self.assertFalse(nucleus.any())
        calibration = self.optics.public_calibration()
        self.assertEqual(calibration["focal_slab_depth_m"], 2e-6)
        self.assertFalse(calibration["thick_specimen_model_calibrated"])

    def test_nominal_planes_are_declared_in_each_scene(self):
        self.assertAlmostEqual(self.state["optical_reference_height_m"], 26e-6)
        adherent = make_task("cell_injection")
        adherent.reset(0)
        self.assertAlmostEqual(adherent.mechanics.image_state(adherent.data)["optical_reference_height_m"], 4e-6)

    def test_status_annotations_do_not_cover_the_observed_needle(self):
        self.state["tip_m"] = [0., 0., self.state["optical_reference_height_m"]]
        self.state["punctured"] = True
        self.state["holding_sealed"] = True
        raw = self.image(self.state)
        annotated = self.optics.render(self.state, annotate=True)
        np.testing.assert_array_equal(np.asarray(raw)[64:-64],
                                      np.asarray(annotated)[64:-64])


if __name__ == "__main__":
    unittest.main()
