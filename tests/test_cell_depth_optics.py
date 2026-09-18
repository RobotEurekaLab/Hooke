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
        shifted["centres_m"][0][2] += 2e-6
        blurred = self.image(shifted)
        self.assertGreater(self.optics.sharpness(original), 2*self.optics.sharpness(blurred))
        shifted["focus_m"] = 2e-6
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
        original = self.image(self.state)
        self.state["tip_m"][2] += 6e-6
        blurred = self.image(self.state)
        self.assertGreater(self.optics.sharpness(original), 2*self.optics.sharpness(blurred))
        self.state["focus_m"] = 6e-6
        np.testing.assert_array_equal(self.image(self.state), original)

    def test_nominal_planes_are_declared_in_each_scene(self):
        self.assertAlmostEqual(self.state["optical_reference_height_m"], 26e-6)
        adherent = make_task("cell_injection")
        adherent.reset(0)
        self.assertAlmostEqual(adherent.mechanics.image_state(adherent.data)["optical_reference_height_m"], 4e-6)


if __name__ == "__main__":
    unittest.main()
