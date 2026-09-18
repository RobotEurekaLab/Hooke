"""Display realism must retain independent, observable tracking and dosing."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))

from microscopy.cell_optics import CellMicroscope
from microscopy.phase_contrast import PhaseContrastOptics


class CellImageFormation(unittest.TestCase):
    def setUp(self):
        self.scope = CellMicroscope()
        self.state = dict(centres_m=[[0., 0., 4e-6]], focus_m=0.,
                          optical_reference_height_m=4e-6, indentation_m=0.,
                          tip_m=[100e-6, 100e-6, 4e-6], delivered_pl=0.,
                          punctured=False, withdrawn=False, time_s=0.)

    def test_unstained_view_is_grayscale_and_dose_does_not_recolour_it(self):
        original = np.asarray(self.scope.render(self.state, annotate=False))
        dosed = dict(self.state, delivered_pl=.5)
        np.testing.assert_array_equal(original, self.scope.render(dosed, annotate=False))
        np.testing.assert_array_equal(original[..., 0], original[..., 1])
        np.testing.assert_array_equal(original[..., 1], original[..., 2])
        fluorescence = np.asarray(self.scope.render(self.state, annotate=False, channel="fluorescence"))
        tracer = np.asarray(self.scope.render(dosed, annotate=False, channel="fluorescence"))
        self.assertGreater(tracer[..., 1].sum(), fluorescence[..., 1].sum())

    def test_tracking_uses_the_supplied_fluorescence_pixels_after_translation(self):
        for centre in ([0., 0., 4e-6], [17e-6, -11e-6, 4e-6]):
            with self.subTest(centre=centre):
                state = deepcopy(self.state)
                state["centres_m"] = [centre]
                image = self.scope.render(state, annotate=False, channel="fluorescence")
                np.testing.assert_allclose(self.scope.locate(image, "cell"), centre[:2], atol=.15e-6)

    def test_empty_optical_path_retains_normalised_flat_field(self):
        plane = np.zeros((64, 64))
        intensity = PhaseContrastOptics().intensity(plane, plane, .2e-6)
        np.testing.assert_allclose(intensity, 1., atol=1e-12)

    def test_channel_and_calibration_boundaries_are_explicit(self):
        with self.assertRaises(ValueError):
            self.scope.render(self.state, channel="unknown")
        calibration = self.scope.public_calibration()
        self.assertFalse(calibration["physical_optics_calibrated"])
        self.assertFalse(calibration["biological_texture_calibrated"])
        self.assertEqual(calibration["observation_channel"], "synthetic_nuclear_fluorescence")


if __name__ == "__main__":
    unittest.main()
