"""Opt-in adherent imaging preserves recorded coordinates and channel contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from microscopy.capillary_optics import aperture_blur_sigma
from microscopy.cell_optics import CellMicroscope
from microscopy.optics import OpticalCalibration
from microscopy.phase_contrast import PhaseContrastOptics


class AdherentOpticsTests(unittest.TestCase):
    @staticmethod
    def specimen():
        y, x = np.mgrid[-32:32, -32:32]
        return 45e-9*np.exp(-.5*((x/9.)**2+(y/6.)**2)), np.zeros((64, 64))

    @staticmethod
    def microscope():
        scope = CellMicroscope()
        scope.calibration = OpticalCalibration(field_width_m=80e-6, pixels=128,
                                                defocus_scale_m=.45e-6)
        return scope

    @staticmethod
    def state():
        return dict(centres_m=[[0., 0., 4e-6]], cell_radii_m=[18e-6, 14e-6, 4e-6],
                    nuclear_radii_m=[6e-6, 4e-6, 2e-6], optical_reference_height_m=4e-6,
                    focus_m=2.49e-6, tip_m=[2.49e-6, 0., 6.49e-6],
                    needle_back=[2**-.5, 0., 2**-.5], indentation_m=2e-6,
                    membrane_contacts=[dict(offset_m=[3.9e-6, 0., 3.9e-6], indentation_m=1.5e-6)],
                    delivered_pl=.2, punctured=True, withdrawn=False, time_s=6.5)

    def test_uniform_phase_remains_flat_and_empty_field_normalised_at_any_depth(self):
        optics = PhaseContrastOptics()
        absorption = np.zeros((32, 48))
        for phase_path in (0., 40e-9):
            plane = np.full_like(absorption, phase_path)
            focused = optics.intensity(plane, absorption, .2e-6)
            for height in (-4e-6, 2.5e-6, 8e-6):
                with self.subTest(phase_path=phase_path, height=height):
                    result = optics.intensity(plane, absorption, .2e-6, defocus_m=height)
                    np.testing.assert_allclose(result, result[0, 0], rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(result, focused)
                    if phase_path == 0.:
                        np.testing.assert_allclose(result, 1., rtol=0, atol=1e-12)

    def test_zero_defocus_is_identical_to_legacy_pupil_image_formation(self):
        optics = PhaseContrastOptics()
        path, absorption = self.specimen()
        scattered = np.fft.fft2(np.exp(-absorption+2j*np.pi*path/optics.wavelength_m)-1.)
        legacy = np.zeros(path.shape)
        for pupil in optics.pupils(path.shape, .2e-6):
            field = np.fft.ifft2(scattered*pupil)-optics.direct_amplitude*1j
            legacy += abs(field)**2
        legacy /= optics.illumination_samples*optics.direct_amplitude**2
        np.testing.assert_array_equal(optics.intensity(path, absorption, .2e-6), legacy)
        np.testing.assert_array_equal(optics.intensity(path, absorption, .2e-6, defocus_m=0.), legacy)

    def test_nonzero_height_changes_phase_image_without_nonfinite_intensity(self):
        optics = PhaseContrastOptics()
        path, absorption = self.specimen()
        before = path.copy()
        focused = optics.intensity(path, absorption, .2e-6)
        for height in (-2.5e-6, 2.5e-6):
            with self.subTest(height=height):
                defocused = optics.intensity(path, absorption, .2e-6, defocus_m=height)
                self.assertTrue(np.isfinite(defocused).all())
                self.assertGreaterEqual(defocused.min(), 0.)
                self.assertGreater(np.abs(defocused-focused).max(), .01)
        np.testing.assert_array_equal(path, before)

    def test_aperture_psf_matches_disk_variance_and_is_invariant_to_pixel_scale(self):
        pixel, depth, na, wavelength, index = .2e-6, 6e-6, .6, 550e-9, 1.333
        sigma = aperture_blur_sigma(depth, pixel, na, wavelength, index)
        diffraction = aperture_blur_sigma(0., pixel, na, wavelength, index)
        self.assertEqual(sigma, aperture_blur_sigma(-depth, pixel, na, wavelength, index))
        self.assertEqual(2*sigma, aperture_blur_sigma(depth, pixel/2, na, wavelength, index))
        # Uniform-area polar quadrature independently checks each-axis variance.
        radius = depth*np.tan(np.arcsin(na/index))
        radii = radius*np.sqrt((np.arange(128)+.5)/128)
        angles = np.arange(256)*2*np.pi/256
        disk_x = radii[:, None]*np.cos(angles)
        measured_variance = np.mean(disk_x**2)-np.mean(disk_x)**2
        inferred_variance = (sigma*pixel)**2-(diffraction*pixel)**2
        np.testing.assert_allclose(inferred_variance, measured_variance, rtol=1e-12, atol=0.)
        self.assertGreater(diffraction, 0.)

    def test_omitted_flags_keep_explicit_legacy_rendering(self):
        scope, state = self.microscope(), self.state()
        default = scope.render(state, annotate=False)
        legacy = scope.render(dict(state, cell_projection_mode="gaussian",
                                   capillary_blur_model="empirical"), annotate=False)
        np.testing.assert_array_equal(default, legacy)

    def test_opt_in_keeps_state_dose_and_fluorescence_contracts(self):
        scope, state = self.microscope(), self.state()
        state.update(cell_projection_mode="phase_defocus", capillary_blur_model="aperture")
        before = deepcopy(state)
        phase = np.asarray(scope.render(state, annotate=False))
        self.assertEqual(phase.shape, (128, 128, 3))
        self.assertTrue(np.isfinite(phase).all())
        self.assertGreater(phase[32:96, 32:96, 0].std(), 2.)
        np.testing.assert_array_equal(phase[..., 0], phase[..., 1])
        np.testing.assert_array_equal(phase[..., 1], phase[..., 2])
        np.testing.assert_array_equal(phase, scope.render(dict(state, delivered_pl=.5), annotate=False))
        legacy = dict(state, cell_projection_mode="gaussian", capillary_blur_model="empirical")
        np.testing.assert_array_equal(scope.render(state, annotate=False, channel="fluorescence"),
                                      scope.render(legacy, annotate=False, channel="fluorescence"))
        self.assertEqual(state, before)

    def test_phase_opt_in_changes_thin_specimen_but_not_thick_specimen_or_its_background(self):
        scope, state = self.microscope(), self.state()
        state["tip_m"] = [100e-6, 100e-6, 100e-6]
        old = np.asarray(scope.render(state, annotate=False))
        new = np.asarray(scope.render(dict(state, cell_projection_mode="phase_defocus"), annotate=False))
        self.assertGreater(np.abs(old.astype(float)-new).mean(), .1)
        thick = dict(state, centres_m=[[0., 0., 18e-6], [27e-6, 20e-6, 18e-6]],
                     cell_radii_m=[18e-6]*3, nuclear_radii_m=[6e-6]*3,
                     optical_reference_height_m=18e-6, membrane_contacts=[])
        np.testing.assert_array_equal(scope.render(thick, annotate=False),
                                      scope.render(dict(thick, cell_projection_mode="phase_defocus"), annotate=False))


if __name__ == "__main__":
    unittest.main()
