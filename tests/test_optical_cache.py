"""Cached convolution remains exact, observes edits and bounds retained data."""

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Hooke'))
from microscopy.optical_cache import PlaneBlurCache
from microscopy.tasks import make_task


class OpticalCache(unittest.TestCase):
    def test_large_defocus_matches_the_discrete_zero_boundary_operator(self):
        values = np.random.default_rng(18).normal(size=(71, 89, 3))*80
        for sigma in ((16., 32., 0.), (2.3, 128., 0.), (64., 1.5, 0.)):
            options = dict(sigma=sigma, mode="constant", cval=0., truncate=4.)
            expected = gaussian_filter(values, **options)
            actual = PlaneBlurCache(payload_budget_bytes=0)(values, **options)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)
            self.assertEqual(actual.shape, values.shape)
            self.assertEqual(actual.dtype, values.dtype)
        # Integer/float32 rounding and nonzero/reflect boundary semantics
        # retain scipy's direct implementation.
        for dtype, mode, boundary in ((np.float32, "constant", 0.),
                                      (np.float64, "constant", 3.),
                                      (np.float64, "reflect", 0.)):
            array = values.astype(dtype)
            options = dict(sigma=(32., 16., 0.), mode=mode, cval=boundary)
            np.testing.assert_array_equal(PlaneBlurCache()(array, **options),
                                          gaussian_filter(array, **options))
        for array in (np.empty((0, 3)), np.array([[np.nan, 1.], [2., 3.]])):
            np.testing.assert_array_equal(PlaneBlurCache()(array, sigma=16.),
                                          gaussian_filter(array, sigma=16., mode="constant"))

    def test_mutable_inputs_and_convolution_parameters_cannot_return_stale_pixels(self):
        values = np.random.default_rng(0).normal(size=(19, 23, 3))
        cache = PlaneBlurCache()
        options = dict(sigma=(2.3, 3.7, 0.), mode='constant', cval=0.)
        for sigma, boundary in (((2.3, 3.7, 0.), 0.), ((1.5, 2., 0.), 0.), ((1.5, 2., 0.), 3.)):
            options.update(sigma=sigma, cval=boundary)
            np.testing.assert_array_equal(cache(values, **options), gaussian_filter(values, **options))
            values[9, 10, 0] += 7
            np.testing.assert_array_equal(cache(values, **options), gaussian_filter(values, **options))
            self.assertFalse(cache(values, **options).flags.writeable)
        self.assertGreater(cache.hits, 0)

    def test_eviction_oversized_inputs_and_disabled_cache_bound_retained_payload(self):
        cache = PlaneBlurCache(payload_budget_bytes=1024, max_entries=2)
        for value in range(10):
            array = np.full((5, 5), value, dtype=float)
            np.testing.assert_array_equal(cache(array, sigma=1.), gaussian_filter(array, sigma=1., mode='constant'))
            self.assertLessEqual(cache.payload_bytes, 1024)
            self.assertLessEqual(cache.statistics()['entries'], 2)
        previous = cache.payload_bytes
        cache(np.zeros((100, 100)), sigma=1.)
        self.assertEqual(cache.payload_bytes, previous)
        disabled = PlaneBlurCache(payload_budget_bytes=0)
        disabled(np.zeros((0, 3)), sigma=1.)
        self.assertEqual(disabled.statistics()['entries'], 0)
        for budget, count in ((True, 2), (-1, 2), (1024, 0)):
            with self.assertRaises(ValueError):
                PlaneBlurCache(budget, count)

    def test_complete_cell_images_match_for_tool_dose_focus_and_height_changes(self):
        with patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS='reference', HOOKE_MICROSCOPY_OPTICS='estimated'):
            task = make_task('suction_injection')
        task.reset(0)
        state = task.mechanics.image_state(task.data)
        state['focus_m'] = 0.
        cache = PlaneBlurCache()
        variants = [deepcopy(state) for _ in range(4)]
        variants[1]['delivered_pl'] = .076
        variants[1]['tip_m'] = np.asarray(variants[1]['tip_m'])+[5e-6, 0, 0]
        variants[2]['focus_m'] = 4e-6
        variants[3]['centres_m'][0][2] += 1e-6
        for variant in variants:
            task.microscope._plane_blur = PlaneBlurCache(payload_budget_bytes=0)
            fresh = np.asarray(task.microscope.render(variant, annotate=False))
            task.microscope._plane_blur = cache
            cold = np.asarray(task.microscope.render(variant, annotate=False))
            warm = np.asarray(task.microscope.render(variant, annotate=False))
            np.testing.assert_array_equal(cold, fresh)
            np.testing.assert_array_equal(warm, fresh)
            self.assertLessEqual(cache.payload_bytes, cache.budget)
        self.assertGreater(cache.hits, 0)
        self.assertEqual(task.data.time, 0.)
        self.assertEqual(task.mechanics.ledger.state('cell').volume_m3, 0.)


if __name__ == '__main__':
    unittest.main()
