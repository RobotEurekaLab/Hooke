"""Bounded reuse of unchanged synthetic object-plane Gaussian inputs.

Cache keys contain the complete input bytes and convolution parameters. No
coordinates, focal heights or intensities are rounded. Each microscope owns
its cache; this helper is intended for serial rendering, not shared threads.
Large float64 zero-boundary kernels use FFT convolution with floating-point
roundoff; other data types and boundary modes retain scipy's direct filter.
"""

from collections import OrderedDict

import numpy as np
from scipy.ndimage import gaussian_filter


def _blur(values, sigma, mode, cval, truncate):
    """Use the same sampled Gaussian kernel for large floating-point planes."""
    if (not values.size or values.dtype != np.dtype("float64") or mode != "constant" or cval != 0
            or not np.isfinite(sigma).all() or not np.isfinite(truncate)
            or truncate <= 0 or not any(s >= 16 for s in sigma)
            or not np.isfinite(values).all()):
        return gaussian_filter(values, sigma=sigma, mode=mode, cval=cval, truncate=truncate)
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import fftconvolve

    result = values
    for axis, width in enumerate(sigma):
        if width <= 1e-15:
            continue
        if width < 16:
            result = gaussian_filter1d(result, width, axis=axis, mode=mode,
                                       cval=cval, truncate=truncate)
            continue
        radius = int(truncate*width+.5)
        offsets = np.arange(-radius, radius+1, dtype=float)
        kernel = np.exp(-.5*(offsets/width)**2)
        kernel /= kernel.sum()
        shape = [1]*values.ndim
        shape[axis] = len(kernel)
        result = fftconvolve(result, kernel.reshape(shape), mode="same", axes=(axis,))
    return result


class PlaneBlurCache:
    def __init__(self, payload_budget_bytes=64*1024*1024, max_entries=16):
        if type(payload_budget_bytes) is not int or payload_budget_bytes < 0:
            raise ValueError("Optical cache payload budget must be a nonnegative integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Optical cache entry limit must be a positive integer")
        self.budget = payload_budget_bytes
        self.max_entries = max_entries
        self.payload_bytes = 0
        self.hits = self.misses = 0
        self._entries = OrderedDict()

    def __call__(self, values, *, sigma, mode="constant", cval=0., truncate=4.):
        values = np.asarray(values)
        sigma = tuple(np.broadcast_to(np.asarray(sigma, dtype=float), (values.ndim,)))
        key = None
        if self.budget and values.nbytes*2 <= self.budget:
            # Bytes equality guards hash collisions as well as mutable inputs.
            key = (values.shape, values.dtype.str, sigma, mode, cval, truncate, values.tobytes())
            if key in self._entries:
                self.hits += 1
                self._entries.move_to_end(key)
                return self._entries[key]
        self.misses += 1
        result = _blur(values, sigma, mode, cval, truncate)
        result.setflags(write=False)
        size = values.nbytes+result.nbytes
        if key is not None and size <= self.budget:
            while self._entries and (self.payload_bytes+size > self.budget
                                     or len(self._entries) >= self.max_entries):
                old_key, old_result = self._entries.popitem(last=False)
                self.payload_bytes -= len(old_key[-1])+old_result.nbytes
            self._entries[key] = result
            self.payload_bytes += size
        return result

    def statistics(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self._entries),
                    payload_bytes=self.payload_bytes, payload_budget_bytes=self.budget,
                    max_entries=self.max_entries)
