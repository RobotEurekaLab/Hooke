"""Scalar thin-object image formation with an estimated annular phase plate.

Each illumination direction is coherent; their intensities are averaged.
This is an uncalibrated projection model, not a volumetric optical solver.
Axial sectioning is applied separately by the microscope's empirical model.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class PhaseContrastOptics:
    wavelength_m: float = 550e-9
    numerical_aperture: float = .6
    illumination_na: float = .35
    ring_halfwidth_na: float = .045
    direct_amplitude: float = .35
    illumination_samples: int = 8

    @lru_cache(maxsize=8)
    def pupils(self, shape, pixel_size_m):
        fy, fx = np.meshgrid(np.fft.fftfreq(shape[0], pixel_size_m),
                             np.fft.fftfreq(shape[1], pixel_size_m), indexing="ij")
        pupils = []
        for angle in np.linspace(0, 2*np.pi, self.illumination_samples, endpoint=False):
            sx, sy = self.illumination_na/self.wavelength_m*np.array([np.cos(angle), np.sin(angle)])
            aperture = self.wavelength_m*np.hypot(fx+sx, fy+sy)
            plate = np.where(abs(aperture-self.illumination_na) < self.ring_halfwidth_na,
                             -self.direct_amplitude*1j, 1.)
            pupil = np.where(aperture <= self.numerical_aperture, plate, 0.)
            pupil.setflags(write=False)
            pupils.append(pupil)
        return tuple(pupils)

    def intensity(self, optical_path_m, absorption, pixel_size_m):
        field = np.exp(-absorption+2j*np.pi*optical_path_m/self.wavelength_m)
        scattered = np.fft.fft2(field-1.)
        intensity = np.zeros(field.shape)
        for pupil in self.pupils(field.shape, pixel_size_m):
            image_field = np.fft.ifft2(scattered*pupil)-self.direct_amplitude*1j
            intensity += abs(image_field)**2
        return intensity / (self.illumination_samples*self.direct_amplitude**2)
