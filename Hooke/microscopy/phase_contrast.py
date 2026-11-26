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
    direct_amplitude: float = .55
    illumination_samples: int = 12

    @lru_cache(maxsize=8)
    def pupils(self, shape, pixel_size_m):
        fy, fx = np.meshgrid(np.fft.fftfreq(shape[0], pixel_size_m),
                             np.fft.fftfreq(shape[1], pixel_size_m), indexing="ij")
        pupils = []
        for angle in np.linspace(0, 2*np.pi, self.illumination_samples, endpoint=False):
            sx, sy = self.illumination_na/self.wavelength_m*np.array([np.cos(angle), np.sin(angle)])
            aperture = self.wavelength_m*np.hypot(fx+sx, fy+sy)
            # Finite phase-ring transitions suppress hard-edged pupil ringing.
            # Parameters describe an estimated objective, not a vendor replica.
            separation = abs(aperture-self.illumination_na)
            ring = .5*(1-np.tanh((separation-self.ring_halfwidth_na)/.008))
            plate = 1+ring*(-self.direct_amplitude*1j-1)
            edge = np.clip((self.numerical_aperture-aperture)/.025, 0., 1.)
            pupil = plate*(edge*edge*(3-2*edge))
            pupil.setflags(write=False)
            pupils.append(pupil)
        return tuple(pupils)

    def intensity(self, optical_path_m, absorption, pixel_size_m, *, defocus_m=0.,
                  medium_index=1.333):
        """Thin-object intensity with optional paraxial axial propagation.

        The zero-defocus path preserves the historical projection exactly.
        Refractive index and annular illumination are estimates, not a measured
        instrument transfer function or a thick-cell scattering model.
        """
        field = np.exp(-absorption+2j*np.pi*optical_path_m/self.wavelength_m)
        scattered = np.fft.fft2(field-1.)
        intensity = np.zeros(field.shape)
        pupils = self.pupils(field.shape, pixel_size_m)
        if defocus_m:
            fy, fx = np.meshgrid(np.fft.fftfreq(field.shape[0], pixel_size_m),
                                 np.fft.fftfreq(field.shape[1], pixel_size_m), indexing="ij")
            propagated = []
            for angle, pupil in zip(np.linspace(0, 2*np.pi, self.illumination_samples, endpoint=False), pupils):
                sx, sy = self.illumination_na/self.wavelength_m*np.array([np.cos(angle), np.sin(angle)])
                # Paraxial propagation relative to each illumination carrier.
                # A uniform specimen therefore retains unit intensity at any z.
                phase = -np.pi*self.wavelength_m*defocus_m/medium_index*(fx*fx+fy*fy+2*(fx*sx+fy*sy))
                propagated.append(pupil*np.exp(1j*phase))
            pupils = propagated
        for pupil in pupils:
            image_field = np.fft.ifft2(scattered*pupil)-self.direct_amplitude*1j
            intensity += abs(image_field)**2
        return intensity / (self.illumination_samples*self.direct_amplitude**2)
