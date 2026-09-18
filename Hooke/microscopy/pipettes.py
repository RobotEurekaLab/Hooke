"""Nominal SI pipette profiles shared by geometry, imaging and flow.

Pulled-glass profiles and mounting angles are independent design estimates,
not manufacturer CAD or calibrated biological parameters.
"""

from dataclasses import dataclass

import numpy as np

NEEDLE_LENGTH_M = .012
BASE_INNER_M = .40e-3
BASE_OUTER_M = .50e-3
TIP_INNER_M = .25e-6
TIP_OUTER_M = .60e-6
HOLDER_INNER_M = 4e-6
HOLDER_OUTER_M = 6e-6


def pipette_back(tool):
    """Unit vector from the mouth toward the holder, above the sample plane."""
    elevation, azimuth = {"injector": (35., 45.), "holder": (30., 180.)}[tool]
    elevation, azimuth = np.deg2rad([elevation, azimuth])
    return np.array([np.cos(elevation)*np.cos(azimuth),
                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])


@dataclass(frozen=True)
class PipetteProfile:
    # Each section is (distance behind the mouth, outer radius, inner radius).
    sections: tuple

    def __post_init__(self):
        points = np.asarray(self.sections, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all() or points[0, 0] != 0
                or np.any(np.diff(points[:, 0]) <= 0)
                or np.any(points[:, 2] <= 0) or np.any(points[:, 1] <= points[:, 2])):
            raise ValueError("Pipette sections need increasing distances and positive inner/outer radii")

    def radius(self, distance_m, *, inner=False):
        points = np.asarray(self.sections)
        return np.interp(distance_m, points[:, 0], points[:, 2 if inner else 1])

    def resistance(self, viscosity_pa_s):
        """Series Poiseuille resistance for piecewise linear circular bores."""
        points = np.asarray(self.sections)
        a, b = points[:-1, 2], points[1:, 2]
        integral = np.sum(np.diff(points[:, 0])*(a*a+a*b+b*b)/(3*a**3*b**3))
        return float(8*viscosity_pa_s/np.pi*integral)


INJECTION_PROFILE = PipetteProfile((
    (0., TIP_OUTER_M, TIP_INNER_M),
    (.1e-3, 1e-6, .6e-6),
    (.3e-3, 2e-6, 1.5e-6),
    (.8e-3, 6e-6, 4.5e-6),
    (3e-3, 70e-6, 54.6e-6),
    (8e-3, BASE_OUTER_M, BASE_INNER_M),
    (NEEDLE_LENGTH_M, BASE_OUTER_M, BASE_INNER_M),
))
HOLDING_PROFILE = PipetteProfile((
    (0., HOLDER_OUTER_M, HOLDER_INNER_M),
    (.15e-3, 6.5e-6, 4.5e-6),
    (.8e-3, 12e-6, 9e-6),
    (3e-3, 70e-6, 54.6e-6),
    (8e-3, BASE_OUTER_M, BASE_INNER_M),
    (NEEDLE_LENGTH_M, BASE_OUTER_M, BASE_INNER_M),
))
