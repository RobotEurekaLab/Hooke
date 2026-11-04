"""Explicit glass centreline and terminal direction, in the tool's SI frame.

The bent profile follows the manufacturer's application layout. Bend length
and radius are independent estimates, not a manufacturer capillary drawing.
"""

from dataclasses import dataclass

import numpy as np

from microscopy.pipettes import NEEDLE_LENGTH_M, pipette_back


@dataclass(frozen=True)
class PipetteGeometry:
    shaft_back: tuple
    tip_back: tuple
    bend_start_m: float = 0.
    bend_end_m: float = 0.

    def __post_init__(self):
        for vector in (self.shaft_back, self.tip_back):
            if np.shape(vector) != (3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.):
                raise ValueError("Pipette directions must be finite unit vectors")
        if not 0 <= self.bend_start_m <= self.bend_end_m <= NEEDLE_LENGTH_M:
            raise ValueError("Pipette bend must fit inside the exposed glass")
        if not self.bend_end_m and not np.allclose(self.shaft_back, self.tip_back):
            raise ValueError("Different shaft and tip directions require a bend")
        if self.bend_end_m and (self.bend_end_m == self.bend_start_m
                               or abs(np.dot(self.shaft_back, self.tip_back)) > 1-1e-10):
            raise ValueError("A bend requires finite arc length and distinct non-opposed directions")

    def centreline(self, distances):
        """Arc-length parameterization with a circular bend in a vertical plane."""
        distance = np.asarray(distances, dtype=float)
        tip, shaft = np.asarray(self.tip_back), np.asarray(self.shaft_back)
        if not self.bend_end_m:
            return distance[..., None]*shaft
        angle = float(np.arccos(np.clip(tip@shaft, -1., 1.)))
        normal = (shaft-tip*np.cos(angle))/np.sin(angle)
        radius = (self.bend_end_m-self.bend_start_m)/angle
        turn = np.clip(distance-self.bend_start_m, 0., self.bend_end_m-self.bend_start_m)/radius
        return (np.minimum(distance, self.bend_start_m)[..., None]*tip
                + radius*(np.sin(turn)[..., None]*tip+(1-np.cos(turn))[..., None]*normal)
                + np.maximum(0., distance-self.bend_end_m)[..., None]*shaft)

    def tangents(self, distances):
        distance = np.asarray(distances, dtype=float)
        tip, shaft = np.asarray(self.tip_back), np.asarray(self.shaft_back)
        if not self.bend_end_m:
            return np.broadcast_to(shaft, distance.shape+(3,))
        angle = float(np.arccos(np.clip(tip@shaft, -1., 1.)))
        normal = (shaft-tip*np.cos(angle))/np.sin(angle)
        turn = np.clip((distance-self.bend_start_m)/(self.bend_end_m-self.bend_start_m), 0., 1.)*angle
        return np.cos(turn)[..., None]*tip+np.sin(turn)[..., None]*normal

    def evidence(self):
        return dict(shaft_back=list(self.shaft_back), tip_back=list(self.tip_back),
                    bend_start_m=self.bend_start_m, bend_end_m=self.bend_end_m,
                    capillary_base_m=self.centreline(NEEDLE_LENGTH_M).tolist(),
                    bend_dimensions_calibrated=False)

    @classmethod
    def from_evidence(cls, values):
        return cls(tuple(values["shaft_back"]), tuple(values["tip_back"]),
                   values["bend_start_m"], values["bend_end_m"])


def for_workstation(tool, *, suspended=False, azimuth_deg=None):
    from microscopy.manipulator_profile import requested_profile

    if requested_profile() != "sutter-mp285-reference":
        back = tuple(pipette_back(tool))
        return PipetteGeometry(back, back)
    elevation = np.deg2rad(35. if tool == "holder" else 45.)
    azimuth = np.deg2rad(azimuth_deg if azimuth_deg is not None else (180. if tool == "holder" else 0.))
    radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
    radial[np.abs(radial) < 1e-15] = 0.
    shaft = tuple(radial*np.cos(elevation)+[0., 0., np.sin(elevation)])
    if suspended:
        return PipetteGeometry(shaft, tuple(radial), .8e-3, 2e-3)
    return PipetteGeometry(shaft, shaft)


def installed_geometry(model, tool):
    from microscopy.cad_assets import asset_evidence

    for assembly in asset_evidence(model).get("assemblies", []):
        if assembly.get("tool") == tool and "pipette_geometry" in assembly:
            values = assembly["pipette_geometry"]
            return PipetteGeometry.from_evidence(values)
    back = tuple(pipette_back(tool))
    return PipetteGeometry(back, back)
