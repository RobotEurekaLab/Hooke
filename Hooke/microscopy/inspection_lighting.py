"""Illustrative room fill for viewing the underside of an inverted microscope."""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import numbers


def add_inspection_fill(world, origin):
    """Light occluded mechanical parts without changing microscope image formation."""
    power = .12
    position = np.asarray(origin) + [-.17, -.23, -.135]
    ET.SubElement(world, "light", name="workstation_inspection_fill",
                  pos=numbers(position), dir=numbers([.145, .23, .095]),
                  directional="false", diffuse=numbers(np.array([.85, .92, 1.])*power),
                  specular=numbers(np.full(3, power*.2)), bulbradius=".02", castshadow="false")
    return dict(name="Illustrative room inspection fill", position_world_m=position.tolist(),
                source_diffuse_peak=power, photometrically_calibrated=False,
                scope="World-camera display only; separate from transmitted illumination and synthetic microscopy",
                native_mapping="Source point light maps to a native sphere; radiometric parity is unverified")
