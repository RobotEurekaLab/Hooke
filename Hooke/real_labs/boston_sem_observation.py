"""Supported illumination for an approximate SEM chamber observation view.

Supra-family specifications document a CCD camera with infrared illumination.
The geometry, mounting positions and intensity here are authored estimates.
The exported emissive surfaces provide a white-light visibility proxy; they do
not model infrared spectral response, calibrated optics or electron imaging.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def add_chamber_observation(builder, base):
    """Add roof-supported hardware behind the existing chamber camera view."""
    prefix = builder.name + "_chamber_observation_"
    asset = builder.root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name=prefix + "white",
        type="2d",
        builtin="flat",
        width="4",
        height="4",
        rgb1="1 1 1",
    )
    ET.SubElement(
        asset,
        "material",
        name=prefix + "illuminator",
        texture=prefix + "white",
        rgba="1 1 1 1",
        emission="50",
        specular="0",
        shininess="0",
    )

    def geom(key, kind, pos, size, material, **extra):
        return ET.SubElement(
            base,
            "geom",
            name=prefix + key,
            type=kind,
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            material=material,
            mass="0",
            contype="1",
            conaffinity="1",
            solref=".02 1",
            **extra,
        )

    # Both housings meet the real roof; emitter faces meet their housings.
    for side in (-1, 1):
        geom(
            f"housing_{side}",
            "box",
            (side * 0.100, 0.105, 1.247),
            (0.013, 0.013, 0.003),
            builder.name + "__mat_dark",
        )
        geom(
            f"emitter_{side}",
            "box",
            (side * 0.100, 0.105, 1.2435),
            (0.010, 0.010, 0.0005),
            prefix + "illuminator",
        )

    # Keep the established camera position and view. Its physical housing sits
    # behind the optical origin and attaches to the chamber roof.
    camera = np.array([0, 0.168, 1.218])
    forward = np.array([0, -0.8881302344836876, -0.45959186959295756])
    quaternion = np.empty(4)
    mujoco.mju_quatZ2Vec(quaternion, forward)
    center = camera - forward * 0.009
    geom(
        "camera_housing",
        "cylinder",
        center,
        (0.008, 0.007),
        builder.name + "__mat_dark",
        quat=" ".join(map(str, quaternion)),
    )
    geom(
        "camera_mount",
        "box",
        (0, center[1], 1.241),
        (0.006, 0.006, 0.009),
        builder.name + "__mat_metal",
    )
    return {
        "revision": 1,
        "camera": "chamber_scope",
        "reference": "https://www.ntnu.edu/documents/140082/1269041159/EM-lab-ProdInfoFE-SEM.pdf/6fb1a853-fa92-4c7b-a653-62fff3fdadf3",
        "documented_feature": "CCD camera with infrared chamber illumination",
        "model": "White-light geometric visibility proxy for an IR chamber viewer",
        "authored_estimates": ["Fixture dimensions", "Mounting positions", "Intensity"],
        "emission_renderer_units": 50,
        "limitations": [
            "No measured optical power, spectral response, reflectance or sensor noise",
            "No electron-optical SEM image or scientific measurement",
            "Source-family feature; installed BU camera configuration is unverified",
        ],
    }
