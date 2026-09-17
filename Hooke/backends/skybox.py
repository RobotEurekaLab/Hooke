"""Convert the original OpenGL cube texture to a Y-up latlong panorama.

MuJoCo's GL cube direction is (world x, world z, -world y). The USD dome
therefore rotates +90 degrees about X in this Z-up scene. Original pixels
remain the source; this only changes the environment map representation.
"""

import numpy as np


def sample_cube(faces, directions):
    faces = np.asarray(faces)
    if faces.ndim != 4 or faces.shape[0] != 6 or faces.shape[1] != faces.shape[2]:
        raise ValueError("Expected six square cube faces")
    xyz = np.asarray(directions, dtype=float)
    if (
        xyz.shape[-1] != 3
        or not np.isfinite(xyz).all()
        or np.any(np.linalg.norm(xyz, axis=-1) == 0)
    ):
        raise ValueError("Expected finite nonzero directions")
    x, y, z = np.moveaxis(xyz, -1, 0)
    axis = np.argmax(abs(xyz), axis=-1)
    ma = np.max(abs(xyz), axis=-1)
    face = np.where(
        axis == 0,
        np.where(x > 0, 0, 1),
        np.where(axis == 1, np.where(y > 0, 2, 3), np.where(z > 0, 4, 5)),
    )
    sc = np.where(
        axis == 0,
        np.where(x > 0, -z, z),
        np.where(axis == 1, x, np.where(z > 0, x, -x)),
    )
    tc = np.where(axis == 0, -y, np.where(axis == 1, np.where(y > 0, z, -z), -y))
    width = faces.shape[1]
    u = np.clip(((sc / ma + 1) * 0.5 * width).astype(int), 0, width - 1)
    v = np.clip(((tc / ma + 1) * 0.5 * width).astype(int), 0, width - 1)
    return faces[face, v, u]


def cube_to_latlong(faces, width=1024, height=512):
    if width < 2 or height < 2:
        raise ValueError("Panorama dimensions are too small")
    longitude = ((np.arange(width) + 0.5) / width * 2 - 1) * np.pi
    latitude = (np.arange(height) + 0.5) / height * np.pi
    phi, theta = np.meshgrid(longitude, latitude)
    directions = np.stack(
        (np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)),
        axis=-1,
    )
    return sample_cube(faces, directions)


def srgb_to_linear(colors):
    colors = np.asarray(colors, dtype=float)
    return np.where(
        colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4
    )
