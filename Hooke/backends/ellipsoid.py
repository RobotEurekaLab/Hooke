"""Closed ellipsoid surface with source dimensions baked into its vertices."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _unit_surface():
    # Smooth display geometry; PhysX cooks a separate, bounded convex hull.
    longitudes, latitudes = 32, 16
    theta = np.arange(1, latitudes) * np.pi / latitudes
    phi = np.arange(longitudes) * 2 * np.pi / longitudes
    rings = np.stack(np.broadcast_arrays(
        np.sin(theta)[:, None] * np.cos(phi),
        np.sin(theta)[:, None] * np.sin(phi),
        np.cos(theta)[:, None]), axis=-1).reshape(-1, 3)
    points = np.vstack(([0., 0., 1.], rings, [0., 0., -1.]))
    faces = []
    for longitude in range(longitudes):
        following = (longitude + 1) % longitudes
        faces.append((0, 1 + longitude, 1 + following))
        for ring in range(latitudes - 2):
            a = 1 + ring * longitudes + longitude
            b = 1 + ring * longitudes + following
            faces.extend(((a, a + longitudes, b), (b, a + longitudes, b + longitudes)))
        last = 1 + (latitudes - 2) * longitudes
        faces.append((len(points) - 1, last + following, last + longitude))
    points.setflags(write=False)
    triangles = np.asarray(faces, dtype=np.int32)
    triangles.setflags(write=False)
    return points, triangles


def ellipsoid_surface(semiaxes):
    """Return metric vertices, outward triangles and analytic unit normals.

    Unlike a nonuniformly scaled sphere primitive, a convex mesh preserves all
    three collision dimensions. The cooked hull is still a tessellation, not
    an analytic ellipsoid, and must be checked for the target contact task.
    """
    axes = np.asarray(semiaxes, dtype=float)
    if axes.shape != (3,) or not np.isfinite(axes).all() or np.any(axes <= 0):
        raise ValueError("Ellipsoid semiaxes must be three finite positive values")
    unit, faces = _unit_surface()
    normals = unit / axes
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return (unit * axes).astype(np.float32), faces, normals.astype(np.float32)
