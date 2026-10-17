"""Nonoverlapping triangular prisms for a bounded inscribed cylinder."""
import math

import numpy as np

RADIAL_BUDGET_M = 40e-6
MIN_SECTORS = 3
MAX_SECTORS = 512


def _radial_deficit(radius, count):
    return radius * 2 * math.sin(math.pi / (2 * count))**2


def sector_count(radius):
    """Smallest count meeting the ideal inscribed radial error budget."""
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Cylinder radius must be finite and positive')
    if _radial_deficit(radius, MAX_SECTORS) > RADIAL_BUDGET_M:
        raise ValueError(f'Cylinder radius {radius:g} m needs more than {MAX_SECTORS} '
                         f'sectors for the {RADIAL_BUDGET_M:g} m radial budget')
    angle = 2 * math.asin(math.sqrt(min(1., RADIAL_BUDGET_M / (2 * radius))))
    count = max(MIN_SECTORS, math.ceil(math.pi / angle))
    while _radial_deficit(radius, count) > RADIAL_BUDGET_M:
        count += 1
    while count > MIN_SECTORS and _radial_deficit(radius, count-1) <= RADIAL_BUDGET_M:
        count -= 1
    return count


def radial_deficit(radius):
    return _radial_deficit(radius, sector_count(radius))


def cylinder_sectors(radius, half_length):
    """Tile an inscribed prism with six irreducible vertices per convex child.

    Shared radial faces separate disjoint sector interiors. Vertices and end
    planes use the source radius and half-length directly. Actual native
    cooking still requires inspection before these meshes can be qualified.
    """
    dimensions = np.asarray([radius, half_length], dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Cylinder radius and half-length must be finite positive values')
    count = sector_count(radius)
    angles = np.arange(count+1) * (2 * math.pi / count)
    ring = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    ring[-1] = ring[0]  # Close the seam without a floating-point sliver.
    faces = np.asarray(((0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3),
                        (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)), dtype=np.int32)
    sectors = []
    for index in range(count):
        triangle = np.vstack((np.zeros(2), ring[index:index+2]))
        points = np.vstack((np.column_stack((triangle, np.full(3, -half_length))),
                            np.column_stack((triangle, np.full(3, half_length)))))
        sectors.append((points, faces))
    return tuple(sectors)


def uses_box_fallback(radius, half_length):
    """Thin source rails use bounded native boxes to avoid lost hull corners.

    The predicate is geometric: radius <= 2 mm and half-length/radius >= 16.
    NASA's two 1.55 mm-radius, 45.26 mm-half-length rails satisfy it; their
    six-vertex native hulls lost one corner. No source names are hard-coded.
    """
    dimensions = np.asarray([radius, half_length], dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Cylinder radius and half-length must be finite positive values')
    return bool(radius <= .002 and half_length / radius >= 16.)
