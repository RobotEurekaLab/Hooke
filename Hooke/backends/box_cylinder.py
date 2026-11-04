"""Bounded cylinder unions built from native boxes, without convex cooking."""
import math

import numpy as np

RADIAL_BUDGET_M = 40e-6
MIN_BOXES = 3
MAX_BOXES = 256


def _radial_excess(radius, count):
    angle = math.pi / (2 * count)
    return radius * (2 * math.sin(angle / 2) ** 2 / math.cos(angle))


def box_count(radius):
    """Smallest bounded count whose ideal outward error is at most 40 micrometres.

    The cap prevents an opt-in precision profile from silently creating an
    excessive number of shapes for large architectural cylinders.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Cylinder radius must be finite and positive')
    if _radial_excess(radius, MAX_BOXES) > RADIAL_BUDGET_M:
        raise ValueError(f'Cylinder radius {radius:g} m needs more than {MAX_BOXES} '
                         f'boxes for the {RADIAL_BUDGET_M:g} m radial budget')
    # Stable half-angle form of acos(r / (r + tolerance)).
    angle = 2 * math.atan(math.sqrt(RADIAL_BUDGET_M / (2 * radius + RADIAL_BUDGET_M)))
    count = max(MIN_BOXES, math.ceil(math.pi / (2 * angle)))
    # Resolve floating-point rounding at exact count boundaries explicitly.
    while _radial_excess(radius, count) > RADIAL_BUDGET_M:
        count += 1
    while count > MIN_BOXES and _radial_excess(radius, count - 1) <= RADIAL_BUDGET_M:
        count -= 1
    return count


def cylinder_boxes(radius, half_length):
    """Return box half extents and axial rotations for a circumscribed union.

    N diameter-aligned boxes cover the source disk and form a regular 2N-sided
    circumscribed prism. The ideal outward error is r*(sec(pi/(2N))-1),
    with no inward gap. Native contact behavior needs separate qualification.
    """
    dimensions = np.asarray([radius, half_length], dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Cylinder radius and half-length must be finite positive values')
    count = box_count(radius)
    return (np.array([radius, radius * math.tan(math.pi / (2 * count)), half_length]),
            np.arange(count) * math.pi / count)


def radial_excess(radius):
    return _radial_excess(radius, box_count(radius))
