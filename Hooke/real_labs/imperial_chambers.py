"""Estimated exteriors from Imperial's historical 2018 Bessemer lab photo.

The two unequal assemblies, bolted ports, inclined branches and parked service
lines are original geometry. Neither their dimensions nor hidden connections
are measured. All apparatus here is fixed; the separate dry access fixture owns
its existing articulated controls. No vacuum or deposition process is modeled.
"""

import math

import numpy as np


REFERENCE_PHOTO = (
    "https://blogs.imperial.ac.uk/photography/files/2018/02/"
    "751_180118_tft_lab_peter_petrov_002.jpg"
)


def _name(b, key):
    return f"{b.name}__imperial_{key}"


def _axial(b, parent, key, start, end, radius, material="bright", collision=True):
    return b.geom(
        parent,
        "cylinder",
        (radius,),
        material=material,
        fromto=tuple(start) + tuple(end),
        collision=collision,
        name=_name(b, key),
    )


def _flange(b, parent, key, center, direction, radius, viewport=False):
    """A flat capped flange and bolts normal to its actual port axis."""
    center = np.asarray(center, dtype=float)
    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    tangent = np.cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (0, 1, 0))
    tangent /= np.linalg.norm(tangent)
    other = np.cross(axis, tangent)
    _axial(b, parent, key, center - axis * 0.009, center + axis * 0.009, radius)
    count = 16 if radius > 0.15 else 12
    for i in range(count):
        angle = math.tau * i / count
        bolt = center + radius * 0.85 * (
            math.cos(angle) * tangent + math.sin(angle) * other
        )
        _axial(
            b,
            parent,
            key + f"_bolt_{i}",
            bolt + axis * 0.009,
            bolt + axis * 0.017,
            min(0.0065, radius * 0.075),
            "metal",
            collision=False,
        )
    if viewport:
        _axial(
            b,
            parent,
            key + "_window",
            center + axis * 0.009,
            center + axis * 0.011,
            radius * 0.66,
            "lens",
            collision=False,
        )


def _frame(b, parent, key, x, y, chamber_bottom):
    """Continuous floor/frame/table/saddle/chamber support chain."""
    for i, dx in enumerate((-0.46, 0.46)):
        for j, dy in enumerate((-0.37, 0.37)):
            b.cylinder(
                parent,
                (x + dx, y + dy, 0.025),
                0.045,
                0.025,
                "rubber",
                collision=True,
                name=_name(b, f"{key}_foot_{i}_{j}"),
            )
            b.box(
                parent,
                (x + dx, y + dy, 0.49),
                (0.021, 0.021, 0.44),
                "bright",
                collision=True,
                name=_name(b, f"{key}_leg_{i}_{j}"),
            )
    b.box(
        parent,
        (x, y, 0.16),
        (0.48, 0.39, 0.025),
        "metal",
        collision=True,
        name=_name(b, key + "_lower_frame"),
    )
    b.box(
        parent,
        (x, y, 0.548),
        (0.436, 0.344, 0.363),
        "cream",
        collision=True,
        name=_name(b, key + "_cabinet"),
    )
    b.box(
        parent,
        (x, y, 0.93),
        (0.49, 0.40, 0.020),
        "metal",
        collision=True,
        name=_name(b, key + "_table"),
    )
    # Thin cover seams and actual fasteners distinguish the cabinet/frame.
    for dx in (-0.405, 0.405):
        for z in (0.22, 0.85):
            _axial(
                b,
                parent,
                f"{key}_panel_bolt_{dx}_{z}",
                (x + dx, y - 0.344, z),
                (x + dx, y - 0.348, z),
                0.004,
                "bright",
                collision=False,
            )
    for i, (dx, dy) in enumerate(((-0.16, -0.13), (0.16, 0.13))):
        b.box(
            parent,
            (x + dx, y + dy, (0.95 + chamber_bottom) / 2),
            (0.037, 0.05, (chamber_bottom - 0.95) / 2),
            "metal",
            collision=True,
            name=_name(b, f"{key}_saddle_{i}"),
        )


def _branch(b, parent, key, start, end, radius):
    """Stepped inclined body, paired flanges and parallel external tie rods."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    axis /= length
    _axial(b, parent, key + "_body", start, end, radius)
    for i, t in enumerate((0.13, 0.48, 0.86, 1.0)):
        _flange(
            b,
            parent,
            f"{key}_flange_{i}",
            start + (end - start) * t,
            axis,
            radius * 1.32,
        )
    side = np.cross(axis, (0, 0, 1))
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    for i in range(4):
        angle = math.tau * i / 4
        offset = radius * 1.18 * (math.cos(angle) * side + math.sin(angle) * other)
        _axial(
            b,
            parent,
            f"{key}_tie_{i}",
            start + axis * length * 0.14 + offset,
            start + axis * length * 0.86 + offset,
            0.006,
        )
    # A narrow collar and termination plate close the representative assembly.
    _axial(b, parent, key + "_end", end, end + axis * 0.055, radius * 0.62, "metal")


def _lead(b, parent, key, points, material):
    """A smooth fixed cable with endpoint positions anchored to connectors."""
    a, c, d, e = [np.asarray(point, dtype=float) for point in points]
    previous = a
    for i in range(32):
        t = (i + 1) / 32
        current = (
            (1 - t) ** 3 * a
            + 3 * (1 - t) ** 2 * t * c
            + 3 * (1 - t) * t**2 * d
            + t**3 * e
        )
        b.rod(
            parent,
            tuple(previous),
            tuple(current),
            0.005,
            material,
            name=_name(b, f"{key}_route_{i}"),
        )
        previous = current


def _chamber(b, parent, key, x, y, radius, bottom, top, ports):
    _frame(b, parent, key, x, y, bottom)
    b.cylinder(
        parent,
        (x, y, (bottom + top) / 2),
        radius,
        (top - bottom) / 2,
        "metal",
        collision=True,
        name=_name(b, key + "_vessel"),
    )
    for suffix, z in (("base_flange", bottom + 0.009), ("lid", top)):
        _flange(b, parent, key + "_" + suffix, (x, y, z), (0, 0, 1), radius + 0.033)
    for i, (azimuth, z, reach, bore, viewport) in enumerate(ports):
        direction = np.array((math.cos(azimuth), math.sin(azimuth), 0))
        start = np.array((x, y, z)) + direction * (radius - 0.018)
        end = np.array((x, y, z)) + direction * reach
        _axial(b, parent, f"{key}_port_{i}", start, end, bore)
        _flange(b, parent, f"{key}_port_{i}_cap", end, direction, bore * 1.43, viewport)
    # The photographed large top feedthrough and smaller neighbouring fitting
    # have separate heights; their dimensions and internal purposes are unknown.
    for i, (dx, height, stem) in enumerate(
        ((-0.085, 0.46, 0.048), (0.14, 0.27, 0.026))
    ):
        p = (x + dx, y, top)
        end = (x + dx, y, top + height)
        _axial(b, parent, f"{key}_top_{i}", p, end, stem)
        _flange(
            b,
            parent,
            f"{key}_top_{i}_base",
            (p[0], p[1], top + 0.027),
            (0, 0, 1),
            stem * 1.55,
        )
        b.box(
            parent,
            (end[0], end[1], end[2] + 0.022),
            (stem * 1.1, stem * 0.90, 0.022),
            "metal",
            collision=True,
            name=_name(b, f"{key}_top_{i}_terminal"),
        )
        endpoint = (x + 0.482, y + 0.22 - i * 0.14, 0.40 + i * 0.10)
        _axial(
            b,
            parent,
            f"{key}_socket_{i}",
            (x + 0.434, endpoint[1], endpoint[2]),
            endpoint,
            0.013,
            "metal",
        )
        startpoint = (end[0] + stem * 1.1, end[1], end[2] + 0.020)
        _lead(
            b,
            parent,
            f"{key}_lead_{i}",
            (
                startpoint,
                (x + 0.85 + i * 0.055, y + 0.36 - i * 0.1, top + 0.63),
                (x + 0.88 + i * 0.055, y + 0.47 - i * 0.1, 0.14 + i * 0.1),
                endpoint,
            ),
            "blue" if i == 0 else "warning",
        )


def historical_chambers(b, parent):
    """Add two source-informed unequal apparatus exteriors at existing centres."""
    _chamber(
        b,
        parent,
        "left",
        -0.95,
        -0.32,
        0.29,
        1.25,
        1.88,
        (
            (-math.pi / 2, 1.68, 0.44, 0.118, False),
            (-math.pi / 6, 1.48, 0.54, 0.145, False),
            (math.pi, 1.66, 0.44, 0.072, True),
            (math.pi / 2, 1.60, 0.40, 0.055, False),
        ),
    )
    _chamber(
        b,
        parent,
        "right",
        0.94,
        0.38,
        0.255,
        1.36,
        1.81,
        (
            (-math.pi / 2, 1.58, 0.50, 0.068, True),
            (0, 1.61, 0.58, 0.083, False),
            (math.pi, 1.56, 0.55, 0.100, False),
            (math.pi / 3, 1.58, 0.46, 0.052, False),
        ),
    )
    for key, x, y in (("left", -0.95, -0.32), ("right", 0.94, 0.38)):
        # Angled assemblies clear the cabinet sides. They are not unsupported
        # cylinders ending in cables, and are not modeled as process actuators.
        if key == "left":
            branches = (
                ((0, -0.24, 1.39), (-0.08, -0.89, 0.45), 0.087),
                ((0.22, -0.20, 1.39), (0.83, -0.90, 0.48), 0.068),
            )
        else:
            branches = (
                ((0.18, -0.13, 1.43), (1.23, -0.48, 0.56), 0.10),
                ((-0.15, 0.16, 1.42), (-0.77, 0.74, 0.68), 0.060),
            )
        for i, (a, c, r) in enumerate(branches):
            _branch(
                b,
                parent,
                f"{key}_branch_{i}",
                (x + a[0], y + a[1], a[2]),
                (x + c[0], y + c[1], c[2]),
                r,
            )
