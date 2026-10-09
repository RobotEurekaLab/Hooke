"""Original exterior detail for instruments with separately defined mechanisms.

All details here are visual, noncolliding geometry. Dimensions without an
explicit source are estimates. Keep mechanical joints, masses, target sites,
and simplified contact geometry in :mod:`real_labs.equipment`.
"""

from __future__ import annotations

import math
from typing import Any
from xml.etree import ElementTree as ET


_X_AXIS = (0.70710678, 0, 0.70710678, 0)
_Y_AXIS = (0.70710678, 0.70710678, 0, 0)


def _screw(b: Any, parent: ET.Element, position, radius=0.0022) -> None:
    """Front-facing captive screw, including its dark socket."""
    x, y, z = position
    b.cylinder(parent, position, radius, 0.0007, "bright", quat=_Y_AXIS)
    b.cylinder(parent, (x, y - 0.0009, z), radius * 0.43, 0.0002, "dark", quat=_Y_AXIS)


def _tip(b: Any, parent: ET.Element, x: float) -> None:
    """Estimated 44 mm taper, retaining the mechanism's existing endpoint."""
    vertices = []
    faces = []
    sides = 12
    for z, radius in ((-0.105, 0.003), (-0.119, 0.0024), (-0.149, 0.00035)):
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vertices.extend(
                (x + radius * math.cos(angle), -0.006 + radius * math.sin(angle), z)
            )
    for row in range(2):
        for i in range(sides):
            a, c = row * sides + i, row * sides + (i + 1) % sides
            faces.extend((a, c, c + sides, a, c + sides, a + sides))
    # Open lower end is deliberate: a disposable tip is a hollow tube.
    mesh = b.unique("pipette_tip_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=" ".join(f"{v:.8g}" for v in vertices),
        face=" ".join(str(v) for v in faces),
    )
    b.geom(parent, "mesh", (), material="cream", mesh=mesh)


def _tray_corner(b: Any, parent: ET.Element, side: int) -> None:
    """One closed quarter-ring wall, without overlapping coplanar caps."""
    vertices, faces = [], []
    segments = 16
    for i in range(segments + 1):
        angle = math.pi * i / (2 * segments)
        for radius, z in (
            (0.017, 0.043),
            (0.031, 0.043),
            (0.017, 0.071),
            (0.031, 0.071),
        ):
            vertices.extend(
                (
                    side * (0.146 + radius * math.sin(angle)),
                    -0.058 - radius * math.cos(angle),
                    z,
                )
            )
    for i in range(segments):
        a, n = i * 4, (i + 1) * 4
        faces.extend(
            (
                (a, n, n + 1),
                (a, n + 1, a + 1),
                (a + 2, a + 3, n + 3),
                (a + 2, n + 3, n + 2),
                (a + 1, n + 1, n + 3),
                (a + 1, n + 3, a + 3),
                (a, a + 2, n + 2),
                (a, n + 2, n),
            )
        )
    end = segments * 4
    faces.extend(
        ((0, 1, 3), (0, 3, 2), (end, end + 2, end + 3), (end, end + 3, end + 1))
    )
    if side < 0:
        faces = [(a, c, d) for a, d, c in faces]
    mesh = b.unique("tray_corner_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=" ".join(f"{v:.8g}" for v in vertices),
        face=" ".join(str(v) for face in faces for v in face),
    )
    b.geom(parent, "mesh", (), material="shell", mesh=mesh)


def liquid_handler_details(b: Any, base, gantry, carriage, pipette, door) -> None:
    """Flex-informed frame/mount details; this remains a generic surrogate."""
    # The dark face, status strip and lower side covers define the enclosure,
    # instead of leaving a white furniture-like box around an exposed rod.
    b.box(base, (0, -0.341, 0.785), (0.417, 0.009, 0.046), "dark")
    b.box(base, (-0.03, -0.351, 0.784), (0.238, 0.001, 0.003), "cyan")
    b.box(base, (0, -0.341, 0.097), (0.416, 0.008, 0.046), "dark")
    b.box(base, (-0.337, -0.354, 0.796), (0.008, 0.003, 0.008), "metal")
    b.cylinder(base, (-0.337, -0.358, 0.796), 0.005, 0.001, "lens", quat=_Y_AXIS)
    # Original estimated mounting bracket. The post stays outside the moving
    # door's right edge; the arm and screen sit ahead of its front surface.
    b.box(
        base,
        (0.416, -0.336, 0.493),
        (0.014, 0.015, 0.035),
        "metal",
        name=f"{b.name}__display_mount_base",
    )
    b.box(
        base,
        (0.412, -0.359, 0.493),
        (0.008, 0.025, 0.01),
        "dark",
        name=f"{b.name}__display_mount_neck",
    )
    b.cylinder(
        base,
        (0.412, -0.38, 0.493),
        0.012,
        0.022,
        "metal",
        name=f"{b.name}__display_mount_pivot",
    )
    b.box(
        base,
        (0.375, -0.38, 0.493),
        (0.039, 0.008, 0.012),
        "dark",
        name=f"{b.name}__display_mount_arm",
    )
    for x in (-0.414, 0.414):
        b.box(base, (x, 0, 0.742), (0.018, 0.316, 0.02), "shell")
        b.box(base, (x, 0, 0.174), (0.019, 0.314, 0.018), "dark")
        b.box(base, (x * 0.925, 0, 0.736), (0.006, 0.27, 0.003), "cream")
        for z in (0.22, 0.68):
            _screw(b, base, (x, -0.343, z))
        side = 1 if x > 0 else -1
        # Side handling recesses and capped fasteners are below the deck.
        for y in (-0.22, 0.22):
            b.box(base, (side * 0.436, y, 0.086), (0.001, 0.05, 0.019), "dark")
            b.cylinder(
                base,
                (side * 0.437, y + 0.066, 0.084),
                0.009,
                0.001,
                "metal",
                quat=_X_AXIS,
            )
        for y in (-0.024, -0.012, 0, 0.012, 0.024):
            b.box(base, (side * 0.436, y, 0.085), (0.001, 0.002, 0.018), "dark")
        # The linear bearing rides on a substantial rail section.
        b.box(base, (side * 0.365, 0, 0.647), (0.014, 0.263, 0.018), "metal")
    for x in (-0.365, 0.365):
        b.box(gantry, (x, 0, -0.012), (0.019, 0.05, 0.047), "dark")
        b.box(gantry, (x, -0.053, -0.01), (0.015, 0.004, 0.035), "metal")
        for z in (-0.032, 0.012):
            _screw(b, gantry, (x, -0.058, z))
    b.box(gantry, (0, -0.038, 0.01), (0.3, 0.003, 0.021), "shell")
    b.box(gantry, (0, -0.039, -0.022), (0.31, 0.002, 0.003), "dark")
    for x in (-0.029, 0.029):
        b.box(carriage, (x, -0.049, -0.034), (0.004, 0.003, 0.071), "metal")
        _screw(b, carriage, (x, -0.053, 0.034))
    b.box(pipette, (0, -0.035, -0.037), (0.03, 0.002, 0.052), "dark")
    b.box(pipette, (0, -0.038, -0.019), (0.022, 0.0007, 0.01), "metal")
    b.box(pipette, (0, -0.039, -0.018), (0.013, 0.0004, 0.0013), "cyan")
    for x in (-0.024, 0.024):
        _screw(b, pipette, (x, -0.039, 0.007), radius=0.0016)
    for i in range(8):
        x = (i - 3.5) * 0.009
        b.cylinder(pipette, (x, -0.006, -0.103), 0.0034, 0.004, "metal")
        _tip(b, pipette, x)
    for i in range(12):
        # Local cable harness moves with the mount; it is not a simulated
        # flexible chain spanning two independently moving bodies.
        z = -0.065 + i * 0.01
        b.box(carriage, (0.051, 0.025, z), (0.009, 0.012, 0.003), "rubber")
    for ix in range(3):
        for iy in range(4):
            x, y = (ix - 1) * 0.146, (iy - 1.5) * 0.109
            for dx in (-0.062, 0.062):
                b.box(base, (x + dx, y, 0.169), (0.002, 0.04, 0.003), "bright")
            b.box(base, (x - 0.059, y + 0.039, 0.172), (0.005, 0.007, 0.003), "blue")
    # A rack at an otherwise empty deck location clarifies the scale of the
    # 9 mm pipette spacing. It does not add a tip-pickup process model.
    rx, ry = 0.146, 0.0545
    b.box(base, (rx, ry, 0.181), (0.0639, 0.0428, 0.014), "blue")
    b.box(base, (rx, ry, 0.196), (0.059, 0.039, 0.002), "cream")
    for row in range(8):
        for col in range(12):
            b.cylinder(
                base,
                (rx + (col - 5.5) * 0.009, ry + (row - 3.5) * 0.009, 0.199),
                0.0027,
                0.001,
                "dark",
            )
    b.box(door, (0, 0, 0.477), (0.381, 0.007, 0.006), "dark")
    for x in (-0.348, 0.348):
        _screw(b, door, (x, -0.012, 0.011))
    b.metadata["visual_revision"] = 3
    b.metadata["visual_evidence"] = (
        "real_labs/evidence/instruments.json#rochester_genomics"
    )


def microtome_details(b: Any, base, stroke, guard, wheel) -> None:
    """Cast envelope, orientation clamp and tray informed by BIOCUT photos."""
    shell = ET.SubElement(base, "body", name=b.unique("cast_housing"), pos="-0.035 0 0")
    b.housing(
        shell,
        [
            (0.027, 0.16, 0.237, 0.073),
            (0.06, 0.162, 0.237, 0.073),
            (0.23, 0.154, 0.209, 0.087),
            (0.281, 0.142, 0.183, 0.1),
            (0.288, 0.136, 0.179, 0.1),
        ],
        radius=0.024,
    )
    # A shallow storage tray follows the documented 235 by 275 mm top area.
    b.box(shell, (0, 0.111, 0.289), (0.1175, 0.1375, 0.002), "metal")
    b.box(shell, (0, 0.111, 0.292), (0.113, 0.133, 0.001), "cream")
    for x in (-0.118, 0.118):
        b.rod(shell, (x, -0.024, 0.294), (x, 0.246, 0.294), 0.002, "metal")
    for y in (-0.026, 0.248):
        b.rod(shell, (-0.118, y, 0.294), (0.118, y, 0.294), 0.002, "metal")
    b.box(base, (-0.035, -0.14, 0.177), (0.044, 0.006, 0.076), "metal")
    for x in (-0.076, 0.006):
        b.box(base, (x, -0.148, 0.184), (0.002, 0.001, 0.067), "dark")
    # Orientation head: machining, round spindle, clamp jaws and adjustment
    # knobs are all attached to the specimen body and follow its full stroke.
    b.cylinder(stroke, (0, -0.006, 0), 0.028, 0.016, "bright", quat=_Y_AXIS)
    b.cylinder(stroke, (0, -0.024, 0), 0.025, 0.003, "metal", quat=_Y_AXIS)
    for x in (-0.031, 0.031):
        b.box(stroke, (x, -0.033, 0), (0.007, 0.009, 0.028), "bright")
        for z in (-0.023, 0.023):
            _screw(b, stroke, (x, -0.043, z), radius=0.0026)
    b.box(stroke, (0, -0.034, -0.024), (0.028, 0.011, 0.004), "bright")
    b.cylinder(stroke, (0, -0.008, 0.039), 0.007, 0.012, "dark")
    b.cylinder(stroke, (0.049, -0.009, 0.006), 0.007, 0.013, "dark", quat=_X_AXIS)
    b.rod(stroke, (-0.041, -0.01, -0.003), (-0.052, -0.01, 0.033), 0.0035, "dark")
    # Rounded, hollow waste tray: use a shallow floor plus an open rim so the
    # blade assembly does not emerge from an opaque solid block.
    tray = ET.SubElement(
        base, "body", name=b.unique("waste_tray"), pos="-0.035 -0.222 0"
    )
    b.housing(
        tray,
        [(0.022, 0.166, 0.084, 0), (0.032, 0.177, 0.089, 0), (0.044, 0.177, 0.089, 0)],
        radius=0.025,
        material="metal",
    )
    b.box(tray, (0, 0, 0.046), (0.151, 0.072, 0.002), "cream")
    for x in (-0.17, 0.17):
        b.box(tray, (x, 0.0095, 0.057), (0.007, 0.0675, 0.014), "shell")
    b.box(tray, (0, -0.082, 0.057), (0.146, 0.007, 0.014), "shell")
    for side in (-1, 1):
        _tray_corner(b, tray, side)
    # Knife holder has a dovetail base, tilted pressure plate, lateral clamp
    # levers and a U-shaped safety guard, rather than a flat silver cuboid.
    b.box(base, (-0.035, -0.223, 0.065), (0.077, 0.041, 0.008), "bright")
    b.box(base, (-0.035, -0.229, 0.076), (0.053, 0.033, 0.008), "dark")
    b.box(
        base,
        (-0.035, -0.22, 0.105),
        (0.057, 0.007, 0.02),
        "dark",
        quat=(0.98480775, 0.17364818, 0, 0),
    )
    for x in (-0.099, 0.029):
        b.cylinder(base, (x, -0.213, 0.093), 0.018, 0.006, "metal", quat=_X_AXIS)
        b.rod(base, (x, -0.226, 0.095), (x, -0.262, 0.116), 0.004, "dark")
    for x in (0, 0.12):
        b.rod(guard, (x, 0, 0), (x, -0.017, -0.018), 0.003, "guard_red")
    for x in (-0.035, 0.003):
        _screw(b, base, (x, -0.267, 0.062), radius=0.002)
    # Coarse feed wheel on the opposite side and thickness-selector dial are
    # static visual controls; no unsupported axes are added to the API.
    b.cylinder(base, (-0.214, 0.108, 0.157), 0.042, 0.009, "dark", quat=_X_AXIS)
    b.cylinder(base, (-0.225, 0.108, 0.157), 0.027, 0.002, "metal", quat=_X_AXIS)
    b.box(base, (0.069, -0.128, 0.22), (0.022, 0.006, 0.03), "metal")
    b.cylinder(base, (0.069, -0.14, 0.207), 0.012, 0.009, "dark", quat=_Y_AXIS)
    b.box(base, (0.069, -0.136, 0.238), (0.014, 0.001, 0.006), "dark")
    for i in range(5):
        b.box(
            base, (0.059 + i * 0.005, -0.138, 0.238), (0.0004, 0.0004, 0.003), "cream"
        )
    b.cylinder(wheel, (0.013, 0, 0), 0.012, 0.002, "metal", quat=_X_AXIS)
    b.ring(wheel, (0, 0, 0), 0.1, 0.002, "rubber", plane="yz", segments=40)
    b.metadata["visual_revision"] = 3
    b.metadata["visual_evidence"] = "real_labs/evidence/instruments.json#penn_pathology"
