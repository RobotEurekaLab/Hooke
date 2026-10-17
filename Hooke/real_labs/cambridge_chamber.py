"""Turner chamber topology from the designer's historical general arrangement.

The published 14 m shell diameter is interpreted here as the nominal inner
diameter; the source does not resolve that datum. Interior details, access
hardware and the drive exterior are authored estimates, not surveyed geometry.
The separate rotor remains an explicitly authored inspection mechanism.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

CONCRETE = (0.49, 0.48, 0.43, 1)
STEEL = (0.32, 0.36, 0.35, 1)
DECK = (0.27, 0.29, 0.27, 1)
CREAM = (0.72, 0.70, 0.59, 1)
INNER_APOTHEM = 7.0
OUTER_APOTHEM = 8.0
PIT_HEIGHT = 3.0
CORRIDOR_WIDTH = 2.0


def _clip(points, axis, limit, keep_less):
    """Clip a convex floor outline to one half-plane."""
    result = []
    for first, second in zip(points, points[1:] + points[:1]):
        a = (first[axis] <= limit) == keep_less
        b = (second[axis] <= limit) == keep_less
        if a:
            result.append(first)
        if a != b:
            t = (limit - first[axis]) / (second[axis] - first[axis])
            result.append(tuple(x + t * (y - x) for x, y in zip(first, second)))
    return result


def _prism(world, asset, name, outline, low, high, color):
    """A closed convex contact mesh with an unambiguous physical interior."""
    n = len(outline)
    vertices = [(x, y, z) for z in (low, high) for x, y in outline]
    faces = []
    for i in range(1, n - 1):
        faces.extend(((0, i + 1, i), (n, n + i, n + i + 1)))
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, j, n + j), (i, n + j, n + i)))
    ET.SubElement(
        asset,
        "mesh",
        name=name + "_mesh",
        vertex=numbers(v for p in vertices for v in p),
        face=" ".join(str(v) for p in faces for v in p),
    )
    return ET.SubElement(
        world,
        "geom",
        name=name,
        type="mesh",
        mesh=name + "_mesh",
        rgba=numbers(color),
        contype="1",
        conaffinity="1",
    )


def _perforated_floor(world, asset, outline, hole, low, high, prefix, color):
    """Four convex pieces leave a real rectangular opening, without a backplate."""
    x0, x1, y0, y1 = hole
    middle = _clip(_clip(outline, 0, x0, False), 0, x1, True)
    pieces = (
        _clip(outline, 0, x0, True),
        _clip(outline, 0, x1, False),
        _clip(middle, 1, y0, True),
        _clip(middle, 1, y1, False),
    )
    for i, polygon in enumerate(pieces):
        _prism(world, asset, f"{prefix}_{i}", polygon, low, high, color)


def _pit(world, asset):
    n = 24
    cosine = math.cos(math.pi / n)
    inner, outer = INNER_APOTHEM / cosine, OUTER_APOTHEM / cosine
    angles = [math.tau * i / n - math.pi / n for i in range(n)]
    outline = [(inner * math.cos(a), inner * math.sin(a)) for a in angles]
    _perforated_floor(
        world,
        asset,
        outline,
        (-1.4, 1.4, -5.25, -3.05),
        -0.22,
        0,
        "arch_floor_turner",
        CONCRETE,
    )
    for i, angle in enumerate(angles):
        other = angle + math.tau / n
        polygon = [
            (inner * math.cos(angle), inner * math.sin(angle)),
            (outer * math.cos(angle), outer * math.sin(angle)),
            (outer * math.cos(other), outer * math.sin(other)),
            (inner * math.cos(other), inner * math.sin(other)),
        ]
        middle = angle + math.pi / n
        cutaway = math.cos(middle) - math.sin(middle) > 0
        prefix = "arch_cutaway_wall" if cutaway else "arch_wall"
        _prism(world, asset, f"{prefix}_turner_sector_{i}", polygon, 0, 3, CONCRETE)
    # Two closed convex slabs avoid artificial internal sector boundaries.
    # The established renderer cutaway hides the roof in external views only.
    roof = [(outer * math.cos(a), outer * math.sin(a)) for a in angles]
    for side, keep_less in (("left", True), ("right", False)):
        _prism(
            world,
            asset,
            f"arch_roof_turner_{side}",
            _clip(roof, 0, -1.12 if keep_less else 1.12, keep_less),
            PIT_HEIGHT,
            3.30,
            CONCRETE,
        )
    for y in (-4.6, -2.3, 0, 2.3, 4.6):
        # Intersect the actual polygon facets, not its smaller inscribed circle.
        # Embed 15 mm into the concrete wall so no support gap is introduced.
        half = (
            min(
                (INNER_APOTHEM - y * math.sin(math.tau * i / n))
                / math.cos(math.tau * i / n)
                for i in range(n)
                if math.cos(math.tau * i / n) > 1e-8
            )
            + 0.015
        )
        box(
            world,
            f"arch_roof_turner_coffer_{y}",
            (half, 0.12, 0.10),
            (0, y, 2.90),
            CONCRETE,
        )
        box(
            world,
            f"arch_luminaire_turner_{y}",
            (0.08, 0.50, 0.025),
            (0, y, 2.77),
            (1, 0.93, 0.72, 1),
            collision=False,
        )
        box(
            world,
            f"arch_roof_turner_light_mount_{y}",
            (0.025, 0.08, 0.0175),
            (0, y, 2.8125),
            STEEL,
        )


def _drive_space(world):
    # Separate below-pit service space, connected to the real grate opening.
    box(world, "turner_drive_floor", (1.6, 3.15, 0.10), (0, -2.2, -1.3), CONCRETE)
    for side in (-1, 1):
        box(
            world,
            f"turner_drive_side_{side}",
            (0.10, 3.15, 0.50),
            (side * 1.50, -2.2, -0.70),
            CONCRETE,
        )
    for y in (-5.25, 0.85):
        box(world, f"turner_drive_end_{y}", (1.5, 0.10, 0.50), (0, y, -0.70), CONCRETE)
    box(world, "turner_drive_motor_mount", (0.48, 0.60, 0.06), (0, -1.7, -1.14), STEEL)
    box(
        world,
        "turner_drive_motor",
        (0.39, 0.50, 0.27),
        (0, -1.7, -0.81),
        (0.20, 0.28, 0.27, 1),
    )
    box(world, "turner_drive_gearbox", (0.39, 0.39, 0.29), (0, -0.5, -0.81), STEEL)
    box(
        world, "turner_drive_gearbox_mount", (0.46, 0.46, 0.05), (0, -0.5, -1.15), STEEL
    )
    ET.SubElement(
        world,
        "geom",
        name="turner_drive_coupling",
        type="cylinder",
        size=".12 .26",
        pos="0 -1.1 -.81",
        euler="1.57079632679 0 0",
        rgba=numbers(STEEL),
        contype="1",
        conaffinity="1",
    )
    # The grate's perimeter rests on the cut floor edges. Individual bars are
    # physical, leaving a measurable opening into the service space below.
    for x in (-1.44, 1.44):
        box(
            world,
            f"turner_grate_frame_x_{x}",
            (0.045, 1.14, 0.035),
            (x, -4.15, 0.01),
            STEEL,
        )
    for y in (-5.25, -3.05):
        box(
            world,
            f"turner_grate_frame_y_{y}",
            (1.40, 0.045, 0.035),
            (0, y, 0.01),
            STEEL,
        )
    for i in range(25):
        x = -1.30 + i * 2.60 / 24
        box(
            world,
            f"turner_service_grate_{i}",
            (0.012, 1.10, 0.018),
            (x, -4.15, 0.027),
            STEEL,
        )
    for y in (-4.9, -4.15, -3.4):
        box(
            world,
            f"turner_grate_crossbar_{y}",
            (1.44, 0.018, 0.018),
            (0, y, -0.009),
            STEEL,
        )


def _gallery(world, asset):
    outline = [(-1.12, -8), (1.12, -8), (1.12, 8), (-1.12, 8)]
    _perforated_floor(
        world,
        asset,
        outline,
        (-0.70, 0.70, 3.10, 4.30),
        3.29,
        3.30,
        "turner_gallery_deck",
        DECK,
    )
    # The source describes 10 mm steel tread cladding. Its bearing ledgers and
    # framing below are explicit estimates, with real contact to the pit roof.
    for side in (-1, 1):
        box(
            world,
            f"turner_gallery_ledger_{side}",
            (0.08, 8.0, 0.04),
            (side * 1.10, 0, 3.16),
            STEEL,
        )
        box(
            world,
            f"turner_gallery_edge_joist_{side}",
            (0.055, 8.0, 0.045),
            (side * 1.06, 0, 3.245),
            STEEL,
        )
        box(
            world,
            f"turner_gallery_hatch_joist_{side}",
            (0.055, 0.655, 0.045),
            (side * 0.755, 3.70, 3.245),
            STEEL,
        )
    for y in (-7.7, -4.6, -2.3, 0, 2.3, 3.045, 4.355, 4.6, 7.7):
        box(
            world,
            f"turner_gallery_cross_joist_{y}",
            (1.14, 0.055, 0.045),
            (0, y, 3.245),
            STEEL,
        )
    for side in (-1, 1):
        for y in (-7.7, -4.5, -2.2, 0, 2.2, 4.5, 7.7):
            box(
                world,
                f"turner_gallery_post_{side}_{y}",
                (0.04, 0.04, 0.70),
                (side * 1.04, y, 4.0),
                STEEL,
            )
        for z in (3.47, 4.10, 4.68):
            box(
                world,
                f"turner_gallery_stringer_{side}_{z}",
                (0.035, 8.0, 0.025),
                (side * 1.08, 0, z),
                STEEL,
            )
        box(
            world,
            f"turner_gallery_glass_{side}",
            (0.008, 7.97, 0.57),
            (side * 1.08, 0, 4.075),
            (0.72, 0.81, 0.77, 0.24),
        )
    # Translucent barrel canopy follows the designer's general arrangement.
    # Panel radii and frame sections are estimates; no weatherproofing claim.
    n = 16
    for i in range(n):
        a, b = math.pi * i / n, math.pi * (i + 1) / n
        x0, z0 = 1.08 * math.cos(a), 4.70 + 1.08 * math.sin(a)
        x1, z1 = 1.08 * math.cos(b), 4.70 + 1.08 * math.sin(b)
        angle = math.atan2(z1 - z0, x1 - x0)
        length = math.hypot(x1 - x0, z1 - z0)
        box(
            world,
            f"arch_roof_turner_canopy_{i}",
            (length / 2, 8.0, 0.015),
            ((x0 + x1) / 2, 0, (z0 + z1) / 2),
            (0.75, 0.81, 0.72, 0.65),
            euler=numbers((0, -angle, 0)),
        )
        for y in (-7.8, -3.4, 0, 3.4, 7.8):
            box(
                world,
                f"turner_canopy_rib_{i}_{y}",
                (length / 2, 0.025, 0.025),
                ((x0 + x1) / 2, y, (z0 + z1) / 2),
                STEEL,
                euler=numbers((0, -angle, 0)),
            )
    # Visible supported guards around an authored inspection opening.
    for x in (-0.75, 0.75):
        for y in (3.05, 4.35):
            box(
                world,
                f"turner_hatch_post_{x}_{y}",
                (0.025, 0.025, 0.48),
                (x, y, 3.78),
                STEEL,
            )
        box(
            world,
            f"turner_hatch_guard_{x}",
            (0.025, 0.67, 0.025),
            (x, 3.70, 4.25),
            STEEL,
        )
    box(world, "turner_hatch_guard_end", (0.775, 0.025, 0.025), (0, 4.35, 4.25), STEEL)
    # A fixed ladder illustrates the access connection, without a locomotion task.
    for x in (-0.32, 0.32):
        box(
            world,
            f"turner_access_ladder_side_{x}",
            (0.025, 0.035, 1.65),
            (x, 4.22, 1.65),
            STEEL,
        )
    for i in range(11):
        box(
            world,
            f"turner_access_ladder_rung_{i}",
            (0.32, 0.035, 0.018),
            (0, 4.22, 0.25 + i * 0.28),
            STEEL,
        )
    for x in (-0.32, 0.32):
        box(
            world,
            f"turner_ladder_top_anchor_{x}",
            (0.035, 0.08, 0.025),
            (x, 4.28, 3.265),
            STEEL,
        )
    for y in (-5.3, 5.3):
        box(
            world,
            f"turner_gallery_console_base_{y}",
            (0.33, 0.35, 0.48),
            (0.48, y, 3.78),
            CREAM,
        )
        box(
            world,
            f"turner_gallery_console_top_{y}",
            (0.38, 0.38, 0.025),
            (0.48, y, 4.285),
            STEEL,
        )
        box(
            world,
            f"turner_gallery_console_display_{y}",
            (0.23, 0.035, 0.18),
            (0.48, y + 0.16, 4.49),
            (0.12, 0.17, 0.17, 1),
        )
        box(
            world,
            f"turner_gallery_console_mount_{y}",
            (0.035, 0.035, 0.105),
            (0.48, y + 0.16, 4.32),
            STEEL,
        )


def build_chamber(world, definition):
    """Replace only this scene's generic square room with the evidenced topology."""
    for node in list(world):
        if node.tag == "geom" and node.get("name", "").startswith("arch_"):
            world.remove(node)
    # Return inline resources through the room-feature hook, alongside geometry.
    asset = ET.Element("asset")
    _pit(world, asset)
    _drive_space(world)
    _gallery(world, asset)
    return list(asset)
