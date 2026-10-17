"""Data-driven room envelopes and workbenches, with real wall openings.

Room dimensions are authored estimates unless the catalogue gives a measured
source. Ceilings are modelled separately so a renderer can cut them away in an
overview while retaining them in human-height interior views.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return " ".join(f"{value:.8g}" for value in values)


def box(parent, name, size, position, color, *, collision=True, **attributes):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        size=numbers(size),
        pos=numbers(position),
        rgba=numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        **attributes,
    )


def _wall_box(
    world, wall, dimensions, name, span, zspan, color, depth=0.075, inset=0.0, **kwargs
):
    width, length, _ = dimensions
    middle = (span[0] + span[1]) / 2
    z = (zspan[0] + zspan[1]) / 2
    halfspan = (span[1] - span[0]) / 2
    halfheight = (zspan[1] - zspan[0]) / 2
    if min(halfspan, halfheight) <= 1e-6:
        return
    if wall == "back":
        position, size = (middle, length / 2 + depth - inset, z), (
            halfspan,
            depth,
            halfheight,
        )
    else:
        position, size = (-width / 2 - depth + inset, middle, z), (
            depth,
            halfspan,
            halfheight,
        )
    box(world, name, size, position, color, **kwargs)


def _wall(world, wall, room):
    dimensions = room["size"]
    length = dimensions[0 if wall == "back" else 1]
    height = dimensions[2]
    openings = [
        o for o in room["architecture"].get("openings", []) if o["wall"] == wall
    ]
    cuts = []
    for opening in sorted(openings, key=lambda o: o["offset"]):
        low = opening["offset"] - opening["width"] / 2
        high = opening["offset"] + opening["width"] / 2
        sill = opening.get("sill", 0.0)
        top = sill + opening["height"]
        if low < -length / 2 or high > length / 2 or sill < 0 or top > height:
            raise ValueError(f"Opening outside {wall} wall: {opening['id']}")
        if cuts and low < cuts[-1][1] - 1e-6:
            raise ValueError(f"Overlapping {wall} wall openings: {opening['id']}")
        cuts.append((low, high, sill, top, opening))
    cursor = -length / 2
    color = room["wall_color"]
    for index, (low, high, sill, top, opening) in enumerate(cuts):
        _wall_box(
            world,
            wall,
            dimensions,
            f"arch_wall_{wall}_{index}",
            (cursor, low),
            (0, height),
            color,
        )
        _wall_box(
            world,
            wall,
            dimensions,
            f"arch_sill_{wall}_{index}",
            (low, high),
            (0, sill),
            color,
        )
        _wall_box(
            world,
            wall,
            dimensions,
            f"arch_header_{wall}_{index}",
            (low, high),
            (top, height),
            color,
        )
        frame_color = (0.29, 0.32, 0.33, 1)
        key = "arch_opening_" + opening["id"]
        for side, at in enumerate((low, high)):
            _wall_box(
                world,
                wall,
                dimensions,
                f"{key}_jamb_{side}",
                (at - 0.025, at + 0.025),
                (sill, top),
                frame_color,
                depth=0.08,
                inset=0.02,
            )
        for side, at in enumerate((sill, top)):
            _wall_box(
                world,
                wall,
                dimensions,
                f"{key}_lintel_{side}",
                (low, high),
                (max(0, at - 0.025), at + 0.025),
                frame_color,
                depth=0.08,
                inset=0.02,
            )
        kind = opening["kind"]
        if kind in ("window", "glazing"):
            _wall_box(
                world,
                wall,
                dimensions,
                key + "_glass",
                (low + 0.025, high - 0.025),
                (sill + 0.025, top - 0.025),
                (0.64, 0.77, 0.79, 0.20),
                depth=0.012,
            )
            count = max(1, math.ceil(opening["width"] / 1.3))
            for part in range(1, count):
                at = low + opening["width"] * part / count
                _wall_box(
                    world,
                    wall,
                    dimensions,
                    f"{key}_mullion_{part}",
                    (at - 0.018, at + 0.018),
                    (sill, top),
                    frame_color,
                    depth=0.025,
                )
        elif kind == "shutter":
            # Raised roller shutter; the doorway below stays traversable.
            _wall_box(
                world,
                wall,
                dimensions,
                key + "_roller",
                (low, high),
                (top - 0.13, top + 0.13),
                (0.40, 0.43, 0.44, 1),
                depth=0.15,
            )
        elif kind != "door":
            raise ValueError(f"Unknown architectural opening: {kind}")
        cursor = high
    _wall_box(
        world,
        wall,
        dimensions,
        f"arch_wall_{wall}_last",
        (cursor, length / 2),
        (0, height),
        color,
    )


def build_room(world, room):
    width, depth, height = room["size"]
    architecture = room["architecture"]
    floor = architecture.get("floor", {})
    color = floor.get("color", [0.48, 0.50, 0.49, 1])
    box(
        world,
        "arch_floor",
        (width / 2, depth / 2, 0.08),
        (0, 0, -0.08),
        color,
        friction=".9 .01 .001",
    )
    pattern = floor.get("pattern", "plain")
    if pattern not in ("plain", "seams", "tiles"):
        raise ValueError(f"Unknown floor pattern: {pattern}")
    if pattern != "plain":
        spacing = floor.get("spacing", 1.2 if pattern == "tiles" else 3.0)
        if spacing <= 0.1:
            raise ValueError("Floor seam spacing must exceed 0.1 metres")
        seam_color = (*[c * 0.80 for c in color[:3]], 1)
        for i, x in enumerate(np.arange(-width / 2 + spacing, width / 2, spacing)):
            box(
                world,
                f"arch_floor_seam_x_{i}",
                (0.0015, depth / 2, 0.0004),
                (x, 0, 0.0004),
                seam_color,
                collision=False,
            )
        for i, y in enumerate(np.arange(-depth / 2 + spacing, depth / 2, spacing)):
            box(
                world,
                f"arch_floor_seam_y_{i}",
                (width / 2, 0.0015, 0.0004),
                (0, y, 0.0004),
                seam_color,
                collision=False,
            )
    for side in ("back", "left"):
        _wall(world, side, room)
    # Complete the physical enclosure. Overview rendering cuts these two faces
    # away; standing-height views retain them instead of looking into a void.
    box(
        world,
        "arch_cutaway_wall_front",
        (width / 2, 0.075, height / 2),
        (0, -depth / 2 - 0.075, height / 2),
        room["wall_color"],
    )
    box(
        world,
        "arch_cutaway_wall_right",
        (0.075, depth / 2, height / 2),
        (width / 2 + 0.075, 0, height / 2),
        room["wall_color"],
    )
    ceiling = architecture.get("ceiling", "none")
    if ceiling not in ("none", "suspended", "exposed"):
        raise ValueError(f"Unknown ceiling construction: {ceiling}")
    if ceiling != "none":
        box(
            world,
            "arch_roof_slab",
            (width / 2, depth / 2, 0.06),
            (0, 0, height + 0.06),
            (0.72, 0.73, 0.70, 1),
        )
    if ceiling == "suspended":
        for i, x in enumerate(np.arange(-width / 2, width / 2, 1.2)):
            box(
                world,
                f"arch_roof_grid_x_{i}",
                (0.012, depth / 2, 0.012),
                (x, 0, height - 0.016),
                (0.58, 0.59, 0.57, 1),
                collision=False,
            )
        for i, y in enumerate(np.arange(-depth / 2, depth / 2, 0.6)):
            box(
                world,
                f"arch_roof_grid_y_{i}",
                (width / 2, 0.012, 0.012),
                (0, y, height - 0.016),
                (0.58, 0.59, 0.57, 1),
                collision=False,
            )
    elif ceiling == "exposed":
        for i, y in enumerate(np.arange(-depth / 2 + 0.8, depth / 2, 3.0)):
            box(
                world,
                f"arch_roof_beam_{i}",
                (width / 2, 0.09, 0.14),
                (0, y, height - 0.14),
                (0.24, 0.28, 0.29, 1),
            )
    # A roofless evidence vignette has no inferred ceiling-mounted equipment.
    # Fixtures and their physical hangers share the roof's cutaway visibility.
    if ceiling != "none":
        for row, y in enumerate(
            np.linspace(-depth * 0.30, depth * 0.30, max(2, round(depth / 3)))
        ):
            for col, x in enumerate((-width * 0.25, width * 0.25)):
                box(
                    world,
                    f"arch_luminaire_{row}_{col}",
                    (0.55, 0.12, 0.025),
                    (x, y, height - 0.13),
                    (0.90, 0.92, 0.91, 1),
                    collision=False,
                )
                for side, offset in enumerate((-0.38, 0.38)):
                    box(
                        world,
                        f"arch_roof_fixture_support_{row}_{col}_{side}",
                        (0.012, 0.020, 0.0525),
                        (x + offset, y, height - 0.0525),
                        (0.55, 0.58, 0.59, 1),
                        collision=False,
                    )
    if architecture.get("service_strip", False):
        _wall_box(
            world,
            "back",
            room["size"],
            "arch_service_raceway",
            (-width / 2 + 0.1, width / 2 - 0.1),
            (1.08, 1.16),
            room["accent"],
            depth=0.025,
            inset=0.055,
        )


def build_bench(world, item, accent, *, asset=None):
    """Keep the scene placement contract separate from reusable furniture detail."""
    from real_labs.bench_details import build_furniture

    parent = ET.SubElement(
        world,
        "body",
        name=item["id"],
        pos=numbers(item["pos"]),
        euler=f"0 0 {item.get('yaw', 0)}",
    )
    return build_furniture(parent, item, accent, asset)
