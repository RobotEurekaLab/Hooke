"""Photo-informed FSEL supports and hall details, with estimated dimensions.

The source shows timber falsework, repeated red portals, braced side columns
and high side windows. These are original meshes/solids, not surveyed building
geometry. Only the primary instrument's existing two joints are actuated.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers


STEEL = (0.24, 0.28, 0.30, 1)
RED = (0.57, 0.16, 0.11, 1)


def timber_cribs(b, base):
    """Two open, braced timber trestles meet the existing metal cap plates."""
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_crib_timber",
        rgba=".43 .29 .16 1",
    )
    for y in (-3.05, 3.05):
        for offset in (-0.31, 0.31):
            for x in (-0.65, 0.65):
                geom = b.box(
                    base,
                    (x, y + offset, 0.425),
                    (0.065, 0.06, 0.425),
                    "crib_timber",
                    collision=True,
                )
                geom.set("name", b.unique("beam_crib_leg"))
            for z in (0.10, 0.90):
                geom = b.box(
                    base,
                    (0, y + offset, z),
                    (0.79, 0.06, 0.05),
                    "crib_timber",
                    collision=True,
                )
                geom.set("name", b.unique("beam_crib_tie"))
            # Diagonals lie on opposite faces, avoiding coplanar timber joints.
            for side in (-1, 1):
                geom = b.box(
                    base,
                    (0, y + offset + side * 0.082, 0.475),
                    (0.035, 0.022, math.hypot(1.30, 0.75) / 2),
                    "crib_timber",
                    collision=True,
                    euler=(0, side * math.atan2(1.30, 0.75), 0),
                )
                geom.set("name", b.unique("beam_crib_brace"))
        for x in (-0.65, 0.65):
            b.box(
                base, (x, y, 0.90), (0.065, 0.31, 0.05), "crib_timber", collision=True
            )
        geom = b.box(base, (0, y, 0.975), (0.84, 0.48, 0.025), "metal", collision=True)
        geom.set("name", b.unique("beam_crib_cap"))


def sensor_riser(b, probe):
    """Join the sled at z=.020 to its crossbar at z=.040 without changing motion."""
    geom = b.box(
        probe, (-0.055, 0, 0.030), (0.035, 0.035, 0.010), "bright", collision=True
    )
    geom.set("name", b.unique("sensor_riser"))
    for y in (-0.021, 0.021):
        b.cylinder(probe, (-0.050, y, 0.093), 0.006, 0.003, "dark")


def _right_windows(world, definition):
    """Keep the right wall's established render cutaway and real window openings."""
    wall = world.find("./geom[@name='arch_cutaway_wall_right']")
    world.remove(wall)
    width, depth, height = definition["room"]["size"]
    color = definition["room"]["wall_color"]
    x, sill, top = width / 2 + 0.075, 6.7, 8.1
    prefix = "arch_cutaway_wall_fsel_right"
    for name, low, high in (("lower", 0, sill), ("upper", top, height)):
        box(
            world,
            f"{prefix}_{name}",
            (0.075, depth / 2, (high - low) / 2),
            (x, 0, (low + high) / 2),
            color,
        )
    cursor = -depth / 2
    for i, center in enumerate((-10.5, -5.5, -0.5, 4.5, 9.5)):
        low, high = center - 1.65, center + 1.65
        box(
            world,
            f"{prefix}_pier_{i}",
            (0.075, (low - cursor) / 2, (top - sill) / 2),
            (x, (low + cursor) / 2, (sill + top) / 2),
            color,
        )
        box(
            world,
            f"{prefix}_glass_{i}",
            (0.012, 1.65, (top - sill) / 2),
            (x, center, (sill + top) / 2),
            (0.64, 0.77, 0.79, 0.20),
        )
        for j, y in enumerate((low, low + 1.1, low + 2.2, high)):
            box(
                world,
                f"{prefix}_mullion_{i}_{j}",
                (0.035, 0.018, (top - sill) / 2),
                (x - 0.03, y, (sill + top) / 2),
                STEEL,
            )
        for j, z in enumerate((sill, (sill + top) / 2, top)):
            box(
                world,
                f"{prefix}_rail_{i}_{j}",
                (0.035, 1.65, 0.020),
                (x - 0.03, center, z),
                STEEL,
            )
        cursor = high
    box(
        world,
        f"{prefix}_pier_last",
        (0.075, (depth / 2 - cursor) / 2, (top - sill) / 2),
        (x, (depth / 2 + cursor) / 2, (sill + top) / 2),
        color,
    )


def add_static_reaction_frame(world, name, x, y, height, half_span):
    """An unpowered reaction frame over an existing supported inert slab."""
    for side in (-1, 1):
        column = x + side * half_span
        box(world, f"{name}_foot_{side}", (0.30, 0.35, 0.07), (column, y, 0.07), RED)
        box(
            world,
            f"{name}_web_{side}",
            (0.10, 0.17, (height - 0.14) / 2),
            (column, y, (height + 0.14) / 2),
            RED,
        )
        for face in (-1, 1):
            box(
                world,
                f"{name}_flange_{side}_{face}",
                (0.22, 0.025, (height - 0.14) / 2),
                (column, y + face * 0.195, (height + 0.14) / 2),
                RED,
            )
        for row in range(int((height - 0.3) / 0.25)):
            for dx in (-0.15, 0.15):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{name}_bolt_{side}_{row}_{dx}",
                    type="cylinder",
                    size=".010 .004",
                    pos=numbers((column + dx, y - 0.225, 0.3 + row * 0.25)),
                    euler=numbers((math.pi / 2, 0, 0)),
                    rgba=".16 .17 .17 1",
                    contype="0",
                    conaffinity="0",
                )
    box(
        world,
        f"{name}_crosshead",
        (half_span + 0.22, 0.17, 0.21),
        (x, y, height - 0.35),
        RED,
    )
    for side in (-1, 1):
        box(
            world,
            f"{name}_crosshead_flange_{side}",
            (half_span + 0.28, 0.25, 0.025),
            (x, y, height - 0.35 + side * 0.235),
            RED,
        )
    ET.SubElement(
        world,
        "geom",
        name=f"{name}_static_ram",
        type="cylinder",
        size=".14 .30",
        pos=numbers((x, y, height - 0.86)),
        rgba=".72 .34 .10 1",
    )
    ET.SubElement(
        world,
        "geom",
        name=f"{name}_static_piston",
        type="cylinder",
        size=".07 .24",
        pos=numbers((x, y, height - 1.35)),
        rgba=".50 .53 .55 1",
    )
    box(
        world,
        f"{name}_static_platen",
        (0.30, 0.28, 0.035),
        (x, y, height - 1.59),
        STEEL,
    )


def hall_details(world, definition):
    """Add reference-visible structure inside the unchanged estimated hall envelope."""
    _right_windows(world, definition)
    for side in (-1, 1):
        x = side * 8.0
        for i, y in enumerate((-10.0, -5.0, 0.0, 5.0, 10.0)):
            for face in (-1, 1):
                box(
                    world,
                    f"fsel_column_flange_{side}_{i}_{face}",
                    (0.26, 0.024, 4.15),
                    (x, y + face * 0.204, 4.15),
                    STEEL,
                )
            box(
                world,
                f"fsel_rail_seat_{side}_{i}",
                (0.32, 0.24, 0.10),
                (x, y, 7.52),
                STEEL,
            )
            if y in (-5.0, 5.0):
                for z in (1.35, 2.10):
                    box(
                        world,
                        f"fsel_service_box_{side}_{i}_{z}",
                        (0.11, 0.26, 0.23),
                        (x - side * 0.36, y, z),
                        (0.48, 0.52, 0.51, 1),
                    )
        # Leave the loading doorway's bay free of structural diagonals.
        for i, y in enumerate((-2.5, 2.5, 7.5)):
            for sign in (-1, 1):
                box(
                    world,
                    f"fsel_wall_brace_{side}_{i}_{sign}",
                    (0.05, 0.045, math.hypot(5.0, 5.4) / 2),
                    (x + side * 0.20, y, 3.25),
                    STEEL,
                    euler=numbers((-sign * math.atan2(5.0, 5.4), 0, 0)),
                )
        # Continuous services sit behind the columns, with wall-connected brackets.
        for i in range(4):
            z = 5.72 + i * 0.13
            ET.SubElement(
                world,
                "geom",
                name=f"fsel_service_conduit_{side}_{i}",
                type="cylinder",
                size=".022",
                fromto=numbers((side * 8.74, -12, z, side * 8.74, 12, z)),
                rgba=".36 .39 .39 1",
                contype="0",
                conaffinity="0",
            )
        for i, y in enumerate((-10, -5, 0, 5, 10)):
            box(
                world,
                f"fsel_service_bracket_{side}_{i}",
                (0.14, 0.023, 0.28),
                (side * 8.87, y, 5.915),
                STEEL,
                collision=False,
            )
    add_static_reaction_frame(world, "fsel_primary_repeated", -2.2, 0.6, 3.65, 1.45)
    add_static_reaction_frame(world, "fsel_secondary_repeated", 4.5, 0.0, 3.45, 1.30)
