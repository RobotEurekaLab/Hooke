"""Source-observed architectural cues for the compact laboratory prototypes.

Dimensions remain layout estimates. These furnishings do not establish surveyed
floor plans or simulate growth, ventilation, tracking, or magnetic levitation.
"""

import math
import xml.etree.ElementTree as ET


def _numbers(values):
    return " ".join(f"{value:.6g}" for value in values)


def _geom(parent, name, shape, size, position, color, **attributes):
    attributes.setdefault("contype", "0")
    attributes.setdefault("conaffinity", "0")
    return ET.SubElement(
        parent,
        "geom",
        name=f"reference_{name}",
        type=shape,
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        **attributes,
    )


def _box(parent, name, size, position, color, **attributes):
    return _geom(parent, name, "box", size, position, color, **attributes)


def _remove_room_features(world, prefixes):
    for element in list(world):
        if element.tag == "geom" and element.get("name", "").startswith(prefixes):
            world.remove(element)


def _robotics_hall(world, room):
    width, depth, height = room["size"]
    _remove_room_features(
        world,
        (
            "back_wall",
            "left_wall",
            "wall_service",
            "wall_socket",
            "socket_hole",
            "window_",
        ),
    )
    metal, glass = (0.34, 0.38, 0.40, 1), (0.52, 0.73, 0.82, 0.14)
    _box(
        world,
        "back_glazing",
        (width / 2, 0.015, height / 2),
        (0, depth / 2, height / 2),
        glass,
        contype="1",
        conaffinity="1",
    )
    _box(
        world,
        "left_glazing",
        (0.015, depth / 2, height / 2),
        (-width / 2, 0, height / 2),
        glass,
        contype="1",
        conaffinity="1",
    )
    for index in range(9):
        _box(
            world,
            f"glass_post_{index}",
            (0.025, 0.035, height / 2),
            (-width / 2 + index * width / 8, depth / 2 - 0.03, height / 2),
            metal,
        )
    for index in range(7):
        _box(
            world,
            f"side_post_{index}",
            (0.035, 0.025, height / 2),
            (-width / 2 + 0.03, -depth / 2 + index * depth / 6, height / 2),
            metal,
        )
    for side in (-1, 1):
        _box(
            world,
            f"gantry_rail_{side}",
            (0.07, depth * 0.42, 0.09),
            (side * width * 0.40, 0, height - 0.45),
            metal,
        )
    _box(
        world,
        "gantry_bridge",
        (width * 0.40, 0.075, 0.11),
        (0, 0.45, height - 0.57),
        (0.74, 0.60, 0.19, 1),
    )
    _box(
        world,
        "gantry_trolley",
        (0.16, 0.15, 0.11),
        (-0.55, 0.45, height - 0.78),
        (0.23, 0.26, 0.28, 1),
    )
    # The published in-floor maglev footprint is 2 x 3 metres.
    _box(
        world,
        "maglev_footprint",
        (1, 1.5, 0.002),
        (-1, -0.45, 0.003),
        (0.17, 0.20, 0.23, 1),
    )
    for index in range(7):
        _box(
            world,
            f"maglev_tile_{index}",
            (0.97, 0.003, 0.001),
            (-1, -1.90 + index * 0.48, 0.006),
            (0.49, 0.52, 0.54, 1),
        )


def _maize(world, key, x, y):
    plant = ET.SubElement(
        world, "body", name=f"reference_maize_{key}", pos=_numbers((x, y, 0.38))
    )
    _geom(
        plant,
        f"pot_{key}",
        "cylinder",
        (0.135, 0.14),
        (0, 0, 0.14),
        (0.35, 0.36, 0.32, 1),
    )
    _geom(
        plant,
        f"pot_rim_{key}",
        "cylinder",
        (0.145, 0.014),
        (0, 0, 0.275),
        (0.42, 0.43, 0.39, 1),
    )
    _geom(
        plant,
        f"soil_{key}",
        "cylinder",
        (0.128, 0.008),
        (0, 0, 0.29),
        (0.25, 0.19, 0.12, 1),
    )
    _geom(
        plant,
        f"stem_{key}",
        "cylinder",
        (0.007, 0.52),
        (0, 0, 0.81),
        (0.26, 0.38, 0.10, 1),
    )
    for leaf in range(8):
        angle = leaf * 2.40 + key * 0.36
        base = 0.36 + leaf * 0.11
        length = 0.42 - leaf * 0.022
        for segment in range(4):
            t = (segment + 0.5) / 4
            radial = length * t
            z = base + 0.22 * math.sin(t * math.pi * 0.85)
            tilt = -0.55 * math.cos(t * math.pi)
            _geom(
                plant,
                f"leaf_{key}_{leaf}_{segment}",
                "ellipsoid",
                (length * 0.18, 0.033 * (1 - t) + 0.004, 0.003),
                (radial * math.cos(angle), radial * math.sin(angle), z),
                (0.21 + 0.03 * (leaf % 3), 0.37 + 0.025 * (leaf % 2), 0.10, 1),
                euler=_numbers((0, tilt, angle)),
            )


def _growth_rows(world, room):
    _remove_room_features(world, ("window_",))
    for row, y in enumerate((1.65, 2.40)):
        for offset in (-0.18, 0.18):
            _box(
                world,
                f"crop_rail_{row}_{offset}",
                (1.65, 0.025, 0.04),
                (1.25, y + offset, 0.31),
                (0.62, 0.65, 0.65, 1),
            )
        for column, x in enumerate((0, 0.80, 1.60, 2.40)):
            key = row * 4 + column
            _box(
                world,
                f"crop_carrier_{key}",
                (0.19, 0.19, 0.025),
                (x, y, 0.365),
                (0.16, 0.18, 0.18, 1),
            )
            _box(
                world,
                f"crop_guide_{key}",
                (0.13, 0.024, 0.018),
                (x, y - 0.21, 0.34),
                (0.72, 0.12, 0.09, 1),
            )
            _maize(world, key, x, y)
            z = min(room["size"][2] - 0.30, 3.15)
            _box(
                world,
                f"grow_lamp_case_{key}",
                (0.24, 0.16, 0.035),
                (x, y, z),
                (0.66, 0.69, 0.67, 1),
            )
            _box(
                world,
                f"grow_lamp_face_{key}",
                (0.21, 0.14, 0.006),
                (x, y, z - 0.038),
                (0.95, 0.94, 0.77, 1),
            )


def _pathology_cabinets(world, room):
    _remove_room_features(world, ("window_",))
    y = room["size"][1] / 2 - 0.30
    for index, x in enumerate((-2.1, -0.95, 0.2, 1.35)):
        cabinet = ET.SubElement(world, "body", name=f"reference_upper_cabinet_{index}")
        for side in (-1, 1):
            _box(
                cabinet,
                f"cabinet_side_{index}_{side}",
                (0.025, 0.23, 0.40),
                (x + side * 0.51, y, 2.15),
                (0.70, 0.73, 0.73, 1),
            )
            _box(
                cabinet,
                f"cabinet_shelf_{index}_{side}",
                (0.51, 0.23, 0.025),
                (x, y, 2.15 + side * 0.40),
                (0.70, 0.73, 0.73, 1),
            )
        _box(
            cabinet,
            f"cabinet_back_{index}",
            (0.51, 0.018, 0.40),
            (x, y + 0.22, 2.15),
            (0.83, 0.85, 0.83, 1),
        )
        _box(
            cabinet,
            f"cabinet_glass_{index}",
            (0.48, 0.012, 0.36),
            (x, y - 0.23, 2.15),
            (0.58, 0.77, 0.80, 0.30),
        )
        _box(
            cabinet,
            f"cabinet_divider_{index}",
            (0.012, 0.018, 0.39),
            (x, y - 0.25, 2.15),
            (0.39, 0.44, 0.45, 1),
        )
        for bottle in range(4):
            _geom(
                cabinet,
                f"stored_bottle_{index}_{bottle}",
                "cylinder",
                (0.044, 0.10),
                (x - 0.34 + bottle * 0.22, y, 1.90),
                (0.67, 0.76, 0.74, 0.70),
            )


def _gas_supply(world):
    for index, x in enumerate((2.83, 3.20)):
        _geom(
            world,
            f"gas_cylinder_{index}",
            "capsule",
            (0.13, 0.58),
            (x, 2.50, 0.73),
            (0.17, 0.29, 0.30, 1),
        )
        _geom(
            world,
            f"gas_valve_{index}",
            "cylinder",
            (0.025, 0.035),
            (x, 2.50, 1.46),
            (0.65, 0.59, 0.35, 1),
        )
        _box(
            world,
            f"cylinder_restraint_{index}",
            (0.16, 0.025, 0.035),
            (x, 2.36, 0.95),
            (0.24, 0.27, 0.28, 1),
        )
    _box(
        world,
        "cylinder_rack",
        (0.39, 0.035, 0.60),
        (3.02, 2.65, 0.64),
        (0.36, 0.40, 0.41, 1),
    )


def add_reference_features(world: ET.Element, definition: dict) -> None:
    """Add inspected visual cues without changing equipment interfaces."""
    identifier, room = definition["id"], definition["room"]
    if identifier == "waterloo_robohub":
        _robotics_hall(world, room)
    elif identifier == "purdue_phenotyping":
        _growth_rows(world, room)
    elif identifier == "penn_pathology":
        _pathology_cabinets(world, room)
    elif identifier == "rochester_genomics":
        for geom in world.iter("geom"):
            name = geom.get("name", "")
            if "_front_" in name or name.endswith("_cabinet"):
                geom.set("rgba", ".53 .35 .20 1")
    elif identifier in ("uw_isolab", "eth_air_quality"):
        _gas_supply(world)
