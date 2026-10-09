"""Observed context and explicit layout estimates for three distinct wet laboratories.

Cabinets, supports and partitions are collision geometry. Small supplies and
service lines are visual context, not additional controllable instruments.
Reference media and placement assumptions are recorded in layouts/wetlabs.json.
"""

import math
import xml.etree.ElementTree as ET


WOOD = (0.62, 0.39, 0.17, 1)
METAL = (0.51, 0.55, 0.55, 1)
DARK = (0.12, 0.15, 0.15, 1)
GLASS = (0.64, 0.78, 0.80, 0.20)


def _numbers(values):
    return " ".join(f"{value:.6g}" for value in values)


def _body(parent, name, position=(0, 0, 0), yaw=0):
    return ET.SubElement(
        parent,
        "body",
        name=f"wetlab_{name}",
        pos=_numbers(position),
        euler=_numbers((0, 0, yaw)),
    )


def _geom(parent, name, shape, size, position, color, collision=False, **extra):
    return ET.SubElement(
        parent,
        "geom",
        name=f"wetlab_{name}",
        type=shape,
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        **extra,
    )


def _box(parent, name, halfsize, position, color, collision=False):
    return _geom(parent, name, "box", halfsize, position, color, collision)


def _line(parent, name, start, end, radius, color, collision=False):
    return ET.SubElement(
        parent,
        "geom",
        name=f"wetlab_{name}",
        type="capsule",
        fromto=_numbers((*start, *end)),
        size=str(radius),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


def _bottle(parent, name, position, radius=0.035, height=0.11, color=GLASS):
    x, y, z = position
    _geom(parent, name, "cylinder", (radius, height / 2), (x, y, z + height / 2), color)
    _geom(
        parent,
        name + "_cap",
        "cylinder",
        (radius * 0.83, 0.01),
        (x, y, z + height + 0.01),
        (0.18, 0.25, 0.39, 1),
    )
    _box(
        parent,
        name + "_label",
        (radius * 0.73, 0.001, height * 0.24),
        (x, y - radius, z + height * 0.53),
        (0.88, 0.89, 0.83, 1),
    )


def _monitor(parent, name, position, yaw=0):
    body = _body(parent, name, position, yaw)
    _box(body, name + "_base", (0.14, 0.095, 0.018), (0, 0, 0.018), DARK)
    _box(body, name + "_stand", (0.022, 0.023, 0.10), (0, 0.015, 0.12), DARK)
    _box(body, name + "_case", (0.23, 0.029, 0.15), (0, 0.035, 0.35), DARK)
    _box(
        body,
        name + "_screen",
        (0.213, 0.002, 0.133),
        (0, 0.003, 0.35),
        (0.12, 0.20, 0.27, 1),
    )
    for row in range(4):
        _box(
            body,
            f"{name}_interface_{row}",
            (0.14 - row * 0.02, 0.002, 0.006),
            (-0.02, -0.0005, 0.43 - row * 0.045),
            (0.56, 0.70, 0.73, 1),
        )
    _box(
        body,
        name + "_keyboard",
        (0.17, 0.062, 0.012),
        (0, -0.18, 0.012),
        (0.68, 0.69, 0.65, 1),
    )
    for row in range(4):
        _box(
            body,
            f"{name}_keys_{row}",
            (0.15, 0.003, 0.001),
            (0, -0.22 + row * 0.024, 0.025),
            (0.29, 0.33, 0.33, 1),
        )


def _storage_cabinet(parent, name, position, width, color, glass=True, open_bay=False):
    body = _body(parent, name, position)
    for side in (-1, 1):
        _box(
            body,
            f"{name}_side_{side}",
            (0.018, 0.18, 0.34),
            (side * (width / 2 - 0.018), 0, 0),
            color,
            True,
        )
        _box(
            body,
            f"{name}_horizontal_{side}",
            (width / 2, 0.18, 0.018),
            (0, 0, side * 0.34),
            color,
            True,
        )
    _box(body, name + "_back", (width / 2, 0.018, 0.34), (0, 0.18, 0), color, True)
    _box(
        body,
        name + "_shelf",
        (width / 2 - 0.02, 0.17, 0.012),
        (0, 0, -0.005),
        color,
        True,
    )
    for side in (-1, 1):
        x = side * width / 4
        if not open_bay:
            _box(
                body,
                f"{name}_door_{side}" + ("_glass" if glass else ""),
                (width / 4 - 0.024, 0.003, 0.30),
                (x, -0.185, 0),
                GLASS if glass else color,
                True,
            )
            for edge in (-1, 1):
                _box(
                    body,
                    f"{name}_rail_{side}_{edge}",
                    (width / 4 - 0.012, 0.013, 0.018),
                    (x, -0.192, edge * 0.317),
                    color,
                )
                _box(
                    body,
                    f"{name}_stile_{side}_{edge}",
                    (0.018, 0.013, 0.317),
                    (x + edge * (width / 4 - 0.018), -0.192, 0),
                    color,
                )
            _box(
                body,
                f"{name}_handle_{side}",
                (0.010, 0.024, 0.065),
                (side * 0.05, -0.204, -0.005),
                METAL,
            )
    _box(body, name + "_divider", (0.013, 0.022, 0.33), (0, -0.19, 0), color)
    # Individually sized stock makes the glass read as a cupboard, not a blue panel.
    for index, fraction in enumerate((-0.36, -0.20, 0.04, 0.30)):
        _bottle(
            body,
            f"{name}_stock_{index}",
            (width * fraction, 0.005, -0.32),
            radius=(0.030, 0.023, 0.042, 0.035)[index],
            height=(0.14, 0.20, 0.10, 0.18)[index],
            color=(0.75, 0.80, 0.74, 0.28),
        )
    for index in range(4):
        x = -width * 0.34 + index * width * 0.21
        height = (0.20, 0.24, 0.17, 0.22)[index]
        color = (0.12, 0.21, 0.40, 1) if index % 2 else (0.68, 0.68, 0.55, 1)
        _box(
            body,
            f"{name}_binder_{index}",
            (0.033, 0.10, height / 2),
            (x, -0.015, 0.01 + height / 2),
            color,
        )
        _box(
            body,
            f"{name}_binder_label_{index}",
            (0.020, 0.001, 0.046),
            (x, -0.116, 0.10 + height / 2),
            (0.90, 0.88, 0.77, 1),
        )


def _cold_storage(world, name, position, width=0.80, depth=0.73, height=1.87):
    body = _body(world, name, position)
    _box(
        body,
        name + "_housing",
        (width / 2, depth / 2, height / 2),
        (0, 0, height / 2),
        (0.82, 0.83, 0.79, 1),
        True,
    )
    for z, halfheight in (
        (height * 0.24, height * 0.225),
        (height * 0.73, height * 0.245),
    ):
        _box(
            body,
            f"{name}_door_{z}",
            (width / 2 - 0.025, 0.018, halfheight),
            (0, -depth / 2 - 0.013, z),
            (0.90, 0.90, 0.85, 1),
            True,
        )
        _line(
            body,
            f"{name}_handle_{z}",
            (width * 0.34, -depth / 2 - 0.058, z - 0.14),
            (width * 0.34, -depth / 2 - 0.058, z + 0.14),
            0.012,
            METAL,
        )
    _box(
        body,
        name + "_panel",
        (0.095, 0.005, 0.04),
        (-0.10, -depth / 2 - 0.034, height * 0.89),
        DARK,
    )


def _ceiling_lights(world, name, positions, halfsize):
    """Replace generic fixtures with the row orientation visible in a room view."""
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_luminaire_"):
            world.remove(geom)
    for index, position in enumerate(positions):
        x, y, z = position
        _box(
            world,
            f"{name}_light_recess_{index}",
            (halfsize[0] + 0.025, halfsize[1] + 0.025, 0.022),
            (x, y, z + 0.028),
            (0.27, 0.28, 0.26, 1),
        )
        _box(
            world,
            f"luminaire_{name}_{index}",
            (*halfsize, 0.006),
            position,
            (0.95, 0.96, 0.92, 1),
        )


def _service_raceway(world, name, position, length, yaw=0):
    body = _body(world, name, position, yaw)
    _box(
        body,
        name + "_channel",
        (length / 2, 0.024, 0.05),
        (0, 0, 0),
        (0.72, 0.73, 0.68, 1),
    )
    for index in range(max(2, round(length / 0.65))):
        x = (
            -length / 2
            + 0.24
            + index * (length - 0.48) / max(1, round(length / 0.65) - 1)
        )
        _box(
            body,
            f"{name}_socket_{index}",
            (0.024, 0.005, 0.034),
            (x, -0.029, 0),
            (0.89, 0.87, 0.77, 1),
        )
        for slot in (-1, 1):
            _box(
                body,
                f"{name}_socket_slot_{index}_{slot}",
                (0.002, 0.001, 0.007),
                (x + slot * 0.009, -0.035, 0.004),
                DARK,
            )


def _clean_work_enclosure(world):
    """Observed glazed cabinet silhouette; dimensions, airflow and model unverified."""
    body = _body(world, "genomics_clean_enclosure", (-1.45, -0.78, 0.92), math.pi)
    shell = (0.78, 0.79, 0.73, 1)
    _box(body, "clean_back", (1.15, 0.026, 0.65), (0, 0.33, 0.65), shell, True)
    _box(body, "clean_worktray", (1.15, 0.34, 0.012), (0, 0, 0.012), METAL, True)
    for side in (-1, 1):
        _box(
            body,
            f"clean_side_{side}_glass",
            (0.004, 0.31, 0.43),
            (side * 1.12, 0, 0.47),
            GLASS,
            True,
        )
        for y in (-0.31, 0.31):
            _box(
                body,
                f"clean_sidepost_{side}_{y}",
                (0.025, 0.025, 0.55),
                (side * 1.12, y, 0.56),
                shell,
                True,
            )
    _box(body, "clean_upper_housing", (1.17, 0.35, 0.21), (0, 0, 1.12), shell, True)
    _box(
        body,
        "clean_sash_glass",
        (1.09, 0.003, 0.31),
        (0, -0.335, 0.57),
        (0.61, 0.80, 0.85, 0.16),
        True,
    )
    _line(
        body,
        "clean_sash_handle",
        (-1.06, -0.354, 0.258),
        (1.06, -0.354, 0.258),
        0.012,
        METAL,
    )
    _box(
        body,
        "clean_control_fascia",
        (0.25, 0.008, 0.076),
        (0.62, -0.358, 1.13),
        (0.64, 0.66, 0.61, 1),
    )
    for index in range(5):
        _geom(
            body,
            f"clean_button_{index}",
            "cylinder",
            (0.010, 0.005),
            (0.46 + index * 0.048, -0.37, 1.13),
            (0.24, 0.33, 0.26, 1) if index < 4 else (0.63, 0.40, 0.15, 1),
            euler=_numbers((math.pi / 2, 0, 0)),
        )
    for index in range(12):
        _box(
            body,
            f"clean_front_vent_{index}",
            (0.055, 0.003, 0.002),
            (-0.97 + index * 0.177, -0.344, 0.052),
            DARK,
        )
    # The reference has bottles, orange tube racks and tissue packs in the cabinet.
    for index in range(4):
        _bottle(
            body,
            f"clean_bottle_{index}",
            (-0.88 + index * 0.13, 0.11, 0.025),
            radius=0.033,
            height=0.10 + index % 2 * 0.045,
            color=(0.78, 0.86, 0.84, 0.34),
        )
    _box(
        body,
        "clean_tube_rack",
        (0.11, 0.075, 0.05),
        (0.18, 0.11, 0.075),
        (0.79, 0.28, 0.08, 1),
    )
    for index in range(8):
        _geom(
            body,
            f"clean_rack_tube_{index}",
            "cylinder",
            (0.009, 0.033),
            (0.105 + index % 4 * 0.048, 0.075 + index // 4 * 0.06, 0.123),
            GLASS,
        )
    _box(
        body,
        "clean_wipe_pack",
        (0.11, 0.067, 0.041),
        (0.58, 0.07, 0.066),
        (0.40, 0.66, 0.54, 1),
    )
    _box(
        body,
        "clean_wipe_label",
        (0.078, 0.001, 0.022),
        (0.58, 0.002, 0.066),
        (0.88, 0.90, 0.80, 1),
    )
    # Neutral visible inspection lamp. No active UV/decontamination simulation.
    _box(
        body,
        "luminaire_clean_work",
        (0.85, 0.035, 0.008),
        (0, 0.08, 0.90),
        (0.90, 0.96, 1, 1),
    )


def _paperwork(parent, name, position, yaw=0):
    body = _body(parent, name, position, yaw)
    for index in range(3):
        _box(
            body,
            f"{name}_page_{index}",
            (0.10, 0.145, 0.0005),
            (index * 0.009, index * 0.006, 0.001 + index * 0.0015),
            (0.86, 0.86, 0.77, 1),
        )
    for index in range(5):
        _box(
            body,
            f"{name}_print_{index}",
            (0.063, 0.0009, 0.0002),
            (0.018, 0.075 - index * 0.022, 0.005),
            (0.40, 0.43, 0.42, 1),
        )


def _rochester(world):
    _ceiling_lights(
        world,
        "genomics",
        [(x, y, 2.79) for x in (-2.4, 0, 2.4) for y in (-0.6, 0.6)],
        (0.55, 0.23),
    )
    _clean_work_enclosure(world)
    _service_raceway(world, "genomics_utility", (-0.9, 1.80, 1.20), 5.4)
    _paperwork(world, "genomics_protocol", (0.47, 1.15, 0.92), 0.11)
    # Long overbench shelving retains >= 0.27 m overhead to the Flex surrogate.
    for column, x in enumerate((-3.61, -0.95, 1.70)):
        _box(
            world,
            f"genomics_shelf_post_{column}",
            (0.022, 0.028, 0.88),
            (x, 1.76, 1.80),
            METAL,
            True,
        )
    for level, z in enumerate((2.06, 2.55)):
        _box(
            world,
            f"genomics_shelf_{level}",
            (2.68, 0.20, 0.025),
            (-0.95, 1.59, z),
            (0.72, 0.69, 0.58, 1),
            True,
        )
        for index in range(12):
            x = -3.37 + index * 0.44
            height = (0.13, 0.09, 0.17)[index % 3]
            color = (
                (0.76, 0.73, 0.58, 1),
                (0.54, 0.68, 0.75, 1),
                (0.85, 0.84, 0.78, 1),
            )[index % 3]
            _box(
                world,
                f"genomics_stock_{level}_{index}",
                (0.13, 0.12, height / 2),
                (x, 1.59, z + 0.025 + height / 2),
                color,
            )
            _box(
                world,
                f"genomics_stock_label_{level}_{index}",
                (0.06, 0.002, 0.02),
                (x, 1.468, z + 0.025 + height / 2),
                (0.91, 0.91, 0.87, 1),
            )
    _monitor(world, "genomics_terminal", (0.94, 1.27, 0.92))
    _cold_storage(world, "genomics_cold_storage", (3.53, -1.23, 0), 0.84, 0.75)
    for index in range(5):
        _bottle(
            world,
            f"genomics_buffer_{index}",
            (-2.85 + index * 0.12, -0.87, 0.92),
            height=0.12 + index % 2 * 0.04,
        )
    # A wood storage bank is visible in the photograph; exact cabinet count is inferred.
    side = _body(world, "genomics_side_cupboards", (-3.88, -0.70, 2.15), math.pi / 2)
    _storage_cabinet(side, "genomics_side_storage", (0, 0, 0), 1.5, WOOD, False)


def _isotope_partition(world):
    # Glazing is a reconstruction cutaway convention, not observed building fabric.
    for name, low, high in (("front", -2.9, -0.70), ("back", 0.50, 2.9)):
        y, half = (low + high) / 2, (high - low) / 2
        _box(
            world,
            f"isotope_partition_{name}_base",
            (0.06, half, 0.53),
            (0.30, y, 0.53),
            (0.74, 0.76, 0.70, 1),
            True,
        )
        _box(
            world,
            f"isotope_partition_{name}_glass",
            (0.014, half - 0.03, 0.79),
            (0.30, y, 1.91),
            (0.68, 0.78, 0.78, 0.16),
            True,
        )
        for edge in (low, high):
            _box(
                world,
                f"isotope_partition_{name}_frame_{edge}",
                (0.035, 0.03, 1.52),
                (0.30, edge, 1.52),
                METAL,
                True,
            )
        _box(
            world,
            f"isotope_partition_{name}_header",
            (0.04, half, 0.12),
            (0.30, y, 2.91),
            (0.74, 0.76, 0.70, 1),
            True,
        )
    _box(
        world,
        "isotope_link_header",
        (0.06, 0.6, 0.38),
        (0.30, -0.10, 2.67),
        (0.74, 0.76, 0.70, 1),
        True,
    )


def _dial(parent, name, position, radius=0.04):
    """Visual pressure gauge: its needle is not a simulated instrument reading."""
    body = _body(parent, name, position)
    for suffix, r, y, color in (
        ("rim", radius, 0, METAL),
        ("face", radius * 0.87, -0.011, (0.90, 0.88, 0.76, 1)),
    ):
        _geom(
            body,
            f"{name}_{suffix}",
            "cylinder",
            (r, 0.006),
            (0, y, 0),
            color,
            euler=_numbers((math.pi / 2, 0, 0)),
        )
    for tick in range(9):
        angle = math.radians(35 + tick * 27.5)
        _line(
            body,
            f"{name}_tick_{tick}",
            (radius * 0.65 * math.cos(angle), -0.019, radius * 0.65 * math.sin(angle)),
            (radius * 0.77 * math.cos(angle), -0.019, radius * 0.77 * math.sin(angle)),
            radius * 0.014,
            DARK,
        )
    _line(
        body,
        name + "_needle",
        (0, -0.022, 0),
        (-radius * 0.43, -0.022, radius * 0.43),
        radius * 0.025,
        DARK,
    )


def _vacuum_line(world):
    # Support rack, branches and shields follow the preparation-video silhouette.
    for index, x in enumerate((1.43, 2.65, 3.93)):
        _line(
            world,
            f"vacuum_post_{index}",
            (x, 2.32, 0.94),
            (x, 2.32, 2.65),
            0.018,
            METAL,
            True,
        )
    for level, z in enumerate((1.32, 1.88, 2.50)):
        _line(
            world,
            f"vacuum_crossbar_{level}",
            (1.43, 2.32, z),
            (3.93, 2.32, z),
            0.015,
            METAL,
            True,
        )
    for level, z in enumerate((1.88, 2.27)):
        _line(
            world,
            f"vacuum_manifold_{level}",
            (1.53, 2.13, z),
            (3.83, 2.13, z),
            0.014,
            (0.73, 0.88, 0.87, 0.58),
        )
        for branch in range(5):
            x = 1.68 + branch * 0.48
            _line(
                world,
                f"vacuum_branch_{level}_{branch}",
                (x, 2.13, z),
                (x, 2.13, z - 0.20),
                0.008,
                (0.71, 0.89, 0.87, 0.63),
            )
            _geom(
                world,
                f"vacuum_valve_{level}_{branch}",
                "cylinder",
                (0.033, 0.009),
                (x, 2.10, z - 0.10),
                (0.18, 0.32, 0.38, 1),
                euler=_numbers((math.pi / 2, 0, 0)),
            )
            _geom(
                world,
                f"vacuum_trap_{level}_{branch}",
                "capsule",
                (0.037, 0.06),
                (x, 2.13, z - 0.26),
                (0.67, 0.80, 0.78, 0.62),
            )
    for index, x in enumerate((2.18, 2.80, 3.42)):
        _geom(
            world,
            f"vacuum_dewar_{index}",
            "cylinder",
            (0.10, 0.17),
            (x, 2.15, 1.09),
            (0.71, 0.73, 0.70, 1),
        )
        _box(
            world,
            f"vacuum_shield_{index}",
            (0.18, 0.016, 0.23),
            (x, 2.02, 1.42),
            (0.63, 0.67, 0.65, 1),
        )
    for index, x in enumerate((1.75, 2.65, 3.52)):
        # Rack-mounted controllers and feed wiring are visible in the prep still.
        _box(
            world,
            f"vacuum_controller_{index}",
            (0.10, 0.052, 0.064),
            (x, 2.245, 2.43),
            (0.75, 0.77, 0.71, 1),
        )
        _box(
            world,
            f"vacuum_controller_display_{index}",
            (0.041, 0.002, 0.023),
            (x - 0.028, 2.191, 2.435),
            DARK,
        )
        _geom(
            world,
            f"vacuum_controller_knob_{index}",
            "cylinder",
            (0.018, 0.008),
            (x + 0.061, 2.185, 2.43),
            (0.21, 0.23, 0.22, 1),
            euler=_numbers((math.pi / 2, 0, 0)),
        )
        _line(
            world,
            f"vacuum_feed_{index}",
            (x, 2.26, 2.36),
            (x + 0.11, 2.24, 1.89),
            0.005,
            (0.14, 0.24, 0.18, 1),
        )
    _dial(world, "vacuum_header_gauge", (3.75, 2.07, 2.31), 0.045)
    pump = _body(world, "vacuum_pump", (0.95, 1.75, 0))
    _box(pump, "vacuum_pump_base", (0.20, 0.23, 0.06), (0, 0, 0.06), DARK, True)
    _geom(
        pump,
        "vacuum_pump_motor",
        "cylinder",
        (0.12, 0.18),
        (0, 0, 0.24),
        (0.29, 0.37, 0.35, 1),
        True,
        euler=_numbers((math.pi / 2, 0, 0)),
    )
    _line(
        world,
        "vacuum_pump_hose",
        (1.04, 1.97, 0.30),
        (1.45, 2.25, 1.36),
        0.013,
        (0.24, 0.26, 0.26, 1),
    )


def _isotope(world):
    _isotope_partition(world)
    _vacuum_line(world)
    _monitor(world, "isotope_analysis_terminal", (-1.00, 1.93, 0.92))
    for index, x in enumerate((-0.90, -0.47)):
        _geom(
            world,
            f"isotope_gas_{index}",
            "capsule",
            (0.12, 0.53),
            (x, 2.73, 0.68),
            ((0.18, 0.40, 0.38, 1), (0.35, 0.15, 0.22, 1))[index],
            True,
        )
        _dial(world, f"isotope_gas_regulator_gauge_{index}", (x + 0.055, 2.59, 1.46))
        _line(
            world,
            f"isotope_gas_regulator_body_{index}",
            (x, 2.73, 1.39),
            (x, 2.59, 1.39),
            0.021,
            (0.61, 0.54, 0.32, 1),
        )
        _geom(
            world,
            f"isotope_gas_valve_{index}",
            "cylinder",
            (0.025, 0.04),
            (x, 2.73, 1.37),
            (0.64, 0.56, 0.32, 1),
        )
        _box(
            world,
            f"isotope_gas_restraint_{index}",
            (0.14, 0.018, 0.025),
            (x, 2.59, 0.98),
            METAL,
        )
        _line(
            world,
            f"isotope_gas_line_{index}",
            (x, 2.73, 1.40),
            (-1.55, 2.62, 1.47 + index * 0.06),
            0.007,
            (0.46, 0.39, 0.28, 1),
        )
    for index in range(6):
        _bottle(
            world,
            f"isotope_standard_{index}",
            (-3.95 + index * 0.10, 1.70, 0.92),
            radius=0.022,
            height=0.075,
        )
    _box(
        world, "isotope_gas_rack", (0.41, 0.025, 0.54), (-0.69, 2.87, 0.66), METAL, True
    )


def _pathology(world):
    _ceiling_lights(
        world,
        "pathology",
        [
            (x, y, 2.745)
            for x in (-0.55, 0.55)
            for y in (-3.05, -1.82, -0.59, 0.64, 1.87, 3.10)
        ],
        (0.18, 0.595),
    )
    for side, x, yaw in (("left", -1.60, math.pi / 2), ("right", 1.60, -math.pi / 2)):
        bank = _body(world, f"pathology_{side}_cupboards", (x, -0.4, 2.16), yaw)
        for index in range(5):
            _storage_cabinet(
                bank,
                f"pathology_{side}_cabinet_{index}",
                (-2.44 + index * 1.22, 0, 0),
                1.18,
                (0.38, 0.43, 0.39, 1),
                open_bay=(index == 1 and side == "left"),
            )
        _service_raceway(world, f"pathology_{side}_utility", (x, -0.4, 1.13), 5.95, yaw)
    _paperwork(world, "pathology_request_forms", (-1.18, 0.95, 0.92), math.pi / 2)
    _paperwork(world, "pathology_accession_forms", (1.15, -1.63, 0.92), -math.pi / 2)
    _monitor(world, "pathology_accession_terminal", (1.38, -1.07, 0.92), -math.pi / 2)
    _monitor(world, "pathology_review_terminal", (-1.38, 0.40, 0.92), math.pi / 2)
    _cold_storage(world, "pathology_rear_storage", (1.29, 3.99, 0), 0.68, 0.73, 1.91)
    for group, (x, y) in enumerate(((-1.30, 2.0), (1.30, -2.80), (1.30, 1.80))):
        _box(
            world,
            f"pathology_rack_{group}",
            (0.13, 0.18, 0.024),
            (x, y, 0.945),
            (0.60, 0.69, 0.72, 1),
        )
        for index in range(8):
            _bottle(
                world,
                f"pathology_vial_{group}_{index}",
                (x - 0.075 + index % 3 * 0.075, y - 0.11 + index // 3 * 0.11, 0.97),
                radius=0.020,
                height=0.055,
                color=(0.75, 0.76, 0.68, 0.6),
            )
    # A single work stool is offset from the central circulation strip.
    seat = _body(world, "pathology_stool", (-0.51, 0.55, 0))
    _geom(
        seat,
        "pathology_stool_seat",
        "cylinder",
        (0.18, 0.035),
        (0, 0, 0.56),
        (0.33, 0.25, 0.29, 1),
        True,
    )
    _geom(
        seat,
        "pathology_stool_post",
        "cylinder",
        (0.024, 0.26),
        (0, 0, 0.28),
        METAL,
        True,
    )
    _line(
        seat,
        "pathology_chair_back_support",
        (0.14, 0, 0.39),
        (0.21, 0, 0.87),
        0.018,
        DARK,
        True,
    )
    _box(
        seat,
        "pathology_chair_back_cushion",
        (0.032, 0.16, 0.12),
        (0.21, 0, 0.85),
        (0.33, 0.25, 0.29, 1),
        True,
    )
    for leg in range(5):
        angle = leg * 2 * math.pi / 5
        _line(
            seat,
            f"pathology_stool_leg_{leg}",
            (0, 0, 0.10),
            (0.22 * math.cos(angle), 0.22 * math.sin(angle), 0.05),
            0.013,
            METAL,
            True,
        )


def add_features(root, world, definition):
    """Add each facility's context without altering equipment interfaces."""
    functions = {
        "rochester_genomics": _rochester,
        "uw_isolab": _isotope,
        "penn_pathology": _pathology,
    }
    function = functions.get(definition["id"])
    if function is not None:
        function(world)
