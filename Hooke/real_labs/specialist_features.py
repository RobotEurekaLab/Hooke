"""Workflow-informed service details for four specialist laboratory designs.

These original geometric fixtures are not scanned facility assets. Their role is
spatial context: no ventilation, vacuum, power or optical isolation is simulated.
Large fixtures retain contact geometry; thin cables and trim are visual only.
"""

import math
import xml.etree.ElementTree as ET


METAL = (0.52, 0.57, 0.60, 1)
DARK = (0.09, 0.105, 0.115, 1)
SHELL = (0.73, 0.75, 0.72, 1)
BLUE = (0.10, 0.24, 0.32, 1)


def _numbers(values):
    return " ".join(f"{value:.7g}" for value in values)


def _geom(world, key, shape, size, position, color, *, collision=False, **extra):
    return ET.SubElement(
        world,
        "geom",
        name=f"specialist_{key}",
        type=shape,
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        **extra,
    )


def _box(world, key, position, halfsize, color, **extra):
    return _geom(world, key, "box", halfsize, position, color, **extra)


def _rod(world, key, start, end, radius, color, *, collision=False):
    return ET.SubElement(
        world,
        "geom",
        name=f"specialist_{key}",
        type="cylinder",
        size=str(radius),
        fromto=_numbers((*start, *end)),
        rgba=_numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


def _line(world, key, points, radius, color):
    for index, (start, end) in enumerate(zip(points, points[1:])):
        _rod(world, f"{key}_{index}", start, end, radius, color)


def _monitor(world, key, x, y, z):
    _box(world, f"{key}_foot", (x, y, z + 0.012), (0.14, 0.10, 0.012), DARK)
    _box(world, f"{key}_neck", (x, y + 0.055, z + 0.13), (0.025, 0.022, 0.12), METAL)
    _box(world, f"{key}_shell", (x, y + 0.05, z + 0.35), (0.265, 0.022, 0.16), DARK)
    _box(world, f"{key}_screen", (x, y + 0.024, z + 0.35), (0.252, 0.002, 0.147), BLUE)
    _box(world, f"{key}_keyboard", (x, y - 0.13, z + 0.012), (0.18, 0.065, 0.012), DARK)


def _stool(world, key, x, y, height=0.51):
    _geom(
        world,
        f"{key}_seat",
        "cylinder",
        (0.19, 0.025),
        (x, y, height),
        DARK,
        collision=True,
    )
    _rod(world, f"{key}_post", (x, y, 0.14), (x, y, height - 0.025), 0.028, METAL)
    for index in range(5):
        angle = index * math.tau / 5
        foot = (x + 0.25 * math.cos(angle), y + 0.25 * math.sin(angle), 0.035)
        _rod(world, f"{key}_leg_{index}", (x, y, 0.14), foot, 0.018, DARK)
        _geom(world, f"{key}_caster_{index}", "sphere", (0.028,), foot, DARK)


def _cylinder_station(world, key, x, y, color):
    _box(world, f"{key}_base", (x, y, 0.025), (0.19, 0.18, 0.025), DARK, collision=True)
    _geom(
        world,
        f"{key}_tank",
        "capsule",
        (0.13, 0.58),
        (x, y, 0.75),
        color,
        collision=True,
    )
    _rod(world, f"{key}_neck", (x, y, 1.42), (x, y, 1.5), 0.033, METAL)
    _rod(world, f"{key}_regulator", (x, y, 1.47), (x, y - 0.15, 1.47), 0.025, METAL)
    _geom(
        world,
        f"{key}_gauge",
        "cylinder",
        (0.045, 0.012),
        (x, y - 0.13, 1.55),
        SHELL,
        euler="1.5707963 0 0",
    )
    for height in (0.53, 1.05):
        _box(
            world,
            f"{key}_strap_{height}",
            (x, y - 0.127, height),
            (0.132, 0.018, 0.023),
            DARK,
        )
        _box(
            world,
            f"{key}_bracket_{height}",
            (x, y + 0.13, height),
            (0.18, 0.04, 0.025),
            METAL,
        )
    _rod(world, f"{key}_rack", (x, y + 0.18, 0.05), (x, y + 0.18, 1.3), 0.025, METAL)


def _vials(world, key, x, y, z, count=6):
    _box(world, f"{key}_rack", (x, y, z + 0.014), (count * 0.017, 0.045, 0.014), SHELL)
    for row in range(2):
        for col in range(count):
            px, py = x + (col - (count - 1) / 2) * 0.032, y + (row - 0.5) * 0.04
            _geom(
                world,
                f"{key}_vial_{row}_{col}",
                "cylinder",
                (0.01, 0.028),
                (px, py, z + 0.045),
                (0.63, 0.72, 0.72, 0.45),
            )
            _geom(
                world,
                f"{key}_cap_{row}_{col}",
                "cylinder",
                (0.0115, 0.005),
                (px, py, z + 0.077),
                (0.75, 0.62, 0.17, 1),
            )


def _imaging_alcove(world):
    # Three sides remain open to the operator and the room camera.
    for x in (-2.76, -0.33):
        _rod(
            world,
            f"curtain_post_{x}",
            (x, 1.46, 0),
            (x, 1.46, 2.4),
            0.026,
            METAL,
            collision=True,
        )
    _rod(
        world, "curtain_back_rail", (-2.76, 1.46, 2.4), (-0.33, 1.46, 2.4), 0.025, METAL
    )
    _rod(
        world, "curtain_left_rail", (-2.76, 0.05, 2.4), (-2.76, 1.46, 2.4), 0.025, METAL
    )
    for index in range(27):
        _box(
            world,
            f"curtain_back_{index}",
            (-2.72 + index * 0.09, 1.46 + 0.016 * (index % 2), 1.60),
            (0.049, 0.022, 0.75),
            (0.055, 0.065, 0.075, 1),
        )
    for index in range(14):
        _box(
            world,
            f"curtain_side_{index}",
            (-2.76, 0.18 + index * 0.09, 1.6),
            (0.025, 0.049, 0.75),
            (0.065, 0.075, 0.085, 1),
        )
    _monitor(world, "imaging_console", -1.0, -1.2, 0.75)
    _stool(world, "imaging", -1.2, -0.34)
    _box(
        world,
        "vacuum_service",
        (-1.45, -1.15, 0.23),
        (0.25, 0.2, 0.18),
        SHELL,
        collision=True,
    )
    for index in range(8):
        _box(
            world,
            f"vacuum_vent_{index}",
            (-1.65 + index * 0.055, -1.353, 0.24),
            (0.013, 0.003, 0.09),
            DARK,
        )
    _line(
        world,
        "vacuum_line",
        [
            (-1.67, -1.15, 0.35),
            (-2.92, -1.15, 0.35),
            (-2.92, 1.2, 0.35),
            (-2.92, 1.2, 1.03),
            (-1.25, 1.2, 1.03),
            (-1.22, 0.8, 1.01),
        ],
        0.009,
        (0.26, 0.31, 0.32, 1),
    )
    _vials(world, "medium", 2.65, 0.5, 0.9, 4)
    _box(world, "prep_splash", (3.05, 0, 1.07), (0.025, 1.45, 0.17), SHELL)


def _mechatronics_bay(world):
    _monitor(world, "microbot_console", 0.97, 1.89, 0.85)
    # Under-table cable trays and a wall-side route keep leads off the specimen.
    _box(world, "optical_cable_tray", (-0.65, 0.33, 0.64), (0.91, 0.065, 0.035), DARK)
    _box(world, "electronics_cable_tray", (-0.3, 2.23, 0.72), (1.85, 0.05, 0.03), DARK)
    for index, color in enumerate(((0.27, 0.08, 0.06, 1), DARK, (0.13, 0.21, 0.29, 1))):
        offset = index * 0.014
        _line(
            world,
            f"microbot_signal_{index}",
            [
                (-0.3, 0.04 + offset, 0.9),
                (-0.1, 0.32 + offset, 0.65),
                (-1.7, 0.32 + offset, 0.65),
                (-2.5, 0.32 + offset, 0.12),
                (-2.5, 2.2 + offset, 0.12),
                (-2.5, 2.2 + offset, 0.89),
                (-0.1, 2.2 + offset, 0.89),
            ],
            0.005,
            color,
        )
    _box(world, "daq_case", (0.98, 1.9, 0.38), (0.24, 0.23, 0.31), DARK, collision=True)
    for index in range(5):
        _box(
            world,
            f"daq_module_{index}",
            (0.98, 1.665, 0.2 + index * 0.095),
            (0.21, 0.012, 0.035),
            METAL,
        )
    _stool(world, "microbot", -0.65, -1.46)
    _vials(world, "microbot_medium", 2.2, -0.2, 0.9, 3)
    # Rail-fixed ancillary breadboards, not a second claimed active mechanism.
    _box(world, "fixture_plate", (-1.38, -0.25, 0.825), (0.20, 0.21, 0.025), METAL)
    for index in range(4):
        for row in range(4):
            _geom(
                world,
                f"fixture_bolt_{row}_{index}",
                "cylinder",
                (0.004, 0.0015),
                (-1.53 + index * 0.1, -0.4 + row * 0.1, 0.852),
                DARK,
            )


def _process_bay(world):
    _cylinder_station(world, "process_argon", -3.45, 1.77, (0.28, 0.37, 0.40, 1))
    _cylinder_station(world, "process_oxygen", -3.05, 1.77, (0.35, 0.39, 0.37, 1))
    _box(
        world,
        "vacuum_pump_base",
        (-2.43, 1.75, 0.065),
        (0.35, 0.25, 0.065),
        DARK,
        collision=True,
    )
    _rod(
        world,
        "vacuum_motor",
        (-2.68, 1.75, 0.26),
        (-2.17, 1.75, 0.26),
        0.135,
        BLUE,
        collision=True,
    )
    _box(
        world,
        "vacuum_pump_head",
        (-2.1, 1.75, 0.27),
        (0.12, 0.16, 0.17),
        METAL,
        collision=True,
    )
    _line(
        world,
        "roughing_line",
        [
            (-1.27, 0.47, 1.04),
            (-1.0, 0.95, 0.82),
            (-1.0, 1.28, 0.4),
            (-2.1, 1.28, 0.4),
            (-2.1, 1.75, 0.4),
        ],
        0.03,
        METAL,
    )
    _box(
        world, "chiller", (-1.38, 1.82, 0.44), (0.26, 0.25, 0.44), SHELL, collision=True
    )
    for index in range(10):
        _box(
            world,
            f"chiller_vent_{index}",
            (-1.38, 1.565, 0.13 + index * 0.055),
            (0.20, 0.007, 0.008),
            DARK,
        )
    _box(world, "chiller_display", (-1.38, 1.562, 0.78), (0.075, 0.006, 0.034), BLUE)
    _box(
        world,
        "process_rack",
        (-0.25, 1.82, 0.91),
        (0.31, 0.3, 0.91),
        DARK,
        collision=True,
    )
    for index, height in enumerate((0.22, 0.5, 0.82, 1.1, 1.4, 1.68)):
        _box(
            world,
            f"process_module_{index}",
            (-0.25, 1.508, height),
            (0.266, 0.018, 0.10),
            METAL,
        )
        _box(
            world,
            f"process_display_{index}",
            (-0.32, 1.485, height),
            (0.10, 0.004, 0.035),
            BLUE,
        )
        for button in range(3):
            _geom(
                world,
                f"process_knob_{index}_{button}",
                "cylinder",
                (0.017, 0.008),
                (-0.08 + button * 0.05, 1.474, height),
                DARK,
                euler="1.5707963 0 0",
            )
    _line(
        world,
        "process_gas",
        [
            (-3.45, 1.62, 1.47),
            (-3.45, 2.25, 1.55),
            (-1.85, 2.25, 1.55),
            (-1.85, 0.22, 1.25),
        ],
        0.009,
        (0.47, 0.44, 0.32, 1),
    )
    # Footprint markers designate this authored machinery area, not regulations.
    for x in (-2.91, -0.34):
        _box(
            world,
            f"skid_marking_x_{x}",
            (x, 0.18, 0.002),
            (0.025, 0.9, 0.001),
            (0.76, 0.61, 0.18, 1),
        )
    _box(
        world,
        "skid_marking_front",
        (-1.625, -0.72, 0.002),
        (1.31, 0.025, 0.001),
        (0.76, 0.61, 0.18, 1),
    )
    _line(
        world,
        "coolant_supply",
        [
            (-1.18, 1.56, 0.82),
            (-0.85, 1.2, 0.8),
            (-0.85, 0.55, 0.77),
            (-1.2, 0.45, 0.95),
        ],
        0.012,
        (0.13, 0.26, 0.39, 1),
    )
    _box(world, "prep_splash", (3.72, 0.1, 1.12), (0.02, 1.5, 0.20), SHELL)


def _sampling_bay(world):
    _cylinder_station(world, "sampling_gas", -2.98, 1.48, (0.24, 0.31, 0.29, 1))
    # Source documents a duct; canopy dimensions and route are authored.
    _box(
        world,
        "capture_canopy",
        (-1.43, 1.3, 2.12),
        (0.73, 0.38, 0.075),
        SHELL,
        collision=True,
    )
    _box(world, "capture_inlet", (-1.43, 1.3, 2.04), (0.62, 0.28, 0.012), DARK)
    _line(
        world,
        "extraction_duct",
        [
            (-1.43, 1.3, 2.2),
            (-1.43, 1.3, 2.55),
            (-1.43, 1.83, 2.55),
            (-1.43, 1.83, 2.95),
        ],
        0.11,
        METAL,
    )
    for index in range(5):
        _geom(
            world,
            f"duct_band_{index}",
            "cylinder",
            (0.118, 0.012),
            (-1.43, 1.83, 2.61 + index * 0.068),
            DARK,
        )
    for x in (-2.05, -0.81):
        _rod(world, f"canopy_hanger_{x}", (x, 1.5, 2.21), (x, 1.5, 2.9), 0.012, METAL)
    _box(
        world, "sampling_service_rail", (-0.15, 1.81, 1.22), (2.38, 0.038, 0.065), SHELL
    )
    for index in range(6):
        _box(
            world,
            f"service_outlet_{index}",
            (-2.23 + index * 0.78, 1.763, 1.22),
            (0.042, 0.012, 0.042),
            DARK,
        )
    _vials(world, "sampling", -0.65, 1.29, 0.9)
    _vials(world, "collection", 1.15, 1.45, 0.9, 5)
    _box(world, "filter_carrier", (2.7, -0.67, 0.84), (0.18, 0.15, 0.02), SHELL)
    _line(
        world,
        "sampling_gas_line",
        [
            (-2.98, 1.35, 1.48),
            (-2.7, 1.75, 1.48),
            (-1.65, 1.75, 1.48),
            (-1.65, 1.38, 1.15),
        ],
        0.007,
        (0.25, 0.32, 0.35, 1),
    )
    _monitor(world, "sampling_console", 1.75, 1.37, 0.9)


_FEATURES = {
    "indiana_microfluidics": _imaging_alcove,
    "epfl_microbiorobotics": _mechatronics_bay,
    "uva_deposition": _process_bay,
    "eth_air_quality": _sampling_bay,
}


def add_features(root, world, definition):
    """Add fixtures for one known layout without modifying instrument models."""
    build = _FEATURES.get(definition["id"])
    if build is not None:
        build(world)
