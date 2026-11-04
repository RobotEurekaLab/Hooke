"""Passive details from Birmingham's photographed installed EDM workroom.

The foreground console, perforated guard, red/black chair, cabinet and service
routes reproduce visible cues. Dimensions and the unidentified secondary tool's
function are estimates. These fixed objects add no operational or OEM claims.
All authored solids are enabled; guard apertures are actual empty spaces.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

PREFIX = "edm_source_"
CREAM = (0.86, 0.85, 0.77, 1)
METAL = (0.47, 0.50, 0.49, 1)
DARK = (0.045, 0.05, 0.045, 1)
RED = (0.50, 0.028, 0.025, 1)


def _geom(parent, key, kind, size, pos=None, color=CREAM, **attrs):
    values = dict(
        name=PREFIX + key,
        type=kind,
        size=numbers(size),
        rgba=numbers(color),
        density="0",
        contype="1",
        conaffinity="1",
    )
    if pos is not None:
        values["pos"] = numbers(pos)
    values.update(attrs)
    return ET.SubElement(parent, "geom", **values)


def _box(parent, key, pos, half, color=CREAM, **attrs):
    return _geom(parent, key, "box", half, pos, color, **attrs)


def _rod(parent, key, a, b, radius=0.009, color=METAL):
    return _geom(
        parent, key, "capsule", (radius,), color=color, fromto=numbers((*a, *b))
    )


def _cylinder(parent, key, pos, radius, halfheight, color=METAL, **attrs):
    return _geom(parent, key, "cylinder", (radius, halfheight), pos, color, **attrs)


def add_apparatus_context(b, base):
    # Raised grille treatment over the retained solid roof. This is a visible
    # cover, not a claim of flow through the unchanged machine roof.
    _box(base, "roof_recess", (0, 0, 1.918), (0.635, 0.445, 0.003), DARK)
    xs = [-0.625 + 0.05 * i for i in range(26)]
    ys = [-0.435 + 0.087 * i for i in range(11)]
    for i, x in enumerate(xs):
        _box(base, f"roof_long_rib_{i}", (x, 0, 1.925), (0.004, 0.445, 0.004))
    # Butt cross-ribs into the long ribs, avoiding coplanar intersecting faces.
    for i, y in enumerate(ys):
        for j in range(len(xs) - 1):
            _box(
                base,
                f"roof_cross_rib_{i}_{j}",
                ((xs[j] + xs[j + 1]) / 2, y, 1.925),
                (0.021, 0.004, 0.004),
            )
    for row in range(3):
        for col in range(8):
            _box(
                base,
                f"main_key_{row}_{col}",
                (0.69 + col * 0.045, -0.285 + row * 0.041, 1.001),
                (0.015, 0.012, 0.006),
                (0.74, 0.73, 0.63, 1),
            )
    # Small status/selector cluster on the spare right edge of the desk.
    for i, color in enumerate(((0.65, 0.16, 0.08, 1), (0.13, 0.28, 0.10, 1))):
        _cylinder(
            base,
            f"main_selector_{i}",
            (1.17, -0.38 + i * 0.075, 1.004),
            0.012,
            0.009,
            color,
        )
    # A supported faceplate gives the main exterior its photographed badge
    # placement without asserting a verified machine generation or model.
    _box(
        base,
        "front_badge_plate",
        (-0.38, -0.522, 0.62),
        (0.105, 0.002, 0.038),
        (0.75, 0.74, 0.67, 1),
    )
    for i in range(4):
        _box(
            base,
            f"front_badge_mark_{i}",
            (-0.45 + 0.046 * i, -0.525, 0.63),
            (0.012, 0.001, 0.013),
            DARK,
        )


def _chair(world):
    cx, cy = 0.59, -0.29
    # A parked static chair. Wheels, gas column and back straps form a physical
    # support chain; no chair dynamics, ergonomics or rated load is inferred.
    for i in range(5):
        a = 2 * math.pi * i / 5
        x, y = cx + 0.24 * math.cos(a), cy + 0.24 * math.sin(a)
        _cylinder(
            world,
            f"chair_castor_{i}",
            (x, y, 0.035),
            0.035,
            0.026,
            DARK,
            euler=numbers((math.pi / 2, 0, a)),
        )
        _rod(world, f"chair_spoke_{i}", (cx, cy, 0.105), (x, y, 0.060), 0.018, DARK)
    _cylinder(world, "chair_column", (cx, cy, 0.27), 0.035, 0.175, DARK)
    _box(world, "chair_seat_mount", (cx, cy, 0.445), (0.15, 0.14, 0.018), DARK)
    _geom(
        world, "chair_seat_red", "ellipsoid", (0.245, 0.228, 0.055), (cx, cy, 0.49), RED
    )
    _geom(
        world,
        "chair_seat_black",
        "ellipsoid",
        (0.224, 0.211, 0.040),
        (cx, cy - 0.008, 0.523),
        DARK,
    )
    for x in (cx - 0.14, cx + 0.14):
        _rod(
            world,
            f"chair_back_strap_{x}",
            (x, cy - 0.13, 0.45),
            (x, cy - 0.20, 0.77),
            0.014,
            DARK,
        )
    _geom(
        world,
        "chair_back_red",
        "ellipsoid",
        (0.235, 0.044, 0.24),
        (cx, cy - 0.208, 0.79),
        RED,
    )
    _geom(
        world,
        "chair_back_black",
        "ellipsoid",
        (0.215, 0.028, 0.215),
        (cx, cy - 0.237, 0.796),
        DARK,
    )


def _foreground(world):
    # Inclined secondary console. Transformation is applied to geoms directly:
    # there is no added body, joint or hidden moving attachment.
    origin = (-0.34, -1.18, 1.10)
    angle = 0.52
    c, s = math.cos(angle), math.sin(angle)

    def panel(key, p, half, color=CREAM):
        x, y, z = p
        return _box(
            world,
            "foreground_" + key,
            (origin[0] + x, origin[1] + c * y - s * z, origin[2] + s * y + c * z),
            half,
            color,
            euler=numbers((angle, 0, 0)),
        )

    for i, x in enumerate((-0.55, -0.13)):
        _box(
            world,
            f"foreground_foot_{i}",
            (x, -1.17, 0.012),
            (0.033, 0.22, 0.012),
            METAL,
        )
        _box(
            world,
            f"foreground_leg_{i}",
            (x, -1.17, 0.491),
            (0.024, 0.026, 0.467),
            METAL,
        )
    _box(
        world,
        "foreground_cross_support",
        (-0.34, -1.17, 0.936),
        (0.235, 0.085, 0.025),
        METAL,
    )
    panel("case", (0, 0, 0), (0.32, 0.23, 0.085))
    panel("control_plate", (0.175, 0, 0.088), (0.122, 0.218, 0.003), DARK)
    for row in range(6):
        for col in range(3):
            panel(
                f"key_{row}_{col}",
                (0.095 + col * 0.069, -0.15 + row * 0.052, 0.095),
                (0.025, 0.017, 0.004),
                (0.76, 0.73, 0.59, 1),
            )
    # Pendant box visibly attached to the front-right support; its controls are
    # passive source detail, not wired to the authored dry XY fixture.
    _box(
        world,
        "foreground_pendant_bracket",
        (-0.13, -1.247, 0.71),
        (0.025, 0.055, 0.02),
        METAL,
    )
    _box(world, "foreground_pendant", (-0.13, -1.31, 0.70), (0.043, 0.025, 0.13), DARK)
    _cylinder(
        world,
        "foreground_emergency_yellow",
        (-0.13, -1.34, 0.752),
        0.027,
        0.005,
        (0.87, 0.68, 0.08, 1),
        euler=numbers((math.pi / 2, 0, 0)),
    )
    _cylinder(
        world,
        "foreground_emergency_red",
        (-0.13, -1.352, 0.752),
        0.018,
        0.007,
        RED,
        euler=numbers((math.pi / 2, 0, 0)),
    )
    for i in range(4):
        _box(
            world,
            f"foreground_pendant_key_{i}",
            (-0.13, -1.338, 0.615 + i * 0.026),
            (0.019, 0.003, 0.008),
            (0.64, 0.65, 0.58, 1),
        )

    # The photo shows a perforated/mesh metal guard. Finite 3 mm strips on a
    # 10 mm pitch leave real 7 mm apertures, with no backing collision plane.
    gx, gy, yaw = 0.40, -1.26, 1.08
    c, s = math.cos(yaw), math.sin(yaw)

    def guard(key, p, half, color=METAL):
        x, y, z = p
        return _box(
            world,
            "guard_" + key,
            (gx + c * x - s * y, gy + s * x + c * y, z),
            half,
            color,
            euler=numbers((0, 0, yaw)),
        )

    for i, x in enumerate((-0.34, 0.34)):
        guard(f"foot_{i}", (x, 0, 0.012), (0.032, 0.15, 0.012))
        guard(f"post_{i}", (x, 0, 0.63), (0.018, 0.017, 0.606))
    for i, z in enumerate((0.515, 1.225)):
        guard(f"edge_{i}", (0, 0, z), (0.322, 0.018, 0.012))
    for i in range(64):
        guard(
            f"vertical_{i}", (-0.315 + i * 0.010, -0.003, 0.87), (0.0015, 0.0015, 0.343)
        )
    for i in range(69):
        guard(f"horizontal_{i}", (0, 0, 0.53 + i * 0.010), (0.322, 0.0015, 0.0015))
    # A cropped red basin surface behind the guard is visible in the source.
    # Its exact function is unknown; it stays a fixed empty tray on four legs.
    _box(
        world,
        "foreground_red_tray_bottom",
        (0.50, -1.26, 0.455),
        (0.14, 0.255, 0.015),
        RED,
    )
    for x in (0.37, 0.63):
        _box(
            world,
            f"foreground_tray_side_{x}",
            (x, -1.26, 0.51),
            (0.010, 0.255, 0.040),
            RED,
        )
    for y in (-1.505, -1.015):
        _box(
            world,
            f"foreground_tray_end_{y}",
            (0.50, y, 0.51),
            (0.12, 0.010, 0.040),
            RED,
        )
    for x in (0.39, 0.61):
        for y in (-1.47, -1.05):
            _box(
                world,
                f"foreground_tray_leg_{x}_{y}",
                (x, y, 0.22),
                (0.020, 0.020, 0.22),
                METAL,
            )


def _services(world):
    # Foreground beacon fixed to the retained secondary casing roof.
    _cylinder(world, "beacon_foot", (1.29, -1.23, 1.811), 0.065, 0.011, DARK)
    _cylinder(world, "beacon_stem", (1.29, -1.23, 1.884), 0.022, 0.063, CREAM)
    _cylinder(world, "beacon_base", (1.29, -1.23, 1.979), 0.055, 0.034, DARK)
    _cylinder(
        world,
        "beacon_amber",
        (1.29, -1.23, 2.056),
        0.055,
        0.043,
        (0.84, 0.31, 0.025, 1),
    )
    _cylinder(world, "beacon_cap", (1.29, -1.23, 2.106), 0.056, 0.007, DARK)
    # Rear-side electrical cupboard: no fabricated internal cabinet machinery.
    _box(world, "cabinet_plinth", (1.64, 1.10, 0.07), (0.26, 0.25, 0.07), DARK)
    _box(
        world,
        "cabinet_case",
        (1.64, 1.10, 0.995),
        (0.26, 0.25, 0.855),
        (0.61, 0.64, 0.65, 1),
    )
    _box(
        world,
        "cabinet_door",
        (1.64, 0.844, 1.00),
        (0.243, 0.006, 0.82),
        (0.65, 0.67, 0.67, 1),
    )
    _box(world, "cabinet_handle", (1.83, 0.829, 1.00), (0.009, 0.009, 0.06), DARK)
    # Enabled standoffs actually meet the inner face x=2 of the retained wall.
    _box(world, "service_wall_rail", (1.990, 0.78, 1.66), (0.010, 0.91, 0.022), METAL)
    for i, y in enumerate((0.15, 0.68, 1.30)):
        _box(
            world,
            f"service_outlet_back_{i}",
            (1.97, y, 1.52),
            (0.030, 0.061, 0.14),
            CREAM,
        )
        _box(
            world,
            f"service_outlet_face_{i}",
            (1.931, y, 1.50),
            (0.009, 0.048, 0.066),
            (0.72, 0.72, 0.64, 1),
        )
        for z in (1.475, 1.527):
            _box(
                world,
                f"service_socket_{i}_{z}",
                (1.920, y, z),
                (0.002, 0.012, 0.007),
                DARK,
            )
    route = [
        (1.92, 0.68, 1.50),
        (1.69, 0.69, 1.44),
        (1.43, 0.70, 1.30),
        (1.18, 0.72, 1.08),
        (1.02, 0.70, 0.94),
        (0.93, 0.65, 0.94),
    ]
    for i, (a, b) in enumerate(zip(route, route[1:])):
        _rod(world, f"service_console_lead_{i}", a, b, 0.007, DARK)
    # Second lead terminates against the rear of the adjacent cabinet.
    route = [
        (1.92, 1.30, 1.50),
        (1.91, 1.49, 1.58),
        (1.90, 1.63, 1.43),
        (1.88, 1.53, 1.22),
        (1.84, 1.34, 1.25),
    ]
    for i, (a, b) in enumerate(zip(route, route[1:])):
        _rod(world, f"service_cabinet_lead_{i}", a, b, 0.008, DARK)

    # Existing small computer cart receives a supported deep beige instrument
    # casing and side selectors. The unidentified screen gets no fake readout.
    _box(world, "computer_support", (-1.42, 0.25, 0.865), (0.06, 0.07, 0.105), METAL)
    _box(world, "computer_rear", (-1.42, 0.345, 1.02), (0.22, 0.105, 0.145), CREAM)
    _box(
        world,
        "computer_side_controls",
        (-1.157, 0.26, 1.02),
        (0.043, 0.085, 0.145),
        CREAM,
    )
    for i in range(4):
        _cylinder(
            world,
            f"computer_selector_{i}",
            (-1.157, 0.17, 0.93 + i * 0.058),
            0.014,
            0.006,
            DARK,
            euler=numbers((math.pi / 2, 0, 0)),
        )
    _box(
        world,
        "rear_table_top",
        (-1.52, 1.36, 0.755),
        (0.34, 0.34, 0.025),
        (0.52, 0.33, 0.17, 1),
    )
    for x in (-1.80, -1.24):
        for y in (1.08, 1.64):
            _box(
                world,
                f"rear_table_leg_{x}_{y}",
                (x, y, 0.365),
                (0.023, 0.023, 0.365),
                METAL,
            )
    _box(world, "rear_table_small_case", (-1.53, 1.47, 0.88), (0.20, 0.16, 0.10), CREAM)
    _box(
        world, "rear_table_case_face", (-1.53, 1.305, 0.88), (0.17, 0.005, 0.075), DARK
    )


def add_room_context(world, definition):
    _chair(world)
    _foreground(world)
    _services(world)
