"""Authored mechanical details informed by three historical Stoltz photographs.

No OEM dimensions, ventilation, chemical process or glove manipulation is implied.
Running clearances are explicit geometry around constrained inspection joints.
"""

import math


def _name(b, geom, key):
    geom.set("name", f"{b.name}__access_{key}")
    return geom


def _box(b, parent, key, pos, half, material="metal", **kwargs):
    return _name(b, b.box(parent, pos, half, material, collision=True, **kwargs), key)


def _rod(b, parent, key, start, end, radius=0.006, material="bright"):
    return _name(b, b.rod(parent, start, end, radius, material, collision=True), key)


def _drawer(b, base):
    drawer = b.moving(
        base,
        "supply_drawer",
        (0, -0.045, 0.754),
        (0, -1, 0),
        (0, 0.23),
        mass=1.1,
        kp=240,
    )
    _box(b, drawer, "drawer_plate", (0, 0, 0), (0.72, 0.35, 0.012))
    _box(b, drawer, "drawer_front", (0, -0.375, 0.017), (0.76, 0.018, 0.073), "cream")
    _rod(
        b, drawer, "drawer_handle", (-0.13, -0.408, 0.025), (0.13, -0.408, 0.025), 0.011
    )
    for x in (-0.11, 0.11):
        _rod(
            b,
            drawer,
            f"drawer_handle_mount_{x}",
            (x, -0.393, 0.025),
            (x, -0.408, 0.025),
            0.006,
        )
    # Cabinet -> rail pedestal -> rail; open shoes meet the plate above the rail.
    # The 1 mm running gap is deliberate, not an unexplained structural break.
    for side in (-1, 1):
        x = side * 0.69
        _box(
            b, base, f"rail_pedestal_{side}", (x, -0.04, 0.6845), (0.010, 0.36, 0.0295)
        )
        _box(
            b,
            base,
            f"drawer_rail_{side}",
            (x, -0.04, 0.721),
            (0.012, 0.36, 0.007),
            "bright",
        )
        for station, y in enumerate((-0.06, 0.22)):
            _box(
                b,
                drawer,
                f"shoe_roof_{side}_{station}",
                (x, y, -0.0185),
                (0.021, 0.045, 0.0065),
                "dark",
            )
            for flank in (-1, 1):
                _box(
                    b,
                    drawer,
                    f"shoe_flank_{side}_{station}_{flank}",
                    (x + flank * 0.017, y, -0.026),
                    (0.004, 0.045, 0.014),
                    "dark",
                )
    _box(
        b,
        drawer,
        "cassette_riser",
        (0.22, 0.04, 0.0185),
        (0.102, 0.077, 0.0065),
        "rubber",
    )
    _box(b, drawer, "cassette", (0.22, 0.04, 0.055), (0.09, 0.065, 0.03), "blue")
    for side in (-1, 1):
        _box(
            b,
            drawer,
            f"cassette_retainer_x_{side}",
            (0.22 + side * 0.096, 0.04, 0.039),
            (0.006, 0.071, 0.014),
            "bright",
        )
        _box(
            b,
            drawer,
            f"cassette_retainer_y_{side}",
            (0.22, 0.04 + side * 0.071, 0.039),
            (0.09, 0.006, 0.014),
            "bright",
        )
    b.site(drawer, "sample_carrier", (0.22, 0.04, 0.055))
    return drawer


def _sash(b, base):
    sash = b.moving(
        base, "sash", (0, -0.418, 1.35), (0, 0, 1), (0, 0.40), mass=2, kp=300, force=150
    )
    _box(b, sash, "sash_glass", (0, 0, 0), (0.746, 0.009, 0.365), "glass")
    for side in (-1, 1):
        x = side * 0.76
        _box(b, sash, f"sash_stile_{side}", (x, 0, 0), (0.009, 0.016, 0.38))
        # U channel is genuinely open around each stile: 1 mm lateral/rear/front.
        _box(
            b,
            base,
            f"sash_channel_side_{side}",
            (side * 0.785, -0.418, 1.58),
            (0.015, 0.037, 0.63),
            "bright",
        )
        for face in (-1, 1):
            y = -0.418 + face * 0.027
            _box(
                b,
                base,
                f"sash_channel_face_{side}_{face}",
                (side * 0.767, y, 1.58),
                (0.017, 0.01, 0.63),
                "bright",
            )
    for side in (-1, 1):
        _box(
            b, sash, f"sash_crossbar_{side}", (0, 0, side * 0.38), (0.767, 0.016, 0.012)
        )
    _rod(b, sash, "sash_handle", (-0.17, -0.046, -0.35), (0.17, -0.046, -0.35), 0.008)
    for x in (-0.15, 0.15):
        _rod(
            b,
            sash,
            f"sash_handle_mount_{x}",
            (x, -0.016, -0.38),
            (x, -0.046, -0.35),
            0.005,
        )
    b.site(sash, "sash_lower_edge", (0, -0.016, -0.392))
    b.site(base, "working_aperture", (0, -0.46, 1.10))
    return sash


def _apparatus(b, base, variant):
    """Inert, retained glassware arrangements; distinct shapes follow photo cues."""
    # Photo-supported metal rack and fixed utility manifold; routing is estimated.
    for x in (-0.70, 0.70):
        _box(b, base, f"rack_foot_{x}", (x, 0.30, 0.949), (0.040, 0.045, 0.014))
        _rod(b, base, f"rack_post_{x}", (x, 0.30, 0.963), (x, 0.30, 1.95), 0.008)
    for z in (1.36, 1.86):
        _rod(b, base, f"rack_crossbar_{z}", (-0.70, 0.30, z), (0.70, 0.30, z), 0.008)
    _rod(b, base, "utility_manifold", (-0.69, 0.34, 1.93), (0.69, 0.34, 1.93), 0.012)
    for index, x in enumerate((-0.47, 0, 0.44)):
        key = f"apparatus_{index}"
        height = (0.11, 0.075, 0.15)[(index + variant) % 3]
        radius = (0.054, 0.044, 0.040)[(index + variant) % 3]
        y = (0.04, -0.015, 0.08)[(index + variant) % 3]
        for dx in (-0.083, 0.083):
            for dy in (-0.087, 0.087):
                _box(
                    b,
                    base,
                    f"{key}_foot_{dx}_{dy}",
                    (x + dx, y + dy, 0.941),
                    (0.013, 0.013, 0.006),
                    "rubber",
                )
        _box(
            b,
            base,
            f"{key}_hotplate",
            (x, y, 0.980),
            (0.112, 0.118, 0.033),
            "blue" if index != 2 else "cream",
        )
        _name(
            b,
            b.cylinder(base, (x, y, 1.023), 0.101, 0.010, "bright", collision=True),
            f"{key}_platen",
        )
        # Bottoms exactly meet the platen; supports are not supplied by transparency.
        center = 1.033 + height / 2
        if (index + variant) % 3 == 1:
            _name(
                b,
                b.cylinder(
                    base, (x, y, center), radius, height / 2, "glass", collision=True
                ),
                f"{key}_vessel",
            )
        else:
            _name(
                b,
                b.geom(
                    base,
                    "ellipsoid",
                    (radius, radius, height / 2),
                    (x, y, center),
                    "glass",
                    collision=True,
                ),
                f"{key}_vessel",
            )
        neck_bottom = 1.033 + height - 0.012
        neck_top = neck_bottom + 0.073
        _name(
            b,
            b.cylinder(
                base,
                (x, y, (neck_top + neck_bottom) / 2),
                0.013,
                0.0365,
                "glass",
                collision=True,
            ),
            f"{key}_neck",
        )
        condenser_bottom = neck_top - 0.005
        condenser_top = condenser_bottom + (0.29, 0.21, 0.34)[index]
        _name(
            b,
            b.cylinder(
                base,
                (x, y, (condenser_bottom + condenser_top) / 2),
                0.019,
                (condenser_top - condenser_bottom) / 2,
                "glass",
                collision=True,
            ),
            f"{key}_condenser",
        )
        clamp_z = condenser_bottom + 0.12
        # Two jaws approach the tube surface; a rear bracket joins the rack crossbar.
        _box(
            b,
            base,
            f"{key}_clamp_jaw",
            (x, y + 0.028, clamp_z),
            (0.034, 0.009, 0.014),
            "dark",
        )
        _rod(
            b,
            base,
            f"{key}_clamp_stem",
            (x, y + 0.037, clamp_z),
            (x, 0.30, clamp_z),
            0.005,
        )
        _rod(b, base, f"{key}_clamp_drop", (x, 0.30, clamp_z), (x, 0.30, 1.36), 0.005)
        for dx in (-0.028, 0.028):
            _rod(
                b,
                base,
                f"{key}_clamp_finger_{dx}",
                (x + dx, y + 0.029, clamp_z),
                (x + dx, y, clamp_z),
                0.009,
            )
        # Both hose endpoints attach to explicit nipples, with no implied fluid model.
        for side, z in ((-1, condenser_bottom + 0.045), (1, condenser_top - 0.045)):
            _rod(
                b,
                base,
                f"{key}_nipple_{side}",
                (x + side * 0.016, y, z),
                (x + side * 0.038, y, z),
                0.004,
                "glass",
            )
            points = [
                (x + side * 0.038, y, z),
                (x + side * 0.073, y + 0.06, z + 0.07),
                (x + side * 0.09, 0.20, 1.79),
                (x + side * 0.09, 0.34, 1.90),
            ]
            for j, (first, last) in enumerate(zip(points, points[1:])):
                _rod(b, base, f"{key}_hose_{side}_{j}", first, last, 0.004, "cyan")
            _rod(
                b,
                base,
                f"{key}_utility_tap_{side}",
                points[-1],
                (points[-1][0], 0.34, 1.93),
                0.007,
            )
        for dx in (-0.072, 0.072):
            _name(
                b,
                b.cylinder(
                    base,
                    (x + dx, y - 0.123, 0.982),
                    0.011,
                    0.007,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                ),
                f"{key}_dial_{dx}",
            )


def add_hood(b, base, params):
    """Sash and supported drawer retain the original position/range contracts."""
    b.feet(base, 1.7, 0.94, 0.085)
    _box(b, base, "cabinet", (0, 0.025, 0.37), (0.81, 0.415, 0.285), "cream")
    for side in (-1, 1):
        _box(
            b,
            base,
            f"deck_support_{side}",
            (side * 0.79, 0.025, 0.76),
            (0.020, 0.415, 0.105),
            "cream",
        )
        _box(
            b,
            base,
            f"cabinet_door_{side}",
            (side * 0.41, -0.417, 0.39),
            (0.388, 0.014, 0.225),
            "shell",
        )
        _rod(
            b,
            base,
            f"cabinet_handle_{side}",
            (side * 0.41 - 0.12, -0.445, 0.54),
            (side * 0.41 + 0.12, -0.445, 0.54),
            0.008,
        )
        for dx in (-0.1, 0.1):
            _rod(
                b,
                base,
                f"cabinet_handle_mount_{side}_{dx}",
                (side * 0.41 + dx, -0.431, 0.54),
                (side * 0.41 + dx, -0.445, 0.54),
                0.005,
            )
    _drawer(b, base)
    _box(b, base, "deck", (0, 0, 0.9), (0.85, 0.47, 0.035), "dark")
    for side in (-1, 1):
        _box(
            b,
            base,
            f"sidewall_{side}",
            (side * 0.82, 0.03, 1.545),
            (0.03, 0.44, 0.61),
            "shell",
        )
    _box(b, base, "backwall", (0, 0.44, 1.54), (0.79, 0.027, 0.605), "cream")
    _box(b, base, "header", (0, 0.04, 2.13), (0.85, 0.43, 0.10), "shell")
    _name(
        b, b.box(base, (0, -0.382, 2.115), (0.64, 0.008, 0.042), "dark"), "header_bezel"
    )
    _sash(b, base)
    _apparatus(b, base, int(params.get("apparatus_variant", 0)))
    _name(
        b,
        b.cylinder(base, (0, 0.18, 2.29), 0.13, 0.07, "metal", collision=True),
        "exhaust_stub",
    )
    b.metadata["capabilities"] = [
        "Measured sash opening and under-deck cassette access",
        "Collision-enabled access motion with full-state observation and return",
    ]
    b.metadata["caltech_access"] = {
        "version": 1,
        "running_clearance_m": 0.001,
        "cassette_attachment": "fixed to visible riser and four retainers",
        "mechanism_basis": "Authored dry access mechanism; not photographed internal hardware",
        "aperture_closed_height_m": 0.023,
        "aperture_width_m": 1.492,
        "apparatus_variant": int(params.get("apparatus_variant", 0)),
    }
    b.metadata["limitations"].append(
        "Historical photographs inform exterior and inert apparatus. Drawer runners, sash channels, cassette fixture and hose routing are authored; no ventilation, chemistry, containment, specimen pickup or validated scientific operation."
    )


def add_glovebox(p):
    """Supported, slanted glazed-front context; all gloves and transfer lid fixed."""
    metal = (0.55, 0.60, 0.61, 1)
    white = (0.85, 0.85, 0.79, 1)
    dark = (0.05, 0.06, 0.06, 1)
    glass = (0.48, 0.68, 0.70, 0.17)
    for x in (-4.72, -3.64):
        for y in (-0.51, 0.31):
            p.cylinder(
                "glovebox_foot", (x, y, 0.017), 0.045, 0.017, dark, collision=True
            )
    p.box(
        "glovebox_support",
        (-4.18, -0.10, 0.48),
        (0.67, 0.53, 0.45),
        white,
        collision=True,
    )
    for x in (-4.69, -3.67):
        p.box(
            "glovebox_support_runner",
            (x, -0.10, 0.9425),
            (0.06, 0.49, 0.0125),
            metal,
            collision=True,
        )
    p.box(
        "glovebox_floor",
        (-4.18, -0.10, 0.98),
        (0.69, 0.52, 0.025),
        metal,
        collision=True,
    )
    p.box(
        "glovebox_back", (-4.18, 0.36, 1.43), (0.67, 0.04, 0.425), metal, collision=True
    )
    p.box(
        "glovebox_roof",
        (-4.18, -0.015, 1.89),
        (0.69, 0.425, 0.03),
        white,
        collision=True,
    )
    # Lower metal front carries the glove flanges; upper glazing slopes rearward.
    p.box(
        "glovebox_port_panel",
        (-4.18, -0.58, 1.16),
        (0.65, 0.025, 0.155),
        metal,
        collision=True,
    )
    angle = -math.atan2(0.22, 0.58)
    center = (-4.18, -0.47, 1.585)
    p.box(
        "glovebox_window",
        center,
        (0.624, 0.014, math.hypot(0.22, 0.58) / 2),
        glass,
        euler=(angle, 0, 0),
        collision=True,
    )
    for x in (-4.83, -3.53):
        p.box(
            "glovebox_side",
            (x, -0.005, 1.43),
            (0.025, 0.355, 0.425),
            metal,
            collision=True,
        )
        p.pipe(
            "glovebox_window_stile", (x, -0.58, 1.295), (x, -0.36, 1.875), 0.024, metal
        )
    for y, z in ((-0.58, 1.295), (-0.36, 1.875)):
        p.pipe("glovebox_window_crossbar", (-4.83, y, z), (-3.53, y, z), 0.024, metal)
    for x in (-4.48, -3.91):
        p.cylinder(
            "glove_port_flange",
            (x, -0.624, 1.17),
            0.122,
            0.024,
            metal,
            euler=(math.pi / 2, 0, 0),
            collision=True,
        )
        p.cylinder(
            "glove_port_cuff",
            (x, -0.66, 1.17),
            0.103,
            0.025,
            dark,
            euler=(math.pi / 2, 0, 0),
        )
        p.pipe("glove_sleeve", (x, -0.675, 1.17), (x + 0.025, -0.91, 1.13), 0.075, dark)
        p.geom(
            "glove_palm",
            "ellipsoid",
            (x + 0.025, -0.96, 1.105),
            (0.052, 0.079, 0.026),
            dark,
        )
        for i in range(4):
            dx = (i - 1.5) * 0.020
            p.pipe(
                "glove_digit",
                (x + 0.025 + dx, -1.005, 1.10),
                (x + 0.025 + dx, -1.065 - abs(i - 1.5) * 0.005, 1.08),
                0.010,
                dark,
            )
        p.pipe(
            "glove_thumb",
            (x + 0.070, -0.955, 1.11),
            (x + 0.094, -1.01, 1.09),
            0.014,
            dark,
        )
    p.cylinder(
        "glovebox_antechamber",
        (-5.04, -0.005, 1.31),
        0.21,
        0.20,
        metal,
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for x in (-4.845, -5.24):
        p.cylinder(
            "glovebox_transfer_flange",
            (x, -0.005, 1.31),
            0.232,
            0.015,
            metal,
            euler=(0, math.pi / 2, 0),
        )
    p.cylinder(
        "glovebox_closed_transfer_lid",
        (-5.267, -0.005, 1.31),
        0.214,
        0.012,
        white,
        euler=(0, math.pi / 2, 0),
    )
    # A cantilever under the transfer barrel visibly ties it to the supported shell.
    p.box(
        "glovebox_transfer_saddle",
        (-5.02, -0.005, 1.075),
        (0.20, 0.07, 0.025),
        metal,
        collision=True,
    )
    p.box(
        "glovebox_transfer_bracket",
        (-4.865, -0.005, 1.035),
        (0.045, 0.09, 0.035),
        metal,
        collision=True,
    )
    for y in (-0.08, 0.07):
        p.pipe(
            "glovebox_transfer_handle_mount",
            (-5.279, y, 1.31),
            (-5.31, y, 1.31),
            0.009,
            metal,
        )
    p.pipe(
        "glovebox_transfer_handle",
        (-5.31, -0.08, 1.31),
        (-5.31, 0.07, 1.31),
        0.012,
        dark,
    )
