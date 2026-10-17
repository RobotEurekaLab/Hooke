"""UW photo-informed commercial covers and passive preparation-line services.

Dimensions, routing, attachment details and unidentified controller functions
are estimates. Original XYZ guides, body inertias and the free vial are kept.
No liquid, vacuum, thermal, electrical or isotope process is synthesized.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

PREFIX = "uw_source_"
CREAM = (0.81, 0.80, 0.72, 1)
METAL = (0.65, 0.68, 0.68, 1)
DARK = (0.06, 0.07, 0.065, 1)
TEAL = (0.12, 0.45, 0.44, 1)


def _geom(parent, key, kind, size, pos=None, color=CREAM, **attrs):
    fields = dict(
        name=PREFIX + key,
        type=kind,
        size=numbers(size),
        rgba=numbers(color),
        density="0",
        contype="1",
        conaffinity="1",
    )
    if pos is not None:
        fields["pos"] = numbers(pos)
    fields.update(attrs)
    return ET.SubElement(parent, "geom", **fields)


def _box(parent, key, pos, half, color=CREAM):
    return _geom(parent, key, "box", half, pos, color)


def _cyl(parent, key, pos, radius, half, color=METAL, **kw):
    return _geom(parent, key, "cylinder", (radius, half), pos, color, **kw)


def _rod(parent, key, a, b, radius=0.005, color=DARK):
    return _geom(
        parent, key, "capsule", (radius,), color=color, fromto=numbers((*a, *b))
    )


def _route(parent, key, points, radius=0.005, color=DARK):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        _rod(parent, f"{key}_{i}", a, b, radius, color)


def add_analyzer_context(root, base):
    x = root.find(".//body[@name='uw_cartesian_v3__body_gantry_x']")
    y = root.find(".//body[@name='uw_cartesian_v3__body_gantry_y']")
    if x is None or y is None:
        raise ValueError("UW context requires the retained version-3 XYZ frames")
    # Three-sided commercial cream cover on the existing XY carrier. The
    # original front head exits below z=.600; the guide mechanism remains open
    # from below and no solid is placed in its original movement aperture.
    for i, xx in enumerate((-0.070, 0.070)):
        _box(y, f"tower_side_{i}", (xx, 0.003, 0.683), (0.003, 0.030, 0.125))
    _box(y, "tower_front", (0, -0.030, 0.704), (0.067, 0.003, 0.104))
    _box(y, "tower_teal_strip", (0, -0.0337, 0.704), (0.020, 0.0007, 0.101), TEAL)
    # Thin side sheets and a front cap cover the XY arm's existing frame;
    # the top remains genuinely open for the transverse carriage.
    for i, xx in enumerate((-0.078, 0.078)):
        _box(x, f"arm_side_{i}", (xx, 0.020, 0.519), (0.003, 0.234, 0.020))
    _box(x, "arm_front", (0, -0.220, 0.519), (0.081, 0.006, 0.020))

    # Explicit connector hardware and parked external cable loops. Routes
    # terminate on the original fixed cabinet; they are not moving tethers.
    for i, yy in enumerate((-0.13, 0.005, 0.14)):
        _cyl(
            base,
            f"case_side_connector_{i}",
            (-0.220, yy, 0.080),
            0.012,
            0.004,
            METAL,
            euler=numbers((0, math.pi / 2, 0)),
        )
        _cyl(
            base,
            f"case_rear_connector_{i}",
            (-0.07 + i * 0.07, 0.222, 0.080),
            0.009,
            0.006,
            DARK,
            euler=numbers((math.pi / 2, 0, 0)),
        )
        _route(
            base,
            f"case_parked_lead_{i}",
            [
                (-0.224, yy, 0.080),
                (-0.262 - i * 0.006, yy - 0.035, 0.065),
                (-0.278 - i * 0.006, yy + 0.02, 0.043),
                (-0.280 - i * 0.006, 0.26 + i * 0.016, 0.043),
                (-0.18, 0.30 + i * 0.014, 0.055),
                (-0.07 + i * 0.07, 0.255, 0.070),
                (-0.07 + i * 0.07, 0.228, 0.080),
            ],
            0.0035,
        )
    for i, yy in enumerate((-0.11, -0.07, -0.03, 0.01, 0.05, 0.09, 0.13)):
        _box(
            base,
            f"case_side_vent_{i}",
            (0.2167, yy, 0.105),
            (0.0007, 0.012, 0.023),
            DARK,
        )


def add_preparation_context(world):
    # The source shows multiple instruments attached to a rack over the black
    # bench. Pads meet the existing bench and connect the retained rack posts.
    for i, xx in enumerate((1.43, 2.65, 3.93)):
        _box(world, f"rack_foot_{i}", (xx, 2.32, 0.935), (0.040, 0.044, 0.015), METAL)
    _box(world, "controller_shelf", (2.68, 2.275, 2.365), (1.25, 0.090, 0.006), METAL)
    # Compact front cases supplement the original three photographed rack
    # positions; no numerical displays or controller identities are fabricated.
    for i, xx in enumerate((1.54, 2.17, 3.08, 3.82)):
        _box(
            world,
            f"controller_case_{i}",
            (xx, 2.255, 2.434),
            (0.090, 0.065, 0.063),
            CREAM if i % 2 else (0.59, 0.62, 0.61, 1),
        )
        _box(
            world,
            f"controller_screen_{i}",
            (xx - 0.025, 2.187, 2.444),
            (0.037, 0.003, 0.023),
            DARK,
        )
        _cyl(
            world,
            f"controller_selector_{i}",
            (xx + 0.059, 2.182, 2.437),
            0.014,
            0.008,
            DARK,
            euler=numbers((math.pi / 2, 0, 0)),
        )
        for j in range(3):
            _box(
                world,
                f"controller_key_{i}_{j}",
                (xx - 0.055 + j * 0.031, 2.187, 2.402),
                (0.010, 0.003, 0.005),
                (0.26, 0.28, 0.26, 1),
            )
    # Rear mounting plates connect retained visual-only housings and visible
    # controls into the new enabled support, without changing old surfaces.
    for i, xx in enumerate((1.75, 2.65, 3.52)):
        _box(
            world,
            f"original_controller_mount_{i}",
            (xx, 2.302, 2.433),
            (0.102, 0.007, 0.068),
            METAL,
        )
        _box(
            world,
            f"original_controller_foot_{i}",
            (xx, 2.245, 2.372),
            (0.100, 0.052, 0.007),
            METAL,
        )
    for i, xx in enumerate((1.54, 1.75, 2.17, 2.65, 3.08, 3.52, 3.82)):
        _cyl(
            world, f"controller_cable_gland_{i}", (xx, 2.24, 2.367), 0.010, 0.014, DARK
        )
        end_z = 2.04 if i % 2 else 1.77
        # A broad sag and return resemble the photographed service loops.
        # Contacts are enabled throughout; endpoints mount to rack brackets.
        _route(
            world,
            f"controller_loop_{i}",
            [
                (xx, 2.24, 2.355),
                (xx + 0.025, 2.08, 2.32),
                (xx + 0.045, 2.03, end_z + 0.05),
                (xx + 0.020, 2.06, end_z - 0.02),
                (xx - 0.015, 2.12, end_z),
                (xx - 0.015, 2.31, end_z + 0.05),
            ],
            0.0045,
            TEAL if i % 2 else (0.30, 0.19, 0.105, 1),
        )
        # Fixed small bracket physically joins the loop return to the nearest
        # retained enabled crossbar rather than to a decorative glass tube.
        target_z = 1.88 if end_z < 2 else 2.50
        _rod(
            world,
            f"loop_mount_{i}",
            (xx - 0.015, 2.31, end_z + 0.05),
            (xx - 0.015, 2.32, target_z),
            0.008,
            METAL,
        )

    # Exterior foil-wrapped trap cue at the left of the vessel row, with a
    # visible clamp to the rack. Finite facets are estimated foil folds, not
    # simulated insulation, a heater or a thermal process.
    _rod(world, "foil_support", (1.91, 2.32, 1.88), (1.91, 2.12, 1.88), 0.013, METAL)
    _rod(world, "foil_stem", (1.91, 2.12, 1.88), (1.91, 2.12, 1.55), 0.023, METAL)
    for i in range(12):
        a = math.tau * i / 12
        radius = 0.079 + (i % 3) * 0.001
        _geom(
            world,
            f"foil_facet_{i}",
            "box",
            (0.010, 0.026, 0.10),
            (1.91 + radius * math.cos(a), 2.12 + radius * math.sin(a), 1.56),
            (0.74 + 0.018 * (i % 3), 0.76 + 0.015 * (i % 3), 0.75 + 0.015 * (i % 3), 1),
            euler=numbers((0, 0, a)),
        )
    _cyl(world, "foil_upper_clamp", (1.91, 2.12, 1.655), 0.090, 0.009, METAL)
    _cyl(world, "foil_lower_clamp", (1.91, 2.12, 1.465), 0.090, 0.009, METAL)
    # Additional outside bands and handles make the three existing metallic
    # vessels read as installed hardware. Their original closed solids remain;
    # no claim of a new open container or cryogenic behavior is made.
    for i, xx in enumerate((2.18, 2.80, 3.42)):
        _cyl(world, f"vessel_foot_{i}", (xx, 2.15, 0.929), 0.103, 0.009, METAL)
        for j, z in enumerate((0.940, 1.055, 1.165, 1.252)):
            points = [
                (
                    xx + 0.102 * math.cos(math.tau * k / 32),
                    2.15 + 0.102 * math.sin(math.tau * k / 32),
                    z,
                )
                for k in range(33)
            ]
            _route(world, f"vessel_band_{i}_{j}", points, 0.005, METAL)
        for k, dx in enumerate((-0.104, 0.104)):
            _route(
                world,
                f"vessel_handle_{i}_{k}",
                [
                    (xx + dx, 2.15, 1.252),
                    (xx + dx * 1.22, 2.15, 1.252),
                    (xx + dx * 1.22, 2.15, 1.055),
                    (xx + dx, 2.15, 1.055),
                ],
                0.005,
                METAL,
            )
        # Vertical seams join all rings to the foot, outside retained shells.
        for k, yy in enumerate((2.048, 2.252)):
            _rod(
                world,
                f"vessel_seam_{i}_{k}",
                (xx, yy, 0.936),
                (xx, yy, 1.253),
                0.005,
                METAL,
            )
