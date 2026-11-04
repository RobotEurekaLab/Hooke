"""Source-informed furnace exterior and an original cold feed inspection mount.

The official close-up supplies visible forms, not scale, hidden mechanisms or
room registration. All dimensions remain authored estimates. Reflector bodies
are closed exterior proxies; no thermal, optical or crystal-growth model.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

HEIGHT_ORIGIN = (0, -0.025, 1.92)
INSPECTION_HEIGHT = 0.040


def _name(b, key):
    return f"{b.name}__jhu_{key}"


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(parent, pos, half, material, collision=True, name=_name(b, key))


def _cyl(b, parent, key, pos, radius, half, material="metal", axis="z"):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=_name(b, key),
        euler=(math.pi / 2, 0, 0) if axis == "y" else (0, 0, 0),
    )


def _rod(b, parent, key, start, end, radius=0.005, material="dark"):
    return b.rod(
        parent, start, end, radius, material, collision=True, name=_name(b, key)
    )


def _reflector(b, parent, side):
    """A continuous rounded housing, not a stack of rendered collision strips.

    Two source-visible front exteriors stand in for four internal reflectors;
    the concealed optical surfaces and their separate axes are not recreated.
    A convex closed shell is appropriate for this inaccessible exterior.
    """
    n = 64
    sections = (
        (-0.205, 0.84),
        (-0.200, 0.93),
        (-0.170, 1),
        (0.270, 1),
        (0.300, 0.93),
        (0.305, 0.84),
    )
    verts = []
    for y, scale in sections:
        for i in range(n):
            a = i * math.tau / n
            c, s = math.cos(a), math.sin(a)
            x = 0.205 * math.copysign(abs(c) ** (2 / 2.6), c) * scale
            z = 0.245 * math.copysign(abs(s) ** (2 / 2.6), s) * scale
            verts.append((side * 0.26 + x, y, 1.5 + z))
    faces = []
    for j in range(len(sections) - 1):
        for i in range(n):
            a, c = j * n + i, j * n + (i + 1) % n
            d, e = c + n, a + n
            faces.extend(((a, c, d), (a, d, e)))
    for j, reverse in ((0, True), (len(sections) - 1, False)):
        center = len(verts)
        verts.append((side * 0.26, sections[j][0], 1.5))
        for i in range(n):
            face = (center, j * n + i, j * n + (i + 1) % n)
            faces.append(tuple(reversed(face)) if reverse else face)
    volume = sum(
        sum(
            verts[a][q]
            * (
                verts[c][(q + 1) % 3] * verts[d][(q + 2) % 3]
                - verts[c][(q + 2) % 3] * verts[d][(q + 1) % 3]
            )
            for q in range(3)
        )
        for a, c, d in faces
    )
    if volume < 0:
        faces = [tuple(reversed(f)) for f in faces]
    mesh = _name(b, f"reflector_{side}_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([x for p in verts for x in p]),
        face=" ".join(str(x) for f in faces for x in f),
    )
    b.geom(
        parent,
        "mesh",
        (),
        material="metal",
        mesh=mesh,
        collision=True,
        name=_name(b, f"reflector_{side}"),
    )
    # Source-visible clamping straps reach the front rods and the metal shell.
    for z in (1.31, 1.70):
        _box(
            b,
            parent,
            f"reflector_strap_{side}_{z}",
            (side * 0.26, -0.24, z),
            (0.23, 0.055, 0.018),
            "bright",
        )
        for x in (side * 0.10, side * 0.42):
            _cyl(
                b,
                parent,
                f"strap_bolt_{side}_{z}_{x}",
                (x, -0.301, z),
                0.007,
                0.009,
                "dark",
                axis="y",
            )
    _cyl(
        b,
        parent,
        f"reflector_face_fastener_{side}",
        (side * 0.26, -0.209, 1.50),
        0.007,
        0.006,
        "dark",
        axis="y",
    )


def _fan(b, base, side):
    x = side * 0.56
    _box(
        b, base, f"fan_mount_{side}", (side * 0.475, -0.155, 1.50), (0.12, 0.13, 0.027)
    )
    _box(b, base, f"fan_case_{side}", (x, -0.29, 1.50), (0.11, 0.06, 0.14), "dark")
    _cyl(b, base, f"fan_hub_{side}", (x, -0.353, 1.50), 0.022, 0.006, "dark", axis="y")
    for r in (0.035, 0.06, 0.085, 0.10):
        b.ring(base, (x, -0.359, 1.50), r, 0.002, "bright", plane="xz", segments=32)
    for sign in (-1, 1):
        _rod(
            b,
            base,
            f"fan_guard_{side}_{sign}",
            (x - 0.092, -0.362, 1.5 - sign * 0.105),
            (x + 0.092, -0.362, 1.5 + sign * 0.105),
            0.0025,
            "bright",
        )
    for dx in (-0.096, 0.096):
        for dz in (-0.124, 0.124):
            _cyl(
                b,
                base,
                f"fan_bolt_{side}_{dx}_{dz}",
                (x + dx, -0.354, 1.50 + dz),
                0.006,
                0.005,
                "metal",
                axis="y",
            )
    bx = side * 0.46
    _box(b, base, f"blower_plinth_{side}", (bx, -0.23, 0.895), (0.135, 0.16, 0.045))
    _box(
        b,
        base,
        f"blower_motor_support_{side}",
        (bx, -0.23, 0.966),
        (0.075, 0.095, 0.026),
        "copper",
    )
    _cyl(
        b,
        base,
        f"blower_body_{side}",
        (bx, -0.23, 1.075),
        0.112,
        0.10,
        "dark",
        axis="y",
    )
    _cyl(
        b,
        base,
        f"blower_face_{side}",
        (bx, -0.333, 1.075),
        0.086,
        0.006,
        "metal",
        axis="y",
    )
    for offset in (-0.06, -0.03, 0, 0.03, 0.06):
        # Grille bars are static appearance; no rotating airflow mechanism.
        length = math.sqrt(0.083**2 - offset**2)
        _rod(
            b,
            base,
            f"blower_grille_{side}_{offset}",
            (bx + offset, -0.341, 1.075 - length),
            (bx + offset, -0.341, 1.075 + length),
            0.003,
            "dark",
        )
    _box(
        b,
        base,
        f"blower_outlet_{side}",
        (bx, -0.17, 1.215),
        (0.052, 0.055, 0.055),
        "dark",
    )
    _rod(
        b,
        base,
        f"blower_fixed_lead_{side}",
        (bx, -0.11, 1.17),
        (side * 0.66, 0.30, 1.12),
        0.008,
    )
    _rod(
        b,
        base,
        f"blower_fixed_tail_{side}",
        (side * 0.66, 0.30, 1.12),
        (side * 0.66, 0.435, 0.94),
        0.008,
    )


def _feed(b, base):
    # The fixed bridge seats on the enclosure sides and anchors both guide rods.
    _box(b, base, "guide_bridge", (0, 0.08, 1.765), (0.72, 0.029, 0.015), "metal")
    for x in (-0.085, 0.085):
        _cyl(b, base, f"guide_lower_mount_{x}", (x, 0.08, 1.783), 0.020, 0.018)
        _cyl(b, base, f"guide_rod_{x}", (x, 0.08, 1.99), 0.010, 0.210, "bright")
        _cyl(b, base, f"guide_top_mount_{x}", (x, 0.08, 2.198), 0.020, 0.018)
    carriage = b.moving(
        base,
        "feed_rod_height",
        HEIGHT_ORIGIN,
        (0, 0, 1),
        (-0.06, 0.06),
        mass=0.22,
        kp=360,
    )
    for x in (-0.085, 0.085):
        annular_mount(
            b,
            carriage,
            f"guide_sleeve_{x}",
            (x, 0.105, 0),
            0.0106,
            0.024,
            0.026,
            axis="z",
            proxies=True,
            material="metal",
            namespace="jhu",
        )
    _box(b, carriage, "carriage_bridge", (0, 0.14, 0), (0.130, 0.012, 0.020))
    _box(b, carriage, "carriage_arm", (0, 0.080, 0), (0.030, 0.050, 0.020))
    annular_mount(
        b,
        carriage,
        "spindle_bearing",
        (0, 0, -0.005),
        0.0186,
        0.036,
        0.030,
        axis="z",
        proxies=True,
        material="metal",
        namespace="jhu",
    )
    rotation = b.moving(
        carriage,
        "feed_rod_rotation",
        (0, 0, -0.005),
        (0, 0, 1),
        (-1, 1),
        kind="hinge",
        mass=0.06,
        kp=90,
    )
    _cyl(b, rotation, "rotor_journal", (0, 0, 0), 0.018, 0.044, "bright")
    _cyl(b, rotation, "rotor_cap", (0, 0, 0.050), 0.034, 0.010, "dark")
    _cyl(b, rotation, "rotor_chuck", (0, 0, -0.050), 0.025, 0.010, "dark")
    _cyl(b, rotation, "feed_shaft", (0, 0, -0.225), 0.004, 0.225, "bright")
    _cyl(b, rotation, "inert_feed_coupon", (0, 0, -0.470), 0.006, 0.020, "orange")
    # Original target point is now on a bonded physical witness face, rather
    # than 2 mm outside the coupon. Dimensions are authored, not furnace data.
    _cyl(
        b,
        rotation,
        "coupon_witness",
        (0, -0.0065, -0.470),
        0.0018,
        0.0015,
        "cream",
        axis="y",
    )
    b.site(rotation, "crystal_coupon", (0, -0.008, -0.470))
    # Fixed lower spindle preserves the existing 25 mm minimum cold gap.
    _cyl(b, base, "lower_pedestal", (0, -0.025, 0.975), 0.045, 0.125)
    _cyl(b, base, "lower_spindle", (0, -0.025, 1.22), 0.004, 0.120, "bright")
    b.site(base, "lower_spindle_top", (0, -0.025, 1.34))
    # Physical surface-height datum, deliberately separate from optical focus.
    _box(b, base, "datum_foot", (-0.031, -0.025, 0.860), (0.014, 0.020, 0.010), "dark")
    _cyl(b, base, "datum_post", (-0.031, -0.025, 1.145), 0.003, 0.275, "metal")
    _box(
        b,
        base,
        "datum_upright",
        (-0.031, -0.025, 1.456),
        (0.004, 0.004, 0.036),
        "metal",
    )
    _box(b, base, "datum_tick", (-0.026, -0.031, 1.485), (0.004, 0.003, 0.001), "cream")
    b.site(base, "inspection_height", (0, -0.025, 1.485))
    b.metadata["feed_interface"] = {
        "coupon_geom": _name(b, "inert_feed_coupon"),
        "witness_geom": _name(b, "coupon_witness"),
        "witness_site": b.metadata["sites"]["crystal_coupon"],
        "lower_spindle_geom": _name(b, "lower_spindle"),
        "lower_spindle_site": b.metadata["sites"]["lower_spindle_top"],
        "height_reference_site": b.metadata["sites"]["inspection_height"],
        "inspection_height_m": INSPECTION_HEIGHT,
        "inspection_azimuth_rad": 0.60,
        "coupon_diameter_m": 0.012,
        "coupon_height_m": 0.040,
        "witness_diameter_m": 0.0036,
        "height_tolerance_m": 0.0001,
        "minimum_lower_gap_m": 0.025,
        "scope": "Cold retained-coupon positioning to an authored physical height datum; no molten zone or optical focus",
    }


def build_furnace(b, base, params):
    for x in (-0.62, 0.62):
        for y in (-0.36, 0.36):
            _box(
                b, base, f"foot_{x}_{y}", (x, y, 0.025), (0.075, 0.075, 0.025), "rubber"
            )
    _box(b, base, "cabinet", (0, 0, 0.39), (0.73, 0.46, 0.34), "cream")
    _box(b, base, "lower_rim", (0, 0, 0.77), (0.75, 0.48, 0.04), "dark")
    _box(b, base, "furnace_optical_deck", (0, 0, 0.83), (0.70, 0.43, 0.02), "metal")
    _box(b, base, "rear_wall", (0, 0.455, 1.54), (0.73, 0.025, 0.69), "dark")
    for side in (-1, 1):
        _box(
            b,
            base,
            f"side_wall_{side}",
            (side * 0.72, 0, 1.54),
            (0.025, 0.45, 0.69),
            "dark",
        )
        for y in (-0.37, 0.37):
            _cyl(
                b,
                base,
                f"outer_post_{side}_{y}",
                (side * 0.64, y, 1.525),
                0.014,
                0.675,
                "bright",
            )
        _cyl(
            b,
            base,
            f"front_rod_{side}",
            (side * 0.18, -0.30, 1.465),
            0.015,
            0.615,
            "bright",
        )
        for z in (1.08, 1.88):
            _box(
                b,
                base,
                f"split_bridge_{side}_{z}",
                (side * 0.38, -0.20, z),
                (0.33, 0.12, 0.022),
                "metal",
            )
        _reflector(b, base, side)
        _fan(b, base, side)
        # Parked flexible service route follows the visible white/grey leads;
        # endpoints attach to fixed housings. It does not imply active cooling.
        points = (
            (side * 0.60, -0.23, 2.18),
            (side * 0.59, -0.25, 2.00),
            (side * 0.56, -0.27, 1.77),
            (side * 0.54, -0.25, 1.64),
        )
        for i, (a, c) in enumerate(zip(points, points[1:])):
            _rod(b, base, f"fixed_cooling_lead_{side}_{i}", a, c, 0.009, "cream")
        _rod(
            b,
            base,
            f"cooling_upper_anchor_{side}",
            (side * 0.60, -0.23, 2.18),
            (side * 0.60, -0.23, 2.205),
            0.012,
            "metal",
        )
        _box(
            b,
            base,
            f"cabinet_front_{side}",
            (side * 0.355, -0.464, 0.40),
            (0.345, 0.008, 0.30),
            "cream",
        )
        _box(
            b,
            base,
            f"cabinet_handle_{side}",
            (side * 0.08, -0.478, 0.52),
            (0.012, 0.008, 0.064),
            "dark",
        )
    _box(b, base, "roof", (0, 0, 2.24), (0.75, 0.48, 0.04), "dark")
    _feed(b, base)
    b.metadata["capabilities"] = [
        "cold inert feed-height positioning",
        "measured coupon azimuth indexing",
        "physical specimen/datum observation and return",
    ]
    b.metadata["limitations"].append(
        "Official furnace detail informs rounded paired reflector exteriors, support rods, fans and fixed leads. Exact room placement and all dimensions/internal guide design are authored estimates. Original controls, masses and gains retained. No heating, optical focusing, pressure, melt, growth kinetics, crystal quality or calibrated machine performance."
    )
