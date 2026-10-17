"""Visible historical Whittle rig context, independently authored from one photo.

Screen pitch, flange sections, access structure and service routing are
estimates. All original enabled geometry and the external dry traverse remain
unchanged. These fixed additions do not model turbine flow or hidden rotors.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers
from real_labs.mounting import annular_mount
from real_labs.whittle_probe_geometry import _casing_normals

PREFIX = "whittle_context_"
STEEL = (0.30, 0.34, 0.31, 1)
PALE = (0.54, 0.57, 0.53, 1)
TEAL = (0.025, 0.33, 0.34, 1)
DARK = (0.08, 0.09, 0.075, 1)


def _box(parent, key, center, half, color=STEEL, **kwargs):
    return box(parent, PREFIX + key, half, center, color, density="0", **kwargs)


def _rod(parent, key, start, end, radius=0.012, color=STEEL):
    return ET.SubElement(
        parent,
        "geom",
        name=PREFIX + key,
        type="capsule",
        fromto=numbers((*start, *end)),
        size=str(radius),
        rgba=numbers(color),
        density="0",
        contype="1",
        conaffinity="1",
    )


def _flange(b, base, key, y, inner, outer, half, material):
    """An open continuous flange and separate enabled sector contact shapes."""
    temporary = ET.Element("body")
    annular_mount(
        b,
        temporary,
        key,
        (0, 0, 0),
        inner,
        outer,
        half,
        axis="z",
        proxies=True,
        material=material,
        namespace="whittle_context",
    )
    mesh = b.root.find(
        f"./asset/mesh[@name='{b.name}__whittle_context_{key}_shell_mesh']"
    )
    _casing_normals(mesh, (inner + outer) / 2)
    for geom in temporary.findall("geom"):
        x, z, axial = map(float, geom.get("pos", "0 0 0").split())
        angle = float(geom.attrib.pop("euler", "0 0 0").split()[2]) / 2
        c, s = math.cos(angle) / math.sqrt(2), math.sin(angle) / math.sqrt(2)
        geom.set("quat", numbers((c, -c, s, s)))
        geom.set("pos", numbers((x, y + axial, 1.90 - z)))
        geom.set("density", "0")
        if geom.get("contype") == "1":
            geom.set("group", "3")
        base.append(geom)
    # Visible fasteners are finite, enabled and intersect the flange face.
    for i in range(24):
        angle = math.tau * i / 24
        radius = (inner + outer) / 2
        start = (
            radius * math.sin(angle),
            y - half - 0.008,
            1.90 + radius * math.cos(angle),
        )
        end = (start[0], y - half + 0.004, start[2])
        _rod(base, f"{key}_bolt_{i}", start, end, 0.010, PALE)


def _transition_panel(b, base, key, lower, upper, side):
    """One thick convex panel; four panels leave a genuinely open transition."""
    z0, hx0, hy0 = lower
    z1, hx1, hy1 = upper
    if side in (0, 1):
        sign = -1 if side == 0 else 1
        face = [
            (sign * hx0, 2.24 - hy0, z0),
            (sign * hx0, 2.24 + hy0, z0),
            (sign * hx1, 2.24 + hy1, z1),
            (sign * hx1, 2.24 - hy1, z1),
        ]
        offset = (-sign * 0.025, 0, 0)
    else:
        sign = -1 if side == 2 else 1
        face = [
            (-hx0, 2.24 + sign * hy0, z0),
            (hx0, 2.24 + sign * hy0, z0),
            (hx1, 2.24 + sign * hy1, z1),
            (-hx1, 2.24 + sign * hy1, z1),
        ]
        offset = (0, -sign * 0.025, 0)
    vertices = face + [tuple(a + v for a, v in zip(p, offset)) for p in face]
    name = PREFIX + key
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=numbers([v for p in vertices for v in p]),
    )
    b.geom(
        base,
        "mesh",
        (),
        mesh=name,
        collision=True,
        density=0,
        name=name,
        rgba=(0.60, 0.63, 0.60, 1),
        material="metal",
    )


def add_apparatus_context(b, base, bands):
    """Retain original physics while replacing only coarse disabled visual nodes."""
    retired = []
    for geom in list(base.findall("geom")):
        if geom.get("type") != "capsule" or geom.get("contype") != "0":
            continue
        ends = list(map(float, geom.get("fromto", "").split()))
        if len(ends) != 6:
            continue
        radius = float(geom.get("size"))
        screen = abs(radius - 0.0028) < 1e-9 and any(
            abs(ends[1] - y) < 1e-8 and abs(ends[4] - y) < 1e-8
            for y in (-2.115, -2.117)
        )
        rim = abs(radius - 0.045) < 1e-9
        if screen or rim:
            # Geometry, density and masks remain exact: only superseded visual
            # rods are hidden, as in the existing continuous-casing renderer.
            geom.set("group", "3")
            geom.set("rgba", "0 0 0 0")
            retired.append(geom.get("name"))
    if len(retired) != 658:  # 82 coarse wires + twelve 48-segment visual rims.
        raise ValueError("Unexpected original Whittle visual screen/rim topology")
    _flange(b, base, "inlet_flange", -2.116, 1.438, 1.595, 0.020, "teal")
    # Finite open weave, not a flat opacity decal. Wires overlap at crossings
    # and meet the enabled inlet rim. Exact pitch and wire size are estimates.
    screen_radius = 1.455
    for i in range(-96, 97):
        t = i * 0.015
        extent = math.sqrt(screen_radius**2 - t**2)
        _rod(
            base,
            f"screen_vertical_{i}",
            (t, -2.130, 1.90 - extent),
            (t, -2.130, 1.90 + extent),
            0.0012,
            PALE,
        )
        _rod(
            base,
            f"screen_horizontal_{i}",
            (-extent, -2.1323, 1.90 + t),
            (extent, -2.1323, 1.90 + t),
            0.0012,
            PALE,
        )
    for i, (y, length, radius, material) in enumerate(bands):
        for end, edge in enumerate((y - length / 2, y + length / 2)):
            if i == 0 and end == 0:
                continue
            _flange(
                b,
                base,
                f"band_{i}_{end}",
                edge,
                radius - 0.027,
                radius + 0.085,
                0.017,
                material,
            )
    # The original stepped bands did not form a continuous load path. This
    # open annular shoulder joins the floor-supported inlet to the adjacent
    # smaller band; its section is an authored estimate, not hidden rotor CAD.
    _flange(b, base, "inlet_reducer", -0.69, 1.205, 1.540, 0.021, "teal")
    # Source-visible rear rectangular casing and rising metal transition.
    # This is its supported exterior, not a reconstruction of hidden airflow.
    for side in (-1, 1):
        _box(
            base,
            f"rear_side_{side}",
            (side * 0.91, 2.24, 1.90),
            (0.035, 0.58, 0.90),
            TEAL,
        )
        for y in (1.86, 2.68):
            _box(
                base,
                f"rear_foot_{side}_{y}",
                (side * 0.75, y, 0.030),
                (0.14, 0.16, 0.030),
                STEEL,
            )
            _box(
                base,
                f"rear_post_{side}_{y}",
                (side * 0.75, y, 0.51),
                (0.045, 0.045, 0.49),
                STEEL,
            )
    _box(base, "rear_bottom", (0, 2.24, 1.02), (0.945, 0.58, 0.035), TEAL)
    _box(base, "rear_back", (0, 2.795, 1.90), (0.945, 0.025, 0.90), TEAL)
    # Open rectangular top rim carries the four-sided transition.
    for side in (-1, 1):
        _box(
            base,
            f"rear_top_x_{side}",
            (side * 0.70, 2.24, 2.785),
            (0.245, 0.58, 0.025),
            TEAL,
        )
        _box(
            base,
            f"rear_top_y_{side}",
            (0, 2.24 + side * 0.52, 2.785),
            (0.72, 0.060, 0.025),
            TEAL,
        )
    for side in range(4):
        _transition_panel(
            b,
            base,
            f"transition_panel_{side}",
            (2.785, 0.70, 0.52),
            (4.17, 1.03, 0.61),
            side,
        )
    for sign in (-1, 1):
        _box(
            base,
            f"transition_top_x_{sign}",
            (sign * 1.03, 2.24, 4.16),
            (0.025, 0.63, 0.025),
            PALE,
        )
        _box(
            base,
            f"transition_top_y_{sign}",
            (0, 2.24 + sign * 0.61, 4.16),
            (1.055, 0.025, 0.025),
            PALE,
        )
    # A casing-side control enclosure and actual continuous supported harness.
    for x in (-1.64, -1.36):
        _box(base, f"control_foot_{x}", (x, 1.18, 0.025), (0.06, 0.09, 0.025))
        _box(base, f"control_post_{x}", (x, 1.18, 0.62), (0.025, 0.035, 0.60))
    _box(base, "control_case", (-1.50, 1.18, 1.49), (0.25, 0.20, 0.28), PALE)
    _box(base, "control_face", (-1.50, 0.974, 1.50), (0.22, 0.008, 0.23), DARK)
    for i in range(3):
        _box(
            base,
            f"control_blank_module_{i}",
            (-1.50, 0.963, 1.37 + i * 0.13),
            (0.17, 0.005, 0.035),
            (0.44, 0.46, 0.39, 1),
        )
    path = [
        (-1.25, 1.18, 1.55),
        (-1.22, 1.18, 1.55),
        (-1.22, 0.62, 1.55),
        (-1.22, 0.62, 1.90),
        (-1.12, 0.62, 1.90),
    ]
    for i, (start, end) in enumerate(zip(path, path[1:])):
        _rod(base, f"control_harness_{i}", start, end, 0.015, DARK)
    _box(
        base, "harness_rig_connector", (-1.105, 0.62, 1.90), (0.025, 0.045, 0.06), STEEL
    )
    b.metadata["source_context"] = dict(
        revision=2,
        source="Historical official Whittle installed-rig photo",
        visible_features=[
            "Fine open inlet screen and flanged bands",
            "Rear rectangular casing with rising transition",
            "Supported side control enclosure and harness",
        ],
        source_dimensions_verified=False,
        screen_pitch_m=0.015,
        wire_radius_m=0.0012,
        original_contact_geometry_preserved=True,
        retired_disabled_visual_geometries=len(retired),
        flow_path_qualification="Exterior only; no hidden rotor or flow model",
    )


def add_hall_context(world, definition):
    """Complete the existing safe stair footprint and restrained hall services."""
    # Keep every original tread/leg and its contact properties. Inclined
    # stringers, bracing and handrails make the footprint an access stair.
    for x in (-2.42, -1.68):
        _rod(
            world, f"stair_stringer_{x}", (x, -0.67, 0.05), (x, 0.43, 0.77), 0.030, PALE
        )
        _rod(
            world, f"stair_rear_leg_{x}", (x, 0.43, 0.035), (x, 0.43, 0.77), 0.025, PALE
        )
        _box(
            world,
            f"stair_foot_front_{x}",
            (x, -0.67, 0.025),
            (0.07, 0.09, 0.025),
            STEEL,
        )
        _box(
            world, f"stair_foot_rear_{x}", (x, 0.43, 0.025), (0.07, 0.09, 0.025), STEEL
        )
        for i in range(4):
            y = -0.50 + i * 0.25
            h = 0.18 * (i + 1)
            _box(
                world,
                f"stair_tread_bearing_{x}_{i}",
                ((x - 2.05) / 2, y, h - 0.025),
                (abs(x + 2.05) / 2 + 0.02, 0.05, 0.022),
                PALE,
            )
        for y, z in ((-0.58, 0.14), (0.39, 0.75)):
            _rod(
                world,
                f"stair_rail_post_{x}_{y}",
                (x, y, z),
                (x, y, z + 0.62),
                0.016,
                PALE,
            )
        _rod(
            world, f"stair_handrail_{x}", (x, -0.58, 0.76), (x, 0.39, 1.37), 0.018, PALE
        )
        _rod(
            world,
            f"stair_underbrace_{x}",
            (x, -0.58, 0.10),
            (x, 0.43, 0.70),
            0.015,
            STEEL,
        )
    _box(world, "stair_landing", (-2.05, 0.42, 0.745), (0.40, 0.20, 0.025), PALE)
    # The historic photograph shows a pale aisle line outside the access
    # stair. This is non-contact paint, not a new floor plate or surveyed path.
    _box(
        world,
        "painted_aisle_line",
        (-2.78, 0.50, 0.00015),
        (0.0225, 1.80, 0.00015),
        (0.82, 0.82, 0.68, 1),
        collision=False,
    )
    # Source background steelwork and roof-connected services. Room bounds
    # remain the existing unsurveyed8m square; no generic extra lab equipment.
    for x in (-3.30, 3.36):
        _box(world, f"hall_foot_{x}", (x, 2.90, 0.025), (0.17, 0.17, 0.025), STEEL)
        _box(world, f"hall_column_{x}", (x, 2.90, 2.50), (0.07, 0.07, 2.50), STEEL)
    _box(world, "hall_crossbeam", (0, 2.90, 4.82), (3.35, 0.085, 0.10), STEEL)
    for x, sign in ((-3.30, 1), (3.36, -1)):
        _rod(
            world,
            f"hall_knee_{x}",
            (x, 2.90, 3.85),
            (x + sign * 0.75, 2.90, 4.82),
            0.030,
            STEEL,
        )
    # The metal transition is floor-supported through its rear casing. Its
    # adjacent ceiling duct retains the existing roof-dependent render role.
    box(
        world,
        "arch_roof_whittle_context_crossduct",
        (0.79, 0.20, 0.18),
        (1.82, 2.24, 4.26),
        (0.62, 0.65, 0.61, 1),
        density="0",
    )
    for x in (1.28, 2.38):
        box(
            world,
            f"arch_roof_whittle_context_crossduct_hanger_{x}",
            (0.018, 0.018, 0.30),
            (x, 2.24, 4.72),
            PALE,
            density="0",
        )
