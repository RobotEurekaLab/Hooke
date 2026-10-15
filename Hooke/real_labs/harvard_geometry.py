"""Original supported XY/head inspection fixture for Harvard CNS references.

Travel and body frames retain the earlier surrogate contract. Internal bearings,
clamps and dimensions are engineering assumptions, not recovered AB-M hardware.
The inert wafer is rigidly clamped to the carriage; it is not a free loading task.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


CLEARANCE_M = 0.0006


def name(b, key):
    return f"{b.name}__cns_{key}"


def box(b, parent, key, position, half, material="metal", **kwargs):
    return b.box(
        parent, position, half, material, collision=True, name=name(b, key), **kwargs
    )


def cylinder(b, parent, key, position, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        position,
        radius,
        half,
        material,
        collision=True,
        name=name(b, key),
        **kwargs,
    )


def shaft(b, parent, key, start, end, radius, material="bright"):
    return b.geom(
        parent,
        "cylinder",
        (radius,),
        material=material,
        fromto=(*start, *end),
        collision=True,
        name=name(b, key),
    )


def _open_shoe(
    b, parent, key, center, radius, roof_top, *, axis="y", half_length=0.018
):
    """Three physical blocks form an open-bottom bearing, with real running gap."""
    x, y, z = center
    bore = radius + CLEARANCE_M
    thickness = 0.006
    roof_bottom = z + bore
    if roof_top <= roof_bottom:
        raise ValueError("Bearing roof must reach the supported plate")
    roof_center = (roof_bottom + roof_top) / 2
    roof_half = (roof_top - roof_bottom) / 2
    if axis == "y":
        box(
            b,
            parent,
            key + "_roof",
            (x, y, roof_center),
            (bore + thickness, half_length, roof_half),
        )
        for side in (-1, 1):
            box(
                b,
                parent,
                key + f"_flank_{side}",
                (x + side * (bore + thickness / 2), y, z),
                (thickness / 2, half_length, bore),
            )
    elif axis == "x":
        box(
            b,
            parent,
            key + "_roof",
            (x, y, roof_center),
            (half_length, bore + thickness, roof_half),
        )
        for side in (-1, 1):
            box(
                b,
                parent,
                key + f"_flank_{side}",
                (x, y + side * (bore + thickness / 2), z),
                (half_length, thickness / 2, bore),
            )
    else:
        raise ValueError("Only horizontal crossed-slide bearings are supported")


def _column_bearing(b, parent, key, center):
    inner, outer, half = 0.0256, 0.041, 0.055
    annular_mount(
        b,
        parent,
        key,
        center,
        inner,
        outer,
        half,
        axis="z",
        namespace="cns",
        proxies=False,
    )
    # Sixteen radial convex contact sectors keep the central bore genuinely
    # empty. Their inner tangent planes retain the declared0.6mm clearance.
    for index in range(16):
        angle = (index + 0.5) * math.tau / 16
        radius = (inner + outer) / 2
        box(
            b,
            parent,
            key + f"_contact_{index}",
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                center[2],
            ),
            ((outer - inner) / 2, outer * math.tan(math.pi / 16), half),
            euler=(0, 0, angle),
            rgba=(0.5, 0.5, 0.5, 0),
        )


def _base(b, base):
    b.feet(base, 1.38, 0.78, height=0.025)
    b.housing(
        base,
        [
            (0.025, 0.66, 0.37, 0),
            (0.055, 0.69, 0.39, 0),
            (0.19, 0.69, 0.39, 0),
            (0.22, 0.67, 0.35, 0),
        ],
        0.026,
    )
    # The enabled core and grounded side webs support the real deck, rather
    # than an invisible ninety-millimetre gap inside the cosmetic enclosure.
    box(b, base, "core", (0, 0, 0.17), (0.67, 0.35, 0.055), "cream")
    for x in (-0.635, 0.635):
        box(b, base, f"core_web_{x}", (x, 0, 0.070), (0.035, 0.34, 0.045), "shell")
    box(b, base, "deck", (0, 0.0175, 0.233), (0.37, 0.325, 0.012))
    for side, x in enumerate((-0.29, 0.29)):
        shaft(b, base, f"y_rail_{side}", (x, -0.30, 0.263), (x, 0.28, 0.263), 0.012)
        for end, y in enumerate((-0.285, 0.265)):
            box(
                b,
                base,
                f"y_rail_mount_{side}_{end}",
                (x, y, 0.2575),
                (0.025, 0.015, 0.0125),
            )
            for dx in (-0.018, 0.018):
                cylinder(
                    b,
                    base,
                    f"y_rail_bolt_{side}_{end}_{dx}",
                    (x + dx, y, 0.273),
                    0.0035,
                    0.003,
                    "dark",
                )
    for side, x in enumerate((-0.24, 0.24)):
        box(b, base, f"column_foot_{side}", (x, 0.35, 0.24), (0.046, 0.047, 0.025))
        shaft(b, base, f"column_{side}", (x, 0.35, 0.265), (x, 0.35, 1.05), 0.025)
    box(b, base, "top_crossmember", (0, 0.35, 1.040), (0.295, 0.07, 0.025), "shell")
    for x in (-0.31, -0.23, -0.15):
        cylinder(
            b,
            base,
            f"front_dial_{x}",
            (x, -0.38, 0.12),
            0.018,
            0.018,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    _reference_exterior(b, base)


def _reference_exterior(b, base):
    """Historical CNS photographs inform appearance, not dimensions or internals."""
    if b.name == "lithography_aligner":
        # AB-M photo: tall left illumination tower, sloping cream front and a
        # black folded lower shroud. The pedestal is an inferred real support.
        box(
            b,
            base,
            "tower_pedestal",
            (-0.525, 0.08, 0.345),
            (0.13, 0.13, 0.12),
            "cream",
        )
        box(
            b,
            base,
            "bellows_core",
            (-0.525, 0.045, 0.565),
            (0.126, 0.155, 0.10),
            "dark",
        )
        for i in range(12):
            box(
                b,
                base,
                f"bellows_fold_{i}",
                (-0.525, 0.045, 0.474 + i * 0.016),
                (0.14, 0.165, 0.004),
                "rubber",
            )
        tower = ET.SubElement(
            base, "body", name=name(b, "illumination_tower"), pos="-0.525 0 0"
        )
        b.housing(
            tower,
            [
                (0.665, 0.155, 0.20, 0.00),
                (0.88, 0.155, 0.15, 0.05),
                (1.30, 0.155, 0.15, 0.05),
            ],
            0.013,
            "cream",
        )
        shape = tower.findall("geom")[-1]
        shape.set("name", name(b, "tower_shell"))
        shape.set("contype", "1")
        shape.set("conaffinity", "1")
        for z in (0.91, 1.23):
            for x in (-0.66, -0.39):
                cylinder(
                    b,
                    base,
                    f"tower_fastener_{x}_{z}",
                    (x, -0.102, z),
                    0.003,
                    0.002,
                    "metal",
                    euler=(math.pi / 2, 0, 0),
                )
        # Real photographed side panels are represented without text or logos.
        for side in (-1, 1):
            panel = ET.SubElement(
                base,
                "body",
                name=name(b, f"panel_{side}"),
                pos=f"{side * .525} -.245 .28",
                euler="0.4 0 0",
            )
            box(b, panel, f"panel_case_{side}", (0, 0, 0), (0.135, 0.03, 0.07), "cream")
            for i, x in enumerate((-0.075, 0, 0.075)):
                if side > 0:
                    cylinder(
                        b,
                        panel,
                        f"gauge_rim_{i}",
                        (x, -0.036, 0.025),
                        0.024,
                        0.006,
                        "bright",
                        euler=(math.pi / 2, 0, 0),
                    )
                    cylinder(
                        b,
                        panel,
                        f"gauge_face_{i}",
                        (x, -0.043, 0.025),
                        0.019,
                        0.001,
                        "cream",
                        euler=(math.pi / 2, 0, 0),
                    )
                    b.rod(
                        panel,
                        (x, -0.045, 0.025),
                        (x + 0.009, -0.045, 0.036),
                        0.001,
                        "dark",
                    )
                else:
                    box(
                        b,
                        panel,
                        f"indicator_{i}",
                        (x, -0.032, 0.025),
                        (0.020, 0.003, 0.014),
                        "dark",
                    )
                cylinder(
                    b,
                    panel,
                    f"panel_knob_{side}_{i}",
                    (x, -0.046, -0.032),
                    0.009,
                    0.014,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
        # The right articulated grip is deliberately outside the whole slide
        # envelope and is attached to a post grounded in the base.
        shaft(b, base, "handle_post", (0.40, 0.25, 0.225), (0.40, 0.25, 0.75), 0.018)
        shaft(b, base, "handle_link_0", (0.40, 0.25, 0.75), (0.54, 0.08, 0.74), 0.014)
        shaft(b, base, "handle_link_1", (0.54, 0.08, 0.74), (0.54, -0.10, 0.70), 0.014)
        shaft(
            b,
            base,
            "handle_grip",
            (0.54, -0.10, 0.70),
            (0.54, -0.10, 0.60),
            0.025,
            "rubber",
        )
    else:
        # A separate MJB4-inspired station has a central rear light housing and
        # asymmetrical black controls. Its retained six-axis family is authored.
        box(b, base, "rear_foundation", (0, 0.395, 0.17), (0.19, 0.105, 0.055), "cream")
        box(
            b,
            base,
            "rear_tower_pedestal",
            (0, 0.44, 0.57),
            (0.15, 0.055, 0.345),
            "cream",
        )
        box(
            b,
            base,
            "rear_light_enclosure",
            (0, 0.445, 1.10),
            (0.17, 0.05, 0.19),
            "dark",
        )
        for side in (-1, 1):
            box(
                b,
                base,
                f"control_support_{side}",
                (side * 0.525, -0.19, 0.34),
                (0.10, 0.065, 0.115),
                "cream",
            )
            box(
                b,
                base,
                f"control_panel_{side}",
                (side * 0.525, -0.26, 0.40),
                (0.12, 0.015, 0.16),
                "dark",
            )
        cylinder(
            b,
            base,
            "power_switch",
            (-0.525, -0.286, 0.47),
            0.040,
            0.01,
            "warning",
            euler=(math.pi / 2, 0, 0),
        )
        b.screen(base, (0.525, -0.284, 0.43), 0.15, 0.12)


def _stages(b, base):
    carrier = b.moving(
        base, "wafer_y", (0, -0.01, 0.30), (0, 1, 0), (-0.09, 0.09), mass=0.55, kp=800
    )
    box(b, carrier, "y_plate", (0, 0, 0), (0.33, 0.23, 0.018), "dark")
    for side, x in enumerate((-0.29, 0.29)):
        for station, y in enumerate((-0.11, 0.11)):
            _open_shoe(
                b, carrier, f"y_shoe_{side}_{station}", (x, y, -0.037), 0.012, -0.018
            )
    for side, y in enumerate((-0.145, 0.145)):
        shaft(b, carrier, f"x_rail_{side}", (-0.25, y, 0.035), (0.25, y, 0.035), 0.010)
        for end, x in enumerate((-0.232, 0.232)):
            box(
                b,
                carrier,
                f"x_rail_mount_{side}_{end}",
                (x, y, 0.0265),
                (0.015, 0.025, 0.0085),
            )
    stage = b.moving(
        carrier, "wafer_x", (0, 0, 0.067), (1, 0, 0), (-0.065, 0.065), mass=0.35, kp=800
    )
    box(b, stage, "x_plate", (0, 0, 0), (0.16, 0.14, 0.017))
    for side, y in enumerate((-0.145, 0.145)):
        for station, x in enumerate((-0.075, 0.075)):
            _open_shoe(
                b,
                stage,
                f"x_shoe_{side}_{station}",
                (x, y, -0.032),
                0.01,
                -0.017,
                axis="x",
            )
    cylinder(b, stage, "chuck", (0, 0, 0.031), 0.098, 0.014, "dark")
    cylinder(b, stage, "wafer_shim", (0, 0, 0.04525), 0.074, 0.00025, "metal")
    cylinder(b, stage, "wafer", (0, 0, 0.047), 0.074, 0.0015, "lens")
    for index in range(3):
        angle = index * math.tau / 3

        def point(radius, z):
            return radius * math.cos(angle), radius * math.sin(angle), z

        box(
            b,
            stage,
            f"clamp_foot_{index}",
            point(0.088, 0.04675),
            (0.006, 0.006, 0.00175),
            euler=(0, 0, angle),
        )
        box(
            b,
            stage,
            f"clamp_cap_{index}",
            point(0.080, 0.0502),
            (0.010, 0.006, 0.0017),
            euler=(0, 0, angle),
        )
        cylinder(
            b, stage, f"clamp_bolt_{index}", point(0.088, 0.052), 0.003, 0.003, "dark"
        )
    for axis in (0, 1):
        half = [0.007, 0.00045, 0.00004]
        if axis:
            half[0], half[1] = half[1], half[0]
        b.box(
            stage, (0, 0, 0.04855), half, "bright", name=name(b, f"wafer_cross_{axis}")
        )
    b.site(stage, "wafer", (0, 0, 0.047))
    b.site(stage, "wafer_surface", (0, 0, 0.0485))
    return stage


def _head(b, base):
    head = b.moving(
        base,
        "alignment_head",
        (0, 0.115, 0.735),
        (0, 0, 1),
        (0, 0.11),
        mass=0.8,
        kp=800,
    )
    b.housing(
        head,
        [
            (-0.06, 0.12, 0.115, 0.015),
            (0.07, 0.12, 0.115, 0.015),
            (0.105, 0.09, 0.08, 0.04),
        ],
        0.025,
    )
    shell = list(head.findall("geom"))[-1]
    shell.set("name", name(b, "head_shell"))
    shell.set("contype", "1")
    shell.set("conaffinity", "1")
    box(b, head, "head_neck", (0, 0.1425, 0), (0.08, 0.060, 0.025))
    box(b, head, "head_bridge", (0, 0.192, 0), (0.242, 0.012, 0.025))
    for side, x in enumerate((-0.24, 0.24)):
        _column_bearing(b, head, f"head_bearing_{side}", (x, 0.235, 0))
    box(b, head, "objective_nose", (0, -0.11, -0.115), (0.09, 0.055, 0.055), "cream")
    for side, x in enumerate((-0.055, 0.055)):
        cylinder(b, head, f"objective_{side}", (x, -0.14, -0.232), 0.03, 0.062, "dark")
        cylinder(
            b, head, f"objective_lens_{side}", (x, -0.14, -0.297), 0.024, 0.003, "lens"
        )
        b.rod(
            head,
            (x, -0.01, 0.075),
            (x, -0.16, 0.17),
            0.027,
            "shell",
            collision=True,
            name=name(b, f"eyetube_{side}"),
        )
        b.rod(
            head,
            (x, -0.16, 0.17),
            (x, -0.19, 0.19),
            0.03,
            "rubber",
            collision=True,
            name=name(b, f"eyecup_{side}"),
        )
    # Datum is the actual right objective's lower face, not an empty arbitrary
    # target. Task checks only its geometric axis, not focus or optical accuracy.
    b.site(head, "inspection_axis", (0.055, -0.14, -0.300))
    return head


def add_aligner(b, base, params):
    _base(b, base)
    _stages(b, base)
    _head(b, base)
    b.metadata["capabilities"] = [
        "Contact-enabled XY retained-wafer positioning",
        "Supported head access motion",
        "Measured geometric axis inspection and return",
    ]
    b.metadata["harvard_access"] = dict(
        version=1,
        wafer_radius_m=0.074,
        wafer_thickness_m=0.003,
        attachment="Rigid clamp assumption: wafer and three visible retainers belong to the X-stage body; no free loading or removal",
        bearing_clearance_m=CLEARANCE_M,
        geometry_basis="Historical CNS 2009 AB-M and MJB4 photographs inform separate exterior silhouettes; all dimensions, bearings, rigid clamps and drive parameters remain authored",
        servo_basis="800 N/m position gain, critical damping from nominal proxy mass, 120 N force bound; finite bandwidth mechanical inspection adapter, not measured OEM performance",
        inspection="Projected axis of the visible right objective onto the retained wafer surface; no focus or exposure model",
    )
    b.metadata["limitations"].append(
        "Ideal constrained open bearings, rigid clamp and motor parameters are authored; no lithography, optical focus, alignment accuracy or fabrication process is validated."
    )
