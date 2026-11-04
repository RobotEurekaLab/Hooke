"""Open, contact-capable guides for the representative EPFL coil microscope.

The cached MICROBS image depicts a different microsurgery mechanism. These
guides and the specimen-mounted coil cage are authored mechanical fixtures,
not a reconstruction of that instrument or an electromagnetic/optical model.
"""

import math


def build_station(b, base, params):
    from real_labs.equipment import _microscope

    stage = _microscope(b, base)
    # Recess the stationary rear column behind the complete Y travel envelope.
    column = base.find("./geom[@name='%s__box_10']" % b.name)
    column.set("pos", "0 .223 .33")
    column.set("size", ".065 .035 .204")
    column.set("name", f"{b.name}__epfl_rear_column")
    for panel in base.findall("geom"):
        if panel.get("pos") == "0.089 0.163 0.345":
            panel.set("pos", ".068 .223 .345")
            panel.set("size", ".003 .035 .072")
    focus = base.find("./body[@name='%s__body_focus']" % b.name)
    xstage = base.find("./body[@name='%s__body_stage_x']" % b.name)
    for body in (focus, xstage, stage):
        for geom in list(body.findall("geom")):
            body.remove(geom)

    def box(body, key, pos, size, material="metal"):
        return b.box(
            body, pos, size, material, collision=True, name=f"{b.name}__epfl_{key}"
        )

    def cylinder(body, key, pos, radius, half_height, material="metal"):
        return b.cylinder(
            body,
            pos,
            radius,
            half_height,
            material,
            collision=True,
            name=f"{b.name}__epfl_{key}",
        )

    # Original base top is z=.122. Four grounded-to-chassis pedestals carry
    # open X guides; no rail is hidden inside a solid moving block.
    box(base, "guide_foundation", (0, 0, 0.126), (0.252, 0.162, 0.004), "shell")
    box(base, "rear_column_foot", (0, 0.203, 0.124), (0.060, 0.027, 0.002), "shell")
    for side in (-1, 1):
        y = side * 0.150
        box(base, f"x_rail_{side}", (0, y, 0.300), (0.245, 0.004, 0.004))
        for end in (-1, 1):
            box(
                base,
                f"x_pedestal_{side}_{end}",
                (end * 0.237, y, 0.213),
                (0.008, 0.012, 0.083),
                "shell",
            )
        box(xstage, f"x_beam_{side}", (0, y, 0), (0.183, 0.012, 0.010), "dark")
        for end in (-1, 1):
            x = end * 0.105
            # .5 mm clearance on every rectangular-rail face.
            box(
                xstage,
                f"x_shoe_top_{side}_{end}",
                (x, y, -0.01475),
                (0.035, 0.012, 0.00475),
            )
            box(
                xstage,
                f"x_shoe_bottom_{side}_{end}",
                (x, y, -0.03125),
                (0.035, 0.012, 0.00275),
            )
            for face in (-1, 1):
                box(
                    xstage,
                    f"x_shoe_side_{side}_{end}_{face}",
                    (x, y + face * 0.00825, -0.024),
                    (0.035, 0.00375, 0.0045),
                )

    # Two cross-members connect the X carriage. The Y rails sit on end
    # pedestals above them; captive shoes connect the upper open frame.
    for side in (-1, 1):
        x = side * 0.125
        box(
            xstage,
            f"x_outer_tie_{side}",
            (side * 0.180, 0, 0),
            (0.003, 0.138, 0.010),
            "dark",
        )
        box(xstage, f"y_rail_{side}", (x, 0, 0.015), (0.003, 0.170, 0.003))
        for end in (-1, 1):
            box(
                xstage,
                f"y_rail_mount_{side}_{end}",
                (x, end * 0.138, 0.011),
                (0.008, 0.012, 0.001),
            )
        box(stage, f"y_side_{side}", (x, 0, 0.011), (0.017, 0.095, 0.007), "dark")
        # Y rail is at stage-local z=-.006; four walls leave its bore open.
        box(stage, f"y_shoe_top_{side}", (x, 0, 0.00075), (0.009, 0.050, 0.00325))
        box(stage, f"y_shoe_bottom_{side}", (x, 0, -0.01175), (0.009, 0.050, 0.00225))
        for face in (-1, 1):
            box(
                stage,
                f"y_shoe_side_{side}_{face}",
                (x + face * 0.00625, 0, -0.006),
                (0.00275, 0.050, 0.0035),
            )
        box(
            stage,
            f"y_crossmember_{side}",
            (0, side * 0.080, 0.011),
            (0.108, 0.015, 0.007),
            "dark",
        )
        # The glass has a real underside seat and upper keepers. A central
        # opening remains for the objective and unscaled microscopic marker.
        box(
            stage, f"slide_seat_{side}", (side * 0.070, 0, 0.010), (0.052, 0.003, 0.003)
        )
        box(
            stage,
            f"slide_keeper_post_{side}",
            (side * 0.024, 0, 0.015),
            (0.002, 0.003, 0.002),
        )
        box(
            stage,
            f"slide_keeper_bridge_{side}",
            (side * 0.021, 0, 0.018),
            (0.005, 0.003, 0.001),
        )
        box(
            stage,
            f"slide_keeper_pad_{side}",
            (side * 0.0185, 0, 0.016),
            (0.0015, 0.003, 0.001),
            "rubber",
        )

    # A real rectangular focus stem runs through four fixed bearing walls.
    # The forward arm clears the top of the fixed sleeve for the full 7 mm
    # stroke, rather than passing through the microscope rear column.
    box(focus, "focus_stem", (0.095, 0.113, -0.020), (0.008, 0.007, 0.037))
    # Same solid union as overlapping bars, with butt joints so duplicate
    # coplanar faces cannot produce dark slots in exported close views.
    box(focus, "focus_arm", (0, 0.0695, 0.012), (0.010, 0.0335, 0.004))
    box(focus, "focus_rear_arm", (0.0475, 0.113, 0.012), (0.0575, 0.010, 0.004))
    box(focus, "focus_turret_support", (0, 0.042, 0.022), (0.012, 0.012, 0.007))
    for side in (-1, 1):
        box(
            base,
            f"focus_guide_side_{side}",
            (0.095 + side * 0.01325, 0.113, 0.250),
            (0.00475, 0.015, 0.042),
        )
    box(base, "focus_guide_front", (0.095, 0.10175, 0.250), (0.0085, 0.00375, 0.042))
    box(base, "focus_guide_back", (0.095, 0.12425, 0.250), (0.0085, 0.00375, 0.042))
    box(base, "focus_guide_mount", (0.075, 0.175, 0.235), (0.038, 0.047, 0.012))
    cylinder(focus, "objective_turret", (0, 0.042, 0), 0.063, 0.015, "dark")
    for index in range(5):
        a = index * 2 * math.pi / 5
        x, y = 0.042 * math.sin(a), 0.042 - 0.042 * math.cos(a)
        cylinder(focus, f"objective_{index}", (x, y, 0.032), 0.012, 0.017, "bright")
        cylinder(focus, f"objective_face_{index}", (x, y, 0.051), 0.007, 0.002, "lens")

    cylinder(stage, "sample_glass", (0, 0, 0.014), 0.020, 0.001, "glass")
    b.geom(
        stage,
        "capsule",
        (0.00006,),
        material="dark",
        collision=True,
        fromto=(-0.0002, 0, 0.017, 0.0002, 0, 0.017),
        name=f"{b.name}__epfl_micro_marker",
    )
    # A visible inert pad contacts the tiny marker; the marker itself is not
    # enlarged or elevated from its original position.
    box(stage, "marker_support", (0, 0, 0.01597), (0.00026, 0.00010, 0.00097), "glass")
    b.site(stage, "micro_agent", (0, 0, 0.017), 0.0001)

    # Keeping the old fixed coils would sweep the wide XY frame through their
    # lower arcs. The entire dry coil fixture now travels with the specimen,
    # with explicit standoffs and a clear optical axis. No flexible live leads
    # or field calibration is claimed.
    def ring(key, center, radius, tube, plane):
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        for i in range(24):
            points = []
            for angle in (i * math.tau / 24, (i + 1) * math.tau / 24):
                p = list(center)
                p[axes[0]] += radius * math.cos(angle)
                p[axes[1]] += radius * math.sin(angle)
                points.append(p)
            b.rod(
                stage,
                points[0],
                points[1],
                tube,
                "copper",
                collision=True,
                name=f"{b.name}__epfl_{key}_{i}",
            )

    for side in (-1, 1):
        for offset in (-0.006, 0, 0.006):
            ring(
                f"coil_x_{side}_{offset}",
                (side * 0.069 + offset, 0, 0.063),
                0.035,
                0.0025,
                "yz",
            )
            ring(
                f"coil_y_{side}_{offset}",
                (0, side * 0.070 + offset, 0.063),
                0.035,
                0.0025,
                "xz",
            )
        box(
            stage,
            f"coil_x_mount_{side}",
            (side * 0.091, 0, 0.02275),
            (0.034, 0.009, 0.00475),
            "dark",
        )
        box(
            stage,
            f"coil_y_mount_{side}",
            (0, side * 0.070, 0.02275),
            (0.012, 0.011, 0.00475),
            "dark",
        )
        for sx in (-1, 1):
            box(
                stage,
                f"coil_upright_{side}_{sx}",
                (sx * 0.039, side * 0.039, 0.052),
                (0.003, 0.003, 0.0415),
                "dark",
            )
            box(
                stage,
                f"coil_upright_arm_{side}_{sx}",
                (sx * 0.039, side * 0.052, 0.011),
                (0.005, 0.013, 0.007),
                "dark",
            )
    ring("coil_z_low", (0, 0, 0.036), 0.055, 0.0035, "xy")
    ring("coil_z_high", (0, 0, 0.090), 0.055, 0.0035, "xy")

    b.box(base, (0.42, 0.095, 0.083), (0.14, 0.13, 0.06), "shell", collision=True)
    b.screen(base, (0.42, -0.037, 0.095), 0.16, 0.065)
    for x in (0.315, 0.525):
        for y in (0.005, 0.185):
            box(
                base,
                f"controller_foot_{x}_{y}",
                (x, y, 0.0115),
                (0.012, 0.012, 0.0115),
                "rubber",
            )
    # A thin sole joins all four original feet (top .024) to the original
    # enabled chassis core (bottom .028), including the forward-offset feet.
    box(base, "chassis_sole", (0, 0.015, 0.026), (0.170, 0.240, 0.002), "rubber")
    b.metadata["capabilities"] = [
        "XY specimen positioning",
        "objective focus",
        "optical sample camera",
        "retained microscopic marker and supported inert coil cage",
    ]
    b.metadata["limitations"].append(
        "Authored open guides use 0.5 mm running clearances and ideal original joints. The inert specimen-mounted coil cage is not documented EPFL hardware, electrically connected actuation or a field model. The retained marker (0.4 mm centerline, 0.52 mm capsule end-to-end length) and original camera do not establish calibrated microscope optics or microrobot locomotion."
    )
    b.metadata["mechanical_interface"] = dict(
        revision=2,
        guide_clearance_m=0.0005,
        coils_move_with_specimen=True,
        coil_attachment_changed=True,
        marker_centerline_length_m=0.0004,
        marker_total_length_m=0.00052,
        marker_diameter_m=0.00012,
        scientific_model=False,
    )
