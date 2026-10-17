"""Authored connected microscope guides and a disconnected dry service setup.

These internal dimensions are engineering estimates, not a reconstruction of
Indiana's microscope. No fluid transfer, optical response or biology is modeled.
"""

import math

from real_labs.optical_stages import aperture_plate


PREFIX = "indiana_"


def _box(b, body, key, pos, half, material="metal"):
    return b.box(
        body, pos, half, material, collision=True, name=f"{b.name}__{PREFIX}{key}"
    )


def _cylinder(b, body, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        body,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=f"{b.name}__{PREFIX}{key}",
        **kwargs,
    )


def _guides(b, base, stage):
    """Replace only this station's moving assemblies; preserve all frames."""
    focus = base.find(f"body[@name='{b.name}__body_focus']")
    stage_x = base.find(f"body[@name='{b.name}__body_stage_x']")
    if focus is None or stage_x is None:
        raise ValueError("Indiana guide repair requires the retained microscope frames")
    for body in (focus, stage_x, stage):
        for geom in list(body.findall("geom")):
            body.remove(geom)
    # A 0.2 mm guide clearance represents the ideal joint bearing, avoiding
    # redundant dry-friction contacts between already constrained guide faces.
    _box(b, base, "focus_column_guide", (0, 0.1236, 0.291), (0.03, 0.0044, 0.025))
    _box(b, focus, "focus_arm", (0, 0.092, 0), (0.025, 0.027, 0.01))
    _cylinder(b, focus, "nosepiece", (0, 0.026, 0), 0.047, 0.015, "dark")
    for i in range(5):
        angle = i * math.tau / 5
        x, y = 0.020 * math.sin(angle), 0.020 - 0.020 * math.cos(angle)
        # A longer active objective and shorter parked barrels clear the entire
        # retained focus travel. Their working distance is an authored estimate.
        center, half, collar, face = (
            (0.028, 0.020, 0.039, 0.052) if i == 0 else (0.019, 0.012, 0.025, 0.035)
        )
        _cylinder(b, focus, f"objective_{i}", (x, y, center), 0.012, half, "bright")
        _cylinder(
            b,
            focus,
            f"objective_collar_{i}",
            (x, y, collar),
            0.0125,
            0.003,
            "blue" if i % 2 else "warning",
        )
        _cylinder(b, focus, f"objective_face_{i}", (x, y, face), 0.007, 0.004, "lens")
    b.site(focus, "inspection_objective", (0, 0, 0.056), 0.001)

    # Column-mounted crossrail supports the X carriage along its full travel.
    _box(b, base, "x_column_rail", (0, 0.1136, 0.3238), (0.235, 0.0144, 0.012))
    for side in (-1, 1):
        _box(
            b,
            base,
            f"x_rail_bracket_{side}",
            (side * 0.08, 0.1135, 0.291),
            (0.012, 0.0145, 0.021),
        )
        _box(
            b,
            stage_x,
            f"x_crossbar_{side}",
            (0, side * 0.085, 0),
            (0.17, 0.014, 0.012),
            "dark",
        )
        _box(
            b,
            stage_x,
            f"x_side_{side}",
            (side * 0.165, 0, 0),
            (0.005, 0.071, 0.012),
            "dark",
        )
        # These bearing strips bridge the former 2 mm gap between the XY plates.
        _box(
            b,
            stage_x,
            f"y_bearing_rail_{side}",
            (side * 0.104, 0, 0.013),
            (0.005, 0.119, 0.001),
            "bright",
        )
    aperture_plate(
        b,
        stage,
        PREFIX + "xy_plate",
        (0, 0, 0),
        (0.135, 0.079, 0.007),
        (0.073, 0.053),
        "dark",
    )
    for side in (-1, 1):
        _box(
            b,
            stage,
            f"slide_end_support_{side}",
            (side * 0.0605, 0, 0.00915),
            (0.0295, 0.0125, 0.00215),
        )
        # A continuous end support and over-edge keeper leave the optical centre open.
        _box(
            b,
            stage,
            f"slide_keeper_post_{side}",
            (side * 0.0395, 0, 0.01225),
            (0.001, 0.008, 0.00095),
        )
        _box(
            b,
            stage,
            f"slide_keeper_{side}",
            (side * 0.038, 0, 0.01345),
            (0.0025, 0.008, 0.00025),
        )
        _box(
            b,
            stage,
            f"slide_keeper_pad_{side}",
            (side * 0.0365, 0, 0.01295),
            (0.001, 0.008, 0.00025),
            "rubber",
        )


def _chip(b, stage, params):
    _box(b, stage, "glass_slide", (0, 0, 0.012), (0.0375, 0.0125, 0.0007), "glass")
    width = float(params.get("channel_width_m", 0.0001))
    if not 0 < width <= 0.0002:
        raise ValueError(
            "Indiana trace width must remain a positive submillimetre dimension"
        )
    for branch in (-1, 1):
        points = [
            (-0.031, 0, 0.01275),
            (-0.028, branch * 0.003, 0.01275),
            (0.025, branch * 0.003, 0.01275),
            (0.028, 0, 0.01275),
        ]
        for index, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(
                stage,
                a,
                c,
                width / 2,
                "cyan",
                name=f"{b.name}__{PREFIX}channel_trace_{branch}_{index}",
            )
    for x, key in ((-0.031, "chip_inlet"), (0.028, "chip_outlet")):
        _cylinder(b, stage, key + "_fitting", (x, 0, 0.016), 0.0025, 0.003, "cream")
        _cylinder(b, stage, key + "_dry_cap", (x, 0, 0.0195), 0.0028, 0.0005, "blue")
        b.site(stage, key, (x, 0, 0.019), 0.001)
    return width


def _controller(b, base):
    _box(
        b, base, "controller_shell", (0.35, 0.068, 0.09), (0.13, 0.122, 0.075), "shell"
    )
    for x in (0.245, 0.455):
        for y in (-0.032, 0.168):
            _cylinder(
                b,
                base,
                f"controller_foot_{x}_{y}",
                (x, y, 0.0075),
                0.013,
                0.0075,
                "rubber",
            )
    b.screen(base, (0.35, -0.058, 0.111), 0.155, 0.075)
    for i in range(4):
        x = 0.275 + i * 0.05
        _cylinder(b, base, f"pressure_fitting_{i}", (x, 0.075, 0.185), 0.011, 0.021)
        _cylinder(
            b, base, f"pressure_collar_{i}", (x, 0.075, 0.21), 0.012, 0.003, "blue"
        )
        _cylinder(b, base, f"pressure_nipple_{i}", (x, 0.075, 0.2145), 0.003, 0.0015)
        b.site(base, f"pressure_port_{i+1}", (x, 0.075, 0.216))
        if i >= 2:
            _cylinder(
                b, base, f"pressure_cap_{i}", (x, 0.075, 0.217), 0.004, 0.001, "dark"
            )
    docks = []
    for i in range(2):
        start = (0.275 + i * 0.05, 0.075, 0.216)
        end = (0.245 + i * 0.05, -0.010, 0.201)
        _cylinder(
            b,
            base,
            f"parked_hose_dock_{i}",
            (end[0], end[1], 0.181),
            0.006,
            0.016,
            "dark",
        )
        _cylinder(
            b, base, f"parked_hose_nipple_{i}", (end[0], end[1], 0.199), 0.003, 0.002
        )
        b.site(base, f"parked_hose_end_{i}", end, 0.001)
        # Both ends are fixed to the controller. This is a parked service lead,
        # not a fake rigid hose joining an independently moving specimen stage.
        points = []
        for n in range(17):
            t = n / 16
            points.append(
                tuple((1 - t) * a + t * c for a, c in zip(start, end))[:2]
                + ((1 - t) * start[2] + t * end[2] + 0.065 * math.sin(math.pi * t),)
            )
        for n, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(
                base,
                a,
                c,
                0.0015,
                "cyan",
                collision=True,
                name=f"{b.name}__{PREFIX}parked_hose_{i}_{n}",
            )
        docks.append(
            dict(
                source=f"{b.name}__site_pressure_port_{i+1}",
                destination=f"{b.name}__site_parked_hose_end_{i}",
                body=base.get("name"),
            )
        )
    q = (0.70710678, 0.70710678, 0, 0)
    _cylinder(
        b, base, "valve_bezel", (0.435, -0.0545, 0.037), 0.018, 0.0005, "metal", quat=q
    )
    valve = b.moving(
        base,
        "flow_valve",
        (0.435, -0.063, 0.037),
        (0, 1, 0),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.02,
    )
    _cylinder(b, valve, "valve_knob", (0, -0.002, 0), 0.016, 0.01, "dark", quat=q)
    _box(b, valve, "valve_pointer", (0, -0.014, 0), (0.012, 0.002, 0.002), "bright")
    return docks


def complete_station(b, base, stage, params):
    _guides(b, base, stage)
    width = _chip(b, stage, params)
    docks = _controller(b, base)
    b.metadata["inspection_interface"] = dict(
        objective_site=f"{b.name}__site_inspection_objective",
        objective_face=f"{b.name}__{PREFIX}objective_face_0",
        slide_geom=f"{b.name}__{PREFIX}glass_slide",
        trace_width_m=width,
        trace_representation="True-scale coloured centreline traces on glass; no resolved fluid-channel walls or fluid solver",
        services_connected_to_chip=False,
        service_configuration="Chip ports capped; controller leads parked on fixed supported docks for dry inspection",
        parked_leads=docks,
        incomplete_capabilities=[
            "Pressure-connected tubing that follows XY travel",
            "Fluid transport",
            "Calibrated optical focus or fluorescence",
        ],
        guide_basis="Original ideal-joint-supported bearing geometry and estimated objective lengths; no OEM CAD or verified Indiana apparatus reconstruction",
        required_contact_geoms=[
            geom.get("name")
            for geom in base.iter("geom")
            if geom.get("name", "").startswith(f"{b.name}__{PREFIX}")
            and geom.get("contype") == "1"
            and geom.get("conaffinity") == "1"
        ],
    )
    b.metadata["capabilities"] = [
        "XY inert-carrier registration",
        "7 mm objective positioning",
        "Supported dry slide and capped chip-port targets",
        "Mechanical valve-knob command",
        "Top-view geometric inspection camera",
    ]
    b.metadata["limitations"].append(
        "The dry configuration has disconnected pressure services; parked leads do not qualify live moving-hose or microfluidic operation"
    )
