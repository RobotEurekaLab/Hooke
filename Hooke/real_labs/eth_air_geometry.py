"""Original open filter-access fixture; all internal dimensions are authored.

The ETH tour thumbnail supports a crowded benchtop setting, not this device's
identity or internal mechanism. The generic monitor/manifold remains explicitly
separate from the unidentified cream apparatus visible in that reference.
"""

import math


def _annulus(b, parent, name, center, inner, outer, halfheight, count=24):
    """Contact-capable convex sectors leave a genuine open axial bore."""
    import xml.etree.ElementTree as ET

    for i in range(count):
        a, c = 2 * math.pi * i / count, 2 * math.pi * (i + 1) / count
        points = [
            (radius * math.cos(angle), radius * math.sin(angle), z)
            for z in (-halfheight, halfheight)
            for radius, angle in ((inner, a), (outer, a), (outer, c), (inner, c))
        ]
        mesh = f"{b.name}__{name}_mesh_{i}"
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=mesh,
            vertex=" ".join(f"{v:.10g}" for p in points for v in p),
            face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7",
        )
        b.geom(
            parent,
            "mesh",
            (),
            center,
            "metal",
            mesh=mesh,
            collision=True,
            name=f"{b.name}__{name}_{i}",
        )


def build_sampler(b, base, params):
    """Retain two original joint frames/ranges and the free carrier interface."""

    def box(parent, key, pos, size, material="metal", **kwargs):
        return b.box(
            parent,
            pos,
            size,
            material,
            collision=True,
            name=f"{b.name}__{key}",
            **kwargs,
        )

    def cyl(parent, key, pos, radius, halfheight, material="metal", **kwargs):
        return b.cylinder(
            parent,
            pos,
            radius,
            halfheight,
            material,
            collision=True,
            name=f"{b.name}__{key}",
            **kwargs,
        )

    # An open recessed top replaces the former solid cap that intersected the
    # carriage. The existing published monitor envelope is only a size reference;
    # this exposed filter fixture and its guides are original surrogate hardware.
    for x in (-0.21, -0.08):
        for y in (-0.06, 0.09):
            cyl(base, f"monitor_foot_{x}_{y}", (x, y, 0.0085), 0.011, 0.0085, "rubber")
    box(base, "monitor_lower", (-0.14, 0.016, 0.0675), (0.108, 0.112, 0.0505), "blue")
    for sign in (-1, 1):
        x = -0.14 + sign * 0.094
        box(
            base,
            f"monitor_cheek_{sign}",
            (x, 0.016, 0.1335),
            (0.014, 0.112, 0.0155),
            "blue",
        )
    box(base, "monitor_rear", (-0.14, 0.098, 0.139), (0.080, 0.030, 0.021), "blue")
    b.screen(base, (-0.14, -0.100, 0.070), 0.132, 0.077)
    for side in (-1, 1):
        for row in range(6):
            b.box(
                base,
                (-0.14 + side * 0.1082, -0.047 + row * 0.020, 0.082),
                (0.0003, 0.006, 0.012),
                "dark",
            )
        for y in (-0.076, 0.099):
            cyl(
                base,
                f"cover_fastener_{side}_{y}",
                (-0.14 + side * 0.094, y, 0.150),
                0.0025,
                0.001,
                "bright",
            )

    # Two full-length supported rectangular rails, with moving four-sided open
    # bearing shoes. 0.5 mm running gaps are geometric estimates, not tolerances
    # from ETH or a manufacturer. Extended feet carry the front rail supports.
    for sign in (-1, 1):
        x = -0.14 + sign * 0.068
        box(base, f"rail_{sign}", (x, -0.0625, 0.137), (0.004, 0.1425, 0.005))
        for end, y in (("front", -0.202), ("rear", 0.074)):
            if end == "front":
                cyl(base, f"rail_foot_{sign}", (x, y, 0.004), 0.014, 0.004, "rubber")
                box(
                    base,
                    f"rail_post_{sign}_{end}",
                    (x, y, 0.072),
                    (0.009, 0.008, 0.064),
                )
            else:
                box(
                    base,
                    f"rail_post_{sign}_{end}",
                    (x, y, 0.124),
                    (0.009, 0.008, 0.012),
                )

    drawer = b.moving(
        base, "filter_drawer", (-0.14, -0.025, 0.156), (0, -1, 0), (0, 0.14), mass=0.15
    )
    box(drawer, "filter_plate", (0, 0, 0), (0.054, 0.059, 0.009), "dark")
    for sign in (-1, 1):
        x, z = sign * 0.068, -0.019
        box(
            drawer,
            f"bearing_top_{sign}",
            (x, 0.020, z + 0.008),
            (0.0095, 0.017, 0.0025),
        )
        box(
            drawer,
            f"bearing_bottom_{sign}",
            (x, 0.020, z - 0.008),
            (0.0095, 0.017, 0.0025),
        )
        for wall in (-1, 1):
            box(
                drawer,
                f"bearing_side_{sign}_{wall}",
                (x + wall * 0.007, 0.020, z),
                (0.0025, 0.017, 0.0055),
            )
        box(
            drawer,
            f"bearing_mount_{sign}",
            (sign * 0.06375, 0.020, -0.005),
            (0.01375, 0.017, 0.004),
        )
    box(drawer, "filter_front", (0, -0.062, 0.003), (0.054, 0.003, 0.012), "blue")
    for x in (-0.027, 0.027):
        b.rod(
            drawer,
            (x, -0.065, 0.004),
            (x, -0.079, 0.004),
            0.003,
            "metal",
            collision=True,
            name=f"{b.name}__handle_standoff_{x}",
        )
    b.rod(
        drawer,
        (-0.027, -0.079, 0.004),
        (0.027, -0.079, 0.004),
        0.003,
        "metal",
        collision=True,
        name=f"{b.name}__handle_bar",
    )
    cyl(drawer, "filter_puck", (0, 0, 0.011), 0.0185, 0.002, "cream")
    # The retained inert carrier is square. A square bezel avoids the old
    # decorative circular ring cutting through its corners.
    for sign in (-1, 1):
        box(
            drawer,
            f"filter_bezel_x_{sign}",
            (sign * 0.023, 0, 0.011),
            (0.002, 0.025, 0.002),
        )
        box(
            drawer,
            f"filter_bezel_y_{sign}",
            (0, sign * 0.023, 0.011),
            (0.021, 0.002, 0.002),
        )
    b.site(drawer, "filter", (0, 0, 0.017))

    # The original suspended stack is carried by an actual base/stand/crossarm.
    for x in (0.10, 0.25):
        for y in (0.02, 0.15):
            cyl(base, f"stand_foot_{x}_{y}", (x, y, 0.0025), 0.012, 0.0025, "rubber")
    box(base, "stand_base", (0.176, 0.083, 0.025), (0.115, 0.13, 0.020))
    b.rod(
        base,
        (0.245, 0.125, 0.04),
        (0.245, 0.125, 0.60),
        0.008,
        "metal",
        collision=True,
        name=f"{b.name}__stand_post",
    )
    b.rod(
        base,
        (0.245, 0.125, 0.55),
        (0.17, 0.08, 0.55),
        0.008,
        "metal",
        collision=True,
        name=f"{b.name}__stand_arm",
    )
    for z in (0.10, 0.55):
        cyl(base, f"stand_collar_{z}", (0.245, 0.125, z), 0.013, 0.011)
        box(base, f"stand_lock_{z}", (0.260, 0.125, z), (0.007, 0.005, 0.004), "dark")
    cyl(base, "stack_lower", (0.17, 0.08, 0.34), 0.024, 0.090)
    cyl(base, "stack_body", (0.17, 0.08, 0.468), 0.038, 0.038, "bright")
    cyl(base, "stack_neck", (0.17, 0.08, 0.5235), 0.018, 0.0215)
    cyl(base, "stack_cap", (0.17, 0.08, 0.56), 0.051, 0.015)
    cyl(base, "stack_gasket", (0.17, 0.08, 0.576), 0.048, 0.001, "rubber")
    cyl(base, "stack_top", (0.17, 0.08, 0.582), 0.056, 0.005, "dark")
    for i in range(6):
        angle = i * math.pi / 3
        cyl(
            base,
            f"stack_fastener_{i}",
            (0.17 + 0.043 * math.cos(angle), 0.08 + 0.043 * math.sin(angle), 0.589),
            0.0028,
            0.002,
            "bright",
        )
    b.site(base, "sampling_inlet", (0.17, 0.08, 0.54))
    cyl(base, "monitor_connector", (-0.20, 0.08, 0.174), 0.008, 0.025)
    b.rod(
        base,
        (-0.20, 0.08, 0.199),
        (-0.22, 0.08, 0.30),
        0.004,
        "dark",
        collision=True,
        name=f"{b.name}__service_riser",
    )
    b.rod(
        base,
        (-0.22, 0.08, 0.30),
        (0.17, 0.08, 0.34),
        0.004,
        "dark",
        collision=True,
        name=f"{b.name}__service_link",
    )

    # An actual open stem bearing fills the former gap below the knob without
    # placing a solid fixed block through the rotating shaft.
    box(base, "valve_foot", (0.005, -0.083, 0.005), (0.035, 0.035, 0.005), "rubber")
    box(base, "valve_floor", (0.005, -0.083, 0.0265), (0.035, 0.035, 0.0165), "shell")
    for sign in (-1, 1):
        box(
            base,
            f"valve_wall_x_{sign}",
            (0.005 + sign * 0.0205, -0.083, 0.0515),
            (0.0145, 0.035, 0.0085),
            "shell",
        )
        box(
            base,
            f"valve_wall_y_{sign}",
            (0.005, -0.083 + sign * 0.0205, 0.0515),
            (0.006, 0.0145, 0.0085),
            "shell",
        )
    _annulus(b, base, "valve_bearing", (0.005, -0.083, 0.05275), 0.0045, 0.008, 0.00875)
    valve = b.moving(
        base,
        "flow_valve",
        (0.005, -0.083, 0.075),
        (0, 0, 1),
        (0, math.pi),
        kind="hinge",
        mass=0.02,
    )
    cyl(valve, "valve_stem", (0, 0, -0.021), 0.004, 0.010)
    cyl(valve, "valve_knob", (0, 0, 0), 0.018, 0.013, "dark")
    box(valve, "valve_mark", (0, 0, 0.014), (0.014, 0.002, 0.001), "bright")
    b.metadata["capabilities"] = [
        "37 mm inert filter carrier on a contact-supported open drawer",
        "140 mm drawer travel with continuous rail/bearing engagement",
        "flow-valve handle angle; no implemented flow response",
        "sampling-inlet and filter target sites",
    ]
    b.metadata["filter_access_interface"] = {
        "revision": 1,
        "drawer_control": "filter_drawer",
        "exposure_target_m": 0.14,
        "holder_site": b.metadata["sites"]["filter"],
        "bearing_running_gap_m": 0.0005,
        "basis": "Authored open inspection mechanism, not an OEM device replica",
        "physical_scope": "Access to an already loaded free inert carrier; no placement or sampling science",
    }
    b.metadata["limitations"].append(
        "The ETH thumbnail does not identify this monitor/manifold; guides, valve interior, fasteners and service routing are original estimates. No validated flow, filtration or aerosol response."
    )
