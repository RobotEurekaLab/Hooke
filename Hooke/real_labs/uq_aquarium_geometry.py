"""Source-informed aquarium context and an authored inert probe bridge.

The small UQ photograph supports round tanks, yellow racks and suspended
services. It does not establish dimensions or a robotic sampler. The open
bearings, depth datum and ideal drives below are explicit authored interfaces.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


class Parts:
    def __init__(self, builder):
        self.b = builder

    def name(self, key):
        return f"{self.b.name}__uq_{key}"

    def box(self, parent, key, pos, half, material="metal", **kwargs):
        return self.b.box(
            parent, pos, half, material, collision=True, name=self.name(key), **kwargs
        )

    def cylinder(self, parent, key, pos, radius, half, material="metal", **kwargs):
        return self.b.cylinder(
            parent,
            pos,
            radius,
            half,
            material,
            collision=True,
            name=self.name(key),
            **kwargs,
        )

    def rod(self, parent, key, start, end, radius=0.008, material="metal"):
        return self.b.rod(
            parent, start, end, radius, material, collision=True, name=self.name(key)
        )

    def bore(self, parent, key, pos, inner, outer, half, axis="z", material="metal"):
        annular_mount(
            self.b,
            parent,
            key,
            pos,
            inner,
            outer,
            half,
            axis=axis,
            proxies=True,
            material=material,
            namespace="uq",
        )


def _tank(p, base):
    b = p.b
    for i, a in enumerate((0, math.tau / 3, 2 * math.tau / 3)):
        p.box(
            base,
            f"tank_foot_{i}",
            (0.82 * math.cos(a), 0.82 * math.sin(a), 0.065),
            (0.10, 0.10, 0.065),
            "shell",
        )
    p.cylinder(
        base,
        "tank_bottom",
        (0, 0, 0.145),
        1.128,
        0.035,
        "cyan",
        rgba=(0.025, 0.29, 0.29, 1),
    )
    p.bore(base, "tank_wall", (0, 0, 0.68), 1.115, 1.165, 0.50, material="cyan")
    p.bore(base, "tank_rim", (0, 0, 1.18), 1.103, 1.177, 0.026, material="cyan")
    for geom in base.findall("geom"):
        if geom.get("name", "").endswith(("tank_wall_shell", "tank_rim_shell")):
            geom.set("rgba", ".025 .29 .29 1")
    # Fill the authored basin rather than placing an air interface just below
    # an 8 mm slab. A 0.2 mm overlap into opaque walls/floor avoids coincident
    # optical boundaries; the source photograph does not measure this depth.
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_clear_water",
        rgba="1 1 1 .14",
    )
    bottom, top = 0.1798, 1.064
    # Group 2 identifies the static optical medium for geometric ray queries.
    # It has neither rigid mass nor contacts and supplies no fluid forces.
    b.cylinder(
        base,
        (0, 0, (bottom + top) / 2),
        1.1152,
        (top - bottom) / 2,
        "clear_water",
        name=p.name("display_water"),
        group=2,
        mass=0,
    )
    p.cylinder(base, "drain_boss", (0, -0.83, 0.191), 0.043, 0.012, "cream")
    p.cylinder(base, "drain_cap", (0, -0.83, 0.207), 0.031, 0.005, "dark")


def _frame(p, base):
    for i, x in enumerate((-1.27, 1.27)):
        for j, y in enumerate((-0.27, 0.27)):
            p.box(base, f"foot_{i}_{j}", (x, y, 0.015), (0.12, 0.075, 0.015))
            p.box(base, f"post_{i}_{j}", (x, y, 0.835), (0.023, 0.023, 0.805), "bright")
            for k, dx in enumerate((-0.075, 0.075)):
                p.cylinder(
                    base, f"anchor_{i}_{j}_{k}", (x + dx, y, 0.035), 0.008, 0.005
                )
            p.rod(
                base,
                f"brace_{i}_{j}",
                (x, y, 1.36),
                (x * 0.82, y * 0.74, 1.64),
                0.015,
                "bright",
            )
        p.box(base, f"end_crossbar_{i}", (x, 0, 1.65), (0.038, 0.32, 0.025))
    for j, y in enumerate((-0.20, 0.20)):
        p.box(base, f"beam_{j}", (0, y, 1.665), (1.29, 0.025, 0.025))
        p.rod(base, f"x_rail_{j}", (-1.16, y, 1.707), (1.16, y, 1.707), 0.012, "bright")
        for i, x in enumerate((-1.16, 1.16)):
            p.box(base, f"rail_anchor_{i}_{j}", (x, y, 1.700), (0.03, 0.028, 0.017))
    # The datum hangs from the fixed rear beam, outside the full carriage sweep.
    p.box(base, "datum_top_bracket", (0.30, 0.267, 1.635), (0.030, 0.080, 0.010))
    p.box(base, "datum_strip", (0.30, 0.332, 1.245), (0.026, 0.004, 0.390), "cream")
    for i in range(16):
        z = 0.88 + 0.04 * i
        p.box(
            base, f"datum_tick_{i}", (0.30, 0.327, z), (0.018, 0.0015, 0.0015), "dark"
        )
    p.box(base, "datum_witness", (0.30, 0.326, 1.002), (0.020, 0.002, 0.003), "copper")
    p.b.site(base, "position_datum", (0.30, 0.324, 1.002))
    # Visible controller and parked capped connector. No rigid hose pretends to
    # track the carriage; power transmission and flexible wiring are unmodeled.
    p.box(base, "controller_bracket", (1.27, -0.345, 1.35), (0.045, 0.065, 0.11))
    p.box(base, "controller", (1.27, -0.431, 1.35), (0.092, 0.040, 0.12), "shell")
    p.box(
        base, "controller_screen", (1.27, -0.473, 1.39), (0.065, 0.003, 0.039), "dark"
    )
    p.cylinder(
        base,
        "controller_button",
        (1.27, -0.474, 1.285),
        0.013,
        0.004,
        "copper",
        euler=(math.pi / 2, 0, 0),
    )
    p.rod(
        base,
        "parked_controller_lead",
        (1.24, -0.41, 1.23),
        (1.24, -0.27, 1.19),
        0.008,
        "rubber",
    )
    p.cylinder(base, "parked_connector", (1.24, -0.27, 1.185), 0.016, 0.015, "dark")


def _carriage(p, base):
    b = p.b
    carriage = b.moving(
        base,
        "probe_position",
        (0, 0, 1.744),
        (1, 0, 0),
        (-0.55, 0.55),
        mass=1.0,
        kp=250,
    )
    # Four joined pieces make an actual square opening through the deck.
    for i, x in enumerate((-0.089, 0.089)):
        p.box(carriage, f"carriage_side_{i}", (x, 0, 0), (0.061, 0.26, 0.022), "blue")
    for i, y in enumerate((-0.144, 0.144)):
        p.box(carriage, f"carriage_end_{i}", (0, y, 0), (0.028, 0.116, 0.022), "blue")
    for i, x in enumerate((-0.095, 0.095)):
        for j, y in enumerate((-0.20, 0.20)):
            p.bore(
                carriage,
                f"x_bearing_{i}_{j}",
                (x, y, -0.037),
                0.0126,
                0.016,
                0.029,
                "x",
                "dark",
            )
    # Two open guides remain engaged by the longer real shaft at both ends of
    # the original 280 mm stroke. Ideal joints, not radial preload, carry load.
    for i, x in enumerate((-0.066, 0.066)):
        p.box(
            carriage,
            f"guide_upright_{i}",
            (x, 0, -0.110),
            (0.012, 0.022, 0.090),
            "bright",
        )
    for i, z in enumerate((-0.120, -0.180)):
        p.bore(carriage, f"z_bearing_{i}", (0, 0, z), 0.0126, 0.030, 0.019, "z", "dark")
        for j, x in enumerate((-0.043, 0.043)):
            p.box(carriage, f"guide_arm_{i}_{j}", (x, 0, z), (0.022, 0.015, 0.012))
    p.box(carriage, "actuator_foot", (-0.094, 0.080, 0.035), (0.044, 0.045, 0.016))
    p.box(
        carriage,
        "actuator_housing",
        (-0.094, 0.080, 0.112),
        (0.040, 0.040, 0.061),
        "shell",
    )
    p.cylinder(carriage, "actuator_cap", (-0.094, 0.080, 0.192), 0.033, 0.019, "dark")
    probe = b.moving(
        carriage, "probe_depth", (0, 0, -0.03), (0, 0, -1), (0, 0.28), mass=0.30
    )
    p.cylinder(probe, "probe_shaft", (0, 0, -0.115), 0.012, 0.355, "bright")
    p.cylinder(probe, "probe_head", (0, 0, -0.500), 0.022, 0.030, "dark")
    p.cylinder(probe, "probe_cap", (0, 0, -0.532), 0.020, 0.004, "cream")
    # Preserve the original central site and add a physical front-surface site.
    b.site(probe, "probe_tip", (0, 0, -0.532))
    b.site(probe, "probe_surface", (0, -0.020, -0.532))


def add_bridge(b, base, params):
    p = Parts(b)
    _tank(p, base)
    _frame(p, base)
    _carriage(p, base)
    b.metadata["capabilities"] = [
        "Inert probe traverse",
        "Measured geometric depth registration and return",
    ]
    b.metadata["probe_interface"] = dict(
        surface_site=f"{b.name}__site_probe_surface",
        datum_site=f"{b.name}__site_position_datum",
        cap_geom=p.name("probe_cap"),
        display_surface_geom=p.name("display_water"),
        datum_geom=p.name("datum_witness"),
        inspection_position_m=0.30,
        inspection_depth_m=0.18,
        registration_tolerance_m=0.0002,
        cap_diameter_m=0.040,
        observation_camera="probe_inspection",
        scope="Inert geometric positioning against an offset fixed datum, not a water sample or sensor reading",
    )
    b.metadata["limitations"].extend(
        [
            "UQ photograph establishes tank/rack/service appearance only; this probe bridge, bearings, datum and dimensions are authored and not verified UQ hardware.",
            "All three tanks retain static water appearance. The probe basin has an estimated clear-water optical volume in Blender, fitted to its authored floor and wall with 0.2 mm hidden overlap; other tanks retain approximate display surfaces. No fluid force, buoyancy, collection, organisms, calibrated optics or environmental measurement is modeled.",
            "The 40 mm cap and original controls remain unchanged in scale; the shaft was lengthened upward for full-stroke guide engagement. The datum is laterally offset, so registration never claims contact.",
            "Ideal joints provide actuation; bearing running clearance is geometric, without real preload, transmission, wiring flex or calibrated hardware accuracy.",
        ]
    )


def add_context(world, definition):
    """Complete existing source-informed racks and service load paths locally."""
    from real_labs.university_facilities import _Parts, METAL, WHITE, DARK, YELLOW

    p = _Parts(world, "uq_detail")
    # Roof-supported lines follow the renderer's existing cut-away convention.
    # Rack-supported plumbing remains present when the roof is hidden.
    for geom in list(world.findall("geom")):
        name = geom.get("name", "")
        if name.startswith("facility_uq_utility_hanger"):
            endpoints = list(map(float, geom.get("fromto").split()))
            endpoints[5] = float(definition["room"]["size"][2])
            geom.set("fromto", " ".join(map(str, endpoints)))
        if name.startswith("facility_uq_") and any(
            k in name
            for k in (
                "water_main",
                "utility_hanger",
                "tank_feed_drop",
                "tank_feed_elbow",
                "filter_return",
            )
        ):
            geom.set("name", "arch_roof_" + name)
    for rack_x in (-4.1, 0, 4.1):
        for side in (-1, 1):
            x = rack_x + side * 1.25
            p.pipe(
                "rack_header_support", (x, 3.56, 2.52), (x, 3.53, 2.36), 0.016, METAL
            )
            p.box(
                "rack_header_bracket",
                (x + side * 0.025, 3.53, 2.36),
                (0.045, 0.035, 0.018),
                METAL,
                collision=True,
            )
        for col in range(4):
            x = rack_x - 0.945 + col * 0.63 + 0.20
            p.pipe("rack_supply_elbow", (x, 3.56, 2.52), (x, 3.50, 2.50), 0.010, WHITE)
        for z in (0.35, 1.02, 1.69):
            # Visible source-like common return below the rack tanks.
            p.pipe(
                "rack_return",
                (rack_x - 1.25, 3.52, z + 0.08),
                (rack_x + 1.25, 3.52, z + 0.08),
                0.018,
                WHITE,
            )
            for col in range(4):
                x = rack_x - 0.945 + col * 0.63 + 0.20
                p.pipe(
                    "standpipe_return",
                    (x, 3.40, z + 0.08),
                    (x, 3.52, z + 0.08),
                    0.010,
                    WHITE,
                )
        for side in (-1, 1):
            p.pipe(
                "rack_diagonal",
                (rack_x + side * 1.30, 3.54, 0.37),
                (rack_x - side * 1.30, 3.54, 1.67),
                0.010,
                YELLOW,
            )
    for x in (-3.35, 0, 3.15):
        geom = p.pipe(
            "feed_branch",
            (x + 0.78, -1.60, 2.92),
            (x + 0.78, -0.82, 2.92),
            0.018,
            WHITE,
        )
        geom.set("name", "arch_roof_" + geom.get("name"))
    for x in (5.25, 6.15):
        for y in (-0.82, 0.22):
            p.box("sump_foot", (x, y, 0.035), (0.07, 0.07, 0.035), DARK, collision=True)
    # Source photo shows dense exposed overhead services, not a surveyed route.
    for x in (-5.3, 0, 5.3):
        geom = p.pipe("header_link", (x, -1.6, 2.92), (x, 3.60, 2.92), 0.021, WHITE)
        geom.set("name", "arch_roof_" + geom.get("name"))
    for rack_x in (-4.1, 0, 4.1):
        geom = p.pipe(
            "rack_header_feed", (rack_x, 3.60, 2.92), (rack_x, 3.56, 2.52), 0.021, WHITE
        )
        geom.set("name", "arch_roof_" + geom.get("name"))
    geom = p.pipe(
        "filter_header_link", (5.77, -0.30, 2.91), (5.3, -0.30, 2.92), 0.025, WHITE
    )
    geom.set("name", "arch_roof_" + geom.get("name"))
