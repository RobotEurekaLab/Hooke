"""Oxford photo-informed probe fixture; geometry and coarse motion only."""

import math

from real_labs.architecture import box
from real_labs.oxford_probe_geometry import add_probe_holder

SOURCE = "https://interface.web.ox.ac.uk/infrastructure"
SOURCES = {
    "semiconductor_probe_station": {
        "reference": "Oxford Bonilla infrastructure page: inspected metal-stage/contact-probe close-up",
        "url": SOURCE,
        "dimensions_m": [0.52, 0.43, 0.29],
        "dimension_basis": "Authored metric estimate from an undimensioned close-up. Contact tips, support blocks and adjustment axes are representative, not measured OEM geometry.",
    }
}


def _probe_station(b, base, params):
    b.feet(base, 0.46, 0.38, 0.014)
    b.box(base, (0, 0, 0.028), (0.23, 0.19, 0.014), "metal", collision=True)
    b.box(base, (0, 0, 0.045), (0.22, 0.18, 0.003), "bright")
    for x in (-0.21, 0.21):
        for y in (-0.17, 0.17):
            b.cylinder(base, (x, y, 0.052), 0.008, 0.005, "dark")
            b.box(base, (x, y, 0.0572), (0.006, 0.001, 0.0005), "metal")
    for y in (-0.052, 0.052):
        b.box(base, (0, y, 0.055), (0.088, 0.008, 0.007), "bright")
    sample_x = b.moving(
        base, "sample_x", (0, 0, 0.067), (1, 0, 0), (-0.020, 0.020), mass=0.2, kp=260
    )
    b.box(sample_x, (0, 0, 0), (0.071, 0.064, 0.005), "metal", collision=True)
    b.cylinder(sample_x, (0, 0, 0.014), 0.036, 0.009, "dark", collision=True)
    b.box(
        sample_x,
        (0, 0, 0.0235),
        (0.006, 0.006, 0.0005),
        "dark",
        collision=True,
        name=b.unique("probe_inert_die"),
    )
    b.site(sample_x, "inert_die", (0, 0, 0.0241), 0.0002)
    for x in (-0.003, 0.003):
        b.box(
            sample_x,
            (x, 0, 0.02405),
            (0.001, 0.0015, 0.00005),
            "copper",
            collision=True,
            name=b.unique("probe_contact_pad"),
        )
    # Original surrogate slide origins remain; open collars replace solid guide
    # overlaps before enabling contact on the two asymmetric probe assemblies.
    for side in (-1, 1):
        x = side * 0.155
        b.box(base, (x, 0.082, 0.072), (0.043, 0.047, 0.024), "metal", collision=True)
        b.rod(
            base,
            (x, 0.082, 0.094),
            (x, 0.082, 0.251),
            0.008,
            "bright",
            collision=True,
            name=b.unique("probe_guide_post"),
        )
        b.cylinder(base, (x, 0.082, 0.217), 0.010, 0.022, "metal", collision=True)
        b.cylinder(base, (x, 0.082, 0.232), 0.015, 0.007, "dark", collision=True)
        b.rod(
            base,
            (x - side * 0.02, 0.082, 0.188),
            (x + side * 0.045, 0.082, 0.188),
            0.006,
            "copper",
        )
        b.cylinder(
            base,
            (x + side * 0.045, 0.082, 0.188),
            0.016,
            0.016,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        arm = b.moving(
            base,
            "left_approach" if side < 0 else "right_approach",
            (x, 0.082, 0.13),
            (0, 0, 1),
            (-0.005, 0.022),
            mass=0.035,
            kp=180,
            force=20,
        )
        add_probe_holder(b, arm, side)
        b.rod(base, (x, 0.094, 0.15), (x + side * 0.045, 0.17, 0.11), 0.0025, "rubber")
        b.rod(
            base,
            (x + side * 0.045, 0.17, 0.11),
            (x + side * 0.055, 0.19, 0.047),
            0.0025,
            "rubber",
        )
    b.box(base, (0.20, -0.12, 0.071), (0.027, 0.023, 0.023), "cream")
    for z in (0.060, 0.082):
        b.cylinder(
            base, (0.20, -0.145, z), 0.006, 0.004, "metal", euler=(math.pi / 2, 0, 0)
        )
    b.metadata["capabilities"] = [
        "coarse_sample_translation",
        "two_independent_probe_approaches",
        "visible_inert_semiconductor_die",
    ]
    b.metadata["limitations"].append(
        "The probes remain in a clearance pose; contact force, electrical response, charge transport, spectroscopy and device testing are not modeled. Mechanical adjustments are surrogate motor controls for simulation, not documented Oxford automation."
    )
    b.metadata["limitations"].append(
        "Broad foreground shaft, slimmer bent rear holder and collar cues follow a 360 x 162 pixel official close-up. Exact dimensions, open guide bearings and hidden fastening are estimates. The raised translation stage, puck and motorized axes are authored. Enabled collision surfaces do not imply electrical touchdown: fine-tip radius remains 120 micrometres and the centered lower-stop tip-to-pad gap remains 3.78 mm."
    )


BUILDERS = {"semiconductor_probe_station": _probe_station}
SAMPLE_INTERFACES = {
    "semiconductor_probe_station": (
        "inert_die",
        (0.012, 0.012, 0.001),
        "clamped",
        "12 mm inert semiconductor coupon with visible contact-pad geometry",
    )
}


def _features(world, definition):
    # A compact bench vignette; wall positions and auxiliary furniture are authored.
    metal = (0.50, 0.53, 0.54, 1)
    dark = (0.08, 0.11, 0.13, 1)
    box(
        world, "probe_rear_service_strip", (1.2, 0.035, 0.06), (-0.6, 2.02, 1.13), metal
    )
    for i in range(5):
        x = -1.6 + i * 0.43
        box(
            world,
            f"probe_power_socket_{i}",
            (0.043, 0.008, 0.051),
            (x, 1.978, 1.13),
            (0.85, 0.85, 0.8, 1),
            collision=False,
        )
    # Generic measurement-console enclosure, deliberately not a branded device.
    box(
        world,
        "probe_readout_case",
        (0.21, 0.19, 0.12),
        (0.5, 0.60, 1.02),
        (0.72, 0.73, 0.70, 1),
    )
    box(
        world,
        "probe_readout_panel",
        (0.18, 0.005, 0.086),
        (0.5, 0.405, 1.03),
        dark,
        collision=False,
    )
    box(
        world,
        "probe_readout_display",
        (0.105, 0.002, 0.035),
        (0.46, 0.397, 1.055),
        (0.05, 0.22, 0.24, 1),
        collision=False,
    )
    for i in range(3):
        box(
            world,
            f"probe_readout_button_{i}",
            (0.014, 0.003, 0.007),
            (0.38 + i * 0.05, 0.393, 0.985),
            (0.71, 0.72, 0.66, 1),
            collision=False,
        )
    box(
        world,
        "probe_sample_case",
        (0.12, 0.08, 0.015),
        (-1.12, 0.38, 0.915),
        (0.2, 0.24, 0.27, 1),
    )
    for i in range(4):
        box(
            world,
            f"probe_empty_sample_slot_{i}",
            (0.018, 0.025, 0.001),
            (-1.195 + i * 0.05, 0.38, 0.931),
            (0.42, 0.43, 0.43, 1),
            collision=False,
        )
    for x in (-1.55, 0.90):
        box(
            world, f"probe_shelf_post_{x}", (0.025, 0.027, 0.49), (x, 1.15, 1.37), metal
        )
    box(
        world,
        "probe_upper_shelf",
        (1.28, 0.19, 0.018),
        (-0.325, 1.15, 1.87),
        (0.73, 0.76, 0.74, 1),
    )
    for i in range(5):
        box(
            world,
            f"probe_archive_box_{i}",
            (0.10, 0.11, 0.11),
            (-1.15 + i * 0.43, 1.15, 1.997),
            (0.67, 0.69, 0.62, 1),
            collision=False,
        )


FEATURES = {"oxford_bonilla_semiconductor": _features}
