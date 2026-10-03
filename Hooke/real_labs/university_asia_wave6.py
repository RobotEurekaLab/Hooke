"""Yonsei MICS close-up RF probe station with inert mechanical inspection axes."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


SOURCE = "https://mics.yonsei.ac.kr/equipment/"


def _coax_head(b, parent, side):
    """Supported dark probe body and fine metal tip, with a segmented coaxial lead."""
    x = side * 0.085
    b.box(
        parent, (x, 0, 0.108), (0.036, 0.028, 0.018), "dark", euler=(0, 0, side * 0.22)
    )
    b.box(
        parent,
        (side * 0.060, -0.015, 0.097),
        (0.030, 0.010, 0.012),
        "dark",
        euler=(0, -side * 0.24, 0),
    )
    b.rod(
        parent,
        (side * 0.050, -0.014, 0.091),
        (side * 0.019, -0.002, 0.082),
        0.0020,
        "metal",
    )
    for offset in (-0.0011, 0, 0.0011):
        b.rod(
            parent,
            (side * 0.020, -0.002 + offset, 0.082),
            (side * 0.003, offset, 0.0824),
            0.00028,
            "bright",
        )
    b.cylinder(parent, (side * 0.098, -0.008, 0.121), 0.010, 0.008, "metal")
    for a in range(12):
        angle = a * math.tau / 12
        b.rod(
            parent,
            (
                side * 0.098 + 0.009 * math.cos(angle),
                -0.008 + 0.009 * math.sin(angle),
                0.115,
            ),
            (
                side * 0.098 + 0.009 * math.cos(angle),
                -0.008 + 0.009 * math.sin(angle),
                0.128,
            ),
            0.0006,
            "bright",
        )
    b.rod(
        parent,
        (side * 0.087, 0.015, 0.116),
        (side * 0.125, 0.042, 0.158),
        0.012,
        "metal",
    )
    b.rod(
        parent,
        (side * 0.125, 0.042, 0.158),
        (side * 0.175, 0.079, 0.205),
        0.013,
        "rubber",
    )
    b.rod(
        parent,
        (side * 0.175, 0.079, 0.205),
        (side * 0.191, 0.12, 0.19),
        0.012,
        "rubber",
    )


def _rf_station(b, base, params):
    # Original estimated 0.44 x 0.36 m footprint; the official photo establishes only a close-up.
    b.box(base, (0, 0, 0.012), (0.22, 0.18, 0.012), "dark", collision=True)
    for x in (-0.195, 0.195):
        for y in (-0.155, 0.155):
            b.cylinder(base, (x, y, 0.024), 0.008, 0.002, "metal")
    b.box(base, (0, 0, 0.035), (0.098, 0.095, 0.011), "metal", collision=True)
    for y in (-0.066, 0.066):
        b.box(base, (0, y, 0.047), (0.092, 0.006, 0.005), "bright")
    chuck = b.moving(
        base, "chuck_x", (0, 0, 0.058), (1, 0, 0), (-0.003, 0.003), mass=0.07, kp=250
    )
    b.box(chuck, (0, 0, 0), (0.086, 0.076, 0.006), "dark", collision=True)
    b.cylinder(chuck, (0, 0, 0.011), 0.068, 0.010, "metal")
    b.cylinder(chuck, (0, 0, 0.015), 0.082, 0.006, "bright", collision=True)
    b.cylinder(chuck, (0, 0, 0.0213), 0.070, 0.0003, "metal")
    b.box(chuck, (0, 0, 0.0224), (0.010, 0.008, 0.0008), "lens")
    b.box(chuck, (0, 0, 0.0234), (0.0024, 0.0020, 0.0002), "copper", collision=True)
    for y in (-0.0015, -0.0005, 0.0005, 0.0015):
        b.box(chuck, (0, y, 0.02362), (0.0018, 0.00011, 0.00002), "bright")
    b.site(chuck, "rf_coupon", (0, 0, 0.0234), 0.00015)
    for x in (-0.062, 0.062):
        b.cylinder(chuck, (x, -0.044, 0.023), 0.004, 0.002, "dark")
    # Each probe has its own anchored positioner, including visible guide blocks.
    for side in (-1, 1):
        b.box(
            base,
            (side * 0.153, 0.016, 0.044),
            (0.041, 0.058, 0.020),
            "dark",
            collision=True,
        )
        for y in (-0.022, 0.052):
            b.box(base, (side * 0.139, y, 0.072), (0.026, 0.008, 0.027), "metal")
        b.cylinder(
            base,
            (side * 0.188, -0.018, 0.078),
            0.011,
            0.009,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        b.rod(
            base,
            (side * 0.166, -0.018, 0.078),
            (side * 0.19, -0.018, 0.078),
            0.0035,
            "metal",
        )
    b.box(base, (-0.12, 0.015, 0.080), (0.032, 0.030, 0.014), "metal")
    _coax_head(b, base, -1)
    probe = b.moving(
        base, "right_probe_lift", (0, 0, 0), (0, 0, 1), (0, 0.010), mass=0.045, kp=180
    )
    b.box(probe, (0.12, 0.015, 0.080), (0.032, 0.030, 0.014), "metal")
    _coax_head(b, probe, 1)
    # Rear ribbon-cable head and fan-out contacts use the photographed three-sided arrangement.
    b.box(base, (0, 0.128, 0.049), (0.052, 0.034, 0.025), "dark", collision=True)
    b.box(base, (0, 0.103, 0.087), (0.038, 0.047, 0.015), "dark")
    b.box(base, (0, 0.066, 0.083), (0.027, 0.032, 0.006), "metal", euler=(0.13, 0, 0))
    for index in range(9):
        x = (index - 4) * 0.005
        b.rod(base, (x, 0.061, 0.084), (x * 0.14, 0.008, 0.0825), 0.00055, "bright")
        b.rod(base, (x, 0.133, 0.097), (x, 0.145, 0.175), 0.0019, "cream")
        b.rod(base, (x, 0.145, 0.175), (x, 0.12, 0.208), 0.0019, "cream")
    b.metadata["capabilities"] = [
        "chuck_translation",
        "independent_right_probe_retraction",
        "visible_inert_rf_coupon",
    ]
    b.metadata["limitations"].append(
        "Close-up-derived geometry with original estimated positioners. Needle clearances are inspection gaps, not calibrated contacts; no RF signals, electromagnetic response, measurement precision or microscope optics."
    )


def _features(world, definition):
    # Close-up source provides no room: retain an explicitly estimated compact electronics bay.
    box(
        world,
        "mics_bench_back_strip",
        (0.80, 0.02, 0.025),
        (0, 1.35, 1.08),
        (0.62, 0.64, 0.63, 1),
        collision=False,
    )
    for index in range(3):
        box(
            world,
            f"mics_blank_outlet_{index}",
            (0.035, 0.009, 0.025),
            (-0.55 + index * 0.5, 1.325, 1.08),
            (0.82, 0.81, 0.76, 1),
            collision=False,
        )


BUILDERS = {"mics_rf_probe_workspace": _rf_station}
SOURCES = {
    "mics_rf_probe_workspace": dict(
        reference="Yonsei MICS official installed RF probe close-up",
        url=SOURCE,
        dimensions_m=[0.44, 0.36, 0.22],
        dimension_basis="All dimensions estimated. Official photo establishes round silver chuck, opposing coaxial heads and rear ribbon probe; hidden positioners and bench are authored.",
    )
}
SAMPLE_INTERFACES = {
    "mics_rf_probe_workspace": (
        "rf_coupon",
        (0.0048, 0.0040, 0.0004),
        "clamped",
        "Visible inert electronic coupon on a 20 x 16 mm carrier",
    )
}
FEATURES = {"yonsei_mics_rf_probe": _features}
