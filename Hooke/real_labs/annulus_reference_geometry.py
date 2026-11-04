"""Authored open registration ring for the ANU vertical profiling probe.

The photographed profiler is retained. This removable-looking but fixed inert
reference is an authored inspection aid, not equipment identified in the photo.
Its open bore permits both retraction above the datum and the original submerged
probe travel; it is not a thermistor, flow sensor or calibrated gauge.
"""

from real_labs.mounting import annular_mount

TIP_RADIUS = 0.002
BORE_RADIUS = 0.008
REFERENCE_CENTER = (0.425, 0.32, 0.3934)
REFERENCE_TOP = 0.3964


def add_registration_ring(b, tank):
    prefix = b.name + "__annulus_reference_"
    b.rod(
        tank,
        (0.475, 0.32, 0.05),
        (0.475, 0.32, 0.3894),
        0.009,
        "metal",
        collision=True,
        name=prefix + "stem",
    )
    b.box(
        tank,
        (0.459, 0.32, 0.3894),
        (0.018, 0.010, 0.004),
        "metal",
        collision=True,
        name=prefix + "arm",
    )
    annular_mount(
        b,
        tank,
        "datum",
        REFERENCE_CENTER,
        BORE_RADIUS,
        0.026,
        0.003,
        axis="z",
        proxies=True,
        material="cream",
        namespace="annulus_reference",
    )
    # Dark index marks stop outside the physical opening.
    for index, (dx, dy, sx, sy) in enumerate(
        (
            (0.017, 0, 0.006, 0.001),
            (-0.017, 0, 0.006, 0.001),
            (0, 0.017, 0.001, 0.006),
            (0, -0.017, 0.001, 0.006),
        )
    ):
        b.box(
            tank,
            (0.425 + dx, 0.32 + dy, REFERENCE_TOP + 0.0001),
            (sx, sy, 0.0001),
            "dark",
            collision=True,
            name=prefix + f"index_{index}",
        )
    b.site(tank, "calibration_target", (0.425, 0.32, REFERENCE_TOP), 0.001)
    b.metadata["registration_reference"] = {
        "revision": 2,
        "shape": "open coaxial ring",
        "inner_radius_m": BORE_RADIUS,
        "probe_radius_m": TIP_RADIUS,
        "nominal_radial_clearance_m": BORE_RADIUS - TIP_RADIUS,
        "datum_top_in_tank_m": REFERENCE_TOP,
        "basis": "Authored inert alignment aid; dimensions and mounting are estimated",
        "scope": "Geometric axial registration and open-bore passage; no sensor or fluid measurement",
    }
