"""Open guide repair for the authored ANU annulus profiling head.

The source photograph establishes a supported twin-rail profiler. Bearing
dimensions and the offset bridge below are inferred mechanical geometry, not
OEM internals, force calibration, thermistor response or fluid simulation.
"""

from real_labs.mounting import annular_mount


def add_profiling_guide(b, tank):
    """Keep the physical probe and drives; use the existing rail's upper travel."""
    prefix = b.name + "__annulus_guide_"

    def box(body, key, pos, half, material="metal"):
        return b.box(body, pos, half, material, collision=True, name=prefix + key)

    box(tank, "rim_bracket", (0.65, 0.32, 0.45), (0.12, 0.045, 0.025))
    for index, x in enumerate((0.59, 0.66)):
        b.rod(
            tank,
            (x, 0.32, 0.45),
            (x, 0.32, 1.0),
            0.009,
            "bright",
            collision=True,
            name=prefix + f"rail_{index}",
        )
    box(tank, "top_bridge", (0.625, 0.32, 1.0), (0.065, 0.045, 0.018), "dark")
    b.cylinder(
        tank,
        (0.625, 0.32, 1.04),
        0.026,
        0.04,
        "dark",
        collision=True,
        name=prefix + "drive_cover",
    )
    probe = b.moving(
        tank,
        "probe_height",
        (0.625, 0.32, 0.77),
        (0, 0, 1),
        (-0.07, 0.17),
        mass=0.08,
        kp=220,
    )
    # A continuous visual shell preserves the true bore. Enabled sectors
    # leave 0.6 mm radial running clearance around each 18 mm diameter rail.
    for index, x in enumerate((-0.035, 0.035)):
        annular_mount(
            b,
            probe,
            f"bearing_{index}",
            (x, 0, 0),
            0.0096,
            0.015,
            0.032,
            axis="z",
            proxies=True,
            namespace="annulus_guide",
            material="dark",
        )
    box(probe, "bearing_web", (0, 0, 0), (0.022, 0.012, 0.020), "dark")
    # Route the arm behind the left bore, then return to the existing stylus.
    # A solid bar through both rail centres would reproduce the old overlap.
    box(probe, "root_elbow", (0, -0.016, 0), (0.014, 0.025, 0.015))
    box(probe, "offset_arm", (-0.11, -0.032, 0), (0.11, 0.011, 0.012))
    box(probe, "stylus_elbow", (-0.20, -0.016, 0), (0.010, 0.016, 0.012))
    b.rod(
        probe,
        (-0.20, 0, 0),
        (-0.20, 0, -0.49),
        0.002,
        "bright",
        collision=True,
        name=prefix + "stylus",
    )
    b.site(probe, "probe_tip", (-0.20, 0, -0.492), 0.0005)
    b.metadata["profiling_guide"] = {
        "revision": 2,
        "rail_radius_m": 0.009,
        "bearing_inner_radius_m": 0.0096,
        "radial_clearance_m": 0.0006,
        "source_basis": "Photographed twin-rail profiler; hidden bearing and offset bridge are authored estimates",
        "scope": "Ideal-joint mechanical guidance only; no bearing stiffness, fluid forces or sensor model",
        "probe_height_range_m": [-0.07, 0.17],
        "range_change": "Revision 2 retains the previous travel and extends upward by 100 mm within the existing rails",
    }
    b.metadata["limitations"].append(
        "Open profiling bearings, support bridge and coaxial registration ring are "
        "authored estimates. Revision 2 increases the upper probe travel within the "
        "unchanged rails; geometric registration is not scientific observation."
    )
    return probe
