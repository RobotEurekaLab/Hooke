"""Authored open guide and retained probe head for the HKUST robot exterior.

The cached photograph resolves the perforated bridge and a narrow hanging
mechanism, not its hidden bearings. Twin supported rails and running gaps here
are explicit mechanical estimates. Original axis frames and drives are kept.
"""

from real_labs.mounting import annular_mount


def add_probe_head(b, trolley):
    """Replace the intersecting central column with two offset guide rails."""
    prefix = b.name + "__gcf_probe_"

    def box(parent, key, pos, half, material="dark"):
        return b.box(parent, pos, half, material, collision=True, name=prefix + key)

    # Each rail is anchored into the existing overhanging trolley bracket.
    # The rails remain outside the rotating disk/stylus sweep, and extend
    # beyond both bearing ends throughout the original 130 mm height travel.
    for side in (-1, 1):
        b.rod(
            trolley,
            (side * 0.065, -0.25, 0.18),
            (side * 0.065, -0.25, -0.30),
            0.008,
            "bright",
            collision=True,
            name=prefix + f"rail_{side}",
        )
    stage = b.moving(
        trolley,
        "probe_height",
        (0, -0.25, -0.18),
        (0, 0, 1),
        (-0.04, 0.09),
        mass=0.20,
        kp=230,
    )
    # Keep the former 110 x 96 x 110 mm outer block envelope, with a real
    # stepped opening. Its bottom recess clears the unchanged rotary flange;
    # an upper annular bearing supports the narrower rotating shaft.
    bore = 0.0356
    for side in (-1, 1):
        box(
            stage,
            f"frame_x_{side}",
            (side * (0.055 + bore) / 2, 0, 0),
            ((0.055 - bore) / 2, 0.048, 0.055),
        )
        box(
            stage,
            f"frame_y_{side}",
            (0, side * (0.048 + bore) / 2, 0),
            (bore, (0.048 - bore) / 2, 0.055),
        )
        annular_mount(
            b,
            stage,
            f"rail_bearing_{side}",
            (side * 0.065, 0, 0),
            0.0086,
            0.016,
            0.038,
            axis="z",
            proxies=True,
            namespace="gcf_probe",
            material="dark",
        )
    annular_mount(
        b,
        stage,
        "rotary_bearing",
        (0, 0, 0.0085),
        0.0146,
        0.041,
        0.0465,
        axis="z",
        proxies=True,
        namespace="gcf_probe",
        material="dark",
    )
    rotor = b.moving(
        stage,
        "probe_rotation",
        (0, 0, -0.055),
        (0, 0, 1),
        (-0.7, 0.7),
        kind="hinge",
        mass=0.05,
        kp=130,
    )
    b.cylinder(
        rotor,
        (0, 0, 0),
        0.035,
        0.015,
        "metal",
        collision=True,
        name=prefix + "rotary_flange",
    )
    b.cylinder(
        rotor,
        (0, 0, 0.0625),
        0.014,
        0.0475,
        "bright",
        collision=True,
        name=prefix + "rotary_shaft",
    )
    # A retained collar rests on the bearing's upper face. Radial guide gaps
    # are 0.6 mm; this deliberate axial tangency is a support interface.
    b.cylinder(
        rotor,
        (0, 0, 0.115),
        0.024,
        0.005,
        "metal",
        collision=True,
        name=prefix + "retaining_collar",
    )
    b.rod(
        rotor,
        (0, 0, -0.015),
        (0, 0, -0.045),
        0.013,
        "bright",
        collision=True,
        name=prefix + "stylus",
    )
    box(rotor, "tip_arm", (0.023, 0, -0.045), (0.025, 0.005, 0.005), "bright")
    b.site(rotor, "probe_tip_surface", (0.023, 0, -0.050), 0.0005)
    b.metadata["limitations"].append(
        "Offset twin rails, open sliding bearings and stepped rotary bearing are "
        "authored hidden hardware, not resolved in the photograph. Ideal joints "
        "carry guide reactions across 0.6 mm running gaps; no bearing stiffness, "
        "preload, drive transmission or structural load rating is calibrated."
    )
