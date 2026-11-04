"""Reference-informed treadmill exterior with authored unloaded mechanisms.

The two Stanford photographs establish layout and exterior cues, not OEM CAD.
Running bearings, lift guides and a rigid inert fixture are original estimates.
Belt decks stay fixed: rotating rollers do not implement belt transport.
"""

import math

from real_labs.mounting import annular_mount


def _box(b, parent, name, pos, half, material="metal"):
    return b.box(
        parent, pos, half, material, collision=True, name=f"{b.name}__gait_{name}"
    )


def _rod(b, parent, name, start, end, radius, material="metal"):
    return b.rod(
        parent,
        start,
        end,
        radius,
        material,
        collision=True,
        name=f"{b.name}__gait_{name}",
    )


def _cylinder(b, parent, name, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=f"{b.name}__gait_{name}",
        **kwargs,
    )


def _roller(b, base, label, x, y):
    moving = y < 0
    parent = (
        b.moving(
            base,
            f"drive_{label}",
            (x, y, 0.188),
            (1, 0, 0),
            None,
            kind="hinge",
            mass=1.2,
            mode="velocity",
        )
        if moving
        else base
    )
    if moving:
        parent.find("joint").set("damping", "0.1")
    center = (0, 0, 0) if moving else (x, y, 0.188)
    key = f"{label}_{'drive' if moving else 'idler'}"
    _cylinder(b, parent, key + "_drum", center, 0.07, 0.29, euler=(0, math.pi / 2, 0))
    # End-face spokes and shaft are attached to the rotating drum, not world.
    for i in range(6):
        a = i * math.pi / 3
        _rod(
            b,
            parent,
            key + f"_spoke_{i}",
            (center[0] + 0.294, center[1], center[2]),
            (
                center[0] + 0.294,
                center[1] + math.cos(a) * 0.055,
                center[2] + math.sin(a) * 0.055,
            ),
            0.004,
            "dark",
        )
    for side in (-1, 1):
        _cylinder(
            b,
            parent,
            key + f"_shaft_{side}",
            (center[0] + side * 0.306, center[1], center[2]),
            0.012,
            0.016,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
        bx = x + side * 0.313
        annular_mount(
            b,
            base,
            key + f"_bearing_{side}",
            (bx, y, 0.188),
            0.0126,
            0.027,
            0.010,
            proxies=True,
            material="dark",
            namespace="gait",
        )
        # Pedestal touches the outer bearing's bottom, and rests on the low rail.
        _box(
            b, base, key + f"_pedestal_{side}", (bx, y, 0.1355), (0.010, 0.025, 0.0255)
        )


def _wearable_exterior(b, base):
    """Source-visible brace/pulley and cable cues on the unloaded rigid fixture.

    No wearer, biological joint or Bowden-cable force transmission is inferred.
    The mechanical fixture remains rigid; only existing rollers/lift are driven.
    """
    for x in (-.29, .29):
        for side in (-1, 1):
            bx = x + side * .046
            # Flat calf rails and padded open cuffs replace the stick-like
            # silhouette without crossing the opposite leg or lift path.
            _rod(b, base, f"brace_rail_{x}_{side}", (bx, .285, 1.14),
                 (bx, .052, .55), .018, "metal")
            for y, z in ((.22, .99), (.11, .70)):
                _box(b, base, f"cuff_side_{x}_{side}_{z}", (bx, y, z),
                     (.018, .060, .038), "dark")
                _cylinder(b, base, f"cuff_fastener_{x}_{side}_{z}",
                          (bx + side * .019, y, z), .006, .002, "bright",
                          euler=(0, math.pi / 2, 0))
            annular_mount(b, base, f"ankle_pulley_{x}_{side}",
                          (x + side * .043, .05, .53), .025, .062, .006,
                          axis="x", proxies=True, material="metal", namespace="gait")
        _box(b, base, f"cuff_back_{x}", (x, .265, .99), (.046, .015, .038), "dark")
        _box(b, base, f"cuff_lower_back_{x}", (x, .155, .70), (.046, .015, .038), "dark")
        # Two parked transmission sheaths terminate on visible frame-mounted
        # amplifiers. Each route has explicit fixed endpoint brackets.
        _box(b, base, f"rear_amplifier_{x}", (x, 1.155, 1.85),
             (.11, .065, .16), "dark")
        for i in range(2):
            points = [(x + .055, .05, .56 + i * .018),
                      (x + .15, .28, .82), (x + .19, .72, 1.30),
                      (x + .12, 1.10, 1.62), (x + .05, 1.09, 1.75 + i * .06)]
            for j, (a, c) in enumerate(zip(points, points[1:])):
                _rod(b, base, f"brace_service_{x}_{i}_{j}", a, c, .004, "rubber")
        for i in range(5):
            _box(b, base, f"amplifier_vent_{x}_{i}", (x, 1.088, 1.78 + i * .028),
                 (.075, .001, .003), "metal")


def build_gait_station(b, base, params):
    b.feet(base, 1.48, 2.52, 0.06)
    # Open chassis: the former solid slab cut through every roller. Low rails
    # support the stationary deck and separate pedestals without hiding contacts.
    _box(b, base, "bottom_pan", (0, 0, 0.070), (0.72, 1.24, 0.010), "shell")
    for x in (-0.685, 0, 0.685):
        _box(b, base, f"lower_rail_{x}", (x, 0, 0.090), (0.027, 1.24, 0.010), "shell")
    for y in (-1.225, 1.225):
        _box(b, base, f"end_frame_{y}", (0, y, 0.142), (0.72, 0.015, 0.062), "shell")
    for label, x in (("left", -0.325), ("right", 0.325)):
        _box(
            b,
            base,
            label + "_deck_support",
            (x, 0, 0.146),
            (0.29, 0.895, 0.066),
            "shell",
        )
        _box(
            b,
            base,
            label + "_static_belt",
            (x, 0, 0.237),
            (0.29, 0.90, 0.025),
            "rubber",
        )
        # Visible strips are attached skins, with contact geometry retained.
        for i, y in enumerate((-0.70, -0.35, 0, 0.35, 0.70)):
            _box(
                b,
                base,
                label + f"_deck_mark_{i}",
                (x, y, 0.2625),
                (0.287, 0.002, 0.0005),
                "dark",
            )
        for y in (-1.0, 1.0):
            _roller(b, base, label, x, y)
    # Small transverse beams bridge low side rails to the bearing pedestals.
    for y in (-1.0, 1.0):
        _box(b, base, f"bearing_base_{y}", (0, y, 0.105), (0.72, 0.035, 0.005), "metal")
    for x in (-0.70, 0.70):
        _box(b, base, f"side_deck_{x}", (x, 0, 0.257), (0.04, 1.19, 0.019))
        for y in (-0.92, 0.88):
            _rod(
                b,
                base,
                f"handrail_post_{x}_{y}",
                (x, y, 0.110),
                (x, y + 0.06, 1.20),
                0.033,
                "shell",
            )
        _rod(
            b, base, f"handrail_{x}", (x, -0.86, 1.20), (x, 0.94, 1.20), 0.031, "shell"
        )
        _box(
            b,
            base,
            f"button_seat_{x}",
            (x, -0.81, 1.225),
            (0.065, 0.065, 0.018),
            "dark",
        )
        _cylinder(
            b, base, f"stop_button_{x}", (x, -0.81, 1.259), 0.025, 0.016, "warning"
        )
    for x in (-0.80, 0.80):
        for y in (-1.15, 1.15):
            _box(b, base, f"frame_foot_{x}_{y}", (x, y, 0.014), (0.11, 0.13, 0.014))
            _box(
                b,
                base,
                f"frame_post_{x}_{y}",
                (x, y, 1.413),
                (0.032, 0.032, 1.385),
                "bright",
            )
            _box(
                b,
                base,
                f"frame_slot_{x}_{y}",
                (x + 0.033, y, 1.41),
                (0.001, 0.015, 1.36),
                "dark",
            )
            for dx in (-0.07, 0.07):
                for dy in (-0.09, 0.09):
                    _cylinder(
                        b,
                        base,
                        f"anchor_{x}_{y}_{dx}_{dy}",
                        (x + dx, y + dy, 0.030),
                        0.007,
                        0.002,
                        "dark",
                    )
        _box(b, base, f"top_side_{x}", (x, 0, 2.78), (0.037, 1.19, 0.038), "bright")
        _rod(
            b,
            base,
            f"fixture_side_support_{x}",
            (x, -1.15, 1.72),
            (x, 1.15, 1.72),
            0.024,
        )
    for y in (-1.15, 1.15):
        _box(b, base, f"top_end_{y}", (0, y, 2.78), (0.83, 0.037, 0.038), "bright")
    _box(b, base, "lift_top_bridge", (0, 0.18, 2.80), (0.80, 0.050, 0.05), "dark")
    # Two guides hang continuously from the overhead beam. Open sleeves leave
    # a true 0.6 mm radial gap and remain on the guide through the full stroke.
    for x in (-0.24, 0.24):
        _cylinder(b, base, f"lift_guide_{x}", (x, 0.18, 2.475), 0.012, 0.275, "bright")
        _cylinder(b, base, f"guide_stop_{x}", (x, 0.18, 2.194), 0.024, 0.006, "dark")
    lift = b.moving(
        base,
        "fixture_lift",
        (0, 0.18, 2.54),
        (0, 0, 1),
        (-0.14, 0.14),
        mass=2.0,
        kp=600,
        force=400,
    )
    _box(b, lift, "lift_crossbar", (0, 0, 0), (0.219, 0.075, 0.045))
    for x in (-0.24, 0.24):
        annular_mount(
            b,
            lift,
            f"lift_sleeve_{x}",
            (x, 0, 0),
            0.0126,
            0.024,
            0.050,
            axis="z",
            proxies=True,
            material="dark",
            namespace="gait",
        )
    for x in (-0.15, 0.15):
        _rod(b, lift, f"hanger_{x}", (x, 0, -0.045), (x, 0, -0.616), 0.005, "dark")
    _box(b, lift, "hanging_crossbar", (0, 0, -0.63), (0.20, 0.06, 0.024), "dark")
    # Original rigid ankle fixture, now continuously supported by side beams.
    _rod(b, base, "fixture_crossbar", (-0.80, 0.67, 1.72), (0.80, 0.67, 1.72), 0.024)
    _rod(
        b,
        base,
        "service_crossbar",
        (-0.80, 1.15, 1.85),
        (0.80, 1.15, 1.85),
        0.018,
        "dark",
    )
    for x in (-0.29, 0.29):
        _box(
            b,
            base,
            f"fixture_mount_{x}",
            (x, 0.67, 1.72),
            (0.053, 0.064, 0.047),
            "dark",
        )
        _rod(
            b,
            base,
            f"upper_fixture_{x}",
            (x, 0.67, 1.68),
            (x, 0.29, 1.20),
            0.026,
            "bright",
        )
        _cylinder(
            b,
            base,
            f"knee_{x}",
            (x, 0.29, 1.20),
            0.07,
            0.039,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        for side in (-1, 1):
            _rod(
                b,
                base,
                f"lower_fixture_{x}_{side}",
                (x + side * 0.035, 0.29, 1.15),
                (x + side * 0.035, 0.05, 0.54),
                0.012,
                "bright",
            )
        _cylinder(
            b,
            base,
            f"ankle_{x}",
            (x, 0.05, 0.53),
            0.046,
            0.036,
            "blue",
            euler=(0, math.pi / 2, 0),
        )
        _box(
            b,
            base,
            f"foot_adapter_{x}",
            (x, 0.05, 0.475),
            (0.030, 0.028, 0.027),
            "metal",
        )
        _box(b, base, f"foot_{x}", (x, -0.06, 0.43), (0.055, 0.18, 0.018), "dark")
        points = [(x, 0.30, 1.2), (x + 0.10, 0.55, 1.65), (x + 0.12, 1.15, 1.85)]
        for i, (a, z) in enumerate(zip(points, points[1:])):
            _rod(b, base, f"fixed_service_{x}_{i}", a, z, 0.008, "rubber")
    _wearable_exterior(b, base)
    b.site(base, "test_fixture", (-0.29, 0.05, 0.53), 0.01)
    b.site(lift, "lift_inspection", (0, -0.075, -0.63), 0.008)
    b.metadata["capabilities"] = [
        "independent_drive_roller_velocity",
        "fixture_support_height",
        "visible_mechanical_ankle_fixture",
        "unloaded_drive_and_lift_inspection",
    ]
    b.metadata["gait_inspection"] = {
        "bearing_radial_clearance_m": 0.0006,
        "drive_radius_m": 0.07,
        "task": "Independent inert roller run-up, measured stop, lift clearance inspection and return",
        "source_basis": "Two official Stanford photos establish exterior arrangement only; bearings, lift and inert leg fixture are authored",
        "belt_contract": "Stationary deck surfaces; rollers are independent unloaded rigid drives, with no belt transport or tension model",
        "lift_contract": "Ideal gravity-compensated slide with two open guide sleeves; no human suspension, bearing stress or rated load claim",
    }
    b.metadata["limitations"].append(
        "Belt decks stay fixed. Unloaded rollers, lift guides and rigid ankle surrogate do not implement walking, human biomechanics, force plates, belt transmission, service dynamics or safety-rated controls."
    )
