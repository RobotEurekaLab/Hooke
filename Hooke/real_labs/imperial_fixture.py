"""Inferred mountings for the original Imperial dry inspection fixture.

This fixture is not equipment copied from the historical vacuum-lab photograph.
The unchanged coupon remains fixed to its carriage; there is no vacuum seal,
deposition, autonomous specimen transfer or validated bearing-stress model.
"""

import math

from real_labs.mounting import annular_mount


RUNNING_CLEARANCE_M = 0.0005


def name(b, key):
    return f"{b.name}__dry_fixture_{key}"


def add_carriage_mounts(b, base, carrier):
    """Support the chamber, then its rails and upper split carriage bearings."""
    for side, x in enumerate((0.615, 1.275)):
        b.box(
            base,
            (x, -0.73, 0.895),
            (0.023, 0.20, 0.005),
            "rubber",
            collision=True,
            name=name(b, f"chamber_runner_{side}"),
        )
    for row, y in enumerate((-0.80, -0.65)):
        for end, x in enumerate((0.69, 1.18)):
            b.box(
                base,
                (x, y, 0.952),
                (0.014, 0.013, 0.011),
                "metal",
                collision=True,
                name=name(b, f"rail_{row}_pedestal_{end}"),
            )
            for side in (-1, 1):
                b.box(
                    base,
                    (x, y + side * 0.010, 0.962),
                    (0.014, 0.003, 0.010),
                    "metal",
                    collision=True,
                    name=name(b, f"rail_{row}_clamp_{end}_{side}"),
                )
        for index, x in enumerate((-0.05, 0.05)):
            annular_mount(
                b,
                carrier,
                f"bearing_{row}_{index}",
                (x, y + 0.73, -0.045),
                0.007 + RUNNING_CLEARANCE_M,
                0.021,
                0.015,
                half=True,
                proxies=True,
                material="metal",
                namespace="dry_fixture",
            )


def add_coupon_retention(b, carrier):
    """A 1 mm shim and visible clamp feet/pads connect the unchanged coupon."""
    b.box(
        carrier,
        (0, 0, 0.0245),
        (0.049, 0.045, 0.0005),
        "metal",
        collision=True,
        name=name(b, "coupon_shim"),
    )
    for side in (-1, 1):
        b.box(
            carrier,
            (side * 0.063, 0, 0.0255),
            (0.010, 0.043, 0.0015),
            "metal",
            collision=True,
            name=name(b, f"clamp_foot_{side}"),
        )
        b.box(
            carrier,
            (side * 0.051, 0, 0.032),
            (0.002, 0.040, 0.007),
            "rubber",
            collision=True,
            name=name(b, f"retaining_pad_{side}"),
        )
        b.box(
            carrier,
            (side * 0.0525, 0, 0.0405),
            (0.0115, 0.030, 0.0015),
            "metal",
            collision=True,
            name=name(b, f"hold_down_{side}"),
        )
        b.cylinder(
            carrier,
            (side * 0.063, 0, 0.052),
            0.004,
            0.001,
            "bright",
            name=name(b, f"clamp_screw_{side}"),
        )


def add_door_mounts(b, base, door):
    """Coaxial hinge with a real bore plus two door-handle standoffs."""
    b.box(
        base,
        (0.609, -0.954, 1.085),
        (0.024, 0.016, 0.20),
        "metal",
        collision=True,
        name=name(b, "hinge_bracket"),
    )
    for side in (-1, 1):
        z = 1.085 + side * 0.185
        b.rod(
            base,
            (0.583, -0.993, z),
            (0.609, -0.954, z),
            0.007,
            "bright",
            collision=True,
            name=name(b, f"hinge_lug_{side}"),
        )
    b.cylinder(
        base,
        (0.583, -0.993, 1.085),
        0.007,
        0.192,
        "bright",
        collision=True,
        name=name(b, "hinge_pin"),
    )
    for side in (-1, 1):
        annular_mount(
            b,
            door,
            f"hinge_hub_{side}",
            (0, 0, side * 0.10),
            0.007 + RUNNING_CLEARANCE_M,
            0.019,
            0.028,
            axis="z",
            proxies=True,
            material="metal",
            namespace="dry_fixture",
        )
        z = side * 0.054
        b.cylinder(
            door,
            (0.60, -0.016, z),
            0.012,
            0.003,
            "bright",
            euler=(math.pi / 2, 0, 0),
            collision=True,
            name=name(b, f"handle_base_{side}"),
        )
        b.rod(
            door,
            (0.60, -0.019, z),
            (0.60, -0.060, z),
            0.006,
            "bright",
            collision=True,
            name=name(b, f"handle_standoff_{side}"),
        )


def describe(b):
    b.metadata["dry_fixture_adaptations"] = {
        "basis": "Original mechanical dry inspection fixture; not copied photographic vacuum hardware",
        "support": (
            "10 mm chamber runners, deck-mounted rail pedestals and open upper "
            "bearing shoes with 0.5 mm radial running clearance; 1 mm coupon shim, "
            "clamp feet, retaining pads and hold-down tabs"
        ),
        "door": "Connected coaxial hinge with 0.5 mm pin/bore clearance and two handle standoffs",
        "limits": (
            "Unchanged fixed coupon, original travel and door reference pose; "
            "no free specimen transfer, bearing-stress, vacuum or deposition model"
        ),
    }
