"""Versioned, closable room-520 cover and supported inert inspection stage.

The institutional photograph establishes the handled cover and exterior only.
Hinges, guides and hidden fixture dimensions are authored, not recovered CAD.
"""

import math

from real_labs.mounting import annular_mount

PREFIX = "covered_optical_table__chicago_v2_"


def _box(b, parent, key, position, half, material="metal"):
    return b.box(parent, position, half, material, collision=True, name=PREFIX + key)


def _handle(b, parent, key, x, y, z):
    for sign in (-1, 1):
        b.rod(
            parent,
            (x + sign * 0.055, y, z),
            (x + sign * 0.055, y, z + 0.05),
            0.005,
            "bright",
            collision=True,
            name=PREFIX + key + f"_leg_{sign}",
        )
    b.rod(
        parent,
        (x - 0.055, y, z + 0.05),
        (x + 0.055, y, z + 0.05),
        0.005,
        "bright",
        collision=True,
        name=PREFIX + key + "_grip",
    )


def _lid(b, base):
    # Version 2: q=0 is horizontal/closed; q=-0.6 is open. The former extra
    # fixed -0.6 rotation is deliberately absent, with migration in metadata.
    lid = b.moving(
        base,
        "access_lid_angle",
        (0, 0, 1.395),
        (1, 0, 0),
        (-0.6, 0),
        kind="hinge",
        mass=0.7,
        kp=240,
    )
    # Folded sheet cover, not a solid 30 mm slab. Its external top remains at
    # the neighboring panels' height; the folded skirt has a real overlap lip.
    _box(b, lid, "lid_sheet", (0, -0.472, 0.014), (0.477, 0.447, 0.001))
    for sign in (-1, 1):
        _box(
            b,
            lid,
            f"lid_side_fold_{sign}",
            (sign * 0.476, -0.472, 0),
            (0.001, 0.447, 0.013),
        )
    for key, y in (("front", -0.918), ("rear", -0.026)):
        _box(b, lid, "lid_" + key + "_fold", (0, y, 0), (0.475, 0.001, 0.013))
    _handle(b, lid, "lid_handle", 0, -0.46, 0.015)
    # Fixed overlap strips support the cover frame and cover its seams. The
    # 0.6 mm running gap is geometric clearance, not a light-tight seal claim.
    _box(b, base, "front_overlap_lip", (0, -0.915, 1.3807), (0.529, 0.015, 0.0007))
    _box(b, base, "rear_overlap_lip", (0, 0, 1.3807), (0.529, 0.04, 0.0007))
    _box(b, base, "rear_header", (0, 0.032, 1.366), (0.529, 0.025, 0.015))
    # Header is connected to both side walls by actual under-roof transverse
    # framing. It does not rely on the floating hinge or a hidden filled bore.
    _box(b, base, "header_crossbeam", (0, 0.032, 1.356), (1.42, 0.025, 0.024))
    for sign in (-1, 1):
        x = sign * 0.42
        b.cylinder(
            base,
            (x, 0, 1.395),
            0.004,
            0.060,
            "bright",
            euler=(0, math.pi / 2, 0),
            collision=True,
            name=PREFIX + f"hinge_pin_{sign}",
        )
        for end in (-1, 1):
            _box(
                b,
                base,
                f"hinge_fixed_ear_{sign}_{end}",
                (x + end * 0.047, 0.0045, 1.391),
                (0.008, 0.0115, 0.014),
            )
        annular_mount(
            b,
            lid,
            f"hinge_knuckle_{sign}",
            (x, 0, 0),
            0.0046,
            0.009,
            0.023,
            proxies=True,
            namespace="chicago_v2",
        )
        _box(b, lid, f"hinge_moving_leaf_{sign}", (x, -0.018, 0), (0.02, 0.011, 0.004))
    b.site(lid, "cover_front", (0, -0.918, 0.014), 0.004)


def _stage(b, base):
    _box(b, base, "stage_base", (0, -0.40, 0.949), (0.24, 0.18, 0.025), "dark")
    stage = b.moving(
        base,
        "mineral_coupon_x",
        (0, -0.40, 1.009),
        (1, 0, 0),
        (-0.07, 0.07),
        mass=0.18,
        kp=340,
    )
    _box(b, stage, "stage_plate", (0, 0, 0), (0.13, 0.12, 0.013))
    b.cylinder(
        stage,
        (0, 0, 0.030),
        0.065,
        0.017,
        "dark",
        collision=True,
        name=PREFIX + "coupon_retainer",
    )
    b.geom(
        stage,
        "ellipsoid",
        (0.020, 0.016, 0.010),
        (0, 0, 0.057),
        "orange",
        collision=True,
        name=PREFIX + "inert_coupon",
    )
    b.site(stage, "mineral_coupon", (0, 0, 0.068))
    for sign in (-1, 1):
        y = -0.40 + sign * 0.09
        b.rod(
            base,
            (-0.22, y, 0.988),
            (0.22, y, 0.988),
            0.004,
            "bright",
            collision=True,
            name=PREFIX + f"stage_rail_{sign}",
        )
        for end in (-1, 1):
            _box(
                b,
                base,
                f"rail_support_{sign}_{end}",
                (end * 0.22, y, 0.981),
                (0.012, 0.012, 0.007),
            )
        annular_mount(
            b,
            stage,
            f"stage_bearing_{sign}",
            (0, sign * 0.09, -0.021),
            0.0046,
            0.010,
            0.040,
            proxies=True,
            namespace="chicago_v2",
        )
        _box(
            b,
            stage,
            f"bearing_crown_{sign}",
            (0, sign * 0.09, -0.012),
            (0.041, 0.01, 0.003),
        )
    # A physical index target defines inert centering; it is not an optic or
    # laser path inferred from a photograph whose interior is concealed.
    _box(b, base, "datum_tab", (0, -0.575, 0.989), (0.007, 0.014, 0.015), "dark")
    b.site(base, "stage_center_datum", (0, -0.40, 1.077), 0.002)


def build_enclosure(b, base):
    for x in (-1.02, 1.02):
        for y in (-0.64, 0.64):
            b.cylinder(base, (x, y, 0.03), 0.20, 0.03, "rubber", collision=True)
            b.cylinder(
                base,
                (x, y, 0.41),
                0.18,
                0.35,
                "dark",
                collision=True,
                name=b.unique("optical_isolator"),
            )
            for z in (0.14, 0.68):
                b.ring(base, (x, y, z), 0.183, 0.006, "metal", segments=20)
    b.box(
        base,
        (0, 0, 0.83),
        (1.45, 0.94, 0.07),
        "dark",
        collision=True,
        name=b.unique("covered_table"),
    )
    _box(b, base, "deck", (0, 0, 0.912), (1.42, 0.91, 0.012))
    for sign in (-1, 1):
        _box(b, base, f"side_wall_{sign}", (sign * 1.435, 0, 1.14), (0.015, 0.94, 0.24))
    _box(b, base, "rear_wall", (0, 0.925, 1.14), (1.42, 0.015, 0.24))
    for i, (x, half) in enumerate(((-0.98, 0.45), (0, 0.529), (0.98, 0.45))):
        _box(
            b, base, f"retained_front_panel_{i}", (x, -0.925, 1.14), (half, 0.015, 0.24)
        )
        # Supported U-shaped panel grip; the front covers remain retained.
        for z in (1.09, 1.23):
            b.rod(
                base,
                (x, -0.94, z),
                (x, -0.972, z),
                0.005,
                "bright",
                collision=True,
                name=PREFIX + f"front_handle_{i}_{z}",
            )
        b.rod(
            base,
            (x, -0.972, 1.09),
            (x, -0.972, 1.23),
            0.005,
            "bright",
            collision=True,
            name=PREFIX + f"front_handle_{i}_grip",
        )
    for i, x in enumerate((-0.96, 0, 0.96)):
        for j, y in enumerate((-0.46, 0.46)):
            if x == 0 and y < 0:
                continue
            _box(
                b,
                base,
                f"fixed_roof_panel_{i}_{j}",
                (x, y, 1.395),
                (0.479, 0.45, 0.015),
            )
            _handle(b, base, f"roof_handle_{i}_{j}", x, y, 1.410)
    _lid(b, base)
    _stage(b, base)
    # Preserve the existing illustrative rear optics, explicitly outside the
    # measured operation; the source does not reveal an interior beam path.
    for x in (-0.40, 0.40):
        _box(
            b,
            base,
            f"illustrative_optic_foot_{x}",
            (x, 0.28, 0.934),
            (0.05, 0.05, 0.01),
            "dark",
        )
        b.cylinder(base, (x, 0.28, 1.0545), 0.009, 0.1105, "bright", collision=True)
        b.ring(base, (x, 0.28, 1.18), 0.030, 0.008, "dark", plane="xz", segments=12)
        b.cylinder(
            base, (x, 0.28, 1.18), 0.024, 0.005, "lens", euler=(math.pi / 2, 0, 0)
        )
    for x in (-0.60, 0.60):
        _box(
            b, base, f"monitor_foot_{x}", (x, 0.57, 1.421), (0.13, 0.10, 0.011), "dark"
        )
        _box(
            b, base, f"monitor_stand_{x}", (x, 0.60, 1.472), (0.02, 0.02, 0.040), "dark"
        )
        _box(
            b, base, f"monitor_case_{x}", (x, 0.60, 1.622), (0.22, 0.035, 0.11), "dark"
        )
        b.box(base, (x, 0.56, 1.622), (0.197, 0.004, 0.089), "screen")
    b.metadata["mechanical_interface"] = dict(
        version=2,
        parameter="enclosure_revision",
        legacy_revision=1,
        lid_control="access_lid_angle",
        control_range_rad=[-0.6, 0],
        old_physical_lid_angle="q - 0.6 radians",
        new_physical_lid_angle="q radians; 0 closed, -0.6 open",
        migration="No automatic remapping. Revision 2 keeps the control ID and limits but explicitly changes the physical zero; legacy geometry is available only with enclosure_revision=1.",
        canonical_reset="Closed roof and retained front covers; operate roof before inspecting the concealed fixture.",
        support_clearance_m=0.0006,
        sample="Original clamped 40 x 32 x 20 mm inert coupon and unchanged X stage frame/range/inertia/drives",
        source_boundary="Photo 69 establishes exterior cover/handles/supports only. Hinge, guide, datum and all hidden dimensions are original estimates, not OEM internals.",
        scope="Roof access and retained inert stage centering; no front-cover removal, light-tight seal, laser, mineral, pressure or spectral physics.",
    )
    b.metadata["capabilities"] = [
        "closable roof access",
        "supported inert coupon translation and centering",
    ]
    b.metadata["limitations"].append(
        "Version 2 restores retained front covers and physically closable roof access. Hidden stage/hinge/datum are authored; the illustrative rear optics are not an aligned or calibrated optical path. No laser heating, pressure, spectroscopy or mineral response."
    )
