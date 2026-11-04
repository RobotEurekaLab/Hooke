"""Authored dry wire/slot inspection assembly for the photographed workroom.

The open slot and its generic true-scale wire are mechanical teaching aids,
not recovered OEM internals or a material-removal model. All sizes are metres.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


RUNNING_CLEARANCE_M = 0.0005
WIRE_RADIUS_M = 0.0001
SLOT_HALF_WIDTH_M = 0.004
WIRE_Y_M = 0.08


def _name(b, key):
    return f"{b.name}__wire_fixture_{key}"


def _box(b, parent, key, pos, half, material="metal", **kwargs):
    return b.box(
        parent, pos, half, material, collision=True, name=_name(b, key), **kwargs
    )


def _bearing(b, parent, key, center, radius, outer, *, along_y=False):
    # Rotate a fixed child frame so the open upper bore remains above its rail.
    frame = parent
    if along_y:
        frame = ET.SubElement(
            parent,
            "body",
            name=_name(b, key + "_frame"),
            pos=" ".join(map(str, center)),
            euler=f"0 0 {math.pi / 2}",
        )
        center = (0, 0, 0)
    annular_mount(
        b,
        frame,
        key,
        center,
        radius + RUNNING_CLEARANCE_M,
        outer,
        0.025,
        half=True,
        proxies=True,
        namespace="wire_fixture",
    )


def add_fixture(b, base):
    """Crossed slides, open carrier and a fixed supported wire inspection span."""
    for row, y in enumerate((-0.30, 0.30)):
        _box(b, base, f"bed_runner_{row}", (0, y, 0.7825), (0.44, 0.035, 0.0025))
    _box(b, base, "bed", (0, 0, 0.82), (0.50, 0.39, 0.035))
    for row, y in enumerate((-0.27, 0.27)):
        for end, x in enumerate((-0.39, 0.39)):
            _box(
                b, base, f"x_pedestal_{row}_{end}", (x, y, 0.859), (0.03, 0.024, 0.004)
            )
        b.rod(
            base,
            (-0.42, y, 0.875),
            (0.42, y, 0.875),
            0.012,
            "bright",
            collision=True,
            name=_name(b, f"x_rail_{row}"),
        )
    xstage = b.moving(
        base,
        "fixture_x",
        (0, 0, 0.92),
        (1, 0, 0),
        (-0.12, 0.12),
        mass=1,
        kp=600,
        force=800,
    )
    _box(b, xstage, "x_plate", (0, 0, 0), (0.24, 0.30, 0.025))
    for row, y in enumerate((-0.27, 0.27)):
        for shoe, x in enumerate((-0.10, 0.10)):
            _bearing(b, xstage, f"x_bearing_{row}_{shoe}", (x, y, -0.045), 0.012, 0.020)
            _box(
                b,
                xstage,
                f"x_shoe_mount_{row}_{shoe}",
                (x, y, -0.022),
                (0.025, 0.010, 0.004),
            )
    for row, x in enumerate((-0.1725, 0.1725)):
        for end, y in enumerate((-0.235, 0.235)):
            _box(
                b,
                xstage,
                f"y_pedestal_{row}_{end}",
                (x, y, 0.026),
                (0.015, 0.02, 0.001),
            )
        b.rod(
            xstage,
            (x, -0.255, 0.033),
            (x, 0.255, 0.033),
            0.006,
            "bright",
            collision=True,
            name=_name(b, f"y_rail_{row}"),
        )
    ystage = b.moving(
        xstage,
        "fixture_y",
        (0, 0, 0.065),
        (0, 1, 0),
        (-0.10, 0.10),
        mass=0.8,
        kp=400,
        force=600,
    )
    # Rear-open frame: a lower guide can enter without a convex mesh filling
    # the aperture. Side beams and the front crossmember carry the specimen.
    for row, x in enumerate((-0.1725, 0.1725)):
        _box(b, ystage, f"y_side_{row}", (x, 0, 0.010), (0.0175, 0.15, 0.025), "dark")
        for shoe, y in enumerate((-0.065, 0.065)):
            _bearing(
                b,
                ystage,
                f"y_bearing_{row}_{shoe}",
                (x, y, -0.032),
                0.006,
                0.012,
                along_y=True,
            )
            _box(
                b,
                ystage,
                f"y_shoe_mount_{row}_{shoe}",
                (x, y, -0.017),
                (0.010, 0.025, 0.004),
            )
    _box(b, ystage, "y_crossmember", (0, -0.135, 0.010), (0.1725, 0.015, 0.025), "dark")
    for side in (-1, 1):
        _box(
            b,
            ystage,
            f"coupon_pad_{side}",
            (side * 0.115, 0, 0.0275),
            (0.060, 0.045, 0.0025),
        )
        _box(
            b,
            ystage,
            f"coupon_clamp_{side}",
            (side * 0.090, 0, 0.060),
            (0.015, 0.040, 0.030),
        )
        _box(
            b,
            ystage,
            f"coupon_tab_{side}",
            (side * 0.0775, 0, 0.083),
            (0.0275, 0.040, 0.003),
        )
        _box(
            b,
            ystage,
            f"coupon_side_{side}",
            (side * 0.0395, 0, 0.055),
            (0.0355, 0.060, 0.025),
            "bright",
        )
    _box(b, ystage, "coupon_front", (0, -0.050, 0.055), (0.004, 0.010, 0.025), "bright")
    b.site(ystage, "inert_coupon", (0, 0, 0.055))
    b.site(ystage, "calibration_slot", (0, 0, 0.055))
    b.site(ystage, "coupon_surface", (0.04, 0, 0.08))
    b.metadata["sample_datum_semantics"] = {
        "inert_coupon": "Legacy coupon envelope centre; now in the open slot, not a material surface",
        "calibration_slot": "Alignment datum at the open-slot centre",
        "coupon_surface": "Actual upper metal surface on the positive-X coupon half",
    }

    # Cantilever supported by the rear machine wall, below the moving coupon.
    _box(b, base, "lower_arm", (0, 0.262, 0.966), (0.010, 0.183, 0.010))
    annular_mount(
        b,
        base,
        "lower_guide",
        (0, WIRE_Y_M, 0.991),
        0.0003,
        0.006,
        0.015,
        axis="z",
        proxies=True,
        namespace="wire_fixture",
    )
    for side in (-1, 1):
        _box(
            b,
            base,
            f"upper_fork_{side}",
            (side * 0.024, 0.2625, 1.43),
            (0.011, 0.1825, 0.018),
        )
    annular_mount(
        b,
        base,
        "upper_guide",
        (0, WIRE_Y_M, 1.43),
        0.0003,
        0.014,
        0.025,
        axis="z",
        proxies=True,
        namespace="wire_fixture",
    )
    # This is a fixed calibration wire with authored bonded terminations,
    # not a running EDM feed path. The upper cap bears on the guide end face;
    # the lower termination meets the supported arm below the open guide bore.
    b.cylinder(
        base,
        (0, WIRE_Y_M, 1.458),
        0.014,
        0.003,
        "metal",
        collision=True,
        name=_name(b, "upper_wire_termination"),
    )
    b.cylinder(
        base,
        (0, WIRE_Y_M, 1.2155),
        WIRE_RADIUS_M,
        0.2395,
        "bright",
        collision=True,
        name=_name(b, "wire"),
    )
    b.metadata["inspection_targets"] = {"fixture_x": -0.06, "fixture_y": 0.08}
    b.metadata["wire_alignment"] = {
        "basis": "Authored dry inspection aid, not photographed OEM internals",
        "wire_radius_m": WIRE_RADIUS_M,
        "wire_dimension_basis": "Generic 0.20 mm choice within GF CUT 2000/3000 S published 0.05–0.30 mm capability; not a Vertex-specific claim",
        "wire_dimension_source": "https://www.gfms.com/content/dam/gfms/pdf/edm/wire-cut/cut-x000/en/cut20003000s-brochure.pdf",
        "facility_equipment_source": "https://www.birmingham.ac.uk/research/activity/mechanical-engineering/advanced-manufacturing/micro-manufacturing",
        "guide_bore_diameter_m": 0.0006,
        "guide_wire_radial_clearance_m": 0.0002,
        "calibration_slot_is_not_a_kerf": True,
        "wire_axis_base_m": [0, WIRE_Y_M],
        "wire_z_span_m": [0.976, 1.455],
        "wire_end_attachment": "Authored fixed bonded calibration terminations: lower arm face and supported upper end cap; no feed, take-up or tension model",
        "coupon_outer_size_m": [0.15, 0.12, 0.05],
        "rear_open_slot_width_m": 2 * SLOT_HALF_WIDTH_M,
        "slot_local_y_interval_m": [-0.04, 0.06],
        "slide_bearing_radial_clearance_m": RUNNING_CLEARANCE_M,
        "safe_route_joint_targets_m": [[0, 0], [0, 0.08], [0, 0]],
        "arbitrary_xy_requires_collision_planning": True,
        "retention": "Rigid coupon pieces on visible support pads and clamp tabs",
    }


def add_facade_details(b, base):
    """Photo-informed fixed silhouette, with explicitly authored mounting parts."""
    for side in (-1, 1):
        x = side * 0.47
        _box(
            b, base, f"wheel_back_spacer_{side}", (x, 0.355, 1.40), (0.055, 0.095, 0.29)
        )
        _box(b, base, f"wheel_plate_{side}", (x, 0.23, 1.40), (0.09, 0.03, 0.32))
        for level, (z, radius) in enumerate(((1.28, 0.095), (1.68, 0.043))):
            b.rod(
                base,
                (x, 0.14, z),
                (x, 0.27, z),
                0.013,
                "bright",
                collision=True,
                name=_name(b, f"wheel_axle_{side}_{level}"),
            )
            b.cylinder(
                base,
                (x, 0.15, z),
                radius,
                0.035,
                "dark" if level == 0 else "metal",
                euler=(math.pi / 2, 0, 0),
                collision=True,
                name=_name(b, f"wheel_{side}_{level}"),
            )
            b.cylinder(
                base,
                (x, 0.11, z),
                radius * 0.58,
                0.008,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
        _box(
            b,
            base,
            f"panel_{side}",
            (side * 0.22, 0.28, 1.42),
            (0.08, 0.018, 0.30),
            "bright",
        )
        for end, z in enumerate((1.18, 1.65)):
            b.rod(
                base,
                (side * 0.22, 0.298, z),
                (side * 0.22, 0.445, z),
                0.012,
                collision=True,
                name=_name(b, f"panel_standoff_{side}_{end}"),
            )
        for z in (1.22, 1.30, 1.38, 1.46, 1.54, 1.62):
            b.box(base, (side * 0.22, 0.260, z), (0.07, 0.002, 0.004), "dark")
    # The photo shows a sloped work tray; these thin faces clear the full slide.
    _box(
        b,
        base,
        "tray_front",
        (0, -0.35, 0.920),
        (0.46, 0.008, 0.070),
        "bright",
        euler=(0.35, 0, 0),
    )
    for side in (-1, 1):
        _box(
            b,
            base,
            f"tray_side_{side}",
            (side * 0.445, 0, 0.920),
            (0.008, 0.35, 0.070),
            "bright",
            euler=(0, side * 0.25, 0),
        )
    for i in range(12):
        b.box(base, (-0.30 + i * 0.045, 0.10, 1.916), (0.013, 0.24, 0.001), "dark")
    for side in (-1, 1):
        for level, z in enumerate((1.15, 1.44, 1.72)):
            b.cylinder(
                base,
                (side * 0.635, -0.509, z),
                0.017,
                0.009,
                "dark",
                euler=(math.pi / 2, 0, 0),
                collision=True,
                name=_name(b, f"frame_knob_{side}_{level}"),
            )
    # The inclined display bears on the console; keys/button receive real pads.
    _box(b, base, "display_bracket", (0.92, 0.015, 1.025), (0.25, 0.08, 0.030), "metal")
    for i in range(6):
        _box(
            b,
            base,
            f"key_pad_{i}",
            (0.70 + i * 0.08, -0.35, 0.9965),
            (0.022, 0.034, 0.0015),
            "dark",
        )
    b.cylinder(
        base,
        (1.12, -0.25, 0.9985),
        0.020,
        0.0035,
        "metal",
        collision=True,
        name=_name(b, "button_pad"),
    )
