"""Source-informed sliding chamber access for the BU SEM inspection bay.

The institutional photo establishes the vented column and multiport chamber.
Door-carried stage access follows public Supra-family operating descriptions;
dimensions and hidden bearings are authored, not OEM CAD or a vacuum seal.
Revision two replaces the external-only cassette with an explicit new control.
"""

import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


def _name(b, key):
    return f"{b.name}__sem_access_{key}"


def _box(b, body, key, pos, half, material="metal"):
    return b.box(body, pos, half, material, collision=True, name=_name(b, key))


def _cylinder(b, body, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        body, pos, radius, half, material, collision=True, name=_name(b, key), **kwargs
    )


def _bearing(b, body, key, pos, inner, outer, half, *, along_y=False):
    parent = ET.SubElement(
        body, "body", name=_name(b, key + "_frame"), pos=" ".join(map(str, pos))
    )
    if along_y:
        parent.set("quat", ".7071067811865476 0 0 .7071067811865476")
    annular_mount(
        b,
        parent,
        key,
        (0, 0, 0),
        inner,
        outer,
        half,
        axis="x" if along_y else "z",
        proxies=True,
        namespace="sem_access",
    )


def add_chamber(b, base):
    """A real cavity and continuous cabinet support replace the closed skin."""
    _box(b, base, "floor", (0, -0.31, 0.79), (0.31, 0.63, 0.01))
    _box(b, base, "rear", (0, 0.235, 1.0475), (0.275, 0.035, 0.2475), "cream")
    for side in (-1, 1):
        _box(
            b,
            base,
            f"side_{side}",
            (side * 0.2675, -0.02, 1.0475),
            (0.0275, 0.22, 0.2475),
            "cream",
        )
    _box(b, base, "roof", (0, 0.025, 1.3075), (0.295, 0.265, 0.0125), "cream")
    # Nominal pole exterior supplies a geometric datum only, with no beam model.
    _cylinder(b, base, "pole_shoulder", (0, 0.025, 1.264), 0.083, 0.031)
    _cylinder(b, base, "pole_tip", (0, 0.025, 1.218), 0.037, 0.015, "bright")
    b.site(base, "column_axis", (0, 0.025, 1.203), 0.001)
    for side in (-1, 1):
        x = side * 0.145
        for end, y in enumerate((-0.91, 0.19)):
            _box(
                b, base, f"rail_post_{side}_{end}", (x, y, 0.812), (0.022, 0.009, 0.012)
            )
        _cylinder(
            b,
            base,
            f"rail_{side}",
            (x, -0.36, 0.825),
            0.009,
            0.55,
            "bright",
            quat=(0.7071067811865476, 0.7071067811865476, 0, 0),
        )


def add_stage(b, base):
    """A 600 mm door-carried stroke reaches the chamber and exterior access."""
    drawer = b.moving(
        base,
        "chamber_access",
        (0, 0.025, 0.871),
        (0, -1, 0),
        (0, 0.60),
        mass=0.22,
        kp=360,
    )
    _box(b, drawer, "carrier_plate", (0, 0, 0), (0.19, 0.14, 0.017), "dark")
    for side in (-1, 1):
        for end, y in enumerate((-0.095, 0.095)):
            _bearing(
                b,
                drawer,
                f"slide_bearing_{side}_{end}",
                (side * 0.145, y, -0.046),
                0.0096,
                0.022,
                0.035,
                along_y=True,
            )
            _box(
                b,
                drawer,
                f"bearing_crown_{side}_{end}",
                (side * 0.145, y, -0.023),
                (0.020, 0.035, 0.009),
                "dark",
            )
    # The door and specimen platform move as one connected rigid assembly.
    for side in (-1, 1):
        _box(
            b,
            drawer,
            f"door_stay_{side}",
            (side * 0.17, -0.206, 0.010),
            (0.016, 0.070, 0.016),
        )
    # Split the lower skirt around both fixed rails: real through-openings,
    # with no convex/visual panel crossing the guide axes during extraction.
    _box(b, drawer, "door_panel", (0, -0.281, 0.1943), (0.275, 0.015, 0.2297), "cream")
    _box(b, drawer, "door_skirt", (0, -0.281, -0.0633), (0.275, 0.015, 0.0067), "cream")
    for index, (left, right) in enumerate(
        ((-0.275, -0.1556), (-0.1344, 0.1344), (0.1556, 0.275))
    ):
        _box(
            b,
            drawer,
            f"door_rail_web_{index}",
            ((left + right) / 2, -0.281, -0.046),
            ((right - left) / 2, 0.015, 0.0106),
            "cream",
        )
    for side in (-1, 1):
        _cylinder(
            b,
            drawer,
            f"handle_post_{side}",
            (side * 0.080, -0.319, 0.195),
            0.009,
            0.023,
            "dark",
            quat=(0.7071067811865476, 0.7071067811865476, 0, 0),
        )
    _cylinder(
        b,
        drawer,
        "handle",
        (0, -0.342, 0.195),
        0.012,
        0.09,
        "dark",
        quat=(0.7071067811865476, 0, 0.7071067811865476, 0),
    )
    b.site(drawer, "chamber_handle", (0, -0.354, 0.195), 0.002)
    for side in (-1, 1):
        _cylinder(
            b,
            drawer,
            f"door_port_{side}",
            (side * 0.20, -0.306, 0.299),
            0.040,
            0.010,
            "bright",
            quat=(0.7071067811865476, 0.7071067811865476, 0, 0),
        )
        b.cylinder(
            drawer,
            (side * 0.20, -0.318, 0.299),
            0.022,
            0.002,
            "dark",
            quat=(0.7071067811865476, 0.7071067811865476, 0, 0),
            name=_name(b, f"door_port_face_{side}"),
        )
    # Continuous spindle / truly open bearing; retained original stub dimensions.
    _bearing(b, drawer, "rotary_bearing", (0, 0, 0.02975), 0.0126, 0.039, 0.01275)
    stage = b.moving(
        drawer,
        "specimen_azimuth",
        (0, 0, 0.055),
        (0, 0, 1),
        (-1.2, 1.2),
        kind="hinge",
        mass=0.05,
        kp=80,
    )
    _cylinder(b, stage, "spindle", (0, 0, -0.0242), 0.012, 0.0132)
    _cylinder(b, stage, "platen", (0, 0, 0), 0.065, 0.012)
    _cylinder(b, stage, "stub", (0, 0, 0.018), 0.015, 0.006, "orange")
    b.box(
        stage,
        (0.004, 0, 0.025),
        (0.0015, 0.011, 0.001),
        "dark",
        name=_name(b, "stub_mark"),
    )
    b.site(stage, "sem_stub", (0, 0, 0.027))
    b.metadata["mechanism_revision"] = {
        "version": 2,
        "retired_controls": ["specimen_exchange"],
        "new_control": "chamber_access",
        "automatic_legacy_mapping": False,
        "migration": "The old external-only 160 mm cassette cannot enter the chamber. Use chamber_access: zero places the stub under the column; 0.60 m opens the door and exposes the stage. Old saved actions require explicit replanning.",
        "original_azimuth_range_rad": [-1.2, 1.2],
        "radial_running_clearance_m": 0.0006,
        "closed_door_clearance_m": 0.001,
        "support_model": "Ideal-joint open bearings; no preload, structural stress or vacuum-seal performance",
    }
    b.metadata["capabilities"] = [
        "door-carried specimen stage enters and leaves the chamber",
        "retained 30 mm inert stub azimuth indexing",
        "geometric column-axis registration",
    ]
    b.metadata["limitations"].append(
        "Version two retires specimen_exchange without automatic action mapping. The dry door remains 1 mm clear of the chamber; it is not a vacuum seal. Internal mechanics and motion bounds are estimated. The original 30 mm stub is fixed; no automatic sample pickup, five-axis positioning, electron image formation or scientific measurement is implemented."
    )
    return drawer
