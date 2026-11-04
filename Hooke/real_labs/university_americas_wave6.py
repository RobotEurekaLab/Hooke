"""Photograph-informed plasma, vacuum, crystal and coating inspection mechanics.

All process physics is out of scope; each station is a dry mechanical prototype.
Source-specific dimensions are identified independently of estimated envelopes.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

UCLA = "https://plasma.physics.ucla.edu/large-plasma-device.html"
PENN = "https://www.nano.upenn.edu/resources/equipmentinstrumentation/"
JHU = "https://www.paradim.org/bulkcrystal"
ALBERTA = "https://www.nanofab.ualberta.ca/new-spray-coating-tool-available/"


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _lapd(b, base, params):
    from real_labs.ucla_probe_geometry import build_segment

    build_segment(b, base, params)


BUILDERS = {"ucla_lapd_diagnostic_segment": _lapd}
SOURCES = {
    "ucla_lapd_diagnostic_segment": {
        "reference": "UCLA BaPSF apparatus photograph linked by the official UCLA newsroom; upgraded facility specification",
        "url": UCLA,
        "dimensions_m": [4.55, 7.80, 2.40],
        "dimension_basis": "Estimated local exterior segment and dry alignment mechanism; verified 18 m length / 0.75 m diameter refer to upgraded plasma volume, not this modeled segment or machine envelope.",
    }
}
SAMPLE_INTERFACES = {
    "ucla_lapd_diagnostic_segment": (
        "probe_target",
        (0.020, 0.050, 0.050),
        "clamped",
        "retained inert probe head; probe_target is a physical off-axis rear-face inspection land",
    )
}
FEATURES = {}


def _denton(b, base, params):
    for x in (-0.74, 0.20, 0.94):
        for y in (-0.32, 0.32):
            b.cylinder(base, (x, y, 0.025), 0.035, 0.025, "metal", collision=True)
    b.box(base, (0.10, 0, 0.075), (0.88, 0.42, 0.025), "cream", collision=True)
    b.box(base, (-0.28, 0, 0.475), (0.49, 0.40, 0.375), "cream", collision=True)
    b.box(base, (-0.28, -0.408, 0.50), (0.45, 0.008, 0.30), "shell")
    for x in (-0.57, 0.02):
        b.box(base, (x, -0.42, 0.70), (0.08, 0.01, 0.012), "dark")
    b.box(base, (-0.28, 0, 0.868), (0.49, 0.40, 0.018), "bright", collision=True)
    # Open chamber shell makes the inert 150 mm wafer mechanically visible.
    for x in (-0.575, 0.015):
        b.box(base, (x, 0.02, 1.14), (0.020, 0.29, 0.254), "cream", collision=True)
    b.box(base, (-0.28, 0.29, 1.14), (0.275, 0.020, 0.254), "metal", collision=True)
    b.box(base, (-0.28, 0.02, 1.41), (0.315, 0.31, 0.016), "metal", collision=True)
    b.box(base, (-0.28, 0.02, 0.911), (0.275, 0.29, 0.025), "metal", collision=True)
    b.cylinder(base, (-0.28, 0.02, 0.961), 0.11, 0.025, "metal")
    platter = b.moving(
        base,
        "wafer_platen_index",
        (-0.28, 0.02, 1.001),
        (0, 0, 1),
        (-1.0, 1.0),
        kind="hinge",
        mass=0.13,
        kp=120,
    )
    b.cylinder(platter, (0, 0, 0), 0.098, 0.015, "metal", collision=True)
    _tag(
        b,
        b.cylinder(platter, (0, 0, 0.01535), 0.075, 0.00035, "lens", collision=True),
        "supported_150mm_wafer",
    )
    b.box(platter, (0.018, 0, 0.0162), (0.004, 0.055, 0.0005), "orange")
    b.site(platter, "pvd_wafer", (0, 0, 0.017))
    # Hinge origin and pre-open inspection pose are inferred, not OEM CAD.
    door = b.moving(
        base,
        "chamber_access_angle",
        (-0.595, -0.30, 1.15),
        (0, 0, 1),
        (-0.75, 0),
        kind="hinge",
        mass=0.8,
        kp=260,
    )
    panel = ET.SubElement(door, "body", euler="0 0 -.55", gravcomp="1")
    b.box(panel, (0.305, 0, 0), (0.30, 0.018, 0.235), "cream", collision=True)
    b.cylinder(
        panel, (0.305, -0.026, -0.01), 0.105, 0.010, "metal", euler=(math.pi / 2, 0, 0)
    )
    b.cylinder(
        panel, (0.305, -0.038, -0.01), 0.080, 0.003, "lens", euler=(math.pi / 2, 0, 0)
    )
    for i in range(8):
        a = math.tau * i / 8
        b.cylinder(
            panel,
            (0.305 + 0.094 * math.cos(a), -0.039, -0.01 + 0.094 * math.sin(a)),
            0.005,
            0.004,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    for z in (-0.15, 0.15):
        b.cylinder(base, (-0.595, -0.30, 1.15 + z), 0.025, 0.032, "metal")
    for x in (-0.44, -0.10):
        b.cylinder(base, (x, 0.04, 1.547), 0.12, 0.12, "metal", collision=True)
        for z in (1.434, 1.667):
            b.cylinder(base, (x, 0.04, z), 0.143, 0.008, "bright", collision=True)
        b.cylinder(base, (x, 0.04, 1.765), 0.025, 0.09, "bright", collision=True)
        b.cylinder(base, (x, 0.04, 1.915), 0.075, 0.06, "shell", collision=True)
    for i in range(6):
        x = -0.52 + i * 0.095
        material = "dark" if i % 2 else "copper"
        points = (
            (x, 0.10, 1.965),
            (x - 0.13, 0.17, 2.20 + i * 0.018),
            (0.22, 0.23, 2.08),
            (0.18, 0.29, 1.12),
        )
        for p, q in zip(points, points[1:]):
            b.rod(base, p, q, 0.009, material)
    # Black control rack is connected to the common floor base.
    b.box(base, (0.64, 0, 0.98), (0.36, 0.42, 0.88), "cream", collision=True)
    b.box(base, (0.64, -0.43, 0.98), (0.315, 0.010, 0.81), "dark", collision=True)
    b.box(base, (0.64, -0.446, 1.47), (0.265, 0.006, 0.19), "screen")
    b.cylinder(
        base, (0.64, -0.453, 1.10), 0.037, 0.012, "warning", euler=(math.pi / 2, 0, 0)
    )
    b.cylinder(
        base, (0.64, -0.472, 1.10), 0.024, 0.008, "guard_red", euler=(math.pi / 2, 0, 0)
    )
    b.box(base, (0.64, -0.57, 0.91), (0.28, 0.15, 0.018), "dark", collision=True)
    for row in range(3):
        for col in range(9):
            b.box(
                base,
                (0.415 + col * 0.055, -0.65 + row * 0.060, 0.935),
                (0.021, 0.022, 0.006),
                "metal",
            )
    for z in (0.29, 0.46, 0.63):
        b.box(base, (0.54, -0.446, z), (0.10, 0.006, 0.045), "screen")
        b.box(base, (0.81, -0.446, z), (0.065, 0.006, 0.033), "metal")
    b.metadata["capabilities"] = [
        "pre-open dry chamber-door inspection",
        "150 mm inert wafer platen indexing",
    ]
    b.metadata["limitations"].append(
        "Photo-informed Denton Explorer 14 PVD-05 exterior. Door hinge, pre-open pose and internal platter are inferred; maximum 150 mm supported wafer diameter is documented. No sputtering plasma, vacuum, process gas, deposition or film thickness model."
    )


BUILDERS["penn_denton_pvd_inspection"] = _denton
SOURCES["penn_denton_pvd_inspection"] = {
    "reference": "Penn QNF official PVD-05 Denton Explorer 14 photograph and equipment listing",
    "url": PENN,
    "dimensions_m": [1.83, 1.25, 2.31],
    "dimension_basis": "Documented maximum wafer diameter 150 mm. All external dimensions, internal geometry, hinge and inspection strokes estimated.",
}
SAMPLE_INTERFACES["penn_denton_pvd_inspection"] = (
    "pvd_wafer",
    (0.15, 0.15, 0.0007),
    "clamped",
    "inert 150 mm wafer attached to a dry indexed platen",
)


def _four_mirror(b, base, params):
    from real_labs.jhu_furnace_geometry import build_furnace

    build_furnace(b, base, params)


def _jhu_features(world, definition):
    box(
        world,
        "jhu_blue_furnace_base",
        (0.55, 0.43, 0.055),
        (-0.75, 3.28, 0.055),
        (0.12, 0.14, 0.16, 1),
    )
    box(
        world,
        "jhu_blue_furnace_body",
        (0.55, 0.43, 1.12),
        (-0.75, 3.28, 1.23),
        (0.10, 0.30, 0.56, 1),
    )
    for z in (0.9, 1.45):
        box(
            world,
            f"jhu_blue_front_{z}",
            (0.47, 0.009, 0.20),
            (-0.75, 2.841, z),
            (0.12, 0.29, 0.52, 1),
        )
    for y in (0.9, 2.30):
        box(
            world,
            f"jhu_white_furnace_{y}",
            (0.43, 0.50, 1.0),
            (-3.32, y, 1.0),
            (0.76, 0.78, 0.77, 1),
        )
        box(
            world,
            f"jhu_white_front_{y}",
            (0.012, 0.40, 0.24),
            (-2.876, y, 1.14),
            (0.25, 0.28, 0.28, 1),
        )
    for y in (-2.0, 0.1, 2.2):
        ET.SubElement(
            world,
            "geom",
            name=f"arch_roof_jhu_overhead_pipe_{y}",
            type="cylinder",
            size=".06 3.55",
            pos=f"0 {y} 3.48",
            euler=f"0 {math.pi/2} 0",
            rgba=".65 .68 .67 1",
        )
        for x in (-3.0, 3.0):
            box(
                world,
                f"arch_roof_jhu_service_hanger_{y}_{x}",
                (0.02, 0.02, 0.13),
                (x, y, 3.67),
                (0.35, 0.37, 0.36, 1),
            )
    for y in (-1.8, -0.75, 0.30):
        box(
            world,
            f"jhu_inert_prep_case_{y}",
            (0.28, 0.22, 0.06),
            (-0.2, y, 0.96),
            (0.63, 0.64, 0.57, 1),
        )
        box(
            world,
            f"jhu_prep_calibration_block_{y}",
            (0.075, 0.07, 0.04),
            (-0.2, y, 1.06),
            (0.26, 0.28, 0.27, 1),
        )

    # Source-visible utility drops are fixed room context with supported ends;
    # their routing and dimensions are estimated, not an electrical model.
    for y in (-0.9, 1.3):
        ET.SubElement(
            world, "geom", name=f"arch_roof_jhu_suspended_utility_lead_{y}",
            type="capsule", size=".004", fromto=f"-.35 {y} 3.80 -.35 {y} 2.70",
            rgba=".65 .54 .12 1",
        )
        box(world, f"arch_roof_jhu_suspended_utility_box_{y}", (.045,.025,.065),
            (-.35,y,2.635), (.32,.34,.32,1))
        for z in (2.61,2.66):
            box(world, f"arch_roof_jhu_utility_socket_{y}_{z}", (.012,.002,.013),
                (-.35,y-.026,z), (.10,.11,.11,1))
    for x in (-1.00,-.80,-.60,-.40):
        ET.SubElement(
            world,"geom",name=f"jhu_blue_control_knob_{x}",type="cylinder",
            size=".023 .009",pos=f"{x} 2.824 1.45",euler=f"{math.pi/2} 0 0",
            rgba=".12 .13 .14 1",
        )
    box(world,"jhu_blue_status_panel",(.18,.008,.070),(-.75,2.844,1.92),
        (.10,.13,.15,1))


BUILDERS["jhu_four_mirror_feed_inspection"] = _four_mirror
SOURCES["jhu_four_mirror_feed_inspection"] = {
    "reference": "JHU PARADIM official bulk-growth room panorama and separate four-mirror xenon-furnace detail",
    "url": JHU,
    "dimensions_m": [1.55, 1.00, 2.28],
    "dimension_basis": "All dimensions and internal drive layout estimated. Instrument detail is not geometrically registered to the room panorama; placement is authored.",
}
SAMPLE_INTERFACES["jhu_four_mirror_feed_inspection"] = (
    "crystal_coupon",
    (0.012, 0.012, 0.040),
    "clamped",
    "inert feed-rod coupon fixed to the reversible upper feed shaft",
)
FEATURES["jhu_paradim_bulk_crystal_growth"] = _jhu_features


def _exactacoat(b, base, params):
    from real_labs.alberta_coater_geometry import build_coater

    build_coater(b, base, params)


BUILDERS["alberta_exactacoat_dry_alignment"] = _exactacoat
SOURCES["alberta_exactacoat_dry_alignment"] = {
    "reference": "University of Alberta nanoFAB installed ExactaCoat tool photograph and maximum specimen specification",
    "url": ALBERTA,
    "dimensions_m": [1.34, 1.07, 2.80],
    "dimension_basis": "Maximum specimen diameter 300 mm documented. Enclosure, table, duct and dry XY mechanism dimensions estimated from a single photograph.",
}
SAMPLE_INTERFACES["alberta_exactacoat_dry_alignment"] = (
    "coater_wafer",
    (0.30, 0.30, 0.001),
    "clamped",
    "inert 300 mm wafer fixed to the stationary platen below the dry alignment head",
)
