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
    # Local 7.8 m exterior segment, not the whole upgraded 18 m plasma volume.
    b.cylinder(
        base,
        (0, 0, 1.25),
        0.46,
        3.90,
        "metal",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    for y in (-3.5, -1.75, 0, 1.75, 3.5):
        for x in (-0.60, 0.60):
            b.box(base, (x, y, 0.37), (0.05, 0.10, 0.37), "metal", collision=True)
        _tag(
            b,
            b.box(base, (0, y, 0.75), (0.65, 0.11, 0.04), "bright", collision=True),
            "vessel_support",
        )
    for i in range(25):
        y = -3.60 + i * 0.30
        b.box(base, (0, y, 0.3175), (0.20, 0.07, 0.3175), "dark", collision=True)
        for j in range(20):
            a = math.tau * j / 20
            c = math.tau * (j + 1) / 20
            b.geom(
                base,
                "capsule",
                (0.065,),
                material="sample",
                fromto=(
                    0.55 * math.cos(a),
                    y,
                    1.25 + 0.55 * math.sin(a),
                    0.55 * math.cos(c),
                    y,
                    1.25 + 0.55 * math.sin(c),
                ),
                rgba=(0.64, 0.045, 0.34, 1),
            )
        # Repeated cream service loops distinguish a dense operating apparatus.
        b.rod(base, (-0.42, y, 1.61), (-1.0, y, 2.3), 0.009, "cream")
        b.rod(base, (-1.0, y, 2.3), (-1.45, y, 2.35), 0.009, "cream")
    for i in range(10):
        y = -3.40 + i * 0.70
        b.cylinder(
            base, (0.67, y, 1.25), 0.045, 0.22, "bright", euler=(0, math.pi / 2, 0)
        )
        b.cylinder(
            base, (0.90, y, 1.25), 0.082, 0.015, "metal", euler=(0, math.pi / 2, 0)
        )
        for a in (0, math.pi / 2, math.pi, math.pi * 1.5):
            b.cylinder(
                base,
                (0.92, y + 0.060 * math.cos(a), 1.25 + 0.060 * math.sin(a)),
                0.006,
                0.004,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
        if i not in (1, 2):
            b.cylinder(
                base, (1.50, y, 1.25), 0.009, 0.58, "bright", euler=(0, math.pi / 2, 0)
            )
            b.box(base, (2.03, y, 0.035), (0.13, 0.13, 0.035), "metal", collision=True)
            b.cylinder(base, (2.03, y, 0.66), 0.020, 0.59, "dark", collision=True)
            b.box(base, (2.03, y, 1.285), (0.09, 0.11, 0.035), "metal", collision=True)
    # A clearly visible two-axis diagnostic alignment mechanism in the aisle.
    for y in (-2.58, -2.22):
        b.box(base, (2.20, y, 0.03), (0.15, 0.08, 0.03), "metal", collision=True)
        b.box(base, (2.20, y, 0.84), (0.035, 0.035, 0.78), "bright", collision=True)
    b.box(base, (2.20, -2.40, 1.59), (0.08, 0.23, 0.04), "metal", collision=True)
    carriage = b.moving(
        base,
        "probe_vertical_alignment",
        (2.20, -2.40, 1.25),
        (0, 0, 1),
        (-0.04, 0.04),
        mass=0.3,
        kp=380,
    )
    b.box(carriage, (0, 0, 0), (0.08, 0.215, 0.06), "metal")
    for y in (-0.09, 0.09):
        _tag(
            b,
            b.box(carriage, (-0.30, y, -0.045), (0.35, 0.010, 0.010), "bright"),
            "diagnostic_x_guide",
        )
    stage = b.moving(
        carriage,
        "probe_approach",
        (-0.30, 0, -0.017),
        (1, 0, 0),
        (-0.20, 0),
        mass=0.15,
        kp=340,
    )
    b.box(stage, (0, 0, 0), (0.095, 0.12, 0.018), "dark", collision=True)
    b.box(stage, (-0.065, 0, 0.065), (0.030, 0.040, 0.050), "metal", collision=True)
    b.cylinder(
        stage,
        (-0.40, 0, 0.065),
        0.010,
        0.305,
        "bright",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.box(stage, (-0.705, 0, 0.065), (0.010, 0.025, 0.025), "orange", collision=True)
    b.site(stage, "probe_target", (-0.716, 0, 0.065))
    # Grounded utility supports carry the back-side hose manifold.
    for y in (-3.7, 0, 3.7):
        b.box(base, (-1.48, y, 1.18), (0.045, 0.045, 1.18), "metal", collision=True)
    b.cylinder(
        base,
        (-1.48, 0, 2.35),
        0.045,
        3.8,
        "copper",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    for y in (-2.8, 0.6, 2.8):
        b.box(base, (-1.48, y, 1.90), (0.16, 0.20, 0.29), "shell", collision=True)
    for x in (2.4, 2.85):
        for y in (0.20, 0.70):
            b.box(base, (x, y, 0.03), (0.06, 0.06, 0.03), "rubber", collision=True)
    b.box(base, (2.625, 0.45, 0.87), (0.285, 0.32, 0.81), "shell", collision=True)
    for z in (0.50, 0.86, 1.22):
        b.box(base, (2.625, 0.123, z), (0.23, 0.007, 0.11), "dark")
        b.box(base, (2.56, 0.114, z), (0.11, 0.002, 0.05), "screen")
    b.metadata["capabilities"] = [
        "dry diagnostic target vertical alignment",
        "bounded external probe approach",
    ]
    b.metadata["limitations"].append(
        "Estimated 7.8 m local exterior segment, not a complete as-built LAPD. Upgraded 18 m by 0.75 m plasma-volume specification is reference information only. No plasma, field, vacuum, probe response, radiation or fusion solver; external dry target remains outside the vessel."
    )


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
        "inert alignment target attached to the external diagnostic probe carriage",
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
    for x in (-0.62, 0.62):
        for y in (-0.36, 0.36):
            b.box(base, (x, y, 0.025), (0.075, 0.075, 0.025), "rubber", collision=True)
    b.box(base, (0, 0, 0.39), (0.73, 0.46, 0.34), "cream", collision=True)
    b.box(base, (0, 0, 0.77), (0.75, 0.48, 0.04), "dark", collision=True)
    _tag(
        b,
        b.box(base, (0, 0, 0.83), (0.70, 0.43, 0.02), "metal", collision=True),
        "furnace_optical_deck",
    )
    b.box(base, (0, 0.455, 1.54), (0.73, 0.025, 0.69), "dark", collision=True)
    for x in (-0.72, 0.72):
        b.box(base, (x, 0, 1.54), (0.025, 0.45, 0.69), "dark", collision=True)
    for x in (-0.64, 0.64):
        for y in (-0.37, 0.37):
            b.cylinder(base, (x, y, 1.525), 0.014, 0.675, "bright", collision=True)
    b.box(base, (0, 0, 2.24), (0.75, 0.48, 0.04), "dark", collision=True)
    # Four closed reflector housings are exterior surrogates, not optical surfaces.
    for x in (-0.26, 0.26):
        for y in (-0.08, 0.24):
            b.geom(
                base,
                "ellipsoid",
                (0.215, 0.18, 0.245),
                (x, y, 1.5),
                "metal",
                collision=True,
            )
        for z in (1.31, 1.70):
            b.box(base, (x, -0.17, z), (0.24, 0.020, 0.015), "bright", collision=True)
    for x in (-0.18, 0.18):
        b.cylinder(base, (x, -0.12, 1.45), 0.015, 0.60, "bright", collision=True)
    for z in (1.08, 1.88):
        for x in (-0.33, 0.33):
            b.box(base, (x, -0.08, z), (0.21, 0.11, 0.025), "metal", collision=True)
    # Photo-visible paired cooling fans and lower blowers are static exteriors.
    for x in (-0.56, 0.56):
        b.box(
            base, (x * 0.85, -0.16, 1.50), (0.12, 0.13, 0.025), "metal", collision=True
        )
        b.box(base, (x, -0.29, 1.50), (0.11, 0.06, 0.14), "dark", collision=True)
        for r in (0.035, 0.06, 0.085, 0.10):
            b.ring(base, (x, -0.354, 1.50), r, 0.002, "bright", plane="xz", segments=20)
        for z in (1.39, 1.61):
            b.box(base, (x, -0.358, z), (0.11, 0.002, 0.003), "metal")
        b.box(
            base, (x * 0.82, -0.23, 0.91), (0.13, 0.15, 0.06), "metal", collision=True
        )
        b.cylinder(
            base,
            (x * 0.82, -0.24, 1.06),
            0.12,
            0.12,
            "dark",
            euler=(math.pi / 2, 0, 0),
            collision=True,
        )
    # Reversible feed alignment, kept above a separate fixed lower spindle.
    b.cylinder(base, (0, -0.025, 0.975), 0.045, 0.125, "metal", collision=True)
    b.cylinder(base, (0, -0.025, 1.22), 0.004, 0.12, "bright", collision=True)
    for x in (-0.085, 0.085):
        b.cylinder(base, (x, 0.08, 1.975), 0.010, 0.225, "bright", collision=True)
    carriage = b.moving(
        base,
        "feed_rod_height",
        (0, -0.025, 1.92),
        (0, 0, 1),
        (-0.06, 0.06),
        mass=0.22,
        kp=360,
    )
    b.box(carriage, (0, 0.06, 0), (0.115, 0.075, 0.025), "metal")
    rotation = b.moving(
        carriage,
        "feed_rod_rotation",
        (0, 0, -0.005),
        (0, 0, 1),
        (-1.0, 1.0),
        kind="hinge",
        mass=0.06,
        kp=90,
    )
    b.cylinder(rotation, (0, 0, 0), 0.035, 0.035, "dark", collision=True)
    b.cylinder(rotation, (0, 0, -0.225), 0.004, 0.225, "bright", collision=True)
    _tag(
        b,
        b.cylinder(rotation, (0, 0, -0.470), 0.006, 0.020, "orange", collision=True),
        "inert_feed_coupon",
    )
    b.site(rotation, "crystal_coupon", (0, -0.008, -0.470))
    b.metadata["capabilities"] = [
        "dry crystal-feed rod vertical alignment",
        "inert feed-rod azimuth indexing",
    ]
    b.metadata["limitations"].append(
        "Four-mirror housing exterior references a separate official instrument photo, not a registered location in the panorama. All dimensions and feed mechanics estimated. Reflectors are closed exterior proxies; no light focusing, heating, pressure, melt, growth kinetics or crystal quality prediction."
    )


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
            name=f"jhu_overhead_pipe_{y}",
            type="cylinder",
            size=".06 3.55",
            pos=f"0 {y} 3.48",
            euler=f"0 {math.pi/2} 0",
            rgba=".65 .68 .67 1",
        )
        for x in (-3.0, 3.0):
            box(
                world,
                f"jhu_service_hanger_{y}_{x}",
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
    # Estimated enclosure on a continuous open-frame floor table.
    for x in (-0.57, 0.57):
        for y in (-0.40, 0.40):
            b.box(base, (x, y, 0.025), (0.065, 0.065, 0.025), "rubber", collision=True)
            b.box(base, (x, y, 0.435), (0.035, 0.035, 0.385), "cream", collision=True)
        b.box(base, (x, 0, 0.30), (0.035, 0.40, 0.025), "cream", collision=True)
    b.box(base, (0, 0.40, 0.30), (0.57, 0.025, 0.025), "cream", collision=True)
    _tag(
        b,
        b.box(base, (0, 0, 0.85), (0.67, 0.51, 0.03), "dark", collision=True),
        "coater_tabletop",
    )
    b.box(base, (0, 0, 0.925), (0.55, 0.43, 0.045), "cream", collision=True)
    b.box(base, (0, 0.40, 1.50), (0.55, 0.03, 0.53), "cream", collision=True)
    for x in (-0.525, 0.525):
        b.box(base, (x, 0, 1.50), (0.025, 0.40, 0.53), "cream", collision=True)
    b.housing(base, [(2.03, 0.55, 0.43, 0), (2.16, 0.47, 0.37, 0.035)], 0.025, "cream")
    b.box(base, (0, 0, 2.045), (0.53, 0.40, 0.015), "cream", collision=True)
    # The front glass is a fixed inspection window; it is not an actuated door.
    b.box(base, (-0.06, -0.432, 1.50), (0.415, 0.006, 0.475), "glass")
    for x in (-0.49, 0.38):
        b.box(base, (x, -0.45, 1.50), (0.018, 0.018, 0.49), "cream", collision=True)
    for z in (1.00, 2.0):
        b.box(base, (-0.055, -0.45, z), (0.435, 0.018, 0.015), "cream", collision=True)
    for z in (1.08, 1.91):
        b.box(base, (-0.51, -0.48, z), (0.045, 0.02, 0.034), "dark")
    b.rod(base, (0.345, -0.485, 1.40), (0.345, -0.485, 1.63), 0.013, "dark")
    b.box(base, (0.465, -0.445, 1.50), (0.060, 0.025, 0.50), "cream", collision=True)
    for z, mat in ((1.86, "guard_red"), (1.69, "orange"), (1.52, "dark")):
        b.cylinder(
            base, (0.465, -0.482, z), 0.025, 0.014, mat, euler=(math.pi / 2, 0, 0)
        )
    for x in (0.415, 0.53):
        b.box(base, (x, -0.49, 1.20), (0.045, 0.025, 0.072), "dark")
        b.box(base, (x, -0.517, 1.22), (0.022, 0.004, 0.024), "screen")
        # Static coiled hand-controller leads terminate at enclosure connectors.
        for i in range(34):
            z = 1.12 - i * 0.018
            b.rod(
                base,
                (x + 0.015 * math.cos(i * 1.9), -0.51 + 0.015 * math.sin(i * 1.9), z),
                (
                    x + 0.015 * math.cos((i + 1) * 1.9),
                    -0.51 + 0.015 * math.sin((i + 1) * 1.9),
                    z - 0.018,
                ),
                0.005,
                "dark",
            )
    b.box(base, (-0.37, 0.12, 0.235), (0.115, 0.235, 0.235), "dark", collision=True)
    for z in (0.26, 0.30, 0.34):
        b.box(base, (-0.37, -0.118, z), (0.085, 0.003, 0.01), "metal")
    # Rear posts and full-width crossmembers carry the two estimated dry axes.
    for x in (-0.425, 0.425):
        b.box(base, (x, 0.26, 1.335), (0.025, 0.025, 0.365), "metal", collision=True)
    for z in (1.51, 1.63):
        _tag(
            b,
            b.box(base, (0, 0.245, z), (0.44, 0.012, 0.018), "bright", collision=True),
            "coater_x_rail",
        )
    slide = b.moving(
        base,
        "dry_head_x",
        (0, 0.215, 1.57),
        (1, 0, 0),
        (-0.21, 0.21),
        mass=0.35,
        kp=450,
    )
    b.box(slide, (0, 0, 0), (0.068, 0.045, 0.09), "dark")
    for x in (-0.06, 0.06):
        b.box(slide, (x, -0.17, -0.025), (0.012, 0.21, 0.015), "metal")
    carriage = b.moving(
        slide,
        "dry_head_y",
        (0, -0.19, -0.04),
        (0, 1, 0),
        (-0.13, 0.13),
        mass=0.22,
        kp=420,
    )
    b.box(carriage, (0, 0, 0), (0.074, 0.070, 0.028), "metal", collision=True)
    b.cylinder(carriage, (0, 0, -0.105), 0.034, 0.080, "metal", collision=True)
    _tag(
        b,
        b.cylinder(carriage, (0, 0, -0.211), 0.006, 0.026, "bright", collision=True),
        "coater_dry_nozzle",
    )
    b.cylinder(base, (0, 0, 0.997), 0.19, 0.027, "metal", collision=True)
    _tag(
        b,
        b.cylinder(base, (0, 0, 1.0245), 0.15, 0.0005, "lens", collision=True),
        "supported_300mm_wafer",
    )
    b.box(base, (0.07, -0.06, 1.026), (0.012, 0.009, 0.001), "orange")
    b.site(base, "coater_wafer", (0.07, -0.06, 1.028))
    # Simplified cable chain sits on the rear beam; no unsupported moving hose.
    for i in range(16):
        b.box(base, (-0.38 + i * 0.045, 0.29, 1.78), (0.020, 0.025, 0.026), "dark")
    for x in (-0.38, 0.32):
        b.box(base, (x, 0.29, 1.68), (0.020, 0.025, 0.075), "dark")
    b.cylinder(base, (0.28, 0.16, 2.475), 0.085, 0.325, "metal", collision=True)
    for z in (2.2, 2.32, 2.44, 2.56, 2.68, 2.79):
        b.ring(base, (0.28, 0.16, z), 0.087, 0.003, "dark")
    b.metadata["capabilities"] = [
        "dry transverse nozzle alignment",
        "dry fore-aft nozzle alignment",
    ]
    b.metadata["limitations"].append(
        "Sono-Tek ExactaCoat exterior follows the official installed photograph. Only maximum 300 mm specimen diameter is documented; all room, enclosure and drive dimensions are estimates. No liquid, aerosol, spray, deposition, process chemistry or film-uniformity model."
    )


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
