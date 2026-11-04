"""Photograph-informed instrument mechanics for four distinct university bays.

Original metre-scale geometry; dimensions and motorization are estimates unless
explicitly sourced. No plasma, laser propagation or lithography process solver.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

COLUMBIA = "https://www.apam.columbia.edu/about/news/hands-fusion-plasma-lab-hosts-first-hbt-ep-fun-run"
HARVARD = "https://news.harvard.edu/gazette/story/2019/03/snapshots-from-harvards-science-labs/"
ILLINOIS = "https://biophotonics.illinois.edu/about/facilities"
MICHIGAN = (
    "https://alumni.umich.edu/michigan-alum/spaces-the-lurie-nanofabrication-facility/"
)


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _flange(b, base, y, z, radius=0.21, x=0):
    b.cylinder(base, (x, y, z), radius, 0.018, "bright", euler=(math.pi / 2, 0, 0))
    for i in range(12):
        a = math.tau * i / 12
        b.cylinder(
            base,
            (
                x + radius * 0.82 * math.cos(a),
                y - 0.021,
                z + radius * 0.82 * math.sin(a),
            ),
            0.009,
            0.006,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )


def _hbtep(b, base, params):
    # Estimated apparatus envelope, with explicitly inferred load-bearing ties.
    for x in (-1.2, 1.2):
        for y in (-1.05, 1.05):
            b.box(base, (x, y, 0.04), (0.15, 0.15, 0.04), "dark", collision=True)
            b.box(base, (x, y, 1.29), (0.055, 0.055, 1.21), "metal", collision=True)
            for z in (0.38, 0.66, 0.94, 1.7, 1.98, 2.26):
                b.cylinder(
                    base,
                    (x, y - 0.058, z),
                    0.016,
                    0.003,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
        for z in (0.25, 1.16, 2.48):
            b.box(base, (x, 0, z), (0.055, 1.105, 0.04), "metal", collision=True)
    for y in (-1.05, 1.05):
        for z in (0.25, 1.16, 2.48):
            b.box(base, (0, y, z), (1.255, 0.055, 0.04), "metal", collision=True)
    for x in (-0.8, 0.8):
        _tag(
            b,
            b.box(base, (x, 0, 1.16), (0.045, 1.05, 0.04), "bright", collision=True),
            "vessel_saddle_beam",
        )
    # Segmented toroidal vessel retains a visible central opening.
    for i in range(40):
        a, c = math.tau * i / 40, math.tau * (i + 1) / 40
        b.geom(
            base,
            "capsule",
            (0.20,),
            material="metal",
            collision=True,
            fromto=(
                0.85 * math.cos(a),
                0.85 * math.sin(a),
                1.4,
                0.85 * math.cos(c),
                0.85 * math.sin(c),
                1.4,
            ),
        )
    # Coil outlines in radial vertical planes, grounded through the support cage.
    for i in range(10):
        a = math.tau * i / 10
        for offset in (-0.035, 0, 0.035):
            points = []
            for radius, z in (
                (0.49, 0.98),
                (1.12, 0.98),
                (1.12, 1.87),
                (0.49, 1.87),
                (0.49, 0.98),
            ):
                points.append(
                    (
                        radius * math.cos(a) - offset * math.sin(a),
                        radius * math.sin(a) + offset * math.cos(a),
                        z,
                    )
                )
            for p, q in zip(points, points[1:]):
                b.rod(base, p, q, 0.024, "copper")
    # Foreground diagnostic line and open steel support stand.
    for x in (-0.64, 0.64):
        for y in (-2.88, -1.30):
            b.box(base, (x, y, 0.025), (0.10, 0.10, 0.025), "dark", collision=True)
            b.box(base, (x, y, 0.515), (0.035, 0.035, 0.465), "bright", collision=True)
        for z in (0.25, 0.98):
            b.box(base, (x, -2.09, z), (0.035, 0.825, 0.035), "bright", collision=True)
    for y in (-2.88, -1.30):
        b.box(base, (0, y, 0.98), (0.675, 0.035, 0.035), "bright", collision=True)
    for y in (-2.5, -1.5):
        b.box(base, (0, y, 0.98), (0.675, 0.035, 0.035), "bright", collision=True)
        _tag(
            b,
            b.box(base, (0, y, 1.1175), (0.19, 0.085, 0.1025), "metal", collision=True),
            "diagnostic_saddle",
        )
    b.cylinder(
        base,
        (0, -2.0, 1.4),
        0.18,
        0.85,
        "metal",
        collision=True,
        euler=(math.pi / 2, 0, 0),
    )
    for y in (-2.85, -2.55, -1.8, -1.15):
        _flange(b, base, y, 1.4)
    for y in (-2.48, -2.4, -2.32, -2.24):
        b.cylinder(base, (0, y, 1.4), 0.19, 0.025, "cream", euler=(math.pi / 2, 0, 0))
    b.cylinder(base, (0, -2.53, 1.68), 0.10, 0.15, "metal", collision=True)
    b.cylinder(base, (0, -2.53, 1.835), 0.14, 0.018, "bright", collision=True)
    # A dry alignment target beside the port is visible over its complete stroke.
    # This inspection fixture is authored; it is not an HBT-EP OEM actuator claim.
    b.box(base, (0.45, -2.40, 1.055), (0.16, 0.31, 0.04), "dark", collision=True)
    for x in (0.34, 0.56):
        _tag(
            b,
            b.box(base, (x, -2.40, 1.105), (0.014, 0.29, 0.01), "bright"),
            "probe_guide",
        )
    slide = b.moving(
        base,
        "diagnostic_traverse",
        (0.45, -2.58, 1.13),
        (0, 1, 0),
        (0, 0.24),
        mass=0.45,
        kp=450,
    )
    b.box(slide, (0, 0, 0), (0.145, 0.08, 0.015), "dark", collision=True)
    b.box(slide, (0, 0.055, 0.16), (0.028, 0.028, 0.145), "metal", collision=True)
    b.box(slide, (0, 0, 0.31), (0.095, 0.07, 0.025), "metal")
    lift = b.moving(
        slide,
        "target_height",
        (0, -0.05, 0.31),
        (0, 0, 1),
        (0, 0.10),
        mass=0.12,
        kp=350,
    )
    b.cylinder(lift, (0, 0, 0.065), 0.011, 0.10, "bright", collision=True)
    b.box(lift, (0, -0.025, 0.17), (0.028, 0.008, 0.028), "orange", collision=True)
    b.site(lift, "alignment_coupon", (0, -0.034, 0.17))
    # Cabinet feet meet its base, and overhead services connect to the cage.
    for x in (1.75, 2.27):
        for y in (-0.22, 0.32):
            b.box(base, (x, y, 0.025), (0.055, 0.055, 0.025), "rubber", collision=True)
    b.box(base, (2.01, 0.05, 0.82), (0.32, 0.33, 0.77), "shell", collision=True)
    for z in (0.4, 0.67, 0.94, 1.21):
        b.box(base, (2.01, -0.288, z), (0.28, 0.008, 0.095), "dark")
        b.box(base, (1.9, -0.300, z), (0.12, 0.005, 0.045), "screen")
    for i in range(9):
        x = -1.05 + i * 0.25
        b.rod(
            base, (x, 0.95, 2.50), (x, -0.3, 2.74), 0.006, "blue" if i % 2 else "dark"
        )
        b.rod(
            base, (x, -0.3, 2.74), (x, -0.7, 1.75), 0.005, "blue" if i % 2 else "dark"
        )
    b.metadata["capabilities"] = [
        "dry diagnostic target axial traverse",
        "diagnostic target vertical alignment",
    ]
    b.metadata["limitations"].append(
        "Estimated local HBT-EP diagnostic bay, not a measured complete tokamak. Authored dry alignment fixture; no vacuum pumping, magnetic field, plasma, radiation or scientific diagnostic-response model."
    )


def _columbia_features(world, definition):
    for x in (-2.5, 2.5):
        box(
            world,
            f"columbia_service_riser_{x}",
            (0.10, 0.12, 1.6),
            (x, 2.2, 1.6),
            (0.5, 0.53, 0.54, 1),
        )
    box(
        world,
        "columbia_overhead_service",
        (2.6, 0.15, 0.14),
        (0, 2.2, 3.18),
        (0.56, 0.59, 0.59, 1),
    )
    for x in (-1.8, 1.8):
        box(
            world,
            f"columbia_hanger_{x}",
            (0.022, 0.022, 0.24),
            (x, 2.2, 3.56),
            (0.35, 0.37, 0.38, 1),
        )
    # Rear cabinets and cable tray correspond to the photographed utility edge.
    for x in (-2.35, -1.6):
        box(
            world,
            f"columbia_rear_cabinet_{x}",
            (0.34, 0.32, 1.05),
            (x, 2.48, 1.05),
            (0.40, 0.43, 0.42, 1),
        )
        for z in (0.7, 1.4):
            box(
                world,
                f"columbia_cabinet_div_{x}_{z}",
                (0.30, 0.006, 0.008),
                (x, 2.153, z),
                (0.16, 0.18, 0.18, 1),
            )


BUILDERS = {"columbia_hbtep_diagnostic_assembly": _hbtep}
SOURCES = {
    "columbia_hbtep_diagnostic_assembly": {
        "reference": "Columbia HBT-EP 2026 official laboratory event photograph",
        "url": COLUMBIA,
        "dimensions_m": [3.65, 4.085, 2.75],
        "dimension_basis": "All dimensions, hidden supports and dry-fixture strokes estimated from one local apparatus photograph; not calibrated CAD.",
    }
}
SAMPLE_INTERFACES = {
    "columbia_hbtep_diagnostic_assembly": (
        "alignment_coupon",
        (0.056, 0.016, 0.056),
        "clamped",
        "inert diagnostic alignment coupon on authored inspection fixture",
    )
}
FEATURES = {"columbia_fusion_hbt_ep": _columbia_features}


def _optic(b, parent, x, y, deck_z, height=0.14):
    b.box(parent, (x, y, deck_z + 0.009), (0.044, 0.035, 0.009), "dark", collision=True)
    b.cylinder(
        parent,
        (x, y, deck_z + 0.02 + height / 2),
        0.009,
        height / 2,
        "bright",
        collision=True,
    )
    b.ring(
        parent,
        (x, y, deck_z + 0.02 + height),
        0.030,
        0.008,
        "dark",
        plane="xz",
        segments=12,
    )
    b.cylinder(
        parent,
        (x, y, deck_z + 0.02 + height),
        0.025,
        0.004,
        "lens",
        euler=(math.pi / 2, 0, 0),
    )
    b.cylinder(
        parent,
        (x + 0.043, y, deck_z + 0.02 + height),
        0.012,
        0.018,
        "dark",
        euler=(0, math.pi / 2, 0),
    )


def _breadboard(b, base, x, y, z, width, depth):
    _tag(
        b,
        b.box(base, (x, y, z), (width / 2, depth / 2, 0.035), "metal", collision=True),
        "optical_breadboard",
    )
    for i in range(int(width / 0.12)):
        for j in range(int(depth / 0.12)):
            b.cylinder(
                base,
                (
                    x - width / 2 + 0.07 + i * 0.12,
                    y - depth / 2 + 0.07 + j * 0.12,
                    z + 0.036,
                ),
                0.004,
                0.001,
                "dark",
            )


def _doyle(b, base, params):
    for center in (-1.0, 1.0):
        for x in (center - 0.63, center + 0.63):
            for y in (-0.87, 0.87):
                b.cylinder(base, (x, y, 0.035), 0.14, 0.035, "rubber", collision=True)
                b.cylinder(base, (x, y, 0.4575), 0.115, 0.3875, "dark", collision=True)
        _breadboard(b, base, center, 0, 0.88, 1.7, 2.0)
        # Elevated optical deck, supported at four corners from the main board.
        for x in (center - 0.60, center + 0.60):
            for y in (0.14, 0.75):
                b.cylinder(base, (x, y, 1.035), 0.020, 0.12, "bright", collision=True)
        _breadboard(b, base, center, 0.44, 1.19, 1.40, 0.80)
        for x in (center - 0.43, center - 0.11, center + 0.29):
            for y in (0.19, 0.65):
                _optic(b, base, x, y, 1.225, height=0.11 + 0.03 * (y > 0.2))
        b.box(base, (center, -0.72, 1.035), (0.43, 0.18, 0.12), "dark", collision=True)
        b.box(base, (center, -0.905, 1.035), (0.31, 0.006, 0.025), "cream")
        for x in (center - 0.51, center + 0.50):
            _optic(b, base, x, -0.20, 0.915)
        for x in (center - 0.70, center + 0.70):
            b.cylinder(base, (x, 0.93, 1.1), 0.012, 0.185, "bright", collision=True)
    # Freestanding shelf posts continue all the way down to the floor.
    for x in (-1.85, 1.85):
        for y in (-0.9, 0.9):
            b.box(base, (x, y, 0.025), (0.09, 0.09, 0.025), "dark", collision=True)
            b.box(base, (x, y, 1.135), (0.025, 0.025, 1.085), "metal", collision=True)
    for y in (-0.9, 0.9):
        b.box(base, (0, y, 2.245), (1.875, 0.025, 0.025), "metal", collision=True)
    for x in (-1.0, 1.0):
        _tag(
            b,
            b.box(base, (x, 0, 2.22), (0.85, 1.0, 0.035), "dark", collision=True),
            "overhead_shelf",
        )
        for y in (-0.65, 0.0, 0.65):
            b.box(base, (x, y, 2.37), (0.25, 0.20, 0.115), "shell", collision=True)
            b.box(base, (x, y - 0.205, 2.37), (0.14, 0.005, 0.065), "screen")
            for k in range(3):
                b.cylinder(
                    base,
                    (x + 0.20, y - 0.208, 2.33 + k * 0.038),
                    0.008,
                    0.007,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    # Central hanging partition is observed; its rail and hem are inferred.
    b.box(base, (0, 0.15, 2.245), (0.026, 0.78, 0.026), "metal", collision=True)
    for j in range(18):
        b.box(
            base,
            (0.014 * math.sin(j), -0.57 + j * 0.084, 1.62),
            (0.018, 0.044, 0.60),
            "dark",
        )
    # Dry two-axis optical specimen alignment on the front right table.
    b.box(base, (1.0, -0.32, 0.935), (0.24, 0.17, 0.02), "dark", collision=True)
    for y in (-0.43, -0.21):
        _tag(
            b,
            b.box(base, (1.0, y, 0.966), (0.21, 0.011, 0.011), "bright"),
            "doyle_translation_guide",
        )
    stage = b.moving(
        base,
        "optical_translation",
        (1.0, -0.32, 0.987),
        (1, 0, 0),
        (-0.055, 0.055),
        mass=0.22,
        kp=350,
    )
    b.box(stage, (0, 0, 0), (0.12, 0.105, 0.010), "metal", collision=True)
    for x in (-0.085, 0.085):
        b.box(stage, (x, 0, 0.054), (0.015, 0.045, 0.044), "dark", collision=True)
    tilt = b.moving(
        stage,
        "sample_pitch",
        (0, 0, 0.092),
        (1, 0, 0),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.07,
        kp=100,
    )
    b.cylinder(tilt, (0, 0, 0), 0.01, 0.10, "bright", euler=(0, math.pi / 2, 0))
    b.box(tilt, (0, 0, 0.040), (0.057, 0.009, 0.065), "dark", collision=True)
    b.box(tilt, (0, -0.010, 0.040), (0.043, 0.002, 0.049), "orange", collision=True)
    b.site(tilt, "optical_coupon", (0, -0.013, 0.040))
    for side in (-1, 1):
        for i in range(7):
            x = side * (0.35 + i * 0.19)
            b.rod(base, (x, 0.91, 2.18), (x + 0.05, 0.78, 1.72), 0.003, "dark")
            b.rod(
                base,
                (x + 0.05, 0.78, 1.72),
                (x - 0.02, 0.65, 1.24),
                0.003,
                "copper" if i % 3 == 0 else "dark",
            )
    b.metadata["capabilities"] = [
        "dry optical target lateral translation",
        "optical target pitch alignment",
    ]
    b.metadata["limitations"].append(
        "Historical 2019 workstation photograph; dimensions, hidden optics, support framing and motorization estimated. No live laser, ray tracing, ultracold molecule, quantum or scientific measurement model."
    )


def _harvard_features(world, definition):
    # Bounded workstation view: rear services rather than invented full-lab plan.
    box(
        world,
        "doyle_rear_service_trunk",
        (2.3, 0.065, 0.08),
        (0, 2.2, 2.52),
        (0.58, 0.60, 0.58, 1),
    )
    for x in (-2.2, 2.2):
        box(
            world,
            f"doyle_service_anchor_{x}",
            (0.08, 0.35, 0.05),
            (x, 2.45, 2.52),
            (0.33, 0.34, 0.32, 1),
        )


BUILDERS["harvard_doyle_dual_optical_bench"] = _doyle
SOURCES["harvard_doyle_dual_optical_bench"] = {
    "reference": "Harvard Gazette John Doyle laboratory photograph, 2019",
    "url": HARVARD,
    "dimensions_m": [3.93, 2.0, 2.49],
    "dimension_basis": "All sizes estimated from the historical photo; no calibrated optical layout or actual actuator performance claimed.",
}
SAMPLE_INTERFACES["harvard_doyle_dual_optical_bench"] = (
    "optical_coupon",
    (0.086, 0.004, 0.098),
    "clamped",
    "inert optical alignment coupon mounted in a pitch cradle",
)
FEATURES["harvard_doyle_cold_molecule_optics"] = _harvard_features


def _laser_case(b, base, x, y, deck, half):
    w, d, h = half
    b.box(base, (x, y, deck + h), (w, d, h), "dark", collision=True)
    b.box(
        base,
        (x, y - d - 0.002, deck + h),
        (w * 0.88, 0.004, h * 0.83),
        "metal",
        rgba=(0.12, 0.13, 0.14, 1),
    )
    for k in range(7):
        b.box(
            base,
            (x + w + 0.002, y - d * 0.55 + k * 0.032, deck + h * 1.25),
            (0.003, d * 0.11, 0.004),
            "rubber",
        )
    for i in range(3):
        b.cylinder(
            base,
            (x - w * 0.58 + i * w * 0.55, y - d - 0.012, deck + h * 0.65),
            0.012,
            0.010,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )
        b.rod(
            base,
            (x - w * 0.58 + i * w * 0.55, y - d - 0.023, deck + h * 0.65),
            (x - w * 0.58 + i * w * 0.55, y - d - 0.10, deck - 0.15),
            0.004,
            "cream",
        )
    b.box(base, (x, y - d - 0.007, deck + h * 1.4), (0.025, 0.002, 0.025), "warning")
    for dx in (-w * 0.55, w * 0.55):
        b.cylinder(base, (x + dx, y, deck + 2 * h + 0.014), 0.016, 0.014, "dark")


def _illinois(b, base, params):
    for x in (-0.91, 0.91):
        for y in (-1.10, 1.10):
            b.cylinder(base, (x, y, 0.04), 0.15, 0.04, "rubber", collision=True)
            b.cylinder(base, (x, y, 0.4825), 0.12, 0.4025, "dark", collision=True)
    _breadboard(b, base, 0, 0, 0.92, 2.25, 2.65)
    b.box(
        base,
        (0, -1.294, 0.71),
        (1.10, 0.026, 0.175),
        "cream",
        rgba=(0.34, 0.30, 0.23, 1),
        collision=True,
    )
    for x in (-1.1, 1.1):
        b.box(
            base,
            (x, 0, 0.71),
            (0.026, 1.30, 0.175),
            "cream",
            rgba=(0.34, 0.30, 0.23, 1),
            collision=True,
        )
    _laser_case(b, base, -0.59, -0.78, 0.955, (0.40, 0.43, 0.18))
    _laser_case(b, base, 0.57, -0.46, 0.955, (0.40, 0.73, 0.15))
    _laser_case(b, base, 0.64, 0.83, 0.955, (0.21, 0.26, 0.22))
    for x, y in (
        (-0.85, 0.03),
        (-0.61, 0.20),
        (-0.72, 0.64),
        (-0.35, 0.97),
        (0.18, 0.98),
        (0.06, 0.24),
    ):
        _optic(b, base, x, y, 0.955, height=0.12 + 0.05 * (y > 0.5))
    # Standoffs support the smaller rear breadboard above the main optical table.
    for x in (-0.85, -0.35):
        for y in (0.38, 0.80):
            b.cylinder(base, (x, y, 1.08), 0.015, 0.125, "bright", collision=True)
    _breadboard(b, base, -0.60, 0.59, 1.24, 0.68, 0.58)
    for x in (-0.80, -0.46):
        _optic(b, base, x, 0.62, 1.275, 0.12)
    # Open dry alignment area behind the front laser enclosures.
    b.box(base, (0.05, 0.55, 0.980), (0.23, 0.16, 0.025), "dark", collision=True)
    for y in (0.44, 0.66):
        _tag(
            b,
            b.box(base, (0.05, y, 1.016), (0.205, 0.010, 0.011), "bright"),
            "illinois_stage_guide",
        )
    stage = b.moving(
        base,
        "filter_translation",
        (0.05, 0.55, 1.04),
        (1, 0, 0),
        (-0.06, 0.06),
        mass=0.18,
        kp=350,
    )
    b.box(stage, (0, 0, 0), (0.11, 0.11, 0.013), "dark", collision=True)
    b.cylinder(stage, (0, 0, 0.033), 0.080, 0.020, "metal")
    rotation = b.moving(
        stage,
        "filter_rotation",
        (0, 0, 0.069),
        (0, 0, 1),
        (-1.1, 1.1),
        kind="hinge",
        mass=0.06,
        kp=90,
    )
    b.cylinder(rotation, (0, 0, 0), 0.09, 0.016, "dark", collision=True)
    b.cylinder(rotation, (0, 0, 0.018), 0.072, 0.002, "orange", collision=True)
    b.box(rotation, (0.020, 0, 0.021), (0.004, 0.052, 0.001), "cream")
    b.site(rotation, "optical_filter", (0, 0, 0.021))
    # Photo-specific utility shelf really hangs from four ceiling anchors.
    _tag(
        b,
        b.box(base, (0, 0.63, 2.15), (0.78, 0.44, 0.032), "shell", collision=True),
        "suspended_utility_shelf",
    )
    for x in (-0.70, 0.70):
        for y in (0.25, 1.01):
            _tag(
                b,
                b.cylinder(base, (x, y, 2.625), 0.009, 0.443, "metal"),
                "shelf_suspension",
            )
            b.box(base, (x, y, 3.084), (0.04, 0.04, 0.016), "metal")
    for x in (-0.47, 0.01, 0.48):
        b.box(base, (x, 0.66, 2.28), (0.17, 0.20, 0.098), "shell", collision=True)
        b.box(base, (x, 0.455, 2.28), (0.12, 0.004, 0.047), "screen")
    for i in range(8):
        x = -0.66 + i * 0.18
        b.rod(base, (x, 0.99, 2.12), (x + 0.045, 0.95, 1.67), 0.0035, "dark")
        b.rod(base, (x + 0.045, 0.95, 1.67), (x - 0.025, 0.87, 1.12), 0.0035, "dark")
    b.metadata["capabilities"] = [
        "dry optical-filter lateral positioning",
        "optical-filter azimuth indexing",
    ]
    b.metadata["limitations"].append(
        "Only the official Nonlinear Optics Development room is referenced. Laser brands, dimensions, hidden beam paths and actuator specifications are unverified; no active laser, nonlinear optics, imaging or biological preparation model."
    )


def _illinois_features(world, definition):
    # Small floor chiller at the side of the table, rather than a second lab bay.
    for x in (-1.70, -1.32):
        for y in (-0.85, -0.45):
            box(
                world,
                f"illinois_chiller_foot_{x}_{y}",
                (0.04, 0.04, 0.025),
                (x, y, 0.025),
                (0.12, 0.13, 0.14, 1),
            )
    box(
        world,
        "illinois_side_chiller",
        (0.25, 0.29, 0.47),
        (-1.51, -0.65, 0.52),
        (0.78, 0.80, 0.79, 1),
    )
    for i in range(12):
        box(
            world,
            f"illinois_chiller_vent_{i}",
            (0.20, 0.006, 0.006),
            (-1.51, -0.947, 0.23 + i * 0.035),
            (0.20, 0.22, 0.22, 1),
            collision=False,
        )


BUILDERS["illinois_nonlinear_optics_workstation"] = _illinois
SOURCES["illinois_nonlinear_optics_workstation"] = {
    "reference": "BIL official Nonlinear Optics Development Lab room photograph",
    "url": ILLINOIS,
    "dimensions_m": [2.25, 2.65, 3.10],
    "dimension_basis": "Entire geometry and control ranges are photographic estimates. Shelf anchors terminate at the authored 3.10 m ceiling, not a measured room height.",
}
SAMPLE_INTERFACES["illinois_nonlinear_optics_workstation"] = (
    "optical_filter",
    (0.144, 0.144, 0.004),
    "clamped",
    "inert patterned filter disc attached to the alignment stage",
)
FEATURES["illinois_bil_nonlinear_optics"] = _illinois_features


def _lurie_inspection(b, base, params):
    # Source shows a stepped rectangular inspection head, not a stereo microscope.
    b.box(base, (0, 0, 0.025), (0.31, 0.29, 0.025), "rubber", collision=True)
    b.box(base, (0, 0, 0.065), (0.31, 0.29, 0.015), "cream", collision=True)
    b.box(base, (0, 0.205, 0.34), (0.13, 0.070, 0.26), "cream", collision=True)
    b.box(base, (0, 0.02, 0.56), (0.19, 0.235, 0.12), "cream", collision=True)
    b.box(base, (0.06, 0.11, 0.775), (0.11, 0.135, 0.095), "cream", collision=True)
    b.box(base, (0.06, 0.11, 0.89), (0.055, 0.065, 0.02), "dark", collision=True)
    b.cylinder(base, (0, -0.12, 0.407), 0.035, 0.033, "dark", collision=True)
    b.cylinder(base, (0, -0.12, 0.366), 0.029, 0.008, "lens", collision=True)
    # Crossed stage bearings, distinct from the earlier Yale upright microscope.
    b.box(base, (0, -0.075, 0.099), (0.255, 0.19, 0.019), "dark", collision=True)
    for y in (-0.19, 0.035):
        _tag(
            b,
            b.box(base, (0, y, 0.129), (0.235, 0.011, 0.011), "metal"),
            "lurie_x_guide",
        )
    xstage = b.moving(
        base,
        "wafer_x",
        (0, -0.075, 0.153),
        (1, 0, 0),
        (-0.045, 0.045),
        mass=0.30,
        kp=420,
    )
    b.box(xstage, (0, 0, 0), (0.19, 0.165, 0.013), "dark", collision=True)
    for x in (-0.13, 0.13):
        _tag(
            b,
            b.box(xstage, (x, 0, 0.023), (0.011, 0.14, 0.010), "metal"),
            "lurie_y_guide",
        )
    ystage = b.moving(
        xstage, "wafer_y", (0, 0, 0.045), (0, 1, 0), (-0.040, 0.040), mass=0.16, kp=320
    )
    b.box(ystage, (0, 0, 0), (0.175, 0.13, 0.012), "dark", collision=True)
    b.cylinder(ystage, (0, -0.012, 0.017), 0.065, 0.005, "metal", collision=True)
    b.cylinder(ystage, (0, -0.012, 0.023), 0.060, 0.001, "lens", collision=True)
    b.box(ystage, (0.015, -0.012, 0.0245), (0.004, 0.040, 0.0005), "orange")
    for x in (-0.067, 0.067):
        b.box(
            ystage, (x, -0.012, 0.023), (0.008, 0.026, 0.004), "cream", collision=True
        )
    b.site(ystage, "wafer_coupon", (0, -0.012, 0.025))
    b.cylinder(
        base, (0.285, -0.06, 0.14), 0.026, 0.025, "metal", euler=(0, math.pi / 2, 0)
    )
    b.box(base, (-0.50, 0.06, 0.018), (0.13, 0.08, 0.018), "cream", collision=True)
    b.box(base, (-0.50, 0.10, 0.16), (0.025, 0.025, 0.125), "cream", collision=True)
    b.box(base, (-0.50, 0.10, 0.33), (0.19, 0.055, 0.145), "cream", collision=True)
    b.box(base, (-0.50, 0.039, 0.33), (0.155, 0.006, 0.112), "screen")
    b.box(base, (-0.50, -0.15, 0.011), (0.20, 0.075, 0.011), "cream", collision=True)
    for row in range(3):
        for col in range(8):
            b.box(
                base,
                (-0.655 + col * 0.044, -0.20 + row * 0.04, 0.024),
                (0.018, 0.013, 0.003),
                "shell",
            )
    b.metadata["capabilities"] = [
        "inert wafer lateral positioning",
        "inert wafer transverse positioning",
    ]
    b.metadata["limitations"].append(
        "Photograph-informed unbranded inspection head; estimated 120 mm inert wafer and stage stroke. No resist, lithography exposure, contamination control, nanofabrication, optical metrology or chemistry model."
    )


def _lurie_features(world, definition):
    cream = (0.76, 0.69, 0.40, 1)
    dark = (0.12, 0.13, 0.12, 1)
    metal = (0.55, 0.56, 0.51, 1)
    # Windowed storage bank occupies only the left wall; center remains an aisle.
    for j in range(9):
        y = -3.60 + j * 0.82
        box(
            world,
            f"lurie_cabinet_plinth_{j}",
            (0.35, 0.405, 0.055),
            (-1.83, y, 0.055),
            dark,
        )
        box(
            world,
            f"lurie_cabinet_body_{j}",
            (0.35, 0.405, 1.045),
            (-1.83, y, 1.155),
            cream,
        )
        for row in range(5):
            z = 0.34 + row * 0.405
            box(
                world,
                f"lurie_door_{j}_{row}",
                (0.013, 0.385, 0.187),
                (-1.467, y, z),
                cream,
            )
            box(
                world,
                f"lurie_window_{j}_{row}",
                (0.003, 0.297, 0.129),
                (-1.451, y, z),
                (0.12, 0.15, 0.14, 1),
                collision=False,
            )
            box(
                world,
                f"lurie_latch_{j}_{row}",
                (0.027, 0.025, 0.021),
                (-1.425, y + 0.32, z),
                metal,
                collision=False,
            )
            for dz in (-0.12, 0.12):
                box(
                    world,
                    f"lurie_hinge_{j}_{row}_{dz}",
                    (0.019, 0.020, 0.027),
                    (-1.43, y - 0.35, z + dz),
                    dark,
                    collision=False,
                )
        # Rising transparent service sight tube, as seen above the cabinet rows.
        box(
            world,
            f"lurie_service_tube_{j}",
            (0.018, 0.022, 0.27),
            (-1.67, y, 2.47),
            (0.65, 0.69, 0.61, 0.45),
            collision=False,
        )
        box(
            world,
            f"lurie_service_tube_anchor_{j}",
            (0.20, 0.022, 0.018),
            (-1.88, y, 2.72),
            metal,
        )
    # Additional source-visible inspection stations are static context only.
    for j, y in enumerate((0.85, 3.0)):
        box(
            world,
            f"lurie_context_base_{j}",
            (0.23, 0.29, 0.035),
            (1.44, y, 0.865),
            (0.16, 0.17, 0.16, 1),
        )
        box(
            world,
            f"lurie_context_back_{j}",
            (0.055, 0.15, 0.27),
            (1.65, y, 1.17),
            cream,
        )
        box(
            world, f"lurie_context_head_{j}", (0.21, 0.18, 0.12), (1.47, y, 1.50), cream
        )
        box(
            world,
            f"lurie_context_tower_{j}",
            (0.10, 0.09, 0.12),
            (1.59, y, 1.74),
            cream,
        )
        box(
            world,
            f"lurie_context_stage_{j}",
            (0.18, 0.20, 0.025),
            (1.37, y, 0.965),
            dark,
        )
        box(
            world,
            f"lurie_context_stage_support_{j}",
            (0.10, 0.13, 0.025),
            (1.37, y, 0.915),
            metal,
        )
        box(
            world,
            f"lurie_context_console_{j}",
            (0.035, 0.18, 0.14),
            (1.80, y + 0.51, 1.12),
            cream,
        )
        box(
            world,
            f"lurie_context_console_stand_{j}",
            (0.025, 0.035, 0.055),
            (1.80, y + 0.51, 0.885),
            cream,
        )
    # Three five-legged stools remain off the center passage.
    for j, y in enumerate((-1.75, 0.85, 3.0)):
        for k in range(5):
            a = math.tau * k / 5
            start = (0.58, y, 0.12)
            end = (0.58 + 0.24 * math.cos(a), y + 0.24 * math.sin(a), 0.045)
            ET.SubElement(
                world,
                "geom",
                name=f"lurie_stool_leg_{j}_{k}",
                type="capsule",
                size=".016",
                fromto=" ".join(str(v) for v in start + end),
                rgba=".5 .51 .49 1",
            )
            box(
                world,
                f"lurie_stool_caster_{j}_{k}",
                (0.028, 0.023, 0.022),
                (end[0], end[1], 0.022),
                dark,
            )
        box(
            world, f"lurie_stool_stem_{j}", (0.023, 0.023, 0.22), (0.58, y, 0.32), metal
        )
        ET.SubElement(
            world,
            "geom",
            name=f"lurie_stool_seat_{j}",
            type="cylinder",
            size=".18 .027",
            pos=f".58 {y} .567",
            rgba=".13 .14 .13 1",
        )


BUILDERS["michigan_lurie_wafer_inspector"] = _lurie_inspection
SOURCES["michigan_lurie_wafer_inspector"] = {
    "reference": "University of Michigan official Lurie Nanofabrication Facility amber cleanroom photograph",
    "url": MICHIGAN,
    "dimensions_m": [1.04, 0.59, 0.91],
    "dimension_basis": "Photo-informed unbranded instrument exterior; all dimensions, 120 mm wafer size and control strokes are authored estimates.",
}
SAMPLE_INTERFACES["michigan_lurie_wafer_inspector"] = (
    "wafer_coupon",
    (0.12, 0.12, 0.002),
    "clamped",
    "inert patterned wafer clamped to the crossed inspection stage",
)
FEATURES["michigan_lurie_lithography_bay"] = _lurie_features
