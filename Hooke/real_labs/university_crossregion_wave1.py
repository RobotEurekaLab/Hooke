"""Original exterior geometry for four photo-documented optical laboratories.

Only rigid inspection mechanics are simulated. Optical, vacuum and nanoscale
measurement responses require separate scientific models and calibration.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

LMU = "https://www.physik.lmu.de/hybridnano/en/research/facilities/"
COPENHAGEN = "https://nbi.ku.dk/english/research_infrastructure/nbi-cleanroom/"
HEIDELBERG = "https://ultracold.physi.uni-heidelberg.de/02research/"
BERLIN = "https://www.physik.fu-berlin.de/en/langenacht/inhalt/labor/franke.html"
BUILDERS = {}
SOURCES = {}
SAMPLE_INTERFACES = {}
FEATURES = {}


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _optical_table(b, base, width, depth, height=0.82):
    for x in (-width * 0.35, width * 0.35):
        for y in (-depth * 0.34, depth * 0.34):
            b.cylinder(base, (x, y, 0.025), 0.13, 0.025, "rubber", collision=True)
            _tag(
                b,
                b.cylinder(
                    base,
                    (x, y, (height - 0.07 + 0.05) / 2),
                    0.11,
                    (height - 0.07 - 0.05) / 2,
                    "dark",
                    collision=True,
                ),
                "table_isolator",
            )
    _tag(
        b,
        b.box(
            base,
            (0, 0, height - 0.035),
            (width / 2, depth / 2, 0.035),
            "bright",
            collision=True,
        ),
        "optical_tabletop",
    )
    for i in range(int(width / 0.12)):
        for j in range(int(depth / 0.12)):
            b.cylinder(
                base,
                (
                    -width / 2 + 0.08 + i * 0.12,
                    -depth / 2 + 0.08 + j * 0.12,
                    height + 0.0006,
                ),
                0.002,
                0.0006,
                "dark",
            )


def _optic(b, base, x, y, z=0.82, angle=0):
    b.box(base, (x, y, z + 0.010), (0.042, 0.045, 0.010), "dark", collision=True)
    b.cylinder(base, (x, y, z + 0.07), 0.009, 0.05, "bright", collision=True)
    mount = ET.SubElement(base, "body", pos=f"{x} {y} {z+.13}", euler=f"0 0 {angle}")
    b.box(mount, (0, 0, 0), (0.038, 0.012, 0.038), "dark")
    b.cylinder(mount, (0, -0.013, 0), 0.026, 0.004, "lens", euler=(math.pi / 2, 0, 0))
    for a in (-1, 1):
        b.cylinder(
            mount,
            (a * 0.027, -0.025, 0.027),
            0.004,
            0.008,
            "bright",
            euler=(math.pi / 2, 0, 0),
        )


def _snom(b, base, params):
    _optical_table(b, base, 2.9, 1.85)
    # Parallel source enclosures and a cable-dense rear rack establish this room.
    for x, w, h in ((-0.75, 0.28, 0.19), (-0.14, 0.19, 0.15)):
        b.box(base, (x, 0.08, 0.82 + h), (w, 0.68, h), "shell", collision=True)
        for y in (-0.35, 0.45):
            for dx in (-0.075, 0.075):
                b.rod(
                    base,
                    (x + dx, y, 0.82 + 2 * h),
                    (x + dx, y, 0.85 + 2 * h),
                    0.005,
                    "metal",
                )
            b.rod(
                base,
                (x - 0.075, y, 0.85 + 2 * h),
                (x + 0.075, y, 0.85 + 2 * h),
                0.005,
                "metal",
            )
    b.box(base, (0.29, -0.25, 0.935), (0.22, 0.20, 0.115), "blue", collision=True)
    b.box(base, (0.13, -0.63, 0.91), (0.28, 0.10, 0.09), "shell", collision=True)
    for x in (-0.05, 0.12, 0.29):
        b.box(base, (x, -0.738, 0.918), (0.060, 0.008, 0.041), "screen")
        b.cylinder(
            base,
            (x + 0.02, -0.75, 0.884),
            0.011,
            0.008,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    for x, y in (
        (-1.18, -0.55),
        (-1.18, -0.12),
        (-1.12, 0.53),
        (-0.38, -0.72),
        (0.64, -0.70),
        (1.22, -0.55),
        (1.20, 0.55),
    ):
        _optic(b, base, x, y, angle=0.3)
    for x in (-0.25, 0.35):
        for y in (0.95, 1.25):
            b.box(base, (x, y, 1.08), (0.023, 0.023, 1.08), "dark", collision=True)
    for z in (0.10, 0.82, 1.25, 1.58, 1.93):
        b.box(base, (0.05, 1.1, z), (0.33, 0.18, 0.020), "metal", collision=True)
    for z in (0.925, 1.355, 1.685):
        b.box(base, (0.05, 1.1, z), (0.28, 0.16, 0.085), "dark", collision=True)
        b.box(base, (0.05, 0.933, z), (0.17, 0.004, 0.050), "screen")
    for i in range(10):
        x = -0.20 + i * 0.055
        b.rod(base, (x, 0.92, 1.95), (x + 0.09, 0.91, 1.55), 0.005, "dark")
        b.rod(
            base,
            (x + 0.09, 0.91, 1.55),
            (x - 0.07, 0.93, 0.89),
            0.005,
            "copper" if i % 3 == 0 else "dark",
        )
    # Open right-side near-field inspection stage, internally authored.
    b.box(base, (0.89, 0.19, 0.85), (0.34, 0.37, 0.03), "dark", collision=True)
    for x in (0.61, 1.17):
        b.box(base, (x, 0.39, 1.02), (0.023, 0.045, 0.14), "metal", collision=True)
    b.box(base, (0.89, 0.39, 1.18), (0.31, 0.055, 0.025), "dark", collision=True)
    for y in (-0.05, 0.12):
        _tag(
            b,
            b.box(base, (0.89, y, 0.8995), (0.23, 0.014, 0.0195), "bright"),
            "snom_x_guide",
        )
    stage = b.moving(
        base,
        "specimen_translation",
        (0.89, 0.035, 0.935),
        (1, 0, 0),
        (-0.055, 0.055),
        mass=0.13,
        kp=350,
    )
    b.box(stage, (0, 0, 0), (0.12, 0.12, 0.016), "dark", collision=True)
    b.cylinder(stage, (0, 0, 0.023), 0.04, 0.007, "metal", collision=True)
    _tag(
        b,
        b.box(stage, (0, 0, 0.032), (0.009, 0.008, 0.002), "orange", collision=True),
        "snom_inert_coupon",
    )
    b.site(stage, "snom_coupon", (0, -0.010, 0.034))
    probe = b.moving(
        base,
        "inspection_head_height",
        (0.89, 0.04, 1.14),
        (0, 0, 1),
        (-0.02, 0.02),
        mass=0.15,
        kp=360,
    )
    b.box(probe, (0, 0.135, 0), (0.080, 0.20, 0.045), "dark")
    b.cylinder(probe, (0, 0, -0.055), 0.019, 0.020, "bright", collision=True)
    _tag(
        b,
        b.cylinder(probe, (0, 0, -0.080), 0.003, 0.005, "bright", collision=True),
        "snom_probe_tip",
    )
    # Rear sleeves visually connect the head crossmember to its support beam.
    for x in (0.84, 0.94):
        b.cylinder(base, (x, 0.34, 1.145), 0.008, 0.07, "bright")
    b.metadata["capabilities"] = [
        "inert specimen lateral alignment",
        "bounded dry inspection-head height adjustment",
    ]
    b.metadata["limitations"].append(
        "Original inspection-stage internals inferred within the photographed near-field station exterior. All dimensions and travel estimates; probe remains above target. No SNOM image, light propagation or nanometre performance."
    )


def _snom_features(world, definition):
    for i in range(15):
        z = 1.12 + i * 0.085
        box(
            world,
            f"lmu_back_blind_{i}",
            (2.45, 0.012, 0.039),
            (0, 2.57, z),
            (0.12, 0.13, 0.14, 1),
        )
        box(
            world,
            f"lmu_left_blind_{i}",
            (0.012, 1.60, 0.039),
            (-2.67, 0.73, z),
            (0.12, 0.13, 0.14, 1),
        )
    box(
        world,
        "lmu_back_blind_headrail",
        (2.47, 0.035, 0.045),
        (0, 2.55, 2.40),
        (0.79, 0.79, 0.77, 1),
    )
    box(
        world,
        "lmu_left_blind_headrail",
        (0.035, 1.62, 0.045),
        (-2.65, 0.73, 2.40),
        (0.79, 0.79, 0.77, 1),
    )


BUILDERS["lmu_snom_inspection_table"] = _snom
SOURCES["lmu_snom_inspection_table"] = {
    "reference": "LMU Hybrid Nanosystems installed SNOM laboratory photograph",
    "url": LMU,
    "dimensions_m": [2.9, 2.205, 2.16],
    "dimension_basis": "All dimensions estimated from one room-corner photograph; internal alignment mechanism original and unverified.",
}
SAMPLE_INTERFACES["lmu_snom_inspection_table"] = (
    "snom_coupon",
    (0.018, 0.016, 0.004),
    "clamped",
    "inert calibration coupon attached to a dry inspection-stage slider",
)
FEATURES["lmu_hybrid_snom"] = _snom_features


def _sensofar(b, base, params):
    b.box(base, (0, 0, 0.04), (0.53, 0.44, 0.04), "dark", collision=True)
    # The right rear column visibly carries the large overhanging blue head.
    b.box(base, (0.34, 0.43, 0.53), (0.135, 0.12, 0.45), "dark", collision=True)
    b.box(base, (0.07, 0.34, 0.94), (0.405, 0.20, 0.065), "dark", collision=True)
    head = ET.SubElement(base, "body", pos="-.04 .10 0")
    b.housing(
        head,
        [(0.67, 0.25, 0.22, 0.02), (1.09, 0.31, 0.25, 0.02), (1.18, 0.29, 0.23, 0.025)],
        0.032,
        "blue",
    )
    b.box(head, (0, 0.035, 0.91), (0.275, 0.215, 0.24), "blue", collision=True)
    b.box(head, (-0.30, 0.015, 0.935), (0.030, 0.23, 0.245), "shell")
    # Shallow geometric embossing reproduces the recognizable patterned fascia.
    for i in range(5):
        for j in range(5):
            half = 0.013 + 0.002 * ((i + j) % 3)
            b.box(
                head,
                (-0.21 + i * 0.10, -0.208, 0.75 + j * 0.082),
                (half, 0.004, half),
                "blue",
            )
    b.cylinder(base, (-0.02, 0.03, 0.646), 0.10, 0.018, "dark", euler=(0, 0.25, 0))
    for x, y, a in ((-0.07, -0.025, -0.2), (0.055, 0.045, 0.35), (-0.045, 0.115, 0)):
        obj = ET.SubElement(base, "body", pos=f"{x} {y} .595", euler=f"0 {a} 0")
        b.cylinder(obj, (0, 0, 0), 0.025, 0.050, "bright", collision=True)
        b.cylinder(obj, (0, 0, -0.063), 0.016, 0.013, "metal", collision=True)
        _tag(
            b,
            b.cylinder(obj, (0, 0, -0.078), 0.010, 0.003, "lens"),
            "profiler_objective_tip",
        )
        for z in (-0.01, 0.015, 0.035):
            b.ring(obj, (0, 0, z), 0.0255, 0.002, "dark")
    for y in (-0.24, 0.17):
        b.box(base, (0, y, 0.1005), (0.36, 0.018, 0.0205), "bright")
    xstage = b.moving(
        base,
        "sample_x",
        (0, -0.06, 0.14),
        (1, 0, 0),
        (-0.060, 0.060),
        mass=0.35,
        kp=480,
    )
    b.box(xstage, (0, 0, 0), (0.34, 0.29, 0.019), "dark", collision=True)
    for x in (-0.25, 0.25):
        b.box(xstage, (x, 0, 0.033), (0.015, 0.26, 0.014), "bright")
    ystage = b.moving(
        xstage, "sample_y", (0, 0, 0.067), (0, 1, 0), (-0.05, 0.05), mass=0.20, kp=430
    )
    b.box(ystage, (0, 0, 0), (0.29, 0.22, 0.020), "dark", collision=True)
    for x in (-0.20, 0.20):
        for y in (-0.14, 0.14):
            b.cylinder(ystage, (x, y, 0.022), 0.004, 0.002, "metal")
    _tag(
        b,
        b.box(ystage, (0, 0, 0.025), (0.022, 0.019, 0.005), "orange", collision=True),
        "profiler_inert_coupon",
    )
    b.site(ystage, "profiler_coupon", (0, -0.023, 0.030))
    for a in range(8):
        y = 0.33 + 0.025 * math.cos(a * 0.65)
        z = 0.49 + 0.10 * math.sin(a * 0.65)
        b.rod(
            base,
            (0.46, y, z),
            (
                0.46,
                0.33 + 0.025 * math.cos((a + 1) * 0.65),
                0.49 + 0.10 * math.sin((a + 1) * 0.65),
            ),
            0.006,
            "dark",
        )
    b.metadata["capabilities"] = [
        "rigid profiler specimen X alignment",
        "rigid profiler specimen Y alignment",
    ]
    b.metadata["limitations"].append(
        "Source shows a Sensofar-labelled profiler with operator occlusion. Head envelope, rear structural support, XY mechanism and travel are estimates, not OEM CAD. No optical profiling, surface reconstruction, focus response or measurement precision."
    )


def _copenhagen_features(world, definition):
    # Nearby bench details are approximate context, not a full cleanroom layout.
    box(
        world,
        "copenhagen_control_foot",
        (0.16, 0.12, 0.018),
        (1.30, 0.28, 0.918),
        (0.16, 0.18, 0.19, 1),
    )
    box(
        world,
        "copenhagen_control_stem",
        (0.025, 0.025, 0.12),
        (1.30, 0.32, 1.056),
        (0.26, 0.28, 0.28, 1),
    )
    box(
        world,
        "copenhagen_control_monitor",
        (0.24, 0.025, 0.15),
        (1.30, 0.32, 1.326),
        (0.10, 0.13, 0.15, 1),
    )
    box(
        world,
        "copenhagen_control_screen",
        (0.22, 0.003, 0.13),
        (1.30, 0.291, 1.326),
        (0.03, 0.10, 0.14, 1),
    )


BUILDERS["copenhagen_sensofar_xy_profiler"] = _sensofar
SOURCES["copenhagen_sensofar_xy_profiler"] = {
    "reference": "NBI Cleanroom official installed Sensofar optical-profiler photograph",
    "url": COPENHAGEN,
    "dimensions_m": [1.06, 0.99, 1.18],
    "dimension_basis": "All dimensions estimated; exact model and internal mechanisms unverified due partial occlusion.",
}
SAMPLE_INTERFACES["copenhagen_sensofar_xy_profiler"] = (
    "profiler_coupon",
    (0.044, 0.038, 0.010),
    "clamped",
    "inert flat calibration coupon fixed to an authored XY positioning deck",
)
FEATURES["copenhagen_nbi_surface_metrology"] = _copenhagen_features


def _flanged_port(b, parent, euler, extension=0.34, cage=False):
    port = ET.SubElement(parent, "body", euler=" ".join(map(str, euler)))
    b.cylinder(
        port, (0, 0, extension / 2), 0.070, extension / 2, "metal", collision=True
    )
    b.cylinder(port, (0, 0, extension), 0.105, 0.017, "bright", collision=True)
    b.cylinder(port, (0, 0, extension + 0.019), 0.071, 0.003, "lens")
    for i in range(10):
        a = i * math.tau / 10
        b.cylinder(
            port,
            (0.090 * math.cos(a), 0.090 * math.sin(a), extension + 0.022),
            0.007,
            0.006,
            "metal",
        )
    if cage:
        for x in (-0.055, 0.055):
            for y in (-0.055, 0.055):
                b.cylinder(port, (x, y, extension + 0.17), 0.004, 0.14, "bright")
        for z in (extension + 0.06, extension + 0.28):
            b.box(port, (0, 0, z), (0.068, 0.068, 0.010), "dark")
            b.cylinder(port, (0, 0, z + 0.012), 0.039, 0.003, "lens")
    return port


def _hqa(b, base, params):
    _optical_table(b, base, 2.25, 1.60, 0.76)
    for x in (-1.04, 1.04):
        for y in (-0.66, 0.66):
            b.box(base, (x, y, 1.50), (0.025, 0.025, 0.74), "metal", collision=True)
    for y in (-0.66, 0.66):
        b.box(base, (0, y, 2.245), (1.065, 0.025, 0.025), "metal", collision=True)
    for x in (-1.04, 1.04):
        b.box(base, (x, 0, 2.245), (0.025, 0.66, 0.025), "metal", collision=True)
    for i in range(23):
        b.box(base, (-1.01 + i * 0.091, 0, 2.272), (0.013, 0.66, 0.005), "metal")
    for j in range(14):
        b.box(base, (0, -0.63 + j * 0.097, 2.284), (1.04, 0.013, 0.005), "metal")
    b.cylinder(base, (-0.40, 0.14, 0.795), 0.20, 0.035, "dark", collision=True)
    b.cylinder(base, (-0.40, 0.14, 1.145), 0.16, 0.315, "metal", collision=True)
    chamber = ET.SubElement(base, "body", pos="-.40 .14 1.15")
    _flanged_port(b, chamber, (math.pi / 2, 0, 0), 0.30, True)
    _flanged_port(b, chamber, (math.pi / 2 - 0.62, 0, 0.28), 0.28, True)
    _flanged_port(b, chamber, (math.pi / 2 + 0.62, 0, -0.22), 0.28, True)
    _flanged_port(b, chamber, (0, math.pi / 2, 0), 0.34, False)
    _flanged_port(b, chamber, (0, -math.pi / 2, 0), 0.32, True)
    for x in (-0.67, -0.04):
        b.box(base, (x, 0.33, 0.97), (0.13, 0.16, 0.21), "dark", collision=True)
        b.cylinder(base, (x, 0.33, 1.33), 0.080, 0.15, "metal", collision=True)
        b.box(base, (x, 0.33, 1.61), (0.14, 0.14, 0.13), "guard_red", collision=True)
    # Source-visible black horizontal optics levels have continuous metal posts.
    for x in (0.30, 0.89):
        for y in (-0.16, 0.47):
            b.cylinder(base, (x, y, 1.365), 0.018, 0.605, "bright", collision=True)
    for z in (1.04, 1.45, 1.965):
        b.box(base, (0.595, 0.155, z), (0.335, 0.355, 0.018), "dark", collision=True)
    for z in (1.12, 1.53):
        for x in (0.43, 0.76):
            b.cylinder(base, (x, 0.18, z), 0.055, 0.062, "dark", collision=True)
            b.cylinder(base, (x, 0.18, z + 0.067), 0.042, 0.005, "lens")
    # Foreground gimbal is an original dry alignment proxy, not a light solver.
    b.box(base, (0.37, -0.57, 0.78), (0.14, 0.13, 0.020), "dark", collision=True)
    b.cylinder(base, (0.37, -0.57, 0.94), 0.020, 0.14, "bright", collision=True)
    for x in (0.275, 0.465):
        b.box(base, (x, -0.57, 1.13), (0.016, 0.026, 0.067), "dark", collision=True)
    tilt = b.moving(
        base,
        "alignment_tip",
        (0.37, -0.57, 1.13),
        (1, 0, 0),
        (-0.16, 0.16),
        kind="hinge",
        mass=0.06,
        kp=100,
    )
    b.ring(tilt, (0, 0, 0), 0.074, 0.008, "metal", plane="xz")
    rotation = b.moving(
        tilt,
        "alignment_tilt",
        (0, 0, 0),
        (0, 0, 1),
        (-0.20, 0.20),
        kind="hinge",
        mass=0.035,
        kp=85,
    )
    _tag(
        b,
        b.cylinder(
            rotation,
            (0, 0, 0),
            0.046,
            0.008,
            "orange",
            euler=(math.pi / 2, 0, 0),
            collision=True,
        ),
        "hqa_inert_alignment_disc",
    )
    for x in (-0.057, 0.057):
        b.cylinder(
            rotation, (x, 0, 0), 0.005, 0.009, "metal", euler=(0, math.pi / 2, 0)
        )
    b.site(rotation, "hqa_target", (0, -0.011, 0))
    for i in range(5):
        x = -0.90 + i * 0.09
        b.rod(
            base, (x, 0.59, 2.245), (x, 0.57, 1.8), 0.007, "blue" if i % 2 else "dark"
        )
        b.rod(
            base,
            (x, 0.57, 1.8),
            (x + 0.13, 0.40, 1.45),
            0.007,
            "blue" if i % 2 else "dark",
        )
    b.metadata["capabilities"] = [
        "inert optical-target tip adjustment",
        "inert optical-target tilt adjustment",
    ]
    b.metadata["limitations"].append(
        "Estimated local optical-vacuum apparatus exterior with authored foreground gimbal. Port inventory and supports approximate a single photograph; no vacuum, optical propagation, atomic trapping, quantum evolution or calibrated alignment response."
    )


BUILDERS["heidelberg_hqa_alignment_assembly"] = _hqa
SOURCES["heidelberg_hqa_alignment_assembly"] = {
    "reference": "Jochim Labs official HQA installed apparatus photograph",
    "url": HEIDELBERG,
    "dimensions_m": [2.25, 1.60, 2.289],
    "dimension_basis": "All dimensions estimated from local apparatus photograph; full room and internal chamber structure unknown.",
}
SAMPLE_INTERFACES["heidelberg_hqa_alignment_assembly"] = (
    "hqa_target",
    (0.092, 0.016, 0.092),
    "clamped",
    "inert alignment disc in an original foreground two-axis gimbal",
)


def _franke(b, base, params):
    for x in (-0.85, 0.85):
        for y in (-0.66, 0.66):
            b.box(base, (x, y, 0.030), (0.09, 0.09, 0.030), "rubber", collision=True)
            b.box(base, (x, y, 0.39), (0.04, 0.04, 0.33), "metal", collision=True)
        b.box(base, (x, 0, 0.28), (0.035, 0.66, 0.025), "metal", collision=True)
    b.box(base, (0, 0, 0.755), (0.94, 0.74, 0.035), "metal", collision=True)
    for y in (-0.66, 0.66):
        b.box(base, (0, y, 0.28), (0.85, 0.035, 0.025), "metal", collision=True)
    b.cylinder(base, (0.12, 0.10, 1.07), 0.22, 0.28, "bright", collision=True)
    for z in (0.815, 1.34):
        b.cylinder(base, (0.12, 0.10, z), 0.26, 0.025, "metal", collision=True)
        for i in range(14):
            a = i * math.tau / 14
            b.cylinder(
                base,
                (0.12 + 0.236 * math.cos(a), 0.10 + 0.236 * math.sin(a), z + 0.031),
                0.009,
                0.006,
                "bright",
            )
    chamber = ET.SubElement(base, "body", pos=".12 .10 1.06")
    _flanged_port(b, chamber, (math.pi / 2, 0, 0.26), 0.36, False)
    _flanged_port(b, chamber, (math.pi / 2 - 0.55, 0, -0.7), 0.33, False)
    _flanged_port(b, chamber, (0, -math.pi / 2, 0), 0.34, False)
    # Tall stainless column and purple external positioning drive match the photo.
    b.cylinder(base, (0.63, 0.40, 1.565), 0.215, 0.775, "bright", collision=True)
    for z in (0.83, 1.24, 1.86, 2.29):
        b.ring(base, (0.63, 0.40, z), 0.220, 0.010, "metal")
    b.cylinder(base, (0.28, 0.08, 1.445), 0.051, 0.095, "bright", collision=True)
    b.cylinder(
        base,
        (0.28, 0.08, 1.855),
        0.057,
        0.315,
        "sample",
        collision=True,
        rgba=(0.22, 0.08, 0.26, 1),
    )
    for z in (1.56, 2.03, 2.11):
        b.cylinder(
            base, (0.28, 0.08, z), 0.068, 0.020, "sample", rgba=(0.22, 0.08, 0.26, 1)
        )
    b.box(base, (0.455, 0.26, 1.72), (0.175, 0.026, 0.035), "metal", collision=True)
    b.box(base, (0.28, 0.18, 1.72), (0.065, 0.10, 0.035), "cream")
    # Exposed access holder is an authored dry proxy in the source-occluded area.
    for x in (-0.53, -0.27):
        for y in (-0.52, -0.05):
            b.box(base, (x, y, 0.84), (0.025, 0.035, 0.05), "metal", collision=True)
        _tag(
            b,
            b.box(base, (x, -0.28, 0.906), (0.015, 0.29, 0.016), "bright"),
            "franke_access_rail",
        )
    slider = b.moving(
        base,
        "holder_extension",
        (-0.40, -0.18, 0.94),
        (0, -1, 0),
        (0, 0.15),
        mass=0.20,
        kp=390,
    )
    _tag(
        b,
        b.box(slider, (0, 0, 0), (0.155, 0.16, 0.018), "dark", collision=True),
        "franke_slider_plate",
    )
    b.cylinder(slider, (0, 0, 0.040), 0.07, 0.022, "metal", collision=True)
    holder = b.moving(
        slider,
        "holder_azimuth",
        (0, 0, 0.077),
        (0, 0, 1),
        (-1.0, 1.0),
        kind="hinge",
        mass=0.06,
        kp=100,
    )
    b.cylinder(holder, (0, 0, 0), 0.059, 0.015, "bright", collision=True)
    _tag(
        b,
        b.box(holder, (0, 0, 0.021), (0.016, 0.012, 0.006), "orange", collision=True),
        "franke_inert_surface_coupon",
    )
    b.site(holder, "surface_coupon", (0, -0.017, 0.027))
    # A rear monitor is mounted to the same grounded structural deck.
    b.box(base, (-0.53, 0.52, 0.805), (0.20, 0.14, 0.015), "dark", collision=True)
    b.box(base, (-0.53, 0.59, 1.12), (0.025, 0.025, 0.30), "metal", collision=True)
    b.box(base, (-0.53, 0.59, 1.57), (0.29, 0.025, 0.15), "dark", collision=True)
    b.box(base, (-0.53, 0.561, 1.57), (0.26, 0.003, 0.13), "screen")
    for i in range(6):
        x = -0.78 + i * 0.075
        b.rod(base, (x, 0.59, 1.45), (x - 0.03, 0.53, 0.95), 0.006, "dark")
        b.rod(
            base,
            (x - 0.03, 0.53, 0.95),
            (-0.24, 0.185, 1.04 + i * 0.006),
            0.006,
            "dark",
        )
    b.metadata["capabilities"] = [
        "external inert-holder extension",
        "external specimen azimuth indexing",
    ]
    b.metadata["limitations"].append(
        "Local Franke microscope-vacuum exterior only, with source operator omitted. Occluded access mechanics are original dry proxies and do not reconstruct the hidden STM internals. All sizes estimated; no tunneling current, atomic imaging, vacuum or surface-response model."
    )


BUILDERS["fuberlin_franke_external_holder"] = _franke
SOURCES["fuberlin_franke_external_holder"] = {
    "reference": "FU Berlin Franke group official installed surface-physics workstation photograph, credit Bernd Wannenmacher",
    "url": BERLIN,
    "dimensions_m": [1.88, 1.48, 2.34],
    "dimension_basis": "All exterior dimensions and original access-holder strokes estimated. Much of the original apparatus is occluded.",
}
SAMPLE_INTERFACES["fuberlin_franke_external_holder"] = (
    "surface_coupon",
    (0.032, 0.024, 0.012),
    "clamped",
    "inert coupon fixed to a reversible external specimen-holder proxy",
)
