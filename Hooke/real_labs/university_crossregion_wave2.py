"""Photograph-informed spectroscopy, thermobalance and de-energized hall scenes.

Original inspection mechanisms provide rigid-motion tasks. Scientific response,
process operation and electrical test procedures are outside these prototypes.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

SPECTROSCOPY = "https://physics.kfupm.edu.sa/facilities/superconductivity"
THERMOBALANCE = "https://www.zc.iir.isct.ac.jp/en/research/facilities/"
HIGH_VOLTAGE = "https://ri.kfupm.edu.sa/arc-mst/facilities/facility-details/high-voltage-laboratory-%28hvl%29"
BUILDERS = {}
SOURCES = {}
SAMPLE_INTERFACES = {}
FEATURES = {}


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _fluoromax(b, base, params):
    b.box(base, (0, 0, 0.035), (0.51, 0.30, 0.035), "metal", collision=True)
    for x in (-0.32, 0.32):
        shell = ET.SubElement(base, "body", pos=f"{x} 0 0")
        b.housing(
            shell,
            [
                (0.07, 0.19, 0.29, 0),
                (0.245, 0.19, 0.29, 0),
                (0.325, 0.175, 0.26, 0.018),
                (0.375, 0.15, 0.23, 0.035),
            ],
            0.035,
            "cream",
        )
        b.box(shell, (0, 0, 0.19), (0.165, 0.245, 0.12), "cream", collision=True)
    b.box(base, (0, 0.25, 0.18), (0.14, 0.035, 0.11), "cream", collision=True)
    b.box(base, (0, -0.01, 0.084), (0.13, 0.24, 0.014), "dark", collision=True)
    for x in (-0.13, 0.13):
        b.box(base, (x, 0.21, 0.275), (0.015, 0.040, 0.19), "blue", collision=True)
    # Pre-open lid and sample lift are original inspection mechanics.
    hinge = b.moving(
        base,
        "sample_cover_angle",
        (0, 0.22, 0.44),
        (1, 0, 0),
        (-0.50, 0),
        kind="hinge",
        mass=0.15,
        kp=95,
    )
    lid = ET.SubElement(hinge, "body", euler="-.45 0 0", gravcomp="1")
    b.box(lid, (0, -0.19, 0), (0.115, 0.23, 0.014), "blue", collision=True)
    b.box(lid, (0, -0.405, -0.095), (0.115, 0.015, 0.11), "blue", collision=True)
    b.rod(lid, (-0.05, -0.30, 0.017), (-0.05, -0.30, 0.045), 0.006, "blue")
    b.rod(lid, (0.05, -0.30, 0.017), (0.05, -0.30, 0.045), 0.006, "blue")
    b.rod(lid, (-0.05, -0.30, 0.045), (0.05, -0.30, 0.045), 0.006, "blue")
    for x in (-0.13, 0.13):
        b.cylinder(
            base, (x, 0.22, 0.44), 0.026, 0.020, "metal", euler=(0, math.pi / 2, 0)
        )
    _tag(
        b,
        b.cylinder(base, (0, -0.04, 0.122), 0.028, 0.024, "metal", collision=True),
        "spectrometer_lift_sleeve",
    )
    stage = b.moving(
        base, "sample_lift", (0, -0.04, 0.165), (0, 0, 1), (0, 0.045), mass=0.06, kp=270
    )
    _tag(
        b,
        b.cylinder(stage, (0, 0, 0), 0.045, 0.019, "dark", collision=True),
        "spectrometer_lift_platform",
    )
    _tag(
        b,
        b.cylinder(stage, (0, 0, -0.040), 0.018, 0.040, "metal"),
        "spectrometer_lift_shaft",
    )
    _tag(
        b,
        b.box(stage, (0, 0, 0.039), (0.006, 0.006, 0.020), "glass", collision=True),
        "spectrometer_inert_cuvette",
    )
    b.box(stage, (0, 0, 0.061), (0.007, 0.007, 0.002), "orange", collision=True)
    b.site(stage, "spectrometer_cuvette", (0, -0.010, 0.063))
    b.metadata["capabilities"] = [
        "pre-open sample-compartment inspection",
        "inert cuvette vertical positioning",
    ]
    b.metadata["limitations"].append(
        "FluoroMax-4 exterior is photo-informed; internal holder, motorization, pre-open cover and strokes are authored mechanical proxies. Inert empty cuvette only; no excitation, fluorescence, spectra, superconductivity or material response."
    )


def _spectroscopy_features(world, definition):
    box(
        world,
        "kfupm_optics_blue_backdrop",
        (1.07, 0.012, 0.75),
        (-0.35, 1.78, 1.57),
        (0.05, 0.10, 0.20, 1),
    )
    for i in range(24):
        box(
            world,
            f"kfupm_optics_blind_{i}",
            (0.45, 0.012, 0.018),
            (1.18, 1.77, 0.99 + i * 0.052),
            (0.76, 0.77, 0.71, 1),
        )
    box(
        world,
        "kfupm_optics_blind_header",
        (0.47, 0.025, 0.030),
        (1.18, 1.75, 2.27),
        (0.64, 0.65, 0.61, 1),
    )


BUILDERS["kfupm_fluoromax_dry_inspection"] = _fluoromax
SOURCES["kfupm_fluoromax_dry_inspection"] = {
    "reference": "KFUPM superconductivity laboratory installed HORIBA FluoroMax-4 photograph",
    "url": SPECTROSCOPY,
    "dimensions_m": [1.02, 0.63, 0.85],
    "dimension_basis": "All dimensions estimated. Exterior identity visible; hidden sample mechanics and pre-open cover are original proxies.",
}
SAMPLE_INTERFACES["kfupm_fluoromax_dry_inspection"] = (
    "spectrometer_cuvette",
    (0.012, 0.012, 0.04),
    "clamped",
    "empty inert cuvette attached to an original dry vertical holder",
)
FEATURES["kfupm_superconductivity_spectroscopy"] = _spectroscopy_features


def _thermobalance(b, base, params):
    # Integrated wood-topped steel bench and controller bay visible in the source.
    for x in (-0.73, 0.73):
        for y in (-0.40, 0.40):
            b.box(base, (x, y, 0.025), (0.070, 0.070, 0.025), "rubber", collision=True)
            b.box(base, (x, y, 0.405), (0.035, 0.035, 0.355), "metal", collision=True)
        b.box(base, (x, 0, 0.20), (0.035, 0.40, 0.025), "metal", collision=True)
    b.box(base, (0, 0, 0.80), (0.83, 0.47, 0.04), "copper", collision=True)
    b.box(base, (-0.20, 0, 0.895), (0.46, 0.38, 0.055), "dark", collision=True)
    b.box(base, (-0.20, 0, 0.98), (0.38, 0.34, 0.03), "cream", collision=True)
    for x in (-0.55, 0.15):
        b.box(base, (x, 0, 1.28), (0.030, 0.34, 0.27), "cream", collision=True)
    b.box(base, (-0.20, 0.31, 1.28), (0.32, 0.030, 0.27), "cream", collision=True)
    b.box(base, (-0.20, 0, 1.55), (0.38, 0.34, 0.025), "cream", collision=True)
    b.box(base, (-0.20, -0.305, 1.39), (0.32, 0.035, 0.135), "cream", collision=True)
    b.box(base, (-0.20, -0.342, 1.39), (0.15, 0.004, 0.055), "metal")
    for x in (-0.39, -0.01):
        b.box(base, (x, -0.15, 0.9895), (0.024, 0.12, 0.0395), "metal", collision=True)
        _tag(
            b,
            b.box(base, (x, -0.22, 1.045), (0.016, 0.28, 0.016), "bright"),
            "thermobalance_access_rail",
        )
    tray = b.moving(
        base,
        "dry_tray_extension",
        (-0.20, -0.12, 1.08),
        (0, -1, 0),
        (0, 0.18),
        mass=0.20,
        kp=380,
    )
    _tag(
        b,
        b.box(tray, (0, 0, 0), (0.23, 0.15, 0.019), "dark", collision=True),
        "thermobalance_tray_plate",
    )
    b.cylinder(tray, (0, 0, 0.028), 0.028, 0.009, "metal", collision=True)
    _tag(
        b,
        b.cylinder(tray, (0, 0, 0.040), 0.008, 0.003, "orange", collision=True),
        "thermobalance_inert_pellet",
    )
    b.site(tray, "thermal_coupon", (0, -0.012, 0.043))
    # Source-visible upper measurement head and foiled connections; no heating.
    for x in (-0.51, 0.11):
        b.cylinder(base, (x, 0.24, 1.945), 0.012, 0.37, "metal", collision=True)
    b.box(base, (-0.20, 0.24, 2.345), (0.35, 0.05, 0.030), "metal", collision=True)
    head = b.moving(
        base,
        "upper_head_height",
        (-0.20, 0.13, 2.13),
        (0, 0, 1),
        (-0.035, 0.035),
        mass=0.30,
        kp=420,
    )
    b.box(head, (0, 0.11, 0), (0.34, 0.04, 0.025), "metal")
    b.cylinder(head, (0, 0, 0), 0.105, 0.10, "dark", collision=True)
    for z in (-0.09, 0.09):
        b.cylinder(head, (0, 0, z), 0.125, 0.023, "metal", collision=True)
    _tag(
        b,
        b.cylinder(head, (0, 0, -0.170), 0.045, 0.070, "dark", collision=True),
        "thermobalance_head_bellows",
    )
    for z in (-0.13, -0.15, -0.17, -0.19, -0.21):
        b.ring(head, (0, 0, z), 0.047, 0.004, "metal")
    b.cylinder(base, (-0.20, 0.13, 1.745), 0.022, 0.17, "bright")
    for z in (1.61, 1.66, 1.71, 1.76, 1.81, 1.86):
        b.ring(base, (-0.20, 0.13, z), 0.025, 0.004, "metal")
    b.cylinder(base, (-0.20, 0.24, 2.53), 0.026, 0.155, "bright", collision=True)
    for z in (2.4, 2.45, 2.5, 2.55, 2.60, 2.65):
        b.ring(base, (-0.20, 0.24, z), 0.027, 0.004, "metal")
    # Left instrumentation rail is independently floor supported.
    for y in (-0.16, 0.38):
        b.box(base, (-1.02, y, 1.05), (0.027, 0.027, 1.05), "metal", collision=True)
    for z in (0.30, 0.74, 1.25, 1.97):
        b.box(base, (-1.02, 0.11, z), (0.18, 0.29, 0.020), "metal", collision=True)
    for z in (0.825, 1.335):
        b.box(base, (-1.02, 0.11, z), (0.16, 0.26, 0.065), "dark", collision=True)
        b.box(base, (-1.02, -0.153, z), (0.080, 0.004, 0.030), "screen")
    b.cylinder(
        base, (-1.02, -0.01, 2.075), 0.09, 0.023, "metal", euler=(math.pi / 2, 0, 0)
    )
    b.cylinder(
        base, (-1.02, -0.037, 2.075), 0.074, 0.004, "cream", euler=(math.pi / 2, 0, 0)
    )
    b.rod(base, (-1.02, -0.044, 2.075), (-1.06, -0.044, 2.115), 0.004, "dark")
    b.box(base, (-1.02, 0.01, 2.015), (0.028, 0.045, 0.026), "metal", collision=True)
    for i in range(7):
        x = -1.14 + i * 0.035
        b.rod(base, (x, -0.15, 1.38), (x + 0.06, -0.16, 0.85), 0.004, "dark")
        b.rod(base, (x + 0.06, -0.16, 0.85), (x, -0.12, 0.32), 0.004, "dark")
    # Right display and under-bench computer have explicit supporting feet.
    b.box(base, (0.60, 0.14, 0.856), (0.15, 0.11, 0.016), "dark", collision=True)
    b.box(base, (0.60, 0.20, 0.968), (0.023, 0.025, 0.096), "metal", collision=True)
    b.box(base, (0.60, 0.20, 1.244), (0.23, 0.024, 0.18), "dark", collision=True)
    b.box(base, (0.60, 0.173, 1.244), (0.21, 0.003, 0.16), "screen")
    b.box(base, (0.60, -0.22, 0.853), (0.23, 0.11, 0.013), "dark")
    b.box(base, (0.32, 0.05, 0.28), (0.16, 0.26, 0.28), "dark", collision=True)
    b.box(base, (0.32, -0.215, 0.42), (0.095, 0.004, 0.045), "metal")
    b.metadata["capabilities"] = [
        "inert sample-tray inspection travel",
        "bounded dry upper-head vertical alignment",
    ]
    b.metadata["limitations"].append(
        "TG3 thermobalance exterior references the specific storage-metrology photograph only, not other facilities on the page. Tray opening, motorization and internal support dimensions are original estimates. No heating, mass-change, reaction, gas handling, nuclear process or thermochemical storage response."
    )


BUILDERS["science_tokyo_tg3_dry_inspection"] = _thermobalance
SOURCES["science_tokyo_tg3_dry_inspection"] = {
    "reference": "Science Tokyo Zero-Carbon Energy Laboratory official TG3 thermobalance workstation photograph",
    "url": THERMOBALANCE,
    "dimensions_m": [2.05, 0.94, 2.685],
    "dimension_basis": "All dimensions estimated from one installed workstation photo; hidden sample access and vertical adjustment are original proxies.",
}
SAMPLE_INTERFACES["science_tokyo_tg3_dry_inspection"] = (
    "thermal_coupon",
    (0.016, 0.016, 0.006),
    "clamped",
    "inert solid calibration pellet fixed in an authored dry inspection tray",
)


def _insulation_bay(b, base, params):
    # De-energized radiator-faced exterior; source does not identify its circuit.
    for x in (-2.05, -0.35):
        for y in (0.04, 1.36):
            b.box(base, (x, y, 0.025), (0.20, 0.15, 0.025), "metal", collision=True)
    b.box(base, (-1.20, 0.70, 0.30), (0.99, 0.83, 0.25), "metal", collision=True)
    b.box(base, (-1.20, 0.70, 1.65), (0.80, 0.70, 1.10), "metal", collision=True)
    for x in (-2.04, -0.36):
        for j in range(19):
            b.box(
                base,
                (x, 0.04 + j * 0.072, 1.60),
                (0.14, 0.018, 0.80),
                "dark",
                collision=True,
            )
        for z in (0.82, 2.38):
            b.box(base, (x, 0.70, z), (0.14, 0.69, 0.025), "metal", collision=True)
    for x in (-1.74, -0.65):
        b.box(base, (x, -0.020, 1.80), (0.20, 0.020, 0.28), "shell")
    b.box(base, (-1.20, 0.70, 2.775), (0.72, 0.45, 0.025), "metal", collision=True)
    b.cylinder(
        base,
        (-1.20, 0.70, 3.10),
        0.30,
        0.95,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for x in (-2.10, -0.30):
        b.cylinder(
            base, (x, 0.70, 3.10), 0.315, 0.025, "bright", euler=(0, math.pi / 2, 0)
        )
    # A separate small exterior enclosure is visible behind the main unit.
    b.box(base, (-2.65, 0.93, 0.85), (0.35, 0.47, 0.85), "blue", collision=True)
    b.cylinder(
        base,
        (-2.65, 0.93, 1.93),
        0.25,
        0.32,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    # The dry inspection fixture is original and explicitly outside energized use.
    for x in (0.70, 1.60):
        for y in (-1.84, -1.16):
            b.box(base, (x, y, 0.03), (0.09, 0.09, 0.03), "rubber", collision=True)
            b.box(base, (x, y, 0.525), (0.035, 0.035, 0.465), "metal", collision=True)
    b.box(base, (1.15, -1.50, 1.015), (0.57, 0.42, 0.025), "metal", collision=True)
    for y in (-1.73, -1.27):
        _tag(
            b,
            b.box(base, (1.15, y, 1.0675), (0.52, 0.018, 0.0275), "bright"),
            "insulation_fixture_rail",
        )
    stage = b.moving(
        base,
        "fixture_x",
        (1.15, -1.50, 1.12),
        (1, 0, 0),
        (-0.14, 0.14),
        mass=0.40,
        kp=450,
    )
    _tag(
        b,
        b.box(stage, (0, 0, 0), (0.30, 0.32, 0.025), "dark", collision=True),
        "insulation_fixture_plate",
    )
    for y in (-0.13, 0.13):
        b.box(stage, (0, y, 0.15), (0.05, 0.020, 0.125), "metal", collision=True)
        b.cylinder(
            stage, (0, y, 0.33), 0.055, 0.025, "metal", euler=(math.pi / 2, 0, 0)
        )
    pitch = b.moving(
        stage,
        "coupon_pitch",
        (0, 0, 0.33),
        (0, 1, 0),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.10,
        kp=115,
    )
    b.cylinder(pitch, (0, 0, 0), 0.025, 0.14, "cream", collision=True)
    for z in (-0.105, -0.035, 0.035, 0.105):
        b.cylinder(pitch, (0, 0, z), 0.064, 0.012, "copper", collision=True)
    for y in (-0.080, 0.080):
        b.cylinder(pitch, (0, y, 0), 0.012, 0.040, "bright", euler=(math.pi / 2, 0, 0))
    _tag(
        b,
        b.box(pitch, (0.026, 0, 0.04), (0.010, 0.022, 0.025), "orange", collision=True),
        "insulation_inert_marker",
    )
    b.site(pitch, "insulating_coupon", (0.04, -0.025, 0.04))
    # Low exclusion fence stays open at the mechanical inspection approach.
    for y in (-2.35, 1.90):
        for x in (-3.10, -1.20, 0.10, 2.0):
            b.cylinder(base, (x, y, 0.67), 0.033, 0.67, "metal", collision=True)
        b.cylinder(
            base,
            (-1.50 if y < 0 else -0.55, y, 1.34),
            0.035,
            1.60 if y < 0 else 2.55,
            "metal",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
    for z in (0.20, 1.20):
        for x in (-3.1, 2.0):
            b.cylinder(
                base, (x, -0.225, z), 0.012, 2.125, "metal", euler=(math.pi / 2, 0, 0)
            )
    for i in range(34):
        x = -3.05 + i * 0.15
        # Both signs create a mesh impression without claiming a rated barrier.
        for sign in (-1, 1):
            x2 = x + sign * 0.32
            if -3.10 <= x2 <= 2.0:
                b.rod(base, (x, 1.9, 0.20), (x2, 1.9, 1.20), 0.0025, "metal")
    # Front fence leaves a wide opening around the dry fixture, not a safety gate.
    for x in (-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6):
        for sign in (-1, 1):
            b.rod(
                base, (x, -2.35, 0.20), (x + sign * 0.15, -2.35, 1.20), 0.0025, "metal"
            )
    b.metadata["capabilities"] = [
        "de-energized inert fixture lateral positioning",
        "inert insulating coupon pitch alignment",
    ]
    b.metadata["limitations"].append(
        "Apparatus exterior is not identified to a specific voltage source or circuit. Added dry inspection fixture is original; no voltage, discharge, insulation performance, electrical test protocol, electrical safety function or process operation is modeled."
    )


def _hv_features(world, definition):
    # Tall green-screened enclosure follows the photo; exact identity unknown.
    for x in (-0.90, 3.90):
        for y in (1.6, 6.0):
            box(
                world,
                f"hv_enclosure_foot_{x}_{y}",
                (0.16, 0.16, 0.04),
                (x, y, 0.04),
                (0.26, 0.32, 0.27, 1),
            )
            box(
                world,
                f"hv_enclosure_post_{x}_{y}",
                (0.060, 0.060, 3.68),
                (x, y, 3.76),
                (0.28, 0.42, 0.26, 1),
            )
    for z in (0.14, 2.60, 5.0, 7.45):
        for y in (1.6, 6.0):
            box(
                world,
                f"hv_enclosure_cross_{y}_{z}",
                (2.46, 0.040, 0.050),
                (1.5, y, z),
                (0.36, 0.47, 0.26, 1),
            )
        for x in (-0.90, 3.90):
            box(
                world,
                f"hv_enclosure_side_{x}_{z}",
                (0.040, 2.20, 0.050),
                (x, 3.8, z),
                (0.36, 0.47, 0.26, 1),
            )
    for i in range(35):
        x = -0.84 + i * 0.135
        box(
            world,
            f"hv_green_front_fold_{i}",
            (0.069, 0.022, 3.61),
            (x, 1.62, 3.77),
            (0.16 + 0.018 * (i % 2), 0.32 + 0.015 * (i % 2), 0.14, 1),
        )
    for i in range(32):
        y = 1.68 + i * 0.135
        box(
            world,
            f"hv_green_side_fold_{i}",
            (0.022, 0.069, 3.61),
            (3.88, y, 3.77),
            (0.18, 0.35, 0.14, 1),
        )
    # Yellow crane-like structural beams are static context, not operable lifting.
    for y in (-4.0, 2.5, 6.7):
        for x in (-5.65, 5.65):
            box(
                world,
                f"hv_hall_column_{x}_{y}",
                (0.13, 0.15, 4.10),
                (x, y, 4.10),
                (0.54, 0.57, 0.44, 1),
            )
        box(
            world,
            f"hv_yellow_crossbeam_{y}",
            (5.65, 0.20, 0.22),
            (0, y, 8.42),
            (0.75, 0.58, 0.06, 1),
        )
        for x in (-4.8, -2.4, 0, 2.4, 4.8):
            box(
                world,
                f"hv_beam_web_{x}_{y}",
                (0.075, 0.26, 0.22),
                (x, y, 8.42),
                (0.76, 0.61, 0.08, 1),
            )
    for i in range(4):
        y = -4.4 + i * 3.1
        box(
            world,
            f"arch_cutaway_wall_right_hv_clerestory_frame_{i}",
            (0.018, 1.05, 0.54),
            (5.975, y, 7.0),
            (0.45, 0.50, 0.49, 1),
        )
        box(
            world,
            f"arch_cutaway_wall_right_hv_clerestory_glass_{i}",
            (0.019, 0.98, 0.48),
            (5.954, y, 7.0),
            (0.20, 0.36, 0.47, 1),
        )


BUILDERS["kfupm_deenergized_insulation_bay"] = _insulation_bay
SOURCES["kfupm_deenergized_insulation_bay"] = {
    "reference": "KFUPM ARC Metrology official high-voltage laboratory installed hall photograph",
    "url": HIGH_VOLTAGE,
    "dimensions_m": [5.2, 4.32, 3.415],
    "dimension_basis": "All depicted apparatus, fence and hall dimensions estimated. Published 6 by 6 by 9 m fog-chamber dimensions are not assigned to the unidentified green enclosure.",
}
SAMPLE_INTERFACES["kfupm_deenergized_insulation_bay"] = (
    "insulating_coupon",
    (0.128, 0.128, 0.28),
    "clamped",
    "de-energized inert insulating coupon fixed to an original inspection fixture",
)
FEATURES["kfupm_high_voltage_insulation"] = _hv_features
