"""Photo-bounded fluid, electron-microscopy and optical workstation mechanisms.

Only specifically documented dimensions are measured claims. Mechanical
surrogates do not predict airflow, sediment transport, imaging or mineral state.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

BROWN = "https://sites.brown.edu/breuerlab/animal-flight-and-aeromechanics-wind-tunnel/"
UBC = "https://geog.ubc.ca/news/cutting-edge-river-science-and-engineering-tools-in-action-the-hassan-labs-legoflume-launches/"
BOSTON = "https://www.bu.edu/photonics/sharedfacilities/pml/"
CHICAGO = "https://mineralphysics.uchicago.edu/labtour.html"


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _brown(b, base, params):
    # Published clear section is 4.0 m long by 1.2 m wide and 1.2 m tall.
    for x in (-1.8, 0, 1.8):
        for y in (-0.68, 0.68):
            b.box(base, (x, y, 0.035), (0.13, 0.13, 0.035), "blue", collision=True)
            b.box(base, (x, y, 0.393), (0.035, 0.035, 0.323), "blue", collision=True)
        b.box(base, (x, 0, 0.742), (0.055, 0.725, 0.026), "blue", collision=True)
        b.geom(
            base,
            "capsule",
            (0.020,),
            material="blue",
            collision=True,
            fromto=(x, -0.66, 0.15, x, 0.61, 0.68),
        )
    _tag(
        b,
        b.box(base, (0, 0, 0.784), (2.0, 0.625, 0.016), "metal", collision=True),
        "test_floor",
    )
    _tag(
        b,
        b.box(base, (0, 0, 2.016), (2.0, 0.625, 0.016), "glass", collision=True),
        "test_roof",
    )
    for sign in (-1, 1):
        y = sign * 0.614
        for i in range(4):
            x = -1.5 + i
            _tag(
                b,
                b.box(
                    base, (x, y, 1.40), (0.495, 0.014, 0.60), "glass", collision=True
                ),
                "test_side",
            )
            for z in (0.835, 1.965):
                b.box(
                    base,
                    (x, y + sign * 0.017, z),
                    (0.5, 0.025, 0.035),
                    "dark",
                    collision=True,
                )
            for dx in (-0.46, 0.46):
                b.box(
                    base,
                    (x + dx, y + sign * 0.017, 1.40),
                    (0.040, 0.025, 0.53),
                    "dark",
                    collision=True,
                )
            for z in (1.02, 1.78):
                b.box(
                    base,
                    (x - 0.36, y + sign * 0.055, z),
                    (0.010, 0.024, 0.065),
                    "metal",
                )
                b.box(
                    base, (x - 0.30, y + sign * 0.074, z), (0.062, 0.010, 0.010), "dark"
                )
            for dx in (-0.35, 0, 0.35):
                b.cylinder(
                    base,
                    (x + dx, y + sign * 0.048, 0.84),
                    0.008,
                    0.008,
                    "bright",
                    euler=(math.pi / 2, 0, 0),
                )
    # Short duct collars bound this photographed section, not an invented loop.
    for x in (-2.13, 2.13):
        for y in (-0.70, 0.70):
            b.box(base, (x, y, 1.4), (0.13, 0.07, 0.70), "dark", collision=True)
        for z in (0.71, 2.09):
            b.box(base, (x, 0, z), (0.13, 0.63, 0.07), "dark", collision=True)
    # Documented pitch-yaw positioning, reconstructed with inferred internals.
    b.box(base, (0, 0, 0.83), (0.15, 0.15, 0.03), "metal", collision=True)
    b.cylinder(base, (0, 0, 1.065), 0.033, 0.205, "bright", collision=True)
    b.cylinder(base, (0, 0, 1.295), 0.080, 0.025, "dark")
    yaw = b.moving(
        base,
        "model_yaw",
        (0, 0, 1.345),
        (0, 0, 1),
        (-0.45, 0.45),
        kind="hinge",
        mass=0.20,
        kp=160,
    )
    b.cylinder(yaw, (0, 0, 0), 0.078, 0.025, "metal", collision=True)
    for y in (-0.087, 0.087):
        b.box(yaw, (0, y, 0.065), (0.06, 0.016, 0.04), "dark", collision=True)
    pitch = b.moving(
        yaw,
        "model_pitch",
        (0, 0, 0.100),
        (0, 1, 0),
        (-0.32, 0.32),
        kind="hinge",
        mass=0.08,
        kp=100,
    )
    b.cylinder(pitch, (0, 0, 0), 0.012, 0.10, "bright", euler=(math.pi / 2, 0, 0))
    b.geom(
        pitch,
        "ellipsoid",
        (0.28, 0.055, 0.014),
        (0, 0, 0.040),
        "orange",
        collision=True,
    )
    b.box(pitch, (0, 0, 0.020), (0.025, 0.025, 0.020), "metal", collision=True)
    b.site(pitch, "aerodynamic_coupon", (0, 0, 0.055))
    # Nearby instrument rack from the room photograph, with continuous feet.
    for x in (2.65, 3.10):
        for y in (0.75, 1.20):
            b.box(base, (x, y, 0.03), (0.06, 0.06, 0.03), "rubber", collision=True)
    b.box(base, (2.875, 0.975, 0.085), (0.29, 0.29, 0.025), "metal", collision=True)
    for x in (2.61, 3.14):
        b.box(base, (x, 0.975, 0.86), (0.025, 0.29, 0.75), "metal", collision=True)
    for z in (0.36, 0.73, 1.10, 1.47):
        b.box(base, (2.875, 0.975, z), (0.24, 0.27, 0.105), "dark", collision=True)
        b.box(base, (2.875, 0.699, z), (0.15, 0.005, 0.055), "screen")
        b.box(
            base, (2.875, 0.975, z - 0.12), (0.27, 0.28, 0.015), "metal", collision=True
        )
    b.metadata["capabilities"] = [
        "dry engineering-model yaw positioning",
        "dry engineering-model pitch positioning",
    ]
    b.metadata["limitations"].append(
        "Documented 1.2 x 1.2 x 4.0 m clear test section; supports, positioner internals and room envelope estimated. Inert engineering coupon only; no animal workflow, aerodynamic or biological model."
    )


BUILDERS = {"brown_afam_pitch_yaw_section": _brown}
SOURCES = {
    "brown_afam_pitch_yaw_section": {
        "reference": "Brown Breuer Lab AFAM test-section photo and published specifications",
        "url": BROWN,
        "dimensions_m": [5.48, 2.32, 2.16],
        "dimension_basis": "Verified clear test section 4.0 m length and 1.2 x 1.2 m cross-section; casing, support, rack and controls estimated.",
    }
}
SAMPLE_INTERFACES = {
    "brown_afam_pitch_yaw_section": (
        "aerodynamic_coupon",
        (0.56, 0.11, 0.028),
        "clamped",
        "inert rigid engineering coupon mounted on the pitch-yaw fixture",
    )
}
FEATURES = {}


def _legoflume(b, base, params):
    # Estimated local shallow river-bed envelope; no water or movable sediment.
    b.box(base, (0, 0, 0.20), (1.25, 3.2, 0.20), "metal", collision=True)
    _tag(
        b,
        b.box(
            base,
            (0, 0, 0.44),
            (1.22, 3.17, 0.04),
            "cream",
            rgba=(0.45, 0.40, 0.28, 1),
            collision=True,
        ),
        "riverbed_slab",
    )
    for x in (-1.26, 1.26):
        b.box(base, (x, 0, 0.54), (0.035, 3.24, 0.14), "metal", collision=True)
    for row in range(27):
        y = -3.03 + row * 0.23
        for col in range(9):
            x = -1.10 + col * 0.27 + 0.025 * math.sin(row * col + 1)
            shade = 0.37 + 0.06 * math.sin(row + col * 1.8)
            if abs(x - 0.22 * math.sin(y * 1.8)) < 0.26:
                shade -= 0.10
            b.geom(
                base,
                "ellipsoid",
                (0.085 + 0.015 * math.sin(row + col), 0.072, 0.018),
                (x, y, 0.483),
                "cream",
                rgba=(shade + 0.08, shade + 0.06, shade, 1),
            )
    # Working walkway and source-visible metal guardrail on the right.
    b.box(base, (1.89, 0, 0.225), (0.49, 3.20, 0.225), "metal", collision=True)
    for y in (-3.1, -1.6, 0, 1.6, 3.1):
        b.box(base, (1.38, y, 0.78), (0.023, 0.023, 0.78), "bright", collision=True)
    for z in (0.78, 1.50):
        b.box(base, (1.38, 0, z), (0.025, 3.15, 0.025), "bright", collision=True)
    # Dense overhead gantry, with the camera bridge riding on separate rails.
    for x in (-1.60, 1.60):
        for y in (-3.15, -1.05, 1.05, 3.15):
            b.box(base, (x, y, 0.025), (0.13, 0.13, 0.025), "metal", collision=True)
            b.box(base, (x, y, 1.775), (0.03, 0.03, 1.725), "metal", collision=True)
            b.box(
                base, (x * 0.90, y, 3.045), (0.19, 0.03, 0.03), "bright", collision=True
            )
    for y in (-3.15, -1.05, 1.05, 3.15):
        b.box(base, (0, y, 3.50), (1.63, 0.035, 0.035), "bright", collision=True)
    for x in (-1.35, 1.35):
        _tag(
            b,
            b.box(base, (x, 0, 3.05), (0.032, 3.22, 0.03), "bright", collision=True),
            "camera_runway",
        )
    carriage = b.moving(
        base,
        "camera_traverse",
        (0, 0, 3.20),
        (0, 1, 0),
        (-2.2, 2.2),
        mass=1.1,
        kp=500,
        force=300,
    )
    b.box(carriage, (0, 0, 0), (1.44, 0.095, 0.020), "metal", collision=True)
    for x in (-1.35, 1.35):
        for y in (-0.07, 0.07):
            _tag(
                b,
                b.cylinder(
                    carriage,
                    (x, y, -0.07),
                    0.05,
                    0.022,
                    "dark",
                    euler=(0, math.pi / 2, 0),
                ),
                "camera_wheel",
            )
    b.box(carriage, (0, 0, -0.085), (0.075, 0.07, 0.065), "dark", collision=True)
    b.cylinder(carriage, (0, 0, -0.17), 0.033, 0.02, "lens", collision=True)
    # Feed roller, wood chute and a reversible dry inlet-height panel.
    for x in (-1.05, 1.05):
        b.box(base, (x, 2.80, 0.04), (0.12, 0.16, 0.04), "metal", collision=True)
        b.box(base, (x, 2.80, 0.58), (0.05, 0.05, 0.50), "orange", collision=True)
    b.cylinder(
        base,
        (0, 2.80, 1.08),
        0.14,
        1.10,
        "dark",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.box(
        base,
        (0, 2.32, 0.84),
        (0.97, 0.42, 0.025),
        "cream",
        euler=(0.34, 0, 0),
        collision=True,
    )
    for x in (-0.82, 0.82):
        b.box(base, (x, 1.95, 0.57), (0.10, 0.13, 0.17), "metal", collision=True)
        b.box(base, (x, 2.69, 0.665), (0.10, 0.11, 0.265), "metal", collision=True)
    for x in (-0.96, 0.96):
        b.box(base, (x, 1.68, 0.87), (0.027, 0.035, 0.39), "bright", collision=True)
    b.box(base, (0, 1.68, 1.285), (0.99, 0.035, 0.025), "bright", collision=True)
    gate = b.moving(
        base,
        "inlet_panel_height",
        (0, 1.68, 0.78),
        (0, 0, 1),
        (0, 0.22),
        mass=0.6,
        kp=400,
    )
    b.box(gate, (0, 0, 0), (0.90, 0.027, 0.13), "metal", collision=True)
    for x in (-0.935, 0.935):
        _tag(
            b,
            b.box(gate, (x, 0, 0), (0.040, 0.050, 0.06), "metal"),
            "inlet_guide_sleeve",
        )
    for x in (-0.82, 0.82):
        b.box(gate, (x, 0, 0.35), (0.012, 0.018, 0.25), "bright")
    # Fixed bright coupon makes camera alignment observable without flow claims.
    b.box(base, (0.25, -0.55, 0.493), (0.075, 0.075, 0.013), "orange", collision=True)
    b.box(base, (0.25, -0.55, 0.508), (0.009, 0.06, 0.002), "dark")
    b.site(base, "riverbed_target", (0.25, -0.55, 0.512))
    b.metadata["capabilities"] = [
        "overhead camera longitudinal positioning",
        "dry inlet-panel height inspection",
    ]
    b.metadata["limitations"].append(
        "Single local LegoFlume bay informed by two official unveiling photos; all lengths, bed material and actuator strokes estimated. Sediment is fixed decoration; no water, erosion, transport, hydraulic feedback or whole-facility layout model."
    )


def _ubc_features(world, definition):
    for z in (2.65, 2.9, 3.15):
        ET.SubElement(
            world,
            "geom",
            name=f"ubc_wall_pipe_{z}",
            type="capsule",
            size=".035",
            fromto=f"-3.38 -3.8 {z} -3.38 3.8 {z}",
            rgba=".72 .72 .65 1",
        )
        for y in (-3.5, 0, 3.5):
            box(
                world,
                f"ubc_pipe_wall_clip_{z}_{y}",
                (0.055, 0.016, 0.018),
                (-3.445, y, z),
                (0.40, 0.42, 0.40, 1),
            )
    for z in (0.075, 0.15, 0.225, 0.30, 0.375):
        # Nested stair treads ground the raised walkway at its front entrance.
        depth = 0.70 - z * 1.2
        box(
            world,
            f"ubc_walkway_step_{z}",
            (0.49, depth, 0.0375),
            (1.89, -3.2 - depth, z - 0.0375),
            (0.55, 0.57, 0.55, 1),
        )


BUILDERS["ubc_legoflume_camera_inlet"] = _legoflume
SOURCES["ubc_legoflume_camera_inlet"] = {
    "reference": "UBC Geography official LegoFlume unveiling photographs",
    "url": UBC,
    "dimensions_m": [4.03, 6.48, 3.535],
    "dimension_basis": "All geometry, river-bed extent, hidden supports and bounded mechanical strokes are estimates. Official variable geometry is qualitative, not a supplied calibrated plan.",
}
SAMPLE_INTERFACES["ubc_legoflume_camera_inlet"] = (
    "riverbed_target",
    (0.15, 0.15, 0.026),
    "clamped",
    "inert fixed river-bed calibration marker inspected by the overhead camera",
)
FEATURES["ubc_hassan_legoflume"] = _ubc_features


def _boston_sem(b, base, params):
    for x in (-0.36, 0.36):
        for y in (-0.31, 0.31):
            b.box(base, (x, y, 0.025), (0.07, 0.07, 0.025), "rubber", collision=True)
    b.box(base, (0, 0, 0.39), (0.43, 0.38, 0.34), "cream", collision=True)
    b.box(base, (0, 0, 0.755), (0.48, 0.43, 0.025), "dark", collision=True)
    b.housing(
        base,
        [
            (0.78, 0.24, 0.27, 0),
            (0.86, 0.31, 0.30, 0),
            (1.17, 0.31, 0.30, 0),
            (1.31, 0.19, 0.20, 0.02),
        ],
        radius=0.04,
        material="cream",
    )
    # Collidable rear and sides keep a genuine open access region at the front.
    b.box(base, (0, 0.22, 1.045), (0.25, 0.06, 0.25), "metal", collision=True)
    for x in (-0.275, 0.275):
        b.box(base, (x, 0, 1.045), (0.035, 0.23, 0.25), "cream", collision=True)
    b.cylinder(base, (0, 0.025, 1.625), 0.235, 0.315, "cream", collision=True)
    b.box(base, (0.10, -0.222, 1.64), (0.205, 0.025, 0.315), "dark", collision=True)
    for i in range(18):
        for j in range(2):
            b.box(
                base,
                (-0.065 + j * 0.18, -0.251, 1.355 + i * 0.032),
                (0.074, 0.003, 0.007),
                "metal",
                rgba=(0.21, 0.23, 0.23, 1),
            )
    for i in range(18):
        a = math.pi * i / 18
        c = math.pi * (i + 1) / 18
        b.rod(
            base,
            (0.18 * math.cos(a), 0.02, 1.94 + 0.18 * math.sin(a)),
            (0.18 * math.cos(c), 0.02, 1.94 + 0.18 * math.sin(c)),
            0.018,
            "dark",
        )
    b.geom(
        base,
        "capsule",
        (0.06,),
        material="metal",
        collision=True,
        fromto=(-0.27, 0.07, 1.18, -0.47, 0.07, 1.34),
    )
    b.box(
        base,
        (-0.51, 0.07, 1.40),
        (0.14, 0.10, 0.075),
        "cream",
        collision=True,
        euler=(0, -0.65, 0),
    )
    for x in (-0.20, 0.20):
        b.cylinder(
            base,
            (x, -0.32, 1.17),
            0.040,
            0.022,
            "bright",
            euler=(math.pi / 2, 0, 0),
            collision=True,
        )
        b.cylinder(
            base, (x, -0.346, 1.17), 0.022, 0.006, "dark", euler=(math.pi / 2, 0, 0)
        )
    # Open inspection cassette is deliberately a visible dry surrogate.
    b.box(base, (0, -0.58, 0.79), (0.25, 0.33, 0.04), "metal", collision=True)
    for x in (-0.14, 0.14):
        _tag(
            b,
            b.box(base, (x, -0.60, 0.842), (0.012, 0.31, 0.012), "bright"),
            "sem_drawer_guide",
        )
    drawer = b.moving(
        base,
        "specimen_exchange",
        (0, -0.62, 0.871),
        (0, -1, 0),
        (0, 0.16),
        mass=0.22,
        kp=360,
    )
    b.box(drawer, (0, 0, 0), (0.19, 0.18, 0.017), "dark", collision=True)
    b.cylinder(drawer, (0, 0, 0.030), 0.07, 0.013, "metal")
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
    b.cylinder(stage, (0, 0, 0), 0.065, 0.012, "metal", collision=True)
    b.cylinder(stage, (0, 0, 0.018), 0.015, 0.006, "orange", collision=True)
    b.box(stage, (0.004, 0, 0.025), (0.0015, 0.011, 0.001), "dark")
    b.site(stage, "sem_stub", (0, 0, 0.027))
    # Adjacent desk supports the source-specific three-monitor curved console.
    for x in (0.58, 2.05):
        for y in (-0.68, 0.10):
            b.box(base, (x, y, 0.02), (0.06, 0.06, 0.02), "rubber", collision=True)
            b.box(base, (x, y, 0.39), (0.025, 0.025, 0.35), "metal", collision=True)
    b.box(base, (1.315, -0.29, 0.765), (0.82, 0.46, 0.025), "dark", collision=True)
    for x in (0.76, 1.31, 1.86):
        b.box(base, (x, 0.025, 0.803), (0.13, 0.12, 0.013), "dark", collision=True)
        b.box(base, (x, 0.055, 0.8805), (0.025, 0.025, 0.0645), "dark", collision=True)
        b.box(base, (x, 0.055, 1.095), (0.255, 0.035, 0.15), "dark", collision=True)
        b.box(base, (x, 0.015, 1.095), (0.232, 0.005, 0.128), "screen")
    console = ET.SubElement(base, "body", pos="1.30 -.44 .79")
    b.housing(
        console,
        [(0, 0.46, 0.18, 0), (0.032, 0.45, 0.18, 0), (0.06, 0.37, 0.15, 0.02)],
        radius=0.10,
        material="cream",
    )
    b.box(console, (0, -0.018, 0.052), (0.25, 0.073, 0.010), "dark")
    for x in (-0.34, 0.34):
        b.cylinder(console, (x, -0.02, 0.068), 0.050, 0.018, "dark", collision=True)
    for x in (-0.30, -0.18, -0.06, 0.06, 0.18, 0.30):
        b.cylinder(console, (x, 0.10, 0.065), 0.015, 0.018, "dark", collision=True)
    b.metadata["capabilities"] = [
        "open dry SEM specimen exchange",
        "inert stub azimuth indexing",
    ]
    b.metadata["limitations"].append(
        "Photo-informed Zeiss-family SEM exterior; exact model suffix and hidden specimen mechanics unverified. Open cassette geometry is authored for dry inspection. No electron optics, vacuum, imaging, X-ray spectrum or sample response solver."
    )


BUILDERS["boston_pml_sem_inspection"] = _boston_sem
SOURCES["boston_pml_sem_inspection"] = {
    "reference": "BU Photonics PML official installed SEM and operator-console photograph",
    "url": BOSTON,
    "dimensions_m": [2.88, 1.36, 2.12],
    "dimension_basis": "All casing, console, sample and control dimensions estimated; model suffix and hidden exchange mechanism unresolved.",
}
SAMPLE_INTERFACES["boston_pml_sem_inspection"] = (
    "sem_stub",
    (0.030, 0.030, 0.012),
    "clamped",
    "inert specimen stub on the open inspection cassette",
)


def _chicago_optical(b, base, params):
    for x in (-1.02, 1.02):
        for y in (-0.64, 0.64):
            b.cylinder(base, (x, y, 0.03), 0.20, 0.03, "rubber", collision=True)
            _tag(
                b,
                b.cylinder(base, (x, y, 0.41), 0.18, 0.35, "dark", collision=True),
                "optical_isolator",
            )
            for z in (0.14, 0.68):
                b.ring(base, (x, y, z), 0.183, 0.006, "metal", segments=20)
    _tag(
        b,
        b.box(base, (0, 0, 0.83), (1.45, 0.94, 0.07), "dark", collision=True),
        "covered_table",
    )
    b.box(base, (0, 0, 0.912), (1.42, 0.91, 0.012), "metal", collision=True)
    for x in (-1.435, 1.435):
        b.box(base, (x, 0, 1.14), (0.015, 0.94, 0.24), "metal", collision=True)
    b.box(base, (0, 0.925, 1.14), (1.42, 0.015, 0.24), "metal", collision=True)
    # Front side panels remain in place; center is an explicit inspection cutaway.
    for x in (-0.98, 0.98):
        b.box(base, (x, -0.925, 1.14), (0.45, 0.015, 0.24), "metal", collision=True)
        for z in (1.09, 1.23):
            b.rod(base, (x - 0.035, -0.947, z), (x + 0.035, -0.947, z), 0.005, "bright")
        for dx in (-0.035, 0.035):
            b.rod(base, (x + dx, -0.947, 1.09), (x + dx, -0.947, 1.23), 0.005, "bright")
    # Six handled top panels; central front panel is shown lifted for access.
    for x in (-0.96, 0, 0.96):
        for y in (-0.46, 0.46):
            if x == 0 and y < 0:
                continue
            b.box(base, (x, y, 1.395), (0.47, 0.45, 0.015), "metal", collision=True)
            for dx in (-0.055, 0.055):
                b.rod(base, (x + dx, y, 1.41), (x + dx, y, 1.46), 0.005, "bright")
            b.rod(base, (x - 0.055, y, 1.46), (x + 0.055, y, 1.46), 0.005, "bright")
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
    lifted = ET.SubElement(lid, "body", euler="-.60 0 0", gravcomp="1")
    b.box(lifted, (0, -0.46, 0), (0.46, 0.435, 0.015), "metal", collision=True)
    for x in (-0.055, 0.055):
        b.rod(lifted, (x, -0.46, 0.015), (x, -0.46, 0.065), 0.005, "bright")
    b.rod(lifted, (-0.055, -0.46, 0.065), (0.055, -0.46, 0.065), 0.005, "bright")
    for x in (-0.43, 0.43):
        b.cylinder(base, (x, 0, 1.395), 0.022, 0.035, "dark", euler=(0, math.pi / 2, 0))
    # Minimal authored internals, deliberately not an invented recovered beam path.
    b.box(base, (0, -0.40, 0.949), (0.24, 0.18, 0.025), "dark", collision=True)
    for y in (-0.52, -0.28):
        _tag(
            b,
            b.box(base, (0, y, 0.985), (0.21, 0.01, 0.011), "bright"),
            "mineral_stage_guide",
        )
    stage = b.moving(
        base,
        "mineral_coupon_x",
        (0, -0.40, 1.009),
        (1, 0, 0),
        (-0.07, 0.07),
        mass=0.18,
        kp=340,
    )
    b.box(stage, (0, 0, 0), (0.13, 0.12, 0.013), "metal", collision=True)
    b.cylinder(stage, (0, 0, 0.030), 0.065, 0.017, "dark", collision=True)
    b.geom(
        stage,
        "ellipsoid",
        (0.020, 0.016, 0.010),
        (0, 0, 0.057),
        "orange",
        collision=True,
    )
    b.site(stage, "mineral_coupon", (0, 0, 0.068))
    for x in (-0.40, 0.40):
        b.box(base, (x, 0.28, 0.934), (0.05, 0.05, 0.01), "dark", collision=True)
        b.cylinder(base, (x, 0.28, 1.0545), 0.009, 0.1105, "bright", collision=True)
        b.ring(base, (x, 0.28, 1.18), 0.030, 0.008, "dark", plane="xz", segments=12)
        b.cylinder(
            base, (x, 0.28, 1.18), 0.024, 0.005, "lens", euler=(math.pi / 2, 0, 0)
        )
    # Two small monitors visibly rest on the rear enclosure roof.
    for x in (-0.60, 0.60):
        b.box(base, (x, 0.57, 1.421), (0.13, 0.10, 0.011), "dark", collision=True)
        b.box(base, (x, 0.60, 1.472), (0.02, 0.02, 0.040), "dark", collision=True)
        b.box(base, (x, 0.60, 1.622), (0.22, 0.035, 0.11), "dark", collision=True)
        b.box(base, (x, 0.56, 1.622), (0.197, 0.004, 0.089), "screen")
    b.metadata["capabilities"] = [
        "raised optical enclosure access-panel positioning",
        "inert mineral-coupon lateral inspection",
    ]
    b.metadata["limitations"].append(
        "Photo 69 supports the covered optical workstation exterior only. Center front panel is deliberately omitted and one lid is pre-opened for inspection. Internal stage and two optics are authored, not recovered from the unregistered header photo. No laser heating, pressure, spectroscopy or mineral response solver."
    )


def _chicago_features(world, definition):
    desk = next(
        item for item in definition["benches"] if item["id"] == "side_service_desk"
    )
    x, y, _ = desk["pos"]
    h = desk["size"][2]
    for name, half, pos in (
        ("foot", (0.14, 0.12, 0.013), (x, y, h + 0.013)),
        ("stand", (0.025, 0.025, 0.064), (x, y, h + 0.090)),
        ("case", (0.25, 0.035, 0.15), (x, y, h + 0.304)),
    ):
        box(world, "chicago_desk_monitor_" + name, half, pos, (0.12, 0.13, 0.14, 1))
    box(
        world,
        "chicago_desk_display",
        (0.23, 0.005, 0.128),
        (x, y - 0.041, h + 0.304),
        (0.025, 0.10, 0.13, 1),
        collision=False,
    )
    box(
        world,
        "chicago_desk_keyboard",
        (0.23, 0.095, 0.012),
        (x, y - 0.30, h + 0.012),
        (0.12, 0.13, 0.14, 1),
    )
    # Black curtains subdivide the mapped large room; this is only a local bay.
    for x in (-1.65, 2.25):
        box(
            world,
            f"chicago_curtain_foot_{x}",
            (0.16, 0.18, 0.025),
            (x, 1.65, 0.025),
            (0.12, 0.13, 0.13, 1),
        )
        box(
            world,
            f"chicago_curtain_upright_{x}",
            (0.027, 0.027, 1.41),
            (x, 1.65, 1.46),
            (0.26, 0.28, 0.28, 1),
        )
    box(
        world,
        "chicago_curtain_rail",
        (1.98, 0.035, 0.035),
        (0.30, 1.65, 2.89),
        (0.28, 0.29, 0.28, 1),
    )
    for i in range(40):
        box(
            world,
            f"chicago_black_curtain_{i}",
            (0.052, 0.022, 1.30),
            (-1.62 + i * 0.098, 1.65 + 0.013 * math.sin(i), 1.56),
            (0.065, 0.068, 0.07, 1),
            collision=False,
        )
    for x in (-2.50, -1.78):
        box(
            world,
            f"chicago_rack_plinth_{x}",
            (0.32, 0.40, 0.035),
            (x, 1.56, 0.035),
            (0.11, 0.12, 0.12, 1),
        )
        box(
            world,
            f"chicago_instrument_rack_{x}",
            (0.32, 0.40, 1.18),
            (x, 1.56, 1.25),
            (0.15, 0.17, 0.17, 1),
        )
        box(
            world,
            f"chicago_rack_door_{x}",
            (0.285, 0.008, 1.08),
            (x, 1.15, 1.25),
            (0.10, 0.12, 0.12, 1),
        )
        for z in (0.44, 1.0, 1.56, 2.12):
            box(
                world,
                f"chicago_rack_seam_{x}_{z}",
                (0.27, 0.006, 0.008),
                (x, 1.137, z),
                (0.27, 0.29, 0.29, 1),
                collision=False,
            )


BUILDERS["chicago_covered_mineral_optics"] = _chicago_optical
SOURCES["chicago_covered_mineral_optics"] = {
    "reference": "University of Chicago room 520 registered tour photo 69 and official floor plan",
    "url": CHICAGO,
    "dimensions_m": [2.92, 1.90, 2.30],
    "dimension_basis": "All local bay and instrument dimensions estimated. Official room 520 area is 1006 ft² (93.46045824 m²), not the modeled local-bay area or recovered linear dimensions.",
}
SAMPLE_INTERFACES["chicago_covered_mineral_optics"] = (
    "mineral_coupon",
    (0.040, 0.032, 0.020),
    "clamped",
    "inert mineral-shaped coupon fixed to an authored inspection translation stage",
)
FEATURES["chicago_mineralphysics_laser_bay"] = _chicago_features
