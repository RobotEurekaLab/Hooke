"""Photo-informed American laboratory exteriors and mechanical demonstrators.

All lengths are metres. Unknown dimensions and mechanism strokes are authored
estimates; no optics, structural failure, hydrodynamics or electrochemistry solver
is implied. Source photographs are references, never embedded model textures.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


YALE = "https://research.yale.edu/cores/cleanroom"
TEXAS = "https://fsel.engr.utexas.edu/"
BERKELEY = "https://flow-lab-berkeley.netlify.app/physical-model-test-facility"
CMU = "https://www.cmu.edu/me/tpes/facilities/tpes-lab.html"


def _named(geom, name):
    geom.set("name", name)
    return geom


def _tube(b, parent, points, radius=0.004, material="cream"):
    for start, end in zip(points, points[1:]):
        b.rod(parent, start, end, radius, material)


def _inspection_station(b, base, params):
    """Upright stereo head, broad stand and an authored motorized XY fixture."""
    b.box(base, (0, 0, 0.018), (0.25, 0.21, 0.018), "cream", collision=True)
    b.box(base, (0, 0.155, 0.23), (0.052, 0.045, 0.212), "cream", collision=True)
    b.box(base, (0, 0.151, 0.29), (0.021, 0.050, 0.19), "metal")
    for x in (-0.17, 0.17):
        _named(
            b.box(base, (x, -0.035, 0.047), (0.022, 0.14, 0.011), "metal"),
            b.unique("stage_bearing"),
        )
    xstage = b.moving(
        base,
        "stage_x",
        (0, -0.035, 0.066),
        (1, 0, 0),
        (-0.035, 0.035),
        mass=0.3,
        kp=450,
    )
    b.box(xstage, (0, 0, 0), (0.19, 0.14, 0.008), "dark", collision=True)
    for x in (-0.135, 0.135):
        _named(
            b.box(xstage, (x, 0, 0.014), (0.015, 0.12, 0.006), "bright"),
            b.unique("stage_cross_guide"),
        )
    ystage = b.moving(
        xstage, "stage_y", (0, 0, 0.028), (0, 1, 0), (-0.025, 0.025), mass=0.2, kp=400
    )
    b.box(ystage, (0, 0, 0), (0.16, 0.12, 0.008), "cyan", collision=True)
    b.cylinder(ystage, (0, 0, 0.010), 0.045, 0.002, "lens", collision=True)
    # The wafer coupon is visible geometry rigidly held on the XY stage.
    for x in (-0.060, 0.060):
        b.box(ystage, (x, 0, 0.016), (0.014, 0.012, 0.008), "orange")
        b.rod(ystage, (x, 0, 0.027), (x * 0.68, 0, 0.027), 0.004, "bright")
    b.site(ystage, "wafer_coupon", (0, 0, 0.012))
    focus = b.moving(
        base, "focus_height", (0, 0.151, 0.24), (0, 0, 1), (0, 0.025), mass=0.65, kp=650
    )
    b.box(focus, (0, -0.095, 0.032), (0.07, 0.050, 0.035), "cream", collision=True)
    # The guide sleeve slides around the column; its internal bearing surfaces
    # are represented by the prismatic joint rather than overlapping colliders.
    b.box(focus, (0, -0.002, 0.032), (0.063, 0.058, 0.035), "metal")
    head = ET.SubElement(focus, "body", pos="0 -0.172 0", gravcomp="1")
    b.housing(
        head,
        [
            (0.007, 0.060, 0.055, 0),
            (0.08, 0.088, 0.07, 0),
            (0.175, 0.064, 0.046, 0.018),
        ],
        radius=0.024,
        material="cream",
    )
    b.cylinder(head, (0, 0, -0.010), 0.052, 0.018, "dark", collision=True)
    b.cylinder(head, (0, 0, -0.032), 0.043, 0.005, "lens")
    for x in (-0.042, 0.042):
        b.rod(head, (x, -0.01, 0.13), (x, -0.105, 0.205), 0.025, "cream")
        b.rod(head, (x, -0.10, 0.20), (x, -0.135, 0.228), 0.023, "dark")
        b.cylinder(head, (x, -0.137, 0.231), 0.019, 0.005, "lens", euler=(0.90, 0, 0))
    for x in (-0.096, 0.096):
        b.cylinder(head, (x, 0, 0.085), 0.030, 0.014, "dark", euler=(0, math.pi / 2, 0))
    # Orange handling fixture echoes the photographed worktop rather than a
    # conventional inverted microscope or a fictitious commercial model.
    for x in (-0.21, 0.21):
        b.box(base, (x, -0.165, 0.080), (0.017, 0.022, 0.044), "orange")
    b.rod(base, (-0.21, -0.165, 0.108), (0.21, -0.165, 0.108), 0.009, "dark")
    b.metadata["capabilities"] = [
        "XY wafer positioning",
        "inspection head focus-height adjustment",
    ]
    b.metadata["limitations"].append(
        "Motorization and travel are authored stand-ins for the photographed manual fixture; no magnification, wafer metrology or optical transfer function."
    )


def _yale_features(world, definition):
    # Local bay only: the source plan confirms a metrology zone but cannot locate
    # this microscope precisely within that floor plan.
    for i in range(4):
        x = -1.65 + i * 0.77
        box(
            world,
            f"yale_back_panel_{i}",
            (0.018, 0.025, 1.28),
            (x, 1.79, 1.3),
            (0.63, 0.60, 0.42, 1),
        )
    box(
        world,
        "yale_amber_glazing",
        (1.50, 0.012, 0.40),
        (0, 1.765, 1.78),
        (0.78, 0.57, 0.14, 0.28),
        collision=False,
    )
    for i in range(6):
        box(
            world,
            f"yale_filter_slit_{i}",
            (0.29, 0.010, 0.007),
            (0.75, 1.737, 2.26 + i * 0.025),
            (0.26, 0.27, 0.23, 1),
            collision=False,
        )


BUILDERS = {"becton_wafer_inspection": _inspection_station}
SOURCES = {
    "becton_wafer_inspection": {
        "reference": "Yale Becton microscope/handling-fixture photograph; original articulated exterior",
        "url": YALE,
        "dimensions_m": [0.50, 0.42, 0.50],
        "dimension_basis": "Estimated exterior and surrogate travel; source supplies no measured microscope dimensions",
    }
}
SAMPLE_INTERFACES = {
    "becton_wafer_inspection": (
        "wafer_coupon",
        (0.09, 0.09, 0.004),
        "clamped",
        "inert wafer coupon held in an XY inspection fixture",
    )
}
FEATURES = {"yale_becton_cleanroom_metrology": _yale_features}


def _structural_frame(b, base, params):
    """Grounded reaction portal, inert beam and an independent sensor traverse."""
    for x in (-1.45, 1.45):
        b.box(base, (x, 0, 0.075), (0.35, 0.43, 0.075), "guard_red", collision=True)
        b.box(base, (x, 0, 2.15), (0.12, 0.16, 2.0), "guard_red", collision=True)
        for y in (-0.18, 0.18):
            b.box(base, (x, y, 2.15), (0.24, 0.028, 2.0), "guard_red", collision=True)
        for z in (
            0.18,
            0.46,
            0.74,
            1.02,
            1.30,
            1.58,
            1.86,
            2.14,
            2.42,
            2.70,
            2.98,
            3.26,
            3.54,
            3.82,
        ):
            for dx in (-0.16, 0.16):
                b.cylinder(
                    base,
                    (x + dx, -0.211, z),
                    0.014,
                    0.005,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    b.box(base, (0, 0, 4.05), (1.69, 0.20, 0.18), "guard_red", collision=True)
    for z in (3.85, 4.25):
        b.box(base, (0, 0, z), (1.74, 0.28, 0.025), "guard_red", collision=True)
    # The slab sits on a pair of full-depth cribbed pedestals, never on air.
    for y in (-3.05, 3.05):
        for level in range(5):
            z = 0.095 + level * 0.19
            for x in (-0.59, 0.59):
                _named(
                    b.box(
                        base, (x, y, z), (0.11, 0.42, 0.095), "copper", collision=True
                    ),
                    b.unique("beam_crib"),
                )
        _named(
            b.box(base, (0, y, 0.975), (0.84, 0.48, 0.025), "metal", collision=True),
            b.unique("beam_crib_cap"),
        )
    _named(
        b.box(base, (0, 0, 1.20), (0.80, 4.0, 0.20), "cream", collision=True),
        b.unique("inert_concrete_beam"),
    )
    for y in (-3.95, 3.95):
        for x in (-0.63, -0.31, 0, 0.31, 0.63):
            b.cylinder(
                base, (x, y, 1.19), 0.035, 0.055, "dark", euler=(math.pi / 2, 0, 0)
            )
    b.site(base, "beam_specimen", (0, 0, 1.40), size=0.05)
    b.cylinder(base, (0, 0, 3.61), 0.21, 0.42, "orange")
    for z in (3.20, 4.02):
        b.cylinder(base, (0, 0, z), 0.25, 0.035, "dark")
    ram = b.moving(
        base,
        "load_ram",
        (0, 0, 2.20),
        (0, 0, -1),
        (0, 0.70),
        mass=8,
        kp=900,
        force=1600,
    )
    # Piston bearing is a prismatic joint; the shaft overlaps its hollow barrel.
    b.cylinder(ram, (0, 0, 0.92), 0.10, 0.88, "bright")
    b.box(ram, (0, 0, 0), (0.40, 0.38, 0.04), "metal", collision=True)
    b.site(ram, "loading_platen", (0, 0, -0.04), size=0.035)
    _tube(
        b,
        base,
        [(0.2, 0.05, 3.75), (0.75, 0.35, 3.5), (1.5, 0.38, 0.35), (1.8, 0.4, 0.2)],
        0.018,
        "dark",
    )
    # Photo-supported research instrumentation is given an authored rail mount;
    # travel does not imply a measured FSEL sensor system or deformation model.
    for y in (-3.5, 3.5):
        b.box(base, (1.06, y, 0.04), (0.20, 0.25, 0.04), "blue", collision=True)
        b.box(base, (1.06, y, 0.91), (0.045, 0.045, 0.83), "blue", collision=True)
    _named(
        b.box(base, (1.06, 0, 1.765), (0.05, 3.55, 0.025), "metal"),
        b.unique("sensor_rail"),
    )
    probe = b.moving(
        base,
        "sensor_traverse",
        (1.06, 0, 1.81),
        (0, 1, 0),
        (-2.8, 2.8),
        mass=0.7,
        kp=600,
        force=500,
    )
    b.box(probe, (0, 0, 0), (0.095, 0.16, 0.020), "blue", collision=True)
    b.box(probe, (-0.24, 0, 0.065), (0.23, 0.035, 0.025), "bright", collision=True)
    b.box(probe, (-0.43, 0, 0.015), (0.055, 0.055, 0.035), "dark", collision=True)
    b.cylinder(probe, (-0.43, 0, -0.027), 0.025, 0.008, "lens")
    b.site(probe, "measurement_head", (-0.43, 0, -0.035), size=0.015)
    b.metadata["capabilities"] = [
        "loading-platen approach",
        "longitudinal inspection-head traverse",
    ]
    b.metadata["limitations"].append(
        "The beam is rigid and the ram stops above it; no hydraulic pressure, load curve, concrete strain, cracking or structural failure is calculated."
    )


def _texas_features(world, definition):
    steel = (0.24, 0.28, 0.30, 1)
    red = (0.57, 0.16, 0.11, 1)
    for side in (-1, 1):
        x = side * 8.0
        for index, y in enumerate((-10.0, -5.0, 0.0, 5.0, 10.0)):
            box(
                world,
                f"fsel_column_{side}_{index}",
                (0.15, 0.18, 4.15),
                (x, y, 4.15),
                steel,
            )
            box(
                world,
                f"fsel_column_foot_{side}_{index}",
                (0.27, 0.30, 0.08),
                (x, y, 0.08),
                steel,
            )
        box(world, f"fsel_crane_rail_{side}", (0.18, 11.7, 0.18), (x, 0, 7.8), steel)
    box(
        world,
        "fsel_crane_bridge",
        (8.15, 0.33, 0.24),
        (0, 8.0, 8.22),
        (0.91, 0.67, 0.05, 1),
    )
    box(world, "fsel_crane_trolley", (0.6, 0.52, 0.20), (2.3, 8.0, 8.66), steel)
    # A second, lower red portal establishes the multiple-test-bay hall layout.
    for x in (3.2, 5.8):
        box(world, f"fsel_secondary_foot_{x}", (0.30, 0.35, 0.07), (x, 3.0, 0.07), red)
        box(world, f"fsel_secondary_column_{x}", (0.14, 0.16, 1.8), (x, 3.0, 1.94), red)
    box(world, "fsel_secondary_crosshead", (1.48, 0.24, 0.20), (4.5, 3.0, 3.54), red)
    for y in (0.0, 6.0):
        box(
            world,
            f"fsel_secondary_pier_{y}",
            (0.85, 0.42, 0.5),
            (4.5, y, 0.5),
            (0.53, 0.53, 0.50, 1),
        )
    box(
        world,
        "fsel_secondary_beam",
        (0.75, 3.7, 0.20),
        (4.5, 3.0, 1.2),
        (0.67, 0.66, 0.61, 1),
    )
    for i, x in enumerate(range(-7, 8)):
        for j, y in enumerate(range(-11, 12, 2)):
            box(
                world,
                f"fsel_anchor_{i}_{j}",
                (0.028, 0.028, 0.001),
                (x, y, 0.001),
                (0.22, 0.23, 0.22, 1),
                collision=False,
            )


BUILDERS["ferguson_reaction_frame"] = _structural_frame
SOURCES["ferguson_reaction_frame"] = {
    "reference": "Ferguson Structural Engineering Laboratory interior photograph",
    "url": TEXAS,
    "dimensions_m": [3.8, 8.1, 4.3],
    "dimension_basis": "Estimated portal, slab and instrumentation envelope; no surveyed dimensions available",
}
SAMPLE_INTERFACES["ferguson_reaction_frame"] = (
    "beam_specimen",
    (1.6, 8.0, 0.4),
    "clamped",
    "rigid inert beam surrogate supported on cribbed pedestals",
)
FEATURES["texas_ferguson_structural_test_hall"] = _texas_features


def _towing_tank(b, base, params):
    """Published 64 x 2.4 x 1.8 m clear tank with an inert mounted test model."""
    _named(
        b.box(base, (0, 0, 0.06), (1.4, 32.2, 0.06), "dark", collision=True),
        b.unique("tank_foundation"),
    )
    for x in (-1.3, 1.3):
        _named(
            b.box(base, (x, 0, 1.02), (0.10, 32.2, 0.90), "metal", collision=True),
            b.unique("tank_sidewall"),
        )
        _named(
            b.box(base, (x, 0, 1.945), (0.08, 32.2, 0.025), "bright"),
            b.unique("carriage_rail"),
        )
        for y in range(-31, 32, 2):
            b.cylinder(base, (x, y, 1.976), 0.018, 0.007, "dark")
    for y in (-32.1, 32.1):
        _named(
            b.box(base, (0, y, 1.02), (1.2, 0.10, 0.90), "metal", collision=True),
            b.unique("tank_endwall"),
        )
    # A transparent surface marks the observed water, without fluid forces.
    b.box(
        base,
        (0, 0, 1.67),
        (1.198, 31.995, 0.003),
        "glass",
        rgba=(0.10, 0.28, 0.19, 0.34),
    )
    # The floor datum is authored. A solid service platform brings the rail to
    # operator waist height while retaining the source-published tank depth.
    _named(
        b.box(base, (2.20, 0, 0.425), (0.8, 32.2, 0.425), "cream", collision=True),
        b.unique("service_walkway"),
    )
    for i in range(4):
        height = 0.17 * (i + 1)
        b.box(
            base,
            (2.2, -33.35 + i * 0.29, height / 2),
            (0.65, 0.145, height / 2),
            "cream",
            collision=True,
        )
    carriage = b.moving(
        base,
        "carriage_travel",
        (0, -22, 2.22),
        (0, 1, 0),
        (0, 8.0),
        mass=5,
        kp=1100,
        force=1200,
    )
    b.box(carriage, (0, 0, 0), (1.49, 0.43, 0.05), "metal", collision=True)
    for x in (-1.3, 1.3):
        for y in (-0.28, 0.28):
            _named(
                b.cylinder(
                    carriage,
                    (x, y, -0.15),
                    0.10,
                    0.062,
                    "dark",
                    euler=(0, math.pi / 2, 0),
                ),
                b.unique("carriage_wheel"),
            )
        b.box(carriage, (x, 0, 0.28), (0.025, 0.025, 0.23), "warning")
    b.rod(carriage, (-1.3, 0, 0.51), (1.3, 0, 0.51), 0.025, "warning")
    for x in (-0.16, 0.16):
        b.rod(carriage, (x, 0, -0.24), (x, 0, 0.36), 0.014, "bright")
    b.box(carriage, (0, 0, 0.36), (0.20, 0.12, 0.025), "blue")
    specimen = b.moving(
        carriage,
        "model_immersion",
        (0, 0, -0.32),
        (0, 0, -1),
        (0, 0.35),
        mass=0.4,
        kp=600,
    )
    b.geom(specimen, "ellipsoid", (0.14, 0.55, 0.12), material="orange", collision=True)
    b.rod(specimen, (0, 0, 0.11), (0, 0, 0.94), 0.018, "bright")
    b.box(specimen, (0, 0, 0.70), (0.055, 0.055, 0.035), "dark")
    b.site(specimen, "tow_model", (0, 0, 0.12), size=0.03)
    b.site(carriage, "carriage_handle", (0, 0, 0.51), size=0.035)
    # Shallow absorber ramp is an inferred end detail, not a wave generator.
    b.box(base, (0, 30.1, 0.42), (1.15, 1.70, 0.06), "dark", euler=(0.17, 0, 0))
    b.metadata["capabilities"] = [
        "8 m representative carriage positioning window",
        "mounted-model immersion adjustment",
    ]
    b.metadata["limitations"].append(
        "Tank clear dimensions are published; carriage shape, 8 m demonstration stroke, mounting hardware and raised service datum are estimates. Water is visual only, with no buoyancy, drag or wave propagation."
    )


def _berkeley_features(world, definition):
    steel = (0.27, 0.29, 0.28, 1)
    for i, y in enumerate(range(-32, 33, 4)):
        for x in (-3.1, 3.1):
            box(
                world,
                f"richmond_column_{i}_{x}",
                (0.060, 0.075, 2.55),
                (x, y, 2.55),
                steel,
            )
        # Pitched portal rafters follow the long corrugated shed reference.
        for side in (-1, 1):
            box(
                world,
                f"arch_roof_richmond_rafter_{i}_{side}",
                (1.62, 0.065, 0.075),
                (side * 1.55, y, 5.28),
                steel,
                euler=f"0 {side * 0.20} 0",
            )
    for i in range(69):
        y = -34 + i
        box(
            world,
            f"richmond_wall_rib_{i}",
            (0.035, 0.018, 2.4),
            (-3.22, y, 2.4),
            (0.45, 0.46, 0.43, 1),
            collision=False,
        )
    # Repeated short sagging spans represent the visible carriage cable festoon.
    for i in range(16):
        y = -30 + i * 4
        for j in range(6):
            a, c = j / 6, (j + 1) / 6
            start = (-2.90, y + 4 * a, 4.25 - 0.8 * math.sin(math.pi * a))
            end = (-2.90, y + 4 * c, 4.25 - 0.8 * math.sin(math.pi * c))
            ET.SubElement(
                world,
                "geom",
                name=f"richmond_festoon_{i}_{j}",
                type="capsule",
                fromto=" ".join(map(str, start + end)),
                size="0.012",
                rgba=".06 .07 .07 1",
                contype="0",
                conaffinity="0",
            )


BUILDERS["richmond_towing_carriage"] = _towing_tank
SOURCES["richmond_towing_carriage"] = {
    "reference": "Berkeley FLOW Lab Richmond towing tank photograph and published tank dimensions",
    "url": BERKELEY,
    "dimensions_m": [4.5, 65.8, 2.9],
    "dimension_basis": "Tank clear length/width/depth 64/2.40/1.80 m published; assembly envelope, raised walkway, carriage and travel estimated",
}
SAMPLE_INTERFACES["richmond_towing_carriage"] = (
    "tow_model",
    (0.28, 1.10, 0.24),
    "clamped",
    "inert ellipsoidal model held by a rigid vertical mounting sting",
)
FEATURES["berkeley_richmond_towing_tank"] = _berkeley_features


def _electrochemical_station(b, base, params):
    """Recognizable 850e/850BP exteriors around an inert coupon fixture."""
    for x, y, width, depth, height in (
        (0.43, 0.10, 0.43, 0.48, 0.79),
        (-0.47, 0.10, 0.40, 0.38, 0.40),
    ):
        for dx in (-width * 0.38, width * 0.38):
            for dy in (-depth * 0.37, depth * 0.37):
                b.cylinder(
                    base,
                    (x + dx, y + dy, 0.018),
                    0.018,
                    0.018,
                    "rubber",
                    collision=True,
                )
        b.box(
            base,
            (x, y, 0.036 + height / 2),
            (width / 2, depth / 2, height / 2),
            "cream",
            collision=True,
        )
        b.box(
            base,
            (x, y - depth / 2 - 0.007, 0.036 + height / 2),
            (width / 2 - 0.01, 0.007, height / 2 - 0.009),
            "shell",
        )
    # Tall tester front: screen, stacked controller faces, emergency stop and
    # large electrical terminals reproduce visible exterior cues, not firmware.
    b.screen(base, (0.48, -0.156, 0.68), 0.14, 0.055)
    for z in (0.53, 0.41):
        b.screen(base, (0.50, -0.157, z), 0.11, 0.069)
    b.cylinder(
        base, (0.49, -0.165, 0.29), 0.029, 0.012, "guard_red", euler=(math.pi / 2, 0, 0)
    )
    b.ring(base, (0.49, -0.179, 0.29), 0.037, 0.008, "warning", "xz")
    for x, mat in ((0.33, "warning"), (0.43, "blue"), (0.57, "guard_red")):
        b.cylinder(
            base, (x, -0.166, 0.14), 0.015, 0.012, mat, euler=(math.pi / 2, 0, 0)
        )
    b.box(base, (0.43, 0.08, 0.91), (0.23, 0.20, 0.084), "shell", collision=True)
    b.vents(base, (0.43, -0.123, 0.865), 0.38, rows=6)
    b.box(base, (0.44, -0.129, 0.962), (0.09, 0.002, 0.003), "dark")
    # Left pressure module, with two large dials and two smaller regulators.
    for x in (-0.57, -0.36):
        for z, radius in ((0.31, 0.046), (0.17, 0.030)):
            b.cylinder(
                base, (x, -0.106, z), radius, 0.009, "bright", euler=(math.pi / 2, 0, 0)
            )
            b.cylinder(
                base,
                (x, -0.117, z),
                radius * 0.85,
                0.002,
                "cream",
                euler=(math.pi / 2, 0, 0),
            )
            b.rod(
                base,
                (x, -0.121, z),
                (x + radius * 0.42, -0.121, z + radius * 0.37),
                0.002,
                "dark",
            )
        b.cylinder(
            base, (x, -0.113, 0.09), 0.022, 0.019, "dark", euler=(math.pi / 2, 0, 0)
        )
    b.box(base, (-0.73, 0.23, 0.10), (0.13, 0.15, 0.07), "blue", collision=True)
    b.box(base, (-0.73, 0.23, 0.015), (0.10, 0.12, 0.015), "dark", collision=True)
    b.screen(base, (-0.73, 0.071, 0.11), 0.16, 0.047)
    # Compact drawer and lifting compression plate: representative dry fixture,
    # not a claim that the photographed setup contains these exact mechanisms.
    b.box(base, (-0.05, -0.46, 0.012), (0.25, 0.24, 0.012), "metal", collision=True)
    for x in (-0.23, 0.13):
        _named(
            b.box(base, (x, -0.48, 0.040), (0.025, 0.26, 0.016), "bright"),
            b.unique("drawer_guide"),
        )
    drawer = b.moving(
        base,
        "specimen_drawer",
        (-0.05, -0.40, 0.070),
        (0, -1, 0),
        (0, 0.13),
        mass=0.45,
        kp=500,
    )
    b.box(drawer, (0, 0, 0), (0.22, 0.14, 0.014), "dark", collision=True)
    b.box(drawer, (0, 0, 0.020), (0.075, 0.065, 0.006), "sample", collision=True)
    b.site(drawer, "electrode_coupon", (0, 0, 0.026))
    for x in (-0.095, 0.095):
        for y in (-0.085, 0.085):
            b.cylinder(drawer, (x, y, 0.062), 0.006, 0.050, "bright")
            b.cylinder(drawer, (x, y, 0.116), 0.012, 0.005, "dark")
    clamp = b.moving(
        drawer, "clamp_lift", (0, 0, 0.040), (0, 0, 1), (0, 0.045), mass=0.15, kp=350
    )
    # A perimeter clamp keeps the inert coupon visible through its central gap.
    for x in (-0.086, 0.086):
        b.box(clamp, (x, 0, 0), (0.026, 0.10, 0.014), "bright", collision=True)
        b.box(clamp, (x, 0, 0.017), (0.015, 0.060, 0.003), "dark")
    for y in (-0.075, 0.075):
        b.box(clamp, (0, y, 0), (0.060, 0.025, 0.014), "bright", collision=True)
    b.site(clamp, "clamp_handle", (0, -0.10, 0))
    b.rod(drawer, (-0.09, -0.17, 0), (0.09, -0.17, 0), 0.009, "metal")
    for i in range(3):
        _tube(
            b,
            base,
            [
                (-0.56 + i * 0.045, -0.11, 0.40),
                (-0.75 + i * 0.03, -0.35, 0.26),
                (-0.56 + i * 0.03, -0.54, 0.045),
            ],
            0.0035,
        )
    _tube(
        b,
        base,
        [(0.57, -0.17, 0.14), (0.78, -0.44, 0.08), (0.37, -0.64, 0.05)],
        0.006,
        "guard_red",
    )
    b.metadata["capabilities"] = [
        "dry coupon drawer positioning",
        "compression-fixture opening",
    ]
    b.metadata["limitations"].append(
        "850e/850BP external styling follows the photograph, but dimensions and dry-fixture actuation are estimates. No gases, current, pressure, electrochemistry or reaction products are simulated."
    )


def _cmu_features(world, definition):
    wood, steel = (0.65, 0.51, 0.32, 1), (0.68, 0.72, 0.73, 1)
    # Sink island has a recessed basin, not a black rectangle on a solid top.
    box(world, "cmu_sink_cabinet", (0.79, 0.54, 0.40), (-1.1, -0.9, 0.4), wood)
    for x in (-1.64, -0.56):
        box(
            world,
            f"cmu_sink_top_side_{x}",
            (0.26, 0.56, 0.035),
            (x, -0.9, 0.865),
            (0.23, 0.25, 0.25, 1),
        )
    for y in (-1.35, -0.45):
        box(
            world,
            f"cmu_sink_top_end_{y}",
            (0.28, 0.11, 0.035),
            (-1.1, y, 0.865),
            (0.23, 0.25, 0.25, 1),
        )
    box(
        world,
        "cmu_sink_basin_floor",
        (0.28, 0.34, 0.01),
        (-1.1, -0.9, 0.81),
        (0.12, 0.14, 0.15, 1),
    )
    for i, (a, c) in enumerate(
        (
            ((-1.18, -0.47, 0.90), (-1.18, -0.47, 1.12)),
            ((-1.18, -0.47, 1.12), (-1.18, -0.76, 1.12)),
            ((-1.18, -0.76, 1.12), (-1.18, -0.76, 1.07)),
        )
    ):
        ET.SubElement(
            world,
            "geom",
            name=f"cmu_sink_tap_{i}",
            type="capsule",
            fromto=" ".join(map(str, a + c)),
            size=".012",
            rgba=".65 .70 .73 1",
            contype="0",
            conaffinity="0",
        )
    box(
        world,
        "cmu_sink_spine",
        (0.72, 0.035, 0.52),
        (-1.1, -0.39, 1.38),
        (0.90, 0.90, 0.86, 1),
    )
    box(world, "cmu_drying_panel", (0.46, 0.013, 0.34), (-1.1, -0.432, 1.49), steel)
    for row in range(3):
        for col in range(5):
            x, z = -1.46 + col * 0.18, 1.28 + row * 0.19
            ET.SubElement(
                world,
                "geom",
                name=f"cmu_drying_peg_{row}_{col}",
                type="capsule",
                fromto=f"{x} -.447 {z} {x} -.59 {z + .065}",
                size=".009",
                rgba=".88 .88 .84 1",
                contype="0",
                conaffinity="0",
            )
    for i, x in enumerate((1.72, 2.09, 2.46)):
        ET.SubElement(
            world,
            "geom",
            name=f"cmu_cylinder_{i}",
            type="cylinder",
            pos=f"{x} 1.04 .66",
            size=".145 .66",
            rgba=".35 .42 .40 1",
        )
        ET.SubElement(
            world,
            "geom",
            name=f"cmu_cylinder_shoulder_{i}",
            type="sphere",
            pos=f"{x} 1.04 1.32",
            size=".145",
            rgba=".35 .42 .40 1",
        )
        box(
            world,
            f"cmu_cylinder_valve_{i}",
            (0.025, 0.065, 0.035),
            (x, 1.04, 1.48),
            (0.62, 0.46, 0.18, 1),
        )
    for x in (1.49, 2.69):
        box(world, f"cmu_rack_post_{x}", (0.023, 0.025, 0.7), (x, 0.84, 0.7), steel)
        box(world, f"cmu_rack_foot_{x}", (0.09, 0.22, 0.016), (x, 1.04, 0.016), steel)
    for z in (0.24, 0.85):
        box(
            world,
            f"cmu_cylinder_restraint_{z}",
            (0.60, 0.018, 0.018),
            (2.09, 0.84, z),
            steel,
        )
    for x in (-1.20, 1.20):
        box(
            world,
            f"cmu_bench_spine_{x}",
            (0.025, 0.028, 0.52),
            (x, 1.55, 1.42),
            (0.87, 0.87, 0.83, 1),
        )
    box(
        world,
        "cmu_overhead_utility_shelf",
        (1.36, 0.20, 0.025),
        (0, 1.40, 1.965),
        (0.87, 0.87, 0.83, 1),
    )


BUILDERS["edl_fuel_cell_testbench"] = _electrochemical_station
SOURCES["edl_fuel_cell_testbench"] = {
    "reference": "CMU EDL official lab photos with visible 850e and 850BP exteriors",
    "url": CMU,
    "dimensions_m": [1.85, 1.15, 1.0],
    "dimension_basis": "Estimated from laboratory photographs; no OEM dimensions or exact specimen-fixture CAD verified",
}
SAMPLE_INTERFACES["edl_fuel_cell_testbench"] = (
    "electrode_coupon",
    (0.15, 0.13, 0.012),
    "clamped",
    "inert dry rectangular coupon held by a mechanical fixture",
)
FEATURES["cmu_edl_electrochemical_testing"] = _cmu_features
