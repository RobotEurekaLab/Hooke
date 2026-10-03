"""Four photographed engineering workstations with bounded dry mechanics.

Exterior geometry is independently authored from the cited university images.
Unpublished dimensions are estimates. None of these mechanisms implements the
associated aerodynamic, constitutive or manufacturing process.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

AUCKLAND = "https://www.auckland.ac.nz/en/engineering/our-research/engineering-research/research-areas-and-facilities/wind-tunnel-hall.html"
KFUPM = (
    "https://ri.kfupm.edu.sa/irc-cbm/facilities/facility-details/building-materials-lab"
)
WHITTLE = "https://www-g.eng.cam.ac.uk/whittle/rigs/rigs.html"
BIRMINGHAM = "https://www.birmingham.ac.uk/research/centres-institutes/research-in-mechanical-engineering/advanced-manufacturing-research-group/facilities"


def _named(geom, b, tag):
    geom.set("name", b.unique(tag))
    return geom


def _palette(b, key, rgba):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=b.name + "__mat_" + key,
        rgba=numbers(rgba),
        specular="0.35",
        shininess="0.45",
    )


def _tunnel(b, base, params):
    # Only a six-metre local segment of the published twenty-metre test section.
    # Its internal width and clear height preserve the published 3.6 x 2.5 m.
    _named(
        b.box(base, (0, 0, 0.20), (3, 1.8, 0.20), "shell", collision=True),
        b,
        "test_floor",
    )
    _named(
        b.box(base, (0, 0, 2.94), (3, 1.8, 0.04), "shell", collision=True),
        b,
        "test_roof",
    )
    for y in (-1.806, 1.806):
        _named(
            b.box(base, (0, y, 1.65), (3, 0.006, 1.25), "glass", collision=True),
            b,
            "test_side",
        )
        for x in (-3, -1.5, 0, 1.5, 3):
            b.box(
                base, (x, y * 1.025, 1.65), (0.035, 0.035, 1.30), "dark", collision=True
            )
        for z in (0.38, 2.94):
            b.box(base, (0, y * 1.025, z), (3.04, 0.04, 0.035), "dark")
    b.box(base, (0, 0, 0.445), (0.85, 0.47, 0.035), "dark", collision=True)
    for y in (-0.39, 0.39):
        b.rod(base, (-0.75, y, 0.50), (0.75, y, 0.50), 0.016, "bright")
    carriage = b.moving(
        base,
        "specimen_x",
        (0, 0, 0.53),
        (1, 0, 0),
        (-0.22, 0.22),
        mass=4,
        kp=1200,
        force=1600,
    )
    b.box(carriage, (0, 0, 0), (0.46, 0.43, 0.035), "metal", collision=True)
    turn = b.moving(
        carriage,
        "specimen_yaw",
        (0, 0, 0.075),
        (0, 0, 1),
        (-0.30, 0.30),
        kind="hinge",
        mass=2,
        kp=400,
        force=500,
    )
    b.cylinder(turn, (0, 0, 0), 0.42, 0.035, "bright", collision=True)
    b.cylinder(turn, (0, 0, 0.12), 0.085, 0.085, "dark", collision=True)
    # Clear foil and circular end plate are the distinctive photographed article.
    _named(
        b.box(turn, (0, 0, 0.92), (0.045, 0.28, 0.74), "glass", collision=True),
        b,
        "clear_foil",
    )
    b.box(turn, (0.047, 0, 0.92), (0.005, 0.02, 0.74), "bright")
    b.cylinder(turn, (0, 0, 1.68), 0.48, 0.012, "glass", collision=True)
    b.ring(turn, (0, 0, 1.68), 0.48, 0.005, "bright")
    b.site(turn, "clear_specimen", (0, 0, 0.92))
    b.metadata["capabilities"] = [
        "specimen_translation",
        "specimen_yaw",
        "visible_clear_calibration_foil",
    ]
    b.metadata["limitations"].append(
        "Only a 6 m local segment is authored. The published full test section is 20 x 3.6 x 2.5 m; no return circuit, boundary layer, loads or aerodynamic similarity is simulated. The two-axis fixture is an authored mechanical aid."
    )


def _wedge_grip(b, parent, z, upper=False):
    sign = -1 if upper else 1
    b.box(parent, (0, 0, z), (0.23, 0.15, 0.17), "metal", collision=True)
    b.box(parent, (0, -0.155, z), (0.13, 0.008, 0.085), "dark")
    for x in (-0.17, 0.17):
        for dz in (-0.11, 0.11):
            b.cylinder(
                parent,
                (x, -0.169, z + dz),
                0.012,
                0.008,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
    for x in (-0.055, 0.055):
        b.box(parent, (x, 0, z + sign * 0.16), (0.05, 0.10, 0.045), "dark")
    b.cylinder(
        parent, (-0.095, -0.19, z), 0.070, 0.020, "bright", euler=(math.pi / 2, 0, 0)
    )
    b.ring(parent, (-0.095, -0.213, z), 0.055, 0.007, "dark", plane="xz")
    b.rod(parent, (-0.15, -0.221, z), (-0.04, -0.221, z), 0.008, "metal")
    b.rod(
        parent,
        (0.03, -0.195, z - sign * 0.08),
        (0.18, -0.195, z - sign * 0.08),
        0.014,
        "bright",
    )


def _test_frame(b, base, params):
    b.box(base, (0, 0, 0.28), (0.60, 0.43, 0.28), "dark", collision=True)
    b.box(base, (0, 0, 0.59), (0.60, 0.43, 0.03), "metal", collision=True)
    for x in (-0.44, 0.44):
        b.cylinder(base, (x, 0, 1.48), 0.065, 0.86, "bright", collision=True)
        for z in (0.65, 2.3):
            b.cylinder(base, (x, 0, z), 0.095, 0.04, "dark")
    b.box(base, (0, 0, 2.38), (0.59, 0.25, 0.12), "dark", collision=True)
    upper = b.moving(
        base,
        "upper_crosshead_z",
        (0, 0, 2.08),
        (0, 0, 1),
        (-0.09, 0.09),
        mass=8,
        kp=2400,
        force=4000,
    )
    b.box(upper, (0, 0, 0), (0.30, 0.20, 0.075), "metal", collision=True)
    for x in (-0.35, 0.35):
        b.rod(upper, (x, 0, 0), (x * 0.7, 0, 0), 0.035, "metal")
    b.cylinder(upper, (0, 0, -0.14), 0.075, 0.065, "bright", collision=True)
    _wedge_grip(b, upper, -0.34, upper=True)
    # Open guide sleeve admits a real moving stem without a solid overlap.
    for x in (-0.058, 0.058):
        b.box(base, (x, 0, 0.72), (0.010, 0.070, 0.08), "bright", collision=True)
    for y in (-0.058, 0.058):
        b.box(base, (0, y, 0.72), (0.048, 0.010, 0.08), "bright", collision=True)
    lower = b.moving(
        base,
        "lower_grip_z",
        (0, 0, 1.02),
        (0, 0, 1),
        (-0.035, 0.035),
        mass=3,
        kp=1200,
        force=2000,
    )
    b.cylinder(lower, (0, 0, -0.245), 0.04, 0.075, "bright", collision=True)
    _wedge_grip(b, lower, 0)
    # A retained metal strip is visible in the lower grip, with an open upper gap.
    b.box(lower, (0, 0, 0.235), (0.025, 0.008, 0.075), "bright", collision=True)
    b.box(lower, (0, -0.009, 0.26), (0.020, 0.001, 0.002), "dark")
    b.site(lower, "inert_coupon", (0, 0, 0.235))
    b.metadata["capabilities"] = [
        "upper_crosshead_approach",
        "lower_grip_position",
        "retained_visible_coupon",
    ]
    b.metadata["limitations"].append(
        "The two bounded vertical motions are original inspection surrogates. No certified load, true grip actuation, material strain, tensile contact or failure behavior is established."
    )


def _annular_band(b, parent, y, length, radius, material):
    for i in range(48):
        a = i * math.tau / 48
        b.box(
            parent,
            (radius * math.sin(a), y, 1.90 + radius * math.cos(a)),
            (radius * math.tan(math.pi / 48), length / 2, 0.035),
            material,
            euler=(0, a, 0),
            collision=True,
        )
    for end in (y - length / 2, y + length / 2):
        b.ring(
            parent,
            (0, end, 1.90),
            radius + 0.045,
            0.045,
            material,
            plane="xz",
            segments=48,
        )


def _turbine(b, base, params):
    _palette(b, "teal", (0.035, 0.40, 0.40, 1))
    for y, length, radius, material in [
        (-1.40, 1.40, 1.5, "teal"),
        (-0.43, 0.5, 1.25, "teal"),
        (0, 0.32, 1.1, "dark"),
        (0.28, 0.22, 1.08, "guard_red"),
        (0.59, 0.40, 1.06, "bright"),
        (1.24, 0.90, 0.86, "teal"),
    ]:
        _annular_band(b, base, y, length, radius, material)
    # A mesh-like inlet screen and three support spokes, rather than an invented rotor.
    for i in range(-20, 21):
        t = i * 0.07
        extent = math.sqrt(1.44**2 - t**2)
        b.rod(
            base,
            (t, -2.115, 1.90 - extent),
            (t, -2.115, 1.90 + extent),
            0.0028,
            "metal",
        )
        b.rod(
            base,
            (-extent, -2.117, 1.90 + t),
            (extent, -2.117, 1.90 + t),
            0.0028,
            "metal",
        )
    b.cylinder(base, (0, -2.17, 1.90), 0.12, 0.06, "copper", euler=(math.pi / 2, 0, 0))
    for a in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6):
        b.rod(
            base,
            (0, -2.18, 1.90),
            (1.44 * math.cos(a), -2.18, 1.90 + 1.44 * math.sin(a)),
            0.022,
            "dark",
        )
    for y, width, height in [
        (-1.4, 0.82, 0.62),
        (0.20, 0.65, 0.89),
        (1.24, 0.48, 1.12),
    ]:
        for x in (-width, width):
            b.box(base, (x, y, 0.035), (0.19, 0.23, 0.035), "metal", collision=True)
            b.box(
                base,
                (x, y, height / 2),
                (0.055, 0.055, height / 2),
                "metal",
                collision=True,
            )
            b.rod(base, (x, y, 0.15), (-x, y, height), 0.025, "metal")
        b.box(base, (0, y, height), (width + 0.07, 0.10, 0.07), "metal", collision=True)
    # External, floor-supported dry traverse and clamped reference target.
    for y in (0.12, 1.08):
        for x in (1.90, 2.38):
            b.box(base, (x, y, 0.66), (0.035, 0.035, 0.66), "metal", collision=True)
    b.box(base, (2.14, 0.60, 1.35), (0.29, 0.56, 0.03), "metal", collision=True)
    for x in (1.97, 2.31):
        b.rod(base, (x, 0.12, 1.42), (x, 1.08, 1.42), 0.013, "bright")
    slide = b.moving(
        base,
        "probe_y",
        (2.14, 0.60, 1.46),
        (0, 1, 0),
        (-0.23, 0.23),
        mass=1,
        kp=400,
        force=800,
    )
    b.box(slide, (0, 0, 0), (0.20, 0.15, 0.025), "dark", collision=True)
    b.box(slide, (0.17, 0, 0.26), (0.025, 0.08, 0.24), "metal", collision=True)
    probe = b.moving(
        slide,
        "probe_z",
        (0, 0, 0.27),
        (0, 0, 1),
        (-0.05, 0.06),
        mass=0.25,
        kp=200,
        force=300,
    )
    b.box(probe, (0.09, 0, 0), (0.065, 0.045, 0.035), "metal")
    b.rod(probe, (0, 0, 0), (0, 0, -0.10), 0.004, "bright", collision=True)
    b.box(slide, (0, 0, 0.07), (0.045, 0.035, 0.04), "cream", collision=True)
    b.site(slide, "dry_reference", (0, 0, 0.07))
    b.metadata["capabilities"] = [
        "external_probe_traverse",
        "external_probe_height",
        "visible_retained_reference",
    ]
    b.metadata["limitations"].append(
        "Historical visible rig only. No hidden rotor or current hall survey is inferred. The two-axis external dry reference traverse is an explicitly authored task aid; no turbine flow, pressure or efficiency output."
    )


def _wire_machine(b, base, params):
    # A real opening, cream front frame and visible paired wire-guide wheels.
    b.box(base, (0, 0, 0.39), (0.66, 0.52, 0.39), "cream", collision=True)
    b.box(base, (0, 0.49, 1.30), (0.66, 0.06, 0.52), "cream", collision=True)
    b.box(base, (0, 0, 1.86), (0.70, 0.56, 0.055), "cream", collision=True)
    for x in (-0.635, 0.635):
        b.box(base, (x, 0, 1.32), (0.045, 0.50, 0.54), "cream", collision=True)
        b.box(base, (x * 0.90, -0.51, 1.31), (0.014, 0.014, 0.51), "dark")
    for x in (-0.47, 0.47):
        b.box(base, (x, 0.23, 1.39), (0.09, 0.03, 0.32), "metal")
        for z in (1.28, 1.68):
            b.cylinder(
                base, (x, 0.15, z), 0.085, 0.035, "dark", euler=(math.pi / 2, 0, 0)
            )
            b.cylinder(
                base, (x, 0.11, z), 0.055, 0.008, "bright", euler=(math.pi / 2, 0, 0)
            )
    for x in (-0.58, 0.58):
        for z in (1.15, 1.44, 1.72):
            b.cylinder(
                base, (x, -0.526, z), 0.017, 0.009, "dark", euler=(math.pi / 2, 0, 0)
            )
    for x in (-0.22, 0.22):
        b.box(base, (x, 0.28, 1.42), (0.08, 0.018, 0.30), "bright")
        for z in (1.22, 1.30, 1.38, 1.46, 1.54, 1.62):
            b.box(base, (x, 0.258, z), (0.07, 0.002, 0.004), "dark")
    # Passive wire facade; no electrical discharge or cutting behavior.
    b.rod(base, (0, 0.08, 1.63), (0, 0.08, 1.13), 0.001, "bright")
    b.box(base, (0, 0, 0.82), (0.50, 0.39, 0.035), "metal", collision=True)
    for y in (-0.27, 0.27):
        b.rod(base, (-0.42, y, 0.885), (0.42, y, 0.885), 0.012, "bright")
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
    b.box(xstage, (0, 0, 0), (0.24, 0.30, 0.025), "metal", collision=True)
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
    b.box(ystage, (0, 0, 0), (0.19, 0.15, 0.025), "dark", collision=True)
    for x in (-0.10, 0.10):
        b.box(ystage, (x, 0, 0.055), (0.025, 0.075, 0.030), "metal")
    b.box(ystage, (0, 0, 0.055), (0.075, 0.06, 0.025), "bright", collision=True)
    b.site(ystage, "inert_coupon", (0, 0, 0.055))
    # Inclined right-hand controller has a real supporting pedestal.
    b.box(base, (0.92, -0.24, 0.47), (0.10, 0.13, 0.47), "cream", collision=True)
    b.box(base, (0.92, -0.24, 0.95), (0.32, 0.27, 0.045), "cream", collision=True)
    b.box(base, (0.92, -0.04, 1.20), (0.32, 0.04, 0.23), "dark", euler=(-0.30, 0, 0))
    b.box(
        base, (0.92, -0.090, 1.22), (0.22, 0.006, 0.13), "screen", euler=(-0.30, 0, 0)
    )
    for i in range(6):
        b.box(base, (0.70 + i * 0.08, -0.35, 1.006), (0.022, 0.034, 0.008), "dark")
    b.cylinder(base, (1.12, -0.25, 1.02), 0.020, 0.018, "guard_red")
    b.metadata["capabilities"] = [
        "fixture_x_translation",
        "fixture_y_translation",
        "visible_retained_metal_coupon",
    ]
    b.metadata["limitations"].append(
        "Original dry XY fixture surrogate; OEM model, controller semantics, discharge, dielectric flow and material removal are unverified. The wire is a passive visual feature above the target."
    )


def _records_wall(world, definition):
    _display(world, "kfupm_control", 1.4, 1.4, 0.77)
    for i in range(5):
        x = -1.4 + i * 0.55
        box(
            world,
            f"records_frame_{i}",
            (0.22, 0.018, 0.30),
            (x, 2.97, 1.90),
            (0.46, 0.33, 0.17, 1),
        )
        box(
            world,
            f"records_sheet_{i}",
            (0.19, 0.010, 0.27),
            (x, 2.947, 1.90),
            (0.92, 0.9, 0.80, 1),
            collision=False,
        )
        for j in range(4):
            box(
                world,
                f"records_line_{i}_{j}",
                (0.13, 0.002, 0.003),
                (x, 2.933, 1.72 + j * 0.08),
                (0.19, 0.23, 0.21, 1),
                collision=False,
            )


def _whittle_steps(world, definition):
    for i in range(4):
        y = -0.50 + i * 0.25
        h = 0.18 * (i + 1)
        box(
            world,
            f"whittle_step_{i}",
            (0.35, 0.13, 0.025),
            (-2.05, y, h),
            (0.55, 0.57, 0.54, 1),
        )
        for x in (-2.36, -1.74):
            box(
                world,
                f"whittle_step_leg_{i}_{x}",
                (0.02, 0.02, h / 2),
                (x, y, h / 2),
                (0.43, 0.47, 0.48, 1),
            )
    # Repeated overhead service sections stay attached to the real room roof.
    for y in (-2.4, 0, 2.4):
        box(
            world,
            f"arch_roof_whittle_hanger_{y}",
            (0.025, 0.025, 0.22),
            (2.9, y, 4.76),
            (0.5, 0.53, 0.54, 1),
        )
    box(
        world,
        "arch_roof_whittle_duct",
        (0.35, 3.25, 0.25),
        (2.9, 0, 4.35),
        (0.65, 0.67, 0.68, 1),
    )


def _edm_room(world, definition):
    _display(world, "edm_computer", -1.42, 0.20, 0.76)
    # The secondary tool is only the partially visible casing from the reference.
    box(
        world,
        "edm_secondary_base",
        (0.48, 0.40, 0.30),
        (1.12, -1.30, 0.30),
        (0.66, 0.67, 0.62, 1),
    )
    box(
        world,
        "edm_secondary_casing",
        (0.44, 0.36, 0.60),
        (1.12, -1.30, 1.20),
        (0.86, 0.88, 0.85, 1),
    )
    box(
        world,
        "edm_secondary_panel",
        (0.16, 0.020, 0.13),
        (1.12, -1.67, 1.42),
        (0.08, 0.11, 0.12, 1),
    )
    for z in (1.30, 1.48):
        box(
            world,
            f"arch_cutaway_wall_edm_conduit_{z}",
            (0.012, 1.85, 0.012),
            (1.97, 0, z),
            (0.62, 0.62, 0.57, 1),
            collision=False,
        )
    for y in (-1.4, -0.3, 0.8, 1.6):
        box(
            world,
            f"arch_cutaway_wall_edm_clip_{y}",
            (0.017, 0.016, 0.13),
            (1.955, y, 1.39),
            (0.30, 0.33, 0.32, 1),
            collision=False,
        )


def _display(world, name, x, y, z):
    """Passive workstation electronics, supported on a documented side table."""
    for key, pos, half, color in (
        ("foot", (x, y, z + 0.013), (0.13, 0.09, 0.013), (0.20, 0.23, 0.24, 1)),
        (
            "stem",
            (x, y + 0.035, z + 0.10),
            (0.025, 0.025, 0.075),
            (0.35, 0.38, 0.39, 1),
        ),
        ("case", (x, y + 0.035, z + 0.26), (0.22, 0.025, 0.145), (0.75, 0.76, 0.70, 1)),
        (
            "screen",
            (x, y + 0.008, z + 0.26),
            (0.195, 0.002, 0.115),
            (0.09, 0.18, 0.22, 1),
        ),
        (
            "keyboard",
            (x, y - 0.17, z + 0.009),
            (0.22, 0.060, 0.009),
            (0.26, 0.29, 0.30, 1),
        ),
    ):
        box(world, name + "_" + key, half, pos, color, collision=False)


BUILDERS = {
    "auckland_glazed_test_segment": _tunnel,
    "kfupm_wedge_grip_frame": _test_frame,
    "whittle_low_speed_rig": _turbine,
    "birmingham_wire_fixture": _wire_machine,
}
SOURCES = {
    "auckland_glazed_test_segment": dict(
        reference="Auckland boundary-layer tunnel local glazed bay",
        url=AUCKLAND,
        dimensions_m=[6.1, 3.8, 3.0],
        dimension_basis="Published full test-section 20 x 3.6 x 2.5 m; modeled local segment is 6 m, cross-section preserved, exterior and specimen fixture estimated.",
    ),
    "kfupm_wedge_grip_frame": dict(
        reference="KFUPM photographed twin-column frame and wedge grips",
        url=KFUPM,
        dimensions_m=[1.2, 0.86, 2.5],
        dimension_basis="All dimensions and bounded travel estimated; manufacturer geometry and calibration are not verified.",
    ),
    "whittle_low_speed_rig": dict(
        reference="Historical Whittle large-scale low-speed turbine rig photograph",
        url=WHITTLE,
        dimensions_m=[4.3, 4.2, 3.45],
        dimension_basis="Visible exterior proportions estimated; no published rig dimensions recovered and no hidden turbine construction inferred.",
    ),
    "birmingham_wire_fixture": dict(
        reference="Birmingham photographed AGIE wire machine exterior",
        url=BIRMINGHAM,
        dimensions_m=[2.3, 1.15, 1.92],
        dimension_basis="Machine envelope, local room and dry fixture travel estimated; no exact OEM model claim.",
    ),
}
SAMPLE_INTERFACES = {
    "auckland_glazed_test_segment": (
        "clear_specimen",
        (0.09, 0.56, 1.48),
        "clamped",
        "Transparent inert foil fixed to the authored translating yaw stage",
    ),
    "kfupm_wedge_grip_frame": (
        "inert_coupon",
        (0.05, 0.016, 0.15),
        "clamped",
        "Visible metal strip retained in the lower wedge grip, with upper grip clear",
    ),
    "whittle_low_speed_rig": (
        "dry_reference",
        (0.09, 0.07, 0.08),
        "clamped",
        "Inert reference block retained on the external dry traverse, outside the flow duct",
    ),
    "birmingham_wire_fixture": (
        "inert_coupon",
        (0.15, 0.12, 0.05),
        "clamped",
        "Metal coupon held between opposing pads on the dry XY fixture",
    ),
}
FEATURES = {
    "kfupm_building_materials": _records_wall,
    "cambridge_whittle_low_speed_turbine": _whittle_steps,
    "birmingham_advanced_micro_edm": _edm_room,
}
