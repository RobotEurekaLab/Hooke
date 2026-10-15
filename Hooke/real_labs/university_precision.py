"""Original precision instruments reconstructed from university/OEM references.

Only geometry and the declared mechanical axes are implemented. Optical,
material-test, AFM and human biomechanics outputs are deliberately absent.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers
from real_labs.optical_stages import aperture_plate, slide_holder_details


NTU = "https://www.ntu.edu.sg/mse/research/sgsr/facilities"
INSTRON = "https://www.ntu.edu.sg/media/docs/librariesprovider121/research-/sgsr/facilities/lab1/6800series_brochurev1_-2021.pdf?sfvrsn=4dbfbba2_3"
AFM = "https://www.ntu.edu.sg/media/docs/librariesprovider121/research-/sgsr/facilities/lab1/afm.pdf?sfvrsn=cbf338ea_3"
RUSH3D = "https://www.tsinghua.edu.cn/info/3155/119959.htm"
STANFORD = "https://biomechatronics.stanford.edu/"

SOURCES = {
    "precision_load_frame": {
        "reference": "NTU-listed Instron 68TM; original model from linked 6800 brochure, pp. 5 and 21",
        "url": INSTRON,
        "dimensions_m": [0.76, 0.715, 1.64],
        "dimension_basis": "Published standard frame W/D/H; published dashboard configuration can add up to 0.45 m. Capacity/configuration at NTU unconfirmed; restricted surrogate stroke.",
    },
    "afm_scanning_station": {
        "reference": "NTU Bruker Dimension Icon equipment sheet and apparatus photograph",
        "url": AFM,
        "dimensions_m": [0.65, 0.69, 0.65],
        "dimension_basis": "Envelope estimated from photo; 210 mm specimen chuck published by facility. Coarse XY axes represent positioning only, not the 90 micrometre piezo scan.",
    },
    "rush3d_optical_station": {
        "reference": "Tsinghua RUSH3D workstation photograph, July 2025 university feature",
        "url": RUSH3D,
        "dimensions_m": [1.66, 1.18, 1.91],
        "dimension_basis": "Estimated from single apparatus photo; table, objective mount, optical path and internal stages not measured or institution-validated.",
    },
    "split_belt_gait_station": {
        "reference": "Stanford Biomechatronics treadmill, side controls and support-frame photographs",
        "url": STANFORD,
        "dimensions_m": [1.82, 2.56, 2.85],
        "dimension_basis": "Authored envelope based on visible arrangement; manufacturer and actual dimensions unconfirmed. Mechanical leg fixture is original surrogate, not human subject geometry.",
    },
}


def _bolt_grid(b, parent, xs, ys, z, radius=0.003):
    for x in xs:
        for y in ys:
            b.cylinder(parent, (x, y, z), radius, 0.0015, "dark")


def _cable(b, parent, points, radius=0.007, material="rubber"):
    for first, second in zip(points, points[1:]):
        b.rod(parent, first, second, radius, material)


def _load_frame(b, base, params):
    from real_labs.ntu_sgsr_geometry import build_load_frame

    build_load_frame(b, base, params)


def _afm(b, base, params):
    from real_labs.ntu_sgsr_geometry import build_afm

    build_afm(b, base, params)


def _rush3d(b, base, params):
    b.box(base, (0, 0.04, 0.026), (0.78, 0.55, 0.026), "dark", collision=True)
    _bolt_grid(
        b,
        base,
        [i * 0.10 for i in range(-7, 8)],
        [j * 0.10 for j in range(-4, 6)],
        0.054,
    )
    # Tall right pedestal and red bridge are the strong silhouette in the photo.
    b.box(base, (0.49, 0.13, 0.43), (0.25, 0.31, 0.38), "dark", collision=True)
    for z in (0.22, 0.51):
        b.box(base, (0.49, -0.185, z), (0.24, 0.005, 0.003), "metal")
    b.box(base, (0.36, 0.12, 0.83), (0.40, 0.35, 0.033), "guard_red", collision=True)
    for x in (-0.63, -0.11):
        for y in (-0.29, 0.33):
            b.rod(base, (x, y, 0.07), (x, y, 0.88), 0.022, "bright", collision=True)
    for z in (0.25, 0.62, 0.89):
        b.box(base, (-0.37, 0.02, z), (0.31, 0.36, 0.026), "dark", collision=True)
    b.box(base, (-0.36, 0.035, 0.51), (0.15, 0.15, 0.17), "dark")
    for x in (-0.51, -0.21):
        b.box(base, (x, -0.16, 0.52), (0.012, 0.028, 0.155), "bright")
    for z in (0.33, 0.57):
        b.rod(base, (-0.35, -0.02, z), (-0.35, -0.38, z), 0.046, "dark")
        b.ring(base, (-0.35, -0.365, z), 0.050, 0.005, "bright", plane="xz")
    # The photographed lower microscope has a silver front frame, a dark
    # circular selector and side focus/camera adjustments below the sample deck.
    # These static exterior fittings leave the existing XYZ sample path intact.
    for x in (-0.515, -0.205):
        b.box(base, (x, -0.172, 0.46), (0.012, 0.012, 0.142), "bright", collision=True)
    for z in (0.318, 0.602):
        b.box(base, (-0.36, -0.172, z), (0.167, 0.012, 0.012), "bright", collision=True)
    b.cylinder(base, (-0.36, -0.180, 0.445), 0.069, 0.013, "dark",
               euler=(math.pi / 2, 0, 0), collision=True)
    b.cylinder(base, (-0.36, -0.196, 0.445), 0.039, 0.003, "metal",
               euler=(math.pi / 2, 0, 0), density=0)
    for x, sign in ((-0.535, -1), (-0.185, 1)):
        for radius, offset, length, material in ((0.042, 0.018, 0.018, "dark"), (0.028, 0.042, 0.007, "bright"), (0.019, 0.053, 0.004, "dark")):
            b.cylinder(base, (x + sign * offset, -0.03, 0.48), radius, length,
                       material, euler=(0, math.pi / 2, 0), collision=True)
    for x in (-0.49, -0.23):
        for z in (0.335, 0.585):
            b.cylinder(base, (x, -0.187, z), 0.005, 0.002, "dark",
                       euler=(math.pi / 2, 0, 0), density=0)
    # Unequal dovetail-like plates and adjustment screws replace the plain
    # camera-stack silhouette. Dimensions are photo-based engineering estimates.
    for i, z in enumerate((0.965, 1.005, 1.045, 1.085)):
        b.box(base, (0.51, 0.13, z), (0.165 - i * 0.007, 0.21 - i * 0.008, 0.008),
              "metal", collision=True)
        b.cylinder(base, (0.51 + (-1)**i * 0.105, -0.095 + i * 0.007, z),
                   0.014, 0.011, "dark", euler=(math.pi / 2, 0, 0), collision=True)
    for x in (-0.59, -0.07):
        b.rod(base, (x, 0.26, 0.92), (x, 0.26, 1.58), 0.020, "bright", collision=True)
    b.box(base, (-0.33, 0.27, 1.59), (0.33, 0.07, 0.025), "dark")
    b.rod(base, (-0.31, 0.13, 1.40), (0.46, 0.13, 1.40), 0.069, "dark")
    for x in (-0.17, 0.0, 0.40, 0.48):
        b.ring(
            base,
            (x, 0.13, 1.40),
            0.078 if x < 0.35 else 0.112,
            0.009,
            "bright",
            plane="yz",
        )
    b.box(base, (0.52, 0.13, 1.39), (0.053, 0.117, 0.114), "dark")
    b.box(base, (0.52, 0.13, 1.2105), (0.061, 0.071, 0.0655), "metal")
    b.box(base, (0.51, 0.13, 1.12), (0.19, 0.24, 0.025), "dark")
    b.box(base, (0.51, 0.13, 1.02), (0.15, 0.20, 0.075), "dark")
    b.box(base, (0.51, 0.13, 0.905), (0.12, 0.16, 0.042), "dark", collision=True)
    _bolt_grid(b, base, (0.42, 0.60), (0.01, 0.25), 0.947, 0.004)
    for z in (0.98, 1.02, 1.06):
        b.box(base, (0.51, -0.074, z), (0.135, 0.003, 0.003), "metal")
    b.box(base, (-0.31, 0.13, 1.41), (0.098, 0.097, 0.082), "dark")
    b.box(base, (-0.31, 0.22, 1.528), (0.060, 0.040, 0.038), "metal")
    objective = b.moving(
        base,
        "objective_focus",
        (-0.31, 0.13, 1.24),
        (0, 0, 1),
        (-0.018, 0.018),
        mass=0.4,
        kp=400,
    )
    for z, radius, height, material in (
        (0.0, 0.050, 0.053, "dark"),
        (-0.071, 0.042, 0.018, "metal"),
        (-0.102, 0.031, 0.020, "dark"),
    ):
        b.cylinder(objective, (0, 0, z), radius, height, material, collision=True)
    stage_x = b.moving(
        base,
        "sample_x",
        (-0.31, 0.13, 0.96),
        (1, 0, 0),
        (-0.035, 0.035),
        mass=0.8,
        kp=500,
    )
    aperture_plate(
        b,
        base,
        "sample_fixed_plate",
        (-0.31, 0.13, 0.924),
        (0.19, 0.16, 0.008),
        (0.079, 0.059),
        "dark",
    )
    aperture_plate(
        b,
        stage_x,
        "sample_x_plate",
        (0, 0, 0),
        (0.178, 0.19, 0.025),
        (0.044, 0.052),
        "metal",
    )
    # The joint represents the internal rolling guide. Keep the two exterior
    # plates apart; coplanar shell contact would spuriously brake the carriage.
    for y in (-0.13, 0.13):
        b.box(base, (-0.31, 0.13 + y, 0.933), (0.16, 0.006, 0.001), "bright")
        for x in (-0.11, 0.11):
            b.box(stage_x, (x, y, -0.026), (0.024, 0.012, 0.002), "dark")
    for x in (-0.10, 0.10):
        b.box(stage_x, (x, 0, 0.0275), (0.011, 0.13, 0.0025), "bright")
    stage_y = b.moving(
        stage_x, "sample_y", (0, 0, 0.046), (0, 1, 0), (-0.03, 0.03), mass=0.5, kp=400
    )
    aperture_plate(
        b,
        stage_y,
        "sample_y_plate",
        (0, 0, 0),
        (0.17, 0.16, 0.016),
        (0.042, 0.020),
        "dark",
    )
    slide_holder_details(b, stage_y, "sample_holder", 0.016, (0.17, 0.16))
    b.box(stage_y, (0, 0, 0.018), (0.0375, 0.0125, 0.0005), "glass")
    b.box(stage_y, (0, 0, 0.019), (0.001, 0.001, 0.00008), "sample")
    b.site(stage_y, "calibration_target", (0, 0, 0.0191), 0.0001)
    for offset in (-0.027, 0.027):
        b.box(stage_y, (offset, 0, 0.020), (0.003, 0.020, 0.002), "bright")
    for offset in (-0.02, 0.02):
        _cable(
            b,
            base,
            [
                (0.61, 0.16 + offset, 1.46),
                (0.75, 0.16 + offset, 1.20),
                (0.80, 0.26 + offset, 0.65),
                (0.74, 0.39, 0.07),
            ],
            0.013,
            "cream",
        )
    for x in (-0.49, -0.43, -0.37):
        _cable(
            b,
            base,
            [
                (x, 0.17, 1.63),
                (x - 0.15, 0.28, 1.4),
                (x - 0.12, 0.33, 0.6),
                (x - 0.1, 0.24, 0.12),
            ],
            0.005,
        )
    b.cylinder(base, (-0.31, 0.13, 1.62), 0.04, 0.012, "dark")
    b.cylinder(base, (-0.31, 0.13, 1.74), 0.037, 0.115, "dark")
    b.metadata["capabilities"] = [
        "motorized_sample_xy",
        "objective_focus",
        "visible_inert_calibration_slide",
    ]
    b.metadata["limitations"].append(
        "Single-view exterior reconstruction; hidden optical components and motor stages are authored estimates. No mesoscopic imaging, live-cell process or optical transfer function."
    )


def _gait_station(b, base, params):
    from real_labs.stanford_gait_geometry import build_gait_station

    build_gait_station(b, base, params)


BUILDERS = {
    "precision_load_frame": _load_frame,
    "afm_scanning_station": _afm,
    "rush3d_optical_station": _rush3d,
    "split_belt_gait_station": _gait_station,
}
SAMPLE_INTERFACES = {
    "precision_load_frame": (
        "coupon",
        (0.024, 0.008, 0.064),
        "clamped",
        "inert material coupon in lower grip",
    ),
    "afm_scanning_station": (
        "sample_coupon",
        (0.026, 0.026, 0.007),
        "clamped",
        "inert disk on the specimen chuck",
    ),
    "rush3d_optical_station": (
        "calibration_target",
        (0.002, 0.002, 0.00016),
        "clamped",
        "inert millimetre calibration target on 75 mm glass slide",
    ),
    "split_belt_gait_station": (
        "test_fixture",
        (0.07, 0.08, 0.09),
        "clamped",
        "original mechanical ankle fixture, no human specimen",
    ),
}


def _partition(world, name, start, end, color, height=2.55, opening=None):
    """Solid authored partition with a true opening, not a door-shaped decal."""
    axis = 0 if start[1] == end[1] else 1
    lo, hi = sorted((start[axis], end[axis]))
    spans = [(lo, hi, 0, height)]
    if opening:
        centre, width = opening
        spans = [
            (lo, centre - width / 2, 0, height),
            (centre + width / 2, hi, 0, height),
            (centre - width / 2, centre + width / 2, 2.12, height),
        ]
    for index, (a, z, b, h) in enumerate(spans):
        if z - a <= 0:
            continue
        pos = [start[0], start[1], (b + h) / 2]
        pos[axis] = (a + z) / 2
        size = [0.055, 0.055, (h - b) / 2]
        size[axis] = (z - a) / 2
        box(world, f"precision_partition_{name}_{index}", size, pos, color)
        # Color stripe identifies the source's separated functional zones.
        if b == 0:
            p = pos.copy()
            p[2] = 1.25
            s = size.copy()
            s[2] = 0.05
            s[1 - axis] += 0.002
            box(
                world,
                f"precision_partition_{name}_stripe_{index}",
                s,
                p,
                (*[v * 0.75 for v in color[:3]], 1),
                collision=False,
            )


def _monitor(world, key, position, rotation=0):
    parent = ET.SubElement(
        world,
        "body",
        name=f"precision_monitor_{key}",
        pos=numbers(position),
        euler=f"0 0 {rotation}",
    )
    box(
        parent,
        f"precision_{key}_foot",
        (0.13, 0.09, 0.012),
        (0, 0, 0.012),
        (0.12, 0.14, 0.15, 1),
    )
    box(
        parent,
        f"precision_{key}_stand",
        (0.018, 0.026, 0.15),
        (0, 0.04, 0.16),
        (0.35, 0.37, 0.38, 1),
    )
    box(
        parent,
        f"precision_{key}_screen",
        (0.26, 0.025, 0.16),
        (0, 0, 0.34),
        (0.08, 0.10, 0.12, 1),
    )
    box(
        parent,
        f"precision_{key}_display",
        (0.244, 0.002, 0.141),
        (0, -0.027, 0.34),
        (0.06, 0.18, 0.24, 1),
        collision=False,
    )
    for i in range(6):
        box(
            parent,
            f"precision_{key}_trace_{i}",
            (0.15 - i * 0.015, 0.001, 0.003),
            (-0.02, -0.030, 0.43 - i * 0.033),
            (0.38, 0.7, 0.68, 1),
            collision=False,
        )
    box(
        parent,
        f"precision_{key}_keyboard",
        (0.20, 0.075, 0.010),
        (0, -0.25, 0.010),
        (0.19, 0.21, 0.22, 1),
    )


def _ntu_features(world, definition):
    def partition(name, start, end, color, height=2.55, opening=None):
        scaled_opening = (opening[0] * 1.4, opening[1] * 1.4) if opening else None
        _partition(
            world,
            name,
            tuple(v * 1.4 for v in start),
            tuple(v * 1.4 for v in end),
            color,
            height,
            scaled_opening,
        )

    warm = (0.88, 0.86, 0.79, 1)
    blue = (0.65, 0.75, 0.86, 1)
    pink = (0.86, 0.76, 0.79, 1)
    # Relative arrangement follows the plan, metric scale/openings are estimates.
    partition("meeting_east", (-4.65, -1.25), (-4.65, 0.75), warm, opening=(-0.4, 0.9))
    partition("meeting_north", (-6.8, 0.75), (-4.65, 0.75), warm)
    partition("meeting_south", (-6.8, -1.25), (-4.65, -1.25), warm)
    partition("dry_east", (-0.9, -1.85), (-0.9, 3.3), warm, opening=(-0.9, 1.2))
    partition("bsc_north", (-0.85, 1.35), (2.2, 1.35), blue)
    partition("bsc_mid", (0.7, -0.05), (0.7, 1.35), blue)
    partition("bsc_east", (2.2, -0.05), (2.2, 1.35), blue)
    partition(
        "bsc_south_one", (-0.85, -0.05), (0.7, -0.05), blue, opening=(-0.12, 0.85)
    )
    partition("bsc_south_two", (0.7, -0.05), (2.2, -0.05), blue, opening=(1.47, 0.85))
    partition(
        "battery_north",
        (-0.85, -1.50),
        (2.2, -1.50),
        (0.71, 0.82, 0.69, 1),
        opening=(0.65, 1.0),
    )
    partition("battery_west", (-0.85, -3.5), (-0.85, -1.5), (0.71, 0.82, 0.69, 1))
    partition("battery_east", (2.2, -3.5), (2.2, -1.5), (0.71, 0.82, 0.69, 1))
    partition("analytical_west", (3.9, 0.90), (3.9, 3.5), warm, opening=(2.75, 0.95))
    partition("analytical_south", (3.9, 0.90), (6.8, 0.90), warm)
    # Lower right room sequence: wet / confocal / AFM with southern corridor.
    for key, x in (("wet_west", 3.9), ("confocal_west", 4.85), ("afm_west", 5.8)):
        partition(key, (x, -0.7), (x, 0.9), pink)
    for i, (a, z) in enumerate(((3.9, 4.85), (4.85, 5.8), (5.8, 6.8))):
        partition(
            f"analytical_annex_{i}",
            (a, -0.7),
            (z, -0.7),
            pink,
            opening=((a + z) / 2, 0.76),
        )
    partition("optics_west", (3.9, -3.5), (3.9, -1.75), warm)
    partition("xrd_west", (5.2, -3.5), (5.2, -1.75), warm)
    for i, (a, z) in enumerate(((3.9, 5.2), (5.2, 6.8))):
        partition(
            f"south_room_{i}", (a, -1.75), (z, -1.75), warm, opening=((a + z) / 2, 0.88)
        )
    _monitor(world, "ntu_dry", (-5.11, 4.032, 0.84))
    _monitor(world, "ntu_analytical", (6.86, 3.99, 0.90))
    # Empty source-labelled compartments intentionally have no invented inventory.
    for i, (x, y, w, d, c) in enumerate(
        (
            (-5.72, -0.2, 1.7, 1.25, (0.87, 0.84, 0.68, 1)),
            (0.7, -2.55, 2.5, 1.55, (0.59, 0.73, 0.60, 1)),
            (5.4, 2.2, 2.5, 2.1, (0.81, 0.81, 0.71, 1)),
        )
    ):
        box(
            world,
            f"precision_zone_floor_{i}",
            (w * 0.7, d * 0.7, 0.0008),
            (x * 1.4, y * 1.4, 0.002),
            c,
            collision=False,
        )


def _rush_features(world, definition):
    dark = (0.035, 0.040, 0.05, 1)
    metal = (0.62, 0.65, 0.67, 1)
    # Ceiling-height optical enclosure with pleated blackout curtain at left/rear.
    for x in (-1.50, 1.50):
        for y in (-0.60, 2.05):
            box(
                world,
                f"precision_optics_post_{x}_{y}",
                (0.032, 0.032, 1.4),
                (x, y, 1.4),
                metal,
            )
    for x in (-1.50, 1.50):
        box(
            world,
            f"precision_optics_rail_{x}",
            (0.035, 1.35, 0.04),
            (x, 0.725, 2.77),
            metal,
        )
    for y in (-0.60, 2.05):
        box(
            world,
            f"precision_optics_bridge_{y}",
            (1.535, 0.035, 0.04),
            (0, y, 2.77),
            metal,
        )
    for index in range(29):
        x = -1.46 + index * 0.101
        box(
            world,
            f"precision_curtain_back_{index}",
            (0.054, 0.027, 1.30),
            (x, 2.06 + (index % 2) * 0.038, 1.37),
            dark,
            collision=False,
        )
    for index in range(22):
        y = -0.55 + index * 0.12
        box(
            world,
            f"precision_curtain_left_{index}",
            (0.032, 0.067, 1.30),
            (-1.48 - (index % 2) * 0.034, y, 1.37),
            dark,
            collision=False,
        )
    _monitor(world, "rush_operator", (2.05, 0.75, 0.80), rotation=-0.35)
    box(
        world,
        "precision_rush_controller",
        (0.23, 0.23, 0.17),
        (2.05, 1.42, 0.97),
        (0.18, 0.2, 0.22, 1),
    )


def _gait_features(world, definition):
    metal = (0.56, 0.59, 0.6, 1)
    dark = (0.09, 0.11, 0.13, 1)
    _monitor(world, "gait_operator", (2.35, -0.20, 0.77), rotation=-0.45)
    _monitor(world, "gait_operator_second", (2.5, 0.55, 0.77), rotation=-0.65)
    rack = ET.SubElement(world, "body", name="precision_gait_rack", pos="1.6 1.65 0")
    for x in (-0.34, 0.34):
        for y in (-0.30, 0.30):
            box(
                rack,
                f"precision_rack_post_{x}_{y}",
                (0.022, 0.022, 1.05),
                (x, y, 1.11),
                metal,
            )
    for i, z in enumerate((0.10, 0.42, 0.72, 1.02, 1.34, 1.65, 2.14)):
        box(rack, f"precision_rack_shelf_{i}", (0.36, 0.33, 0.012), (0, 0, z), metal)
        if i in (0, 6):
            continue
        box(
            rack,
            f"precision_rack_electronics_{i}",
            (0.30, 0.25, 0.092),
            (0, 0, z + 0.11),
            dark,
        )
        for j in range(7):
            box(
                rack,
                f"precision_rack_vent_{i}_{j}",
                (0.14, 0.002, 0.003),
                (-0.08, -0.253, z + 0.065 + j * 0.015),
                (0.37, 0.40, 0.42, 1),
                collision=False,
            )
        box(
            rack,
            f"precision_rack_status_{i}",
            (0.02, 0.002, 0.025),
            (0.235, -0.254, z + 0.10),
            (0.21, 0.68, 0.68, 1),
            collision=False,
        )
    # Rear laboratory blinds and secured perimeter cable tray follow the photos.
    for i in range(22):
        box(
            world,
            f"precision_gait_blind_{i}",
            (2.4, 0.014, 0.010),
            (-0.9, 2.87, 1.05 + i * 0.064),
            (0.64, 0.65, 0.61, 1),
            collision=False,
        )
    box(
        world,
        "precision_gait_service_tray",
        (2.1, 0.08, 0.035),
        (-0.2, 2.55, 0.085),
        dark,
    )


FEATURES = {
    "ntu_sgsr_characterization": _ntu_features,
    "tsinghua_rush3d": _rush_features,
    "stanford_biomechatronics_gait": _gait_features,
}
