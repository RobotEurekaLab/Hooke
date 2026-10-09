"""Oceania laboratory apparatus reconstructed from inspected local photographs.

Only bounded mechanical positioning is implemented. Optical, thermal-fluid and
X-ray results, surveyed rooms and manufacturer performance are not implied.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

ANU = "https://cfplab.org/facilities/"
SYDNEY = "https://www.sydney.edu.au/engineering/our-research/laboratories-and-facilities/fluids-laboratory.html"
UNSW = "https://www.unsw.edu.au/research/facilities-and-infrastructure/find-an-instrument/micro-ct-instruments/coretom-micro-ct-1"

SOURCES = {
    "coretom_positioning_station": {
        "reference": "UNSW Tyree installed CoreTOM source, sample and detector photograph",
        "url": UNSW,
        "dimensions_m": [2.00, 0.96, 2.05],
        "dimension_basis": "Published source-detector separation 0.970 m. All other dimensions and two-axis motion ranges estimated. Partial exposed assembly, not full room or all nine OEM stages.",
    },
    "convection_imaging_tank": {
        "reference": "University of Sydney photographed side-heated convection tank",
        "url": SYDNEY,
        "dimensions_m": [2.03, 0.78, 2.07],
        "dimension_basis": "Local heated-tank apparatus dimensions estimated from the official photograph. The separately listed 30 m wave flume dimensions do not apply. The two-axis probe gantry is an authored positioning aid.",
    },
    "rotating_annulus_station": {
        "reference": "ANU Climate and Fluid Physics Laboratory installed Large Rotating Annulus",
        "url": ANU,
        "dimensions_m": [1.70, 1.70, 1.86],
        "dimension_basis": "Published tank outer diameter 1.6 m and depth 0.4 m. Pedestal, inner cylinder, profiling support and motion bounds are estimated; not a fluid or thermal simulation.",
    },
}


def _material(b, key, rgba, **attributes):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=b.name + "__mat_" + key,
        rgba=numbers(rgba),
        **{k: str(v) for k, v in attributes.items()},
    )


def _faceted_wall(b, parent, radius, thickness, bottom, height, material, segments=40):
    """Hollow ring panels retain an accessible cavity and explicit colliders."""
    for index in range(segments):
        angle = 2 * math.pi * index / segments
        b.box(
            parent,
            (radius * math.cos(angle), radius * math.sin(angle), bottom + height / 2),
            (thickness / 2, radius * math.tan(math.pi / segments), height / 2),
            material,
            collision=True,
            euler=(0, 0, angle),
        )


def _annular_liquid(b, parent, inner, outer, bottom, top):
    """Static visual volume; no fluid forces, heat transport or generated fields."""
    vertices, faces = [], []
    count = 48
    for z in (bottom, top):
        for radius in (inner, outer):
            vertices.extend(
                (
                    radius * math.cos(i * 2 * math.pi / count),
                    radius * math.sin(i * 2 * math.pi / count),
                    z,
                )
                for i in range(count)
            )
    for index in range(count):
        following = (index + 1) % count
        for a, c in (
            (0, count),
            (2 * count, 3 * count),
            (0, 2 * count),
            (count, 3 * count),
        ):
            faces.extend(
                (
                    (a + index, a + following, c + following),
                    (a + index, c + following, c + index),
                )
            )
    name = b.name + "__annular_liquid"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=numbers(v for point in vertices for v in point),
        face=" ".join(str(v) for face in faces for v in face),
    )
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="mesh",
        mesh=name,
        material=b.name + "__mat_water",
        contype="0",
        conaffinity="0",
    )


def _annulus(b, base, params):
    _material(b, "water", (0.035, 0.25, 0.10, 0.48), specular=0.75, shininess=0.65)
    b.cylinder(base, (0, 0, 0.045), 0.60, 0.045, "dark", collision=True)
    b.cylinder(base, (0, 0, 0.36), 0.33, 0.315, "metal", collision=True)
    b.cylinder(base, (0, 0, 0.715), 0.49, 0.04, "dark", collision=True)
    b.cylinder(base, (0, 0, 0.795), 0.42, 0.045, "bright")
    tank = b.moving(
        base,
        "tank_rotation",
        (0, 0, 0.84),
        (0, 0, 1),
        (-0.50, 0.50),
        kind="hinge",
        mass=18,
        kp=2500,
        force=3000,
    )
    b.cylinder(tank, (0, 0, 0.025), 0.81, 0.025, "metal", collision=True)
    _faceted_wall(b, tank, 0.79, 0.02, 0.05, 0.40, "glass")
    _faceted_wall(b, tank, 0.18, 0.034, 0.05, 0.42, "copper", 32)
    for z in (0.059, 0.437):
        b.ring(tank, (0, 0, z), 0.790, 0.012, "bright", segments=40)
        b.ring(tank, (0, 0, z), 0.181, 0.017, "copper", segments=32)
    for index in range(20):
        a = index * math.pi / 10
        b.cylinder(
            tank,
            (0.794 * math.cos(a), 0.794 * math.sin(a), 0.451),
            0.006,
            0.006,
            "dark",
        )
    _annular_liquid(b, tank, 0.201, 0.777, 0.054, 0.365)
    # Profiling rail is supported by a rim bracket carried by the rotating table.
    b.box(tank, (0.65, 0.32, 0.45), (0.12, 0.045, 0.025), "metal")
    for x in (0.59, 0.66):
        b.rod(tank, (x, 0.32, 0.45), (x, 0.32, 1.0), 0.009, "bright")
    b.box(tank, (0.625, 0.32, 1.0), (0.065, 0.045, 0.018), "dark")
    b.cylinder(tank, (0.625, 0.32, 1.04), 0.026, 0.04, "dark")
    probe = b.moving(
        tank,
        "probe_height",
        (0.625, 0.32, 0.77),
        (0, 0, 1),
        (-0.07, 0.07),
        mass=0.08,
        kp=220,
    )
    b.box(probe, (0, 0, 0), (0.06, 0.035, 0.03), "dark")
    b.box(probe, (-0.11, 0, 0), (0.11, 0.018, 0.012), "metal")
    b.rod(probe, (-0.20, 0, 0), (-0.20, 0, -0.49), 0.002, "bright", collision=True)
    # Raised inert calibration target is visible above the static liquid surface.
    b.rod(tank, (0.39, -0.12, 0.05), (0.39, -0.12, 0.390), 0.009, "metal")
    b.box(tank, (0.39, -0.12, 0.393), (0.035, 0.025, 0.003), "cream")
    b.box(tank, (0.39, -0.12, 0.3962), (0.015, 0.002, 0.0002), "dark")
    b.site(tank, "calibration_target", (0.39, -0.12, 0.3964), 0.001)
    b.metadata["capabilities"] = [
        "bounded_tank_rotation",
        "profiling_probe_height",
        "visible_inert_calibration_target",
    ]
    b.metadata["limitations"].append(
        "The annular liquid is a static visual mesh. There is no fluid motion, heat transport, thermistor response, rotation-rate performance or measured flow field. The raised inert calibration target is an authored mechanical aid."
    )


def _annulus_features(world, definition):
    dark, metal = (0.025, 0.028, 0.034, 1), (0.42, 0.45, 0.47, 1)
    for x in (-2.05, 0.62):
        box(
            world, f"anu_curtain_post_{x}", (0.025, 0.025, 1.15), (x, 1.52, 1.15), metal
        )
    box(world, "anu_curtain_rail", (1.36, 0.022, 0.023), (-0.715, 1.52, 2.30), metal)
    for index in range(36):
        x = -2.05 + index * 0.076
        box(
            world,
            f"anu_curtain_fold_{index}",
            (0.039, 0.018, 1.02),
            (x, 1.50 + 0.025 * math.sin(index), 1.26),
            dark,
            collision=False,
        )
    # The control desk and electronics rack are visible in the source photo.
    box(world, "anu_monitor_foot", (0.18, 0.11, 0.016), (1.29, 0.80, 0.776), dark)
    box(world, "anu_monitor_post", (0.025, 0.022, 0.17), (1.29, 0.84, 0.954), metal)
    box(world, "anu_monitor_bezel", (0.30, 0.032, 0.19), (1.29, 0.84, 1.20), dark)
    box(
        world,
        "anu_monitor_display",
        (0.278, 0.002, 0.165),
        (1.29, 0.805, 1.20),
        (0.08, 0.18, 0.22, 1),
        collision=False,
    )
    box(world, "anu_keyboard", (0.23, 0.085, 0.012), (1.29, 0.56, 0.772), dark)
    for x in (1.93, 2.46):
        for y in (0.40, 0.99):
            box(
                world,
                f"anu_rack_post_{x}_{y}",
                (0.021, 0.021, 0.80),
                (x, y, 0.80),
                metal,
            )
    for index in range(5):
        z = 0.20 + index * 0.28
        box(
            world,
            f"anu_rack_unit_{index}",
            (0.265, 0.29, 0.10),
            (2.195, 0.695, z),
            (0.67, 0.68, 0.67, 1),
        )
        box(
            world,
            f"anu_rack_screen_{index}",
            (0.070, 0.004, 0.03),
            (2.10, 0.397, z + 0.02),
            (0.02, 0.12, 0.10, 1),
            collision=False,
        )
    for index in range(18):
        box(
            world,
            f"anu_bay_mark_{index}",
            (0.05, 0.025, 0.001),
            (-1.40 + index * 0.10, -1.32, 0.002),
            (0.90, 0.70, 0.12, 1) if index % 2 else dark,
            collision=False,
        )


def _convection_tank(b, base, params):
    _material(b, "blue_liquid", (0.05, 0.30, 0.62, 0.16), specular=0.65, shininess=0.5)
    _material(b, "blue_backlight", (0.025, 0.17, 0.42, 1), emission=0.15)
    for x in (-0.94, 0.94):
        for y in (-0.30, 0.30):
            b.box(base, (x, y, 0.40), (0.035, 0.035, 0.40), "dark", collision=True)
            b.box(base, (x, y, 0.018), (0.075, 0.065, 0.018), "metal")
        b.box(base, (x, 0, 0.79), (0.045, 0.37, 0.04), "dark", collision=True)
    for y in (-0.30, 0.30):
        b.box(base, (0, y, 0.78), (0.98, 0.03, 0.04), "dark", collision=True)
        b.box(base, (0, y, 0.30), (0.98, 0.025, 0.025), "metal")
    # Insulating supports connect both tank ends to the rigid stand.
    for x in (-0.67, 0.67):
        b.box(base, (x, 0, 0.845), (0.15, 0.23, 0.025), "cream", collision=True)
    b.box(base, (0, 0, 0.885), (0.81, 0.20, 0.015), "metal", collision=True)
    b.box(base, (0, 0, 1.26), (0.786, 0.168, 0.358), "blue_liquid")
    for y in (-0.19, 0.19):
        b.box(base, (0, y, 1.28), (0.80, 0.012, 0.38), "glass", collision=True)
        for z in (0.90, 1.66):
            b.box(base, (0, y, z), (0.84, 0.018, 0.025), "bright")
            for index in range(19):
                b.cylinder(
                    base,
                    (-0.77 + index * 0.085, y - 0.021, z),
                    0.007,
                    0.006,
                    "dark",
                    euler=(math.pi / 2, 0, 0),
                )
    b.box(base, (-0.80, 0, 1.28), (0.012, 0.19, 0.38), "glass", collision=True)
    b.box(base, (0.825, 0, 1.28), (0.027, 0.23, 0.43), "metal", collision=True)
    b.box(base, (0.786, 0, 1.28), (0.012, 0.17, 0.35), "copper")
    for z in (0.92, 1.06, 1.20, 1.34, 1.48, 1.62):
        for y in (-0.205, 0.205):
            b.cylinder(
                base, (0.858, y, z), 0.010, 0.008, "bright", euler=(0, math.pi / 2, 0)
            )
    for y in (-0.10, 0.10):
        b.rod(base, (0.89, y, 0.88), (0.89, y, 1.67), 0.012, "bright")
        b.rod(base, (0.83, y, 0.92), (0.94, y, 0.92), 0.012, "metal")
        b.rod(base, (0.89, y, 1.64), (0.82, y, 1.64), 0.012, "metal")
        for z in (1.02, 1.53):
            b.box(base, (0.89, y, z), (0.018, 0.035, 0.022), "dark")
    for x in (-0.59, 0.58):
        for z in (0.94, 1.61):
            b.box(base, (x, -0.215, z), (0.24, 0.018, 0.039), "dark")
    # Backlight and probe gantry are carried by the same stand, not floating decor.
    for x in (-0.94, 0.94):
        b.box(base, (x, 0.28, 1.35), (0.028, 0.028, 0.53), "dark", collision=True)
    b.box(base, (0, 0.28, 1.88), (0.97, 0.035, 0.03), "metal")
    b.box(base, (0, 0.315, 1.32), (0.90, 0.012, 0.45), "blue_backlight")
    carrier = b.moving(
        base, "probe_x", (0, 0.28, 1.88), (1, 0, 0), (-0.55, 0.55), mass=0.30, kp=450
    )
    b.box(carrier, (0, 0, 0.018), (0.075, 0.065, 0.025), "dark")
    b.box(carrier, (0, -0.14, 0.025), (0.020, 0.14, 0.018), "metal")
    b.box(carrier, (0, -0.28, 0.045), (0.04, 0.028, 0.12), "dark")
    target = b.moving(
        carrier,
        "probe_height",
        (0, -0.28, 0),
        (0, 0, 1),
        (-0.09, 0.09),
        mass=0.04,
        kp=180,
    )
    b.rod(target, (0, 0, 0.10), (0, 0, -0.64), 0.003, "bright", collision=True)
    b.box(target, (0, 0, -0.65), (0.015, 0.001, 0.01), "cream")
    b.box(target, (0, -0.0012, -0.65), (0.010, 0.0002, 0.001), "dark")
    b.site(target, "calibration_plate", (0, -0.0014, -0.65), 0.001)
    b.metadata["capabilities"] = [
        "probe_horizontal_traverse",
        "probe_height",
        "visible_inert_calibration_plate",
    ]
    b.metadata["limitations"].append(
        "The vertical-plane traverse is an original two-axis inspection proxy. No thermal boundary conditions, convection, water motion, PIV/LIF data, illumination calibration or actual laboratory controller is implemented."
    )


def _convection_features(world, definition):
    # Short insulated-panel seams describe a local bay, not the separate wave hall.
    for index in range(9):
        box(
            world,
            f"sydney_panel_seam_{index}",
            (0.006, 0.002, 1.32),
            (-2.2 + index * 0.55, 2.29, 1.50),
            (0.59, 0.63, 0.66, 1),
            collision=False,
        )
    box(
        world,
        "sydney_service_cabinet",
        (0.32, 0.28, 0.38),
        (1.66, 1.44, 0.38),
        (0.60, 0.64, 0.68, 1),
    )
    box(
        world,
        "sydney_cabinet_base",
        (0.33, 0.29, 0.022),
        (1.66, 1.44, 0.022),
        (0.14, 0.16, 0.18, 1),
    )
    box(
        world,
        "sydney_cabinet_display",
        (0.13, 0.004, 0.055),
        (1.66, 1.153, 0.61),
        (0.03, 0.15, 0.18, 1),
        collision=False,
    )
    # A separate supported camera looks into the tank; it does not synthesize PIV.
    for x in (-1.72, -1.48):
        box(
            world,
            f"sydney_camera_foot_{x}",
            (0.022, 0.24, 0.023),
            (x, -0.28, 0.023),
            (0.20, 0.22, 0.24, 1),
        )
    box(
        world,
        "sydney_camera_crossfoot",
        (0.15, 0.026, 0.022),
        (-1.60, -0.28, 0.052),
        (0.30, 0.33, 0.35, 1),
    )
    box(
        world,
        "sydney_camera_post",
        (0.021, 0.021, 0.67),
        (-1.60, -0.28, 0.742),
        (0.43, 0.47, 0.51, 1),
    )
    box(
        world,
        "sydney_camera_head",
        (0.095, 0.07, 0.06),
        (-1.54, -0.28, 1.41),
        (0.10, 0.12, 0.14, 1),
    )


def _coretom(b, base, params):
    _material(b, "granite", (0.22, 0.24, 0.25, 1), specular=0.10, shininess=0.10)
    _material(b, "cable_red", (0.47, 0.025, 0.035, 1))
    _material(b, "cable_blue", (0.025, 0.17, 0.50, 1))
    for x in (-0.88, 0.88):
        for y in (-0.37, 0.37):
            b.box(base, (x, y, 0.375), (0.055, 0.055, 0.375), "metal", collision=True)
            b.cylinder(base, (x, y, 0.025), 0.075, 0.025, "dark")
    for y in (-0.37, 0.37):
        b.box(base, (0, y, 0.69), (0.93, 0.04, 0.06), "metal", collision=True)
    b.box(base, (0, 0, 0.81), (0.98, 0.48, 0.07), "granite", collision=True)
    b.box(base, (-0.22, 0.34, 1.46), (0.145, 0.11, 0.58), "granite", collision=True)
    for z in (1.11, 1.35, 1.59, 1.83):
        for x in (-0.29, -0.15):
            b.cylinder(
                base, (x, 0.223, z), 0.009, 0.006, "bright", euler=(math.pi / 2, 0, 0)
            )
    # Left source exterior; its aperture faces the detector 970 mm away.
    for y in (-0.12, 0.20):
        b.box(base, (-0.72, y, 1.31), (0.021, 0.021, 0.43), "bright", collision=True)
    b.box(base, (-0.72, 0.04, 1.18), (0.16, 0.20, 0.03), "metal")
    b.box(base, (-0.68, 0.04, 1.46), (0.12, 0.20, 0.03), "metal", collision=True)
    b.box(base, (-0.64, 0.03, 1.59), (0.12, 0.12, 0.12), "metal", collision=True)
    b.box(base, (-0.64, -0.099, 1.59), (0.09, 0.008, 0.085), "dark")
    b.cylinder(
        base, (-0.505, 0, 1.615), 0.033, 0.020, "bright", euler=(0, math.pi / 2, 0)
    )
    b.site(base, "source_aperture", (-0.485, 0, 1.615), 0.002)
    b.cylinder(base, (-0.64, 0.02, 1.38), 0.065, 0.09, "dark")
    for material, zoff in (("cable_red", 0.02), ("cable_blue", 0.13)):
        points = [
            (-0.75, -0.04, 1.60 + zoff),
            (-0.91, -0.06, 1.64 + zoff),
            (-0.96, -0.08, 1.22 + zoff),
            (-0.80, -0.16, 0.93 + zoff),
            (-0.64, -0.06, 1.31),
        ]
        for start, end in zip(points, points[1:]):
            b.rod(base, start, end, 0.006, material)
    for index in range(7):
        b.box(base, (-0.78, -0.04, 1.05 + index * 0.041), (0.024, 0.055, 0.012), "dark")
    # Right upright detector panel and stiff support remain behind its active face.
    b.box(base, (0.68, 0.0, 0.94), (0.21, 0.32, 0.06), "dark", collision=True)
    b.box(base, (0.74, 0.13, 1.44), (0.045, 0.12, 0.44), "metal", collision=True)
    b.box(base, (0.565, 0, 1.615), (0.08, 0.295, 0.34), "dark", collision=True)
    b.box(base, (0.483, 0, 1.615), (0.002, 0.273, 0.317), "screen")
    b.site(base, "detector_plane", (0.485, 0, 1.615), 0.002)
    for y in (-0.285, 0.285):
        b.box(base, (0.475, y, 1.615), (0.013, 0.012, 0.34), "metal")
    # Original Z carrier on visible guides; gaps represent ideal linear bearings.
    b.box(base, (0, 0, 1.02), (0.155, 0.12, 0.14), "cream", collision=True)
    b.box(base, (0, 0.13, 1.13), (0.23, 0.04, 0.025), "metal")
    for x in (-0.19, 0.19):
        b.rod(base, (x, 0.13, 1.14), (x, 0.13, 1.43), 0.010, "bright", collision=True)
    lift = b.moving(
        base,
        "sample_height",
        (0, 0, 1.24),
        (0, 0, 1),
        (-0.035, 0.070),
        mass=0.75,
        kp=500,
        force=500,
    )
    b.box(lift, (0, 0, 0), (0.17, 0.10, 0.022), "dark", collision=True)
    for x in (-0.19, 0.19):
        b.box(lift, (x * 0.80, 0.10, 0), (0.043, 0.025, 0.018), "dark")
        for z in (-0.017, 0.017):
            b.ring(lift, (x, 0.13, z), 0.020, 0.006, "metal", segments=12)
    b.cylinder(lift, (0, 0, 0.048), 0.095, 0.025, "metal")
    rotation = b.moving(
        lift,
        "sample_rotation",
        (0, 0, 0.075),
        (0, 0, 1),
        (-0.8, 0.8),
        kind="hinge",
        mass=0.10,
        kp=160,
    )
    b.cylinder(rotation, (0, 0, 0), 0.089, 0.010, "metal")
    b.cylinder(rotation, (0, 0, 0.055), 0.075, 0.045, "cream", collision=True)
    for z in (0.025, 0.082):
        b.ring(rotation, (0, 0, z), 0.076, 0.0025, "bright", segments=24)
    b.cylinder(rotation, (0, 0, 0.135), 0.042, 0.035, "bright", collision=True)
    b.cylinder(rotation, (0, 0, 0.30), 0.027, 0.13, "dark", collision=True)
    for z in (0.176, 0.427):
        b.cylinder(rotation, (0, 0, z), 0.030, 0.006, "metal")
    # Off-centre seam makes actual sample rotation visibly observable.
    b.box(rotation, (0, -0.027, 0.30), (0.003, 0.0005, 0.10), "copper")
    b.site(rotation, "inert_core", (0, 0, 0.30), 0.003)
    b.metadata["capabilities"] = [
        "sample_vertical_position",
        "sample_rotation",
        "visible_inert_core_capsule",
    ]
    b.metadata["limitations"].append(
        "Two sample axes represent a mechanical subset of the nine stages listed for CoreTOM. No X-ray emission, dose, detector response, tomography reconstruction, core microstructure, fluid transport or certified shielding is implemented. Exposed machinery is an inspection vignette, not an operating procedure."
    )


def _coretom_features(world, definition):
    # Service partition and rail outlines are local context, not certified shielding.
    for x in (-1.35, 1.35):
        box(
            world,
            f"unsw_service_post_{x}",
            (0.035, 0.035, 1.18),
            (x, 1.30, 1.18),
            (0.50, 0.54, 0.56, 1),
        )
    box(
        world,
        "unsw_service_crossbar",
        (1.39, 0.035, 0.035),
        (0, 1.30, 2.36),
        (0.50, 0.54, 0.56, 1),
    )
    box(
        world,
        "unsw_service_panel",
        (1.30, 0.025, 1.13),
        (0, 1.34, 1.16),
        (0.74, 0.76, 0.79, 1),
        collision=False,
    )
    for index in range(8):
        box(
            world,
            f"unsw_floor_tray_{index}",
            (0.22, 0.018, 0.009),
            (-0.75, 0.85 + index * 0.06, 0.016),
            (0.12, 0.14, 0.16, 1),
            collision=False,
        )


BUILDERS = {
    "rotating_annulus_station": _annulus,
    "convection_imaging_tank": _convection_tank,
    "coretom_positioning_station": _coretom,
}
SAMPLE_INTERFACES = {
    "coretom_positioning_station": (
        "inert_core",
        (0.054, 0.054, 0.260),
        "clamped",
        "Visible inert rigid core capsule held by a white rotary chuck",
    ),
    "convection_imaging_tank": (
        "calibration_plate",
        (0.030, 0.002, 0.020),
        "clamped",
        "Visible inert calibration plate held on the supported probe carriage",
    ),
    "rotating_annulus_station": (
        "calibration_target",
        (0.070, 0.050, 0.006),
        "clamped",
        "Raised inert calibration plate rigidly supported by the rotating tank",
    ),
}
FEATURES = {
    "anu_climate_fluid_physics": _annulus_features,
    "sydney_fluids_convection": _convection_features,
    "unsw_tyree_xray": _coretom_features,
}
