"""Observed North-American instrument bays with bounded dry mechanical tasks.

Photographs constrain exterior geometry; all dimensions are estimates. Controls
provide reversible inspection, never calibrated material or acoustic responses.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box

NYU = "https://physics.nyu.edu/experimentalparticle/"
DUKE = "https://mems.duke.edu/impact/facilities/"
PSU = "https://aeroacoustics.psu.edu/excap/"
NORTHWESTERN = "https://sites.northwestern.edu/clammp/instruments/"


def _tag(b, geom, name):
    geom.set("name", b.unique(name))
    return geom


def _nyu_printer(b, base, params):
    # Blue cube enclosure, red front window rim, dark rear/side openings.
    for x in (-0.17, 0.17):
        for y in (-0.15, 0.15):
            b.box(base, (x, y, 0.010), (0.025, 0.025, 0.010), "rubber", collision=True)
    b.box(base, (0, 0, 0.05), (0.22, 0.21, 0.03), "blue", collision=True)
    b.box(base, (0, 0.20, 0.25), (0.22, 0.012, 0.17), "blue", collision=True)
    for x in (-0.208, 0.208):
        b.box(base, (x, 0, 0.25), (0.012, 0.20, 0.17), "blue", collision=True)
        b.box(base, (x * 1.06, -0.02, 0.265), (0.001, 0.11, 0.105), "dark")
    b.box(base, (0, 0, 0.435), (0.22, 0.21, 0.015), "blue", collision=True)
    for x in (-0.198, 0.198):
        b.box(
            base, (x, -0.21, 0.267), (0.018, 0.012, 0.157), "guard_red", collision=True
        )
    for z in (0.11, 0.424):
        b.box(base, (0, -0.21, z), (0.198, 0.012, 0.014), "guard_red", collision=True)
    # Transparent front keeps inert print target inspectable.
    b.box(base, (0, -0.212, 0.268), (0.18, 0.003, 0.142), "glass", collision=True)
    b.box(base, (0.115, -0.218, 0.065), (0.058, 0.005, 0.021), "screen")
    for y in (-0.12, 0.12):
        _tag(
            b,
            b.cylinder(
                base, (0, y, 0.357), 0.006, 0.187, "bright", euler=(0, math.pi / 2, 0)
            ),
            "print_x_rail",
        )
    head = b.moving(
        base, "printhead_x", (0, 0, 0.377), (1, 0, 0), (-0.13, 0.13), mass=0.18, kp=320
    )
    b.box(head, (0, 0, 0), (0.036, 0.07, 0.031), "dark", collision=True)
    b.cylinder(head, (0, 0, -0.045), 0.013, 0.014, "metal", collision=True)
    b.cylinder(head, (0, 0, -0.064), 0.005, 0.005, "copper", collision=True)
    # Bed guide columns connect to the chassis, never an unsupported plate.
    for x in (-0.095, 0.095):
        b.box(base, (x, 0, 0.14), (0.018, 0.17, 0.06), "dark", collision=True)
        _tag(
            b, b.box(base, (x, 0, 0.205), (0.010, 0.17, 0.005), "metal"), "bed_y_guide"
        )
    bed = b.moving(
        base,
        "buildplate_y",
        (0, 0, 0.221),
        (0, 1, 0),
        (-0.055, 0.055),
        mass=0.25,
        kp=360,
    )
    b.box(bed, (0, 0, 0), (0.14, 0.12, 0.011), "metal", collision=True)
    b.box(bed, (0, 0, 0.013), (0.13, 0.11, 0.002), "dark", collision=True)
    b.box(bed, (0, 0, 0.033), (0.027, 0.027, 0.018), "orange", collision=True)
    b.box(bed, (0, 0, 0.057), (0.016, 0.016, 0.006), "orange", collision=True)
    b.site(bed, "calibration_part", (0, 0, 0.063))
    for i in range(6):
        b.box(base, (-0.09 + i * 0.036, 0.214, 0.16), (0.009, 0.002, 0.034), "dark")
    b.metadata["capabilities"] = [
        "dry prototyping head traverse",
        "inert build-plate positioning",
    ]
    b.metadata["limitations"].append(
        "Desktop printer exterior informed by the NYU room photograph; estimated geometry and strokes. No extrusion, temperature, detector response, radiation transport or CERN environment model."
    )


def _nyu_features(world, definition):
    # Yellow hanging retractable power reels characterize this actual room.
    for x in (-1.7, 0, 1.7):
        box(
            world,
            f"nyu_ceiling_rail_{x}",
            (0.05, 2.65, 0.06),
            (x, 0, 2.98),
            (0.66, 0.69, 0.67, 1),
        )
        for y in (-2.5, 2.5):
            box(
                world,
                f"nyu_rail_anchor_{x}_{y}",
                (0.05, 0.04, 0.03),
                (x, y, 3.07),
                (0.45, 0.47, 0.44, 1),
            )
        for y in (-1.5, -0.3, 0.9, 2.1):
            name = f"nyu_reel_{x}_{y}"
            box(
                world,
                name + "_bracket",
                (0.025, 0.035, 0.07),
                (x, y, 2.85),
                (0.45, 0.47, 0.44, 1),
            )
            ET.SubElement(
                world,
                "geom",
                name=name,
                type="cylinder",
                size=".14 .045",
                pos=f"{x} {y} 2.64",
                euler=f"{math.pi/2} 0 0",
                rgba=".80 .60 .07 1",
            )
            ET.SubElement(
                world,
                "geom",
                name=name + "_lead",
                type="capsule",
                size=".005",
                fromto=f"{x} {y} 2.50 {x} {y} 2.08",
                rgba=".05 .06 .05 1",
            )
            box(
                world,
                name + "_outlet",
                (0.045, 0.022, 0.060),
                (x, y, 2.02),
                (0.80, 0.60, 0.08, 1),
            )
    # Instrument boards and computer displays on separate right and rear benches.
    for j, y in enumerate((-0.7, 0.6, 1.8)):
        box(
            world,
            f"nyu_monitor_foot_{j}",
            (0.14, 0.13, 0.013),
            (2.25, y, 0.913),
            (0.13, 0.14, 0.15, 1),
        )
        box(
            world,
            f"nyu_monitor_stem_{j}",
            (0.025, 0.025, 0.07),
            (2.25, y, 0.995),
            (0.13, 0.14, 0.15, 1),
        )
        box(
            world,
            f"nyu_monitor_case_{j}",
            (0.04, 0.25, 0.18),
            (2.25, y, 1.22),
            (0.13, 0.14, 0.15, 1),
        )
        box(
            world,
            f"nyu_monitor_screen_{j}",
            (0.005, 0.23, 0.155),
            (2.205, y, 1.22),
            (0.02, 0.13, 0.14, 1),
            collision=False,
        )
        box(
            world,
            f"nyu_detector_board_{j}",
            (0.17, 0.23, 0.007),
            (1.80, y, 0.907),
            (0.15, 0.34, 0.24, 1),
        )
        for k in range(6):
            box(
                world,
                f"nyu_board_package_{j}_{k}",
                (0.018, 0.025, 0.005),
                (1.70 + (k % 2) * 0.14, y - 0.14 + (k // 2) * 0.12, 0.919),
                (0.12, 0.13, 0.13, 1),
                collision=False,
            )
    # Shelf uprights end on rear counter; boxes sit on the shelves.
    for x in (-1.3, 1.3):
        box(
            world,
            f"nyu_shelf_upright_{x}",
            (0.025, 0.025, 0.6),
            (x, 2.62, 1.5),
            (0.45, 0.48, 0.48, 1),
        )
    for z in (1.45, 2.02):
        box(
            world,
            f"nyu_storage_shelf_{z}",
            (1.38, 0.30, 0.025),
            (0, 2.42, z),
            (0.45, 0.48, 0.48, 1),
        )
        for x in (-0.86, -0.15, 0.64):
            box(
                world,
                f"nyu_storage_box_{z}_{x}",
                (0.26, 0.22, 0.15),
                (x, 2.42, z + 0.175),
                (0.52, 0.43, 0.30, 1),
            )


BUILDERS = {"nyu_detector_prototyping_printer": _nyu_printer}
SOURCES = {
    "nyu_detector_prototyping_printer": {
        "reference": "NYU on-campus EPP laboratory official room photograph",
        "url": NYU,
        "dimensions_m": [0.46, 0.45, 0.45],
        "dimension_basis": "Printer enclosure, inert calibration part and movements are estimated from the local room photo; no OEM CAD used.",
    }
}
SAMPLE_INTERFACES = {
    "nyu_detector_prototyping_printer": (
        "calibration_part",
        (0.054, 0.054, 0.048),
        "clamped",
        "inert prototype calibration coupon fixed to the movable printer bed",
    )
}
FEATURES = {"nyu_epp_detector_assembly": _nyu_features}


def _duke_frame(b, base, params):
    for x in (-0.38, 0.38):
        for y in (-0.28, 0.28):
            b.box(base, (x, y, 0.035), (0.075, 0.075, 0.035), "rubber", collision=True)
    b.box(base, (0, 0, 0.365), (0.46, 0.35, 0.295), "dark", collision=True)
    b.box(
        base,
        (0, -0.357, 0.35),
        (0.34, 0.008, 0.20),
        "metal",
        rgba=(0.13, 0.15, 0.16, 1),
    )
    for z in (0.24, 0.32, 0.40):
        b.box(base, (0, -0.368, z), (0.22, 0.003, 0.008), "rubber")
    b.box(base, (0, 0, 0.70), (0.49, 0.37, 0.04), "metal", collision=True)
    for x in (-0.37, 0.37):
        b.cylinder(base, (x, 0, 1.56), 0.035, 0.82, "bright", collision=True)
        for z in (0.78, 1.66, 2.35):
            b.cylinder(base, (x, 0, z), 0.065, 0.032, "metal", collision=True)
    b.box(base, (0, 0, 1.66), (0.46, 0.28, 0.055), "dark", collision=True)
    b.box(base, (0, 0, 2.35), (0.43, 0.28, 0.050), "metal", collision=True)
    # Tall black axial drive above the open loading window.
    b.box(base, (0, 0.02, 2.045), (0.205, 0.205, 0.255), "dark", collision=True)
    b.cylinder(base, (0, 0, 1.735), 0.105, 0.055, "metal", collision=True)
    for x in (-0.20, 0.20):
        b.cylinder(base, (x, 0.14, 1.16), 0.012, 0.42, "metal", collision=True)
        for z in (0.86, 1.04, 1.22, 1.40):
            b.ring(base, (x, 0.14, z), 0.014, 0.003, "dark", segments=8)
    # Upper drive translates a chuck for inspection; it never loads the specimen.
    ram = b.moving(
        base,
        "axial_chuck_position",
        (0, 0, 1.47),
        (0, 0, -1),
        (0, 0.20),
        mass=0.6,
        kp=500,
    )
    b.cylinder(ram, (0, 0, 0.30), 0.033, 0.30, "bright")
    b.cylinder(ram, (0, 0, 0), 0.066, 0.06, "metal", collision=True)
    for x in (-0.048, 0.048):
        b.box(ram, (x, -0.018, -0.05), (0.015, 0.04, 0.03), "dark", collision=True)
    b.cylinder(base, (0, 0, 0.80), 0.125, 0.060, "dark", collision=True)
    _tag(b, b.cylinder(base, (0, 0, 0.88), 0.09, 0.020, "metal"), "torsion_bearing")
    lower = b.moving(
        base,
        "lower_chuck_rotation",
        (0, 0, 0.934),
        (0, 0, 1),
        (-0.60, 0.60),
        kind="hinge",
        mass=0.35,
        kp=160,
    )
    _tag(
        b,
        b.cylinder(lower, (0, 0, 0), 0.085, 0.034, "metal", collision=True),
        "lower_chuck",
    )
    for x in (-0.042, 0.042):
        b.box(lower, (x, 0, 0.057), (0.014, 0.04, 0.040), "dark", collision=True)
    b.box(lower, (0, 0, 0.131), (0.024, 0.012, 0.07), "orange", collision=True)
    for z in (0.080, 0.182):
        b.box(lower, (0, 0, z), (0.038, 0.014, 0.019), "orange", collision=True)
    b.site(lower, "inert_coupon", (0, -0.015, 0.135))
    # Independent wheeled controller in the source photograph.
    for x in (1.0, 1.46):
        for y in (-0.29, 0.22):
            b.cylinder(
                base,
                (x, y, 0.06),
                0.06,
                0.035,
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
    b.box(base, (1.23, -0.035, 0.14), (0.29, 0.33, 0.02), "metal", collision=True)
    b.box(base, (1.23, -0.035, 0.68), (0.27, 0.31, 0.52), "dark", collision=True)
    for x in (0.95, 1.51):
        b.box(base, (x, -0.035, 0.68), (0.01, 0.33, 0.54), "metal", collision=True)
    for z in (0.70, 0.98):
        b.box(base, (1.23, -0.351, z), (0.20, 0.005, 0.055), "screen")
        for x in (1.05, 1.41):
            b.cylinder(
                base,
                (x, -0.36, z - 0.10),
                0.015,
                0.006,
                "orange" if x > 1.2 else "led",
                euler=(math.pi / 2, 0, 0),
            )
    b.box(base, (1.23, -0.035, 1.235), (0.29, 0.33, 0.015), "cream", collision=True)
    b.box(base, (1.23, 0.02, 1.31), (0.025, 0.025, 0.06), "metal", collision=True)
    b.box(base, (1.23, 0.02, 1.48), (0.22, 0.055, 0.11), "dark", collision=True)
    b.box(base, (1.23, -0.039, 1.48), (0.195, 0.003, 0.09), "screen")
    for i in range(3):
        b.rod(
            base,
            (0.44, 0.20, 0.3 + i * 0.07),
            (0.70, 0.28, 0.15 + i * 0.025),
            0.012,
            "dark",
        )
        b.rod(
            base,
            (0.70, 0.28, 0.15 + i * 0.025),
            (0.95, 0.20, 0.3 + i * 0.07),
            0.012,
            "dark",
        )
    b.metadata["capabilities"] = [
        "bounded upper-chuck axial alignment",
        "lower-chuck angular specimen indexing",
    ]
    b.metadata["limitations"].append(
        "830LE-AT exterior estimated from a 225x300 installed-instrument photograph. Rigid inert coupon remains clear of the upper chuck; no load calibration, torque, stress-strain, fatigue or material failure model."
    )


def _duke_features(world, definition):
    # Local MSRB bay only; separate Research Park Instron installation excluded.
    for x in (-1.20, -0.40, 0.40):
        box(
            world,
            f"duke_rear_storage_{x}",
            (0.38, 0.24, 0.40),
            (x, 1.97, 1.82),
            (0.71, 0.65, 0.43, 1),
        )
        box(
            world,
            f"duke_storage_front_{x}",
            (0.32, 0.006, 0.32),
            (x, 1.725, 1.82),
            (0.54, 0.51, 0.37, 1),
        )
        box(
            world,
            f"duke_wall_anchor_{x}",
            (0.32, 0.10, 0.02),
            (x, 2.26, 1.82),
            (0.35, 0.36, 0.32, 1),
        )


BUILDERS["duke_830leat_alignment_frame"] = _duke_frame
SOURCES["duke_830leat_alignment_frame"] = {
    "reference": "Duke Joint Mechanical Testing Laboratory 830LE-AT installed system photo",
    "url": DUKE,
    "dimensions_m": [2.05, 0.74, 2.40],
    "dimension_basis": "All dimensions and bounded inspection strokes estimated; published load/torque ratings are not used as simulated mechanical performance.",
}
SAMPLE_INTERFACES["duke_830leat_alignment_frame"] = (
    "inert_coupon",
    (0.076, 0.028, 0.14),
    "clamped",
    "rigid dry coupon mounted only in the lower indexing chuck",
)
FEATURES["duke_joint_mechanical_testing"] = _duke_features


def _wedge_mesh(b):
    name = b.unique("acoustic_wedge_mesh")
    # Triangular prism: rear face y=.12 and inward ridge y=-.16.
    vertices = [
        (-0.22, 0.12, -0.22),
        (0.22, 0.12, -0.22),
        (-0.22, 0.12, 0.22),
        (0.22, 0.12, 0.22),
        (-0.22, -0.16, 0),
        (0.22, -0.16, 0),
    ]
    faces = [
        (0, 1, 3),
        (0, 3, 2),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 5),
        (2, 5, 4),
        (0, 2, 4),
        (1, 5, 3),
    ]
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for p in vertices for v in p),
        face=" ".join(str(v) for f in faces for v in f),
    )
    return name


def _propeller(b, parent, name):
    rotor = b.moving(
        parent, name, (0, 0, 0), (0, 0, 1), (-0.8, 0.8), kind="hinge", mass=0.08, kp=100
    )
    b.cylinder(rotor, (0, 0, 0), 0.030, 0.020, "metal", collision=True)
    for sign in (-1, 1):
        # Tapered flat inert blade, not a calibrated aerodynamic profile.
        mesh = b.unique("inert_blade")
        vertices = []
        for z in (-0.006, 0.006):
            vertices.extend(
                [
                    (sign * 0.022, -0.022, z),
                    (sign * 0.38, -0.034, z),
                    (sign * 0.40, 0.006, z),
                    (sign * 0.055, 0.052, z),
                ]
            )
        faces = [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ]
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=mesh,
            vertex=" ".join(str(v) for p in vertices for v in p),
            face=" ".join(str(v) for f in faces for v in f),
        )
        b.geom(rotor, "mesh", (), material="cream", collision=True, mesh=mesh)
    b.box(rotor, (0.34, 0, 0.008), (0.027, 0.021, 0.002), "orange")
    return rotor


def _psu_rotors(b, base, params):
    # Upward assembly rises from a grounded extrusion inside a pink fairing.
    b.box(base, (-0.70, -0.12, 0.035), (0.27, 0.23, 0.035), "metal", collision=True)
    b.box(base, (-0.70, -0.12, 0.655), (0.038, 0.038, 0.585), "bright", collision=True)
    body = ET.SubElement(base, "body", pos="-.70 -.12 0")
    b.housing(
        body,
        [
            (0.72, 0.19, 0.14, 0),
            (0.83, 0.23, 0.17, 0),
            (1.20, 0.23, 0.17, 0),
            (1.27, 0.15, 0.14, 0),
        ],
        radius=0.07,
        material="sample",
    )
    b.box(base, (-0.70, -0.12, 1.27), (0.10, 0.10, 0.030), "metal", collision=True)
    b.cylinder(base, (-0.70, -0.12, 1.34), 0.070, 0.04, "dark", collision=True)
    mount = ET.SubElement(base, "body", pos="-.70 -.12 1.40")
    lower = _propeller(b, mount, "pedestal_rotor_index")
    # Inferred floor-supported gantry closes the load path above the source crop.
    for x in (-1.65, 1.65):
        b.box(base, (x, 1.02, 0.04), (0.15, 0.15, 0.04), "dark", collision=True)
        b.box(base, (x, 1.02, 1.415), (0.040, 0.040, 1.335), "metal", collision=True)
        b.box(base, (x, 0.46, 2.75), (0.040, 0.60, 0.040), "metal", collision=True)
    for y in (-0.10, 1.02):
        b.box(base, (0, y, 2.75), (1.69, 0.040, 0.040), "metal", collision=True)
    b.box(base, (0.70, 0.45, 2.75), (0.040, 0.61, 0.040), "metal", collision=True)
    b.cylinder(base, (0.70, 0.15, 2.49), 0.025, 0.22, "metal", collision=True)
    for z, r, h, material in (
        (2.22, 0.09, 0.07, "metal"),
        (2.08, 0.11, 0.07, "dark"),
        (1.95, 0.095, 0.06, "copper"),
        (1.83, 0.10, 0.06, "metal"),
        (1.73, 0.065, 0.04, "dark"),
    ):
        b.cylinder(base, (0.70, 0.15, z), r, h, material, collision=True)
    b.cylinder(base, (0.70, 0.15, 1.66), 0.025, 0.03, "bright", collision=True)
    upper_mount = ET.SubElement(base, "body", pos=".70 .15 1.61")
    _propeller(b, upper_mount, "suspended_rotor_index")
    b.site(lower, "rotor_marker", (0.34, 0, 0.012))
    for i, x in enumerate((-1.2, -0.35, 0.40, 1.15)):
        b.rod(base, (x, -0.10, 2.71), (x, -0.10, 2.18), 0.004, "dark")
        b.cylinder(base, (x, -0.10, 2.095), 0.014, 0.085, "dark", collision=True)
        b.cylinder(base, (x, -0.10, 1.993), 0.006, 0.017, "metal", collision=True)
    # Acoustic wedges have real triangular profiles rather than painted squares.
    mesh = _wedge_mesh(b)
    b.box(base, (0, 1.72, 1.45), (2.10, 0.05, 1.45), "dark", collision=True)
    for row in range(6):
        for col in range(9):
            b.geom(
                base,
                "mesh",
                (),
                (col * 0.46 - 1.84, 1.55, 0.23 + row * 0.46),
                "copper",
                collision=True,
                mesh=mesh,
                rgba=(0.38, 0.23, 0.12, 1),
            )
    b.box(base, (-2.15, 0.25, 1.45), (0.05, 1.42, 1.45), "dark", collision=True)
    for row in range(6):
        for col in range(6):
            b.geom(
                base,
                "mesh",
                (),
                (-1.98, col * 0.46 - 0.90, 0.23 + row * 0.46),
                "copper",
                collision=True,
                mesh=mesh,
                euler=(0, 0, math.pi / 2),
                rgba=(0.38, 0.23, 0.12, 1),
            )
    b.box(
        base,
        (2.0, 0.45, 1.10),
        (0.12, 0.47, 1.10),
        "dark",
        rgba=(0.32, 0.34, 0.34, 1),
        collision=True,
    )
    b.metadata["capabilities"] = [
        "bounded lower-rotor blade inspection indexing",
        "bounded suspended-rotor blade inspection indexing",
    ]
    b.metadata["limitations"].append(
        "Estimated partial anechoic bay. Upper support outside the source crop is inferred. Propellers are inert, low-speed bounded geometry, without motor/thrust, aerodynamic, sound-pressure or acoustic-absorption models."
    )


BUILDERS["pennstate_anechoic_tandem_rotors"] = _psu_rotors
SOURCES["pennstate_anechoic_tandem_rotors"] = {
    "reference": "Penn State Aeroacoustics tandem forward-flight rotor experiment photograph",
    "url": PSU,
    "dimensions_m": [4.42, 3.24, 2.90],
    "dimension_basis": "All apparatus, acoustic wedge and partial-room geometry estimated; upper load path inferred beyond source crop. No propeller performance claim.",
}
SAMPLE_INTERFACES["pennstate_anechoic_tandem_rotors"] = (
    "rotor_marker",
    (0.054, 0.042, 0.004),
    "clamped",
    "inert inspection marker fixed to the lower rotor blade",
)


def _charpy(b, base, params):
    _tag(
        b,
        b.box(base, (0, 0, 0.08), (0.43, 0.33, 0.08), "dark", collision=True),
        "charpy_plinth",
    )
    b.box(base, (0, 0, 0.22), (0.23, 0.20, 0.06), "metal", collision=True)
    for x in (-0.17, 0.17):
        b.box(base, (x, 0.04, 0.95), (0.033, 0.046, 0.67), "dark", collision=True)
        b.box(base, (x, -0.008, 0.95), (0.016, 0.004, 0.64), "metal")
    # Rounded lower case and sloping shoulders approximate the photographed head.
    b.housing(
        base,
        [
            (1.50, 0.32, 0.070, 0),
            (1.59, 0.43, 0.070, 0),
            (2.11, 0.43, 0.070, 0),
            (2.27, 0.24, 0.070, 0),
        ],
        radius=0.045,
        material="metal",
    )
    b.box(base, (0, 0.035, 1.85), (0.29, 0.07, 0.26), "metal", collision=True)
    for i in range(34):
        a = math.radians(23 + i * 4)
        c = a + math.radians(4)
        b.rod(
            base,
            (0.405 * math.cos(a), -0.082, 1.91 + 0.405 * math.sin(a)),
            (0.405 * math.cos(c), -0.082, 1.91 + 0.405 * math.sin(c)),
            0.031,
            "dark",
        )
        b.rod(
            base,
            (0.390 * math.cos(a), -0.116, 1.91 + 0.390 * math.sin(a)),
            (0.416 * math.cos(a), -0.116, 1.91 + 0.416 * math.sin(a)),
            0.0017,
            "cream",
        )
    b.box(base, (0, -0.075, 1.945), (0.095, 0.003, 0.026), "guard_red")
    b.cylinder(base, (0, -0.13, 1.91), 0.035, 0.10, "bright", euler=(math.pi / 2, 0, 0))
    pendulum = b.moving(
        base,
        "pendulum_inspection_angle",
        (0, -0.24, 1.91),
        (0, 1, 0),
        (-0.35, 0.35),
        kind="hinge",
        mass=0.70,
        kp=240,
    )
    b.cylinder(
        pendulum,
        (0, 0, 0),
        0.045,
        0.023,
        "dark",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    b.geom(
        pendulum,
        "capsule",
        (0.020,),
        material="metal",
        collision=True,
        fromto=(0, 0, 0, 0.44, 0, 0.30),
    )
    b.box(
        pendulum,
        (0.43, 0, 0.30),
        (0.10, 0.028, 0.045),
        "metal",
        collision=True,
        euler=(0, -0.60, 0),
    )
    b.geom(
        pendulum,
        "capsule",
        (0.010,),
        material="dark",
        fromto=(0.46, 0, 0.33, 0.57, 0, 0.41),
    )
    b.geom(pendulum, "sphere", (0.024,), (0.57, 0, 0.41), "dark", collision=True)
    # Dry replaceable-coupon carriage is authored; source does not resolve jaws.
    b.box(base, (0, -0.02, 0.30), (0.18, 0.16, 0.020), "metal", collision=True)
    for y in (-0.12, 0.08):
        _tag(
            b,
            b.box(base, (0, y, 0.33), (0.17, 0.010, 0.010), "bright"),
            "charpy_coupon_guide",
        )
    stage = b.moving(
        base,
        "coupon_alignment",
        (0, -0.02, 0.354),
        (1, 0, 0),
        (-0.035, 0.035),
        mass=0.16,
        kp=320,
    )
    b.box(stage, (0, 0, 0), (0.095, 0.10, 0.014), "dark", collision=True)
    for x in (-0.065, 0.065):
        b.box(stage, (x, 0, 0.032), (0.025, 0.07, 0.018), "metal", collision=True)
    b.box(stage, (0, 0, 0.057), (0.087, 0.019, 0.007), "orange", collision=True)
    b.site(stage, "charpy_coupon", (0, -0.021, 0.057))
    b.metadata["capabilities"] = [
        "bounded unloaded pendulum inspection",
        "dry coupon alignment on an authored carriage",
    ]
    b.metadata["limitations"].append(
        "Apparatus-only P2 reference from a 210x300 image. Dimensions, hidden clamp and translation carriage estimated. Pendulum stays raised and unloaded; no impact, release, energy calibration or fracture mechanics model."
    )


BUILDERS["northwestern_clammp_charpy_inspection"] = _charpy
SOURCES["northwestern_clammp_charpy_inspection"] = {
    "reference": "Northwestern CLaMMP installed Tinius Olson Charpy photograph",
    "url": NORTHWESTERN,
    "dimensions_m": [1.10, 0.66, 2.50],
    "dimension_basis": "Estimated apparatus-only exterior from low-resolution imagery; coupon fixture and bounded inspection strokes authored. No dimensional or impact calibration.",
}
SAMPLE_INTERFACES["northwestern_clammp_charpy_inspection"] = (
    "charpy_coupon",
    (0.174, 0.038, 0.014),
    "clamped",
    "inert rigid bar fixed on the dry alignment carriage",
)
