"""Photo-informed wind-tunnel and water-channel mechanical inspection scenes."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box


ZJU = "https://atc.zju.edu.cn/instrument/detail-4202.html"
HKU = "https://www.civil.hku.hk/h0_facilities_lab.html"
NTU = "https://www.iam.ntu.edu.tw/zh/research/teaching-lab"
OSAKA = "https://www-naoe.eng.osaka-u.ac.jp/naoe/naoe1/e/facility/"


def _material(b, name, color):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_{name}",
        rgba=" ".join(map(str, color)),
    )


def _frame(b, parent, x, half_width, half_height, z, material, thickness=0.035):
    """Open rectangular frame in the YZ plane."""
    for side in (-1, 1):
        b.box(
            parent,
            (x, side * half_width, z),
            (thickness, thickness, half_height + thickness),
            material,
        )
        b.box(
            parent,
            (x, 0, z + side * half_height),
            (thickness, half_width, thickness),
            material,
        )


def _sheet(b, parent, corners, material):
    """A thin watertight trapezoid, preserving all four taper corners."""
    first = [corners[1][i] - corners[0][i] for i in range(3)]
    second = [corners[2][i] - corners[0][i] for i in range(3)]
    normal = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
    length = math.sqrt(sum(value * value for value in normal))
    normal = [value * 0.012 / length for value in normal]
    vertices = [
        tuple(point[i] + sign * normal[i] for i in range(3))
        for sign in (-1, 1)
        for point in corners
    ]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for index in range(4):
        following = (index + 1) % 4
        faces.extend(
            [(index, following, following + 4), (index, following + 4, index + 4)]
        )
    name = b.unique("duct_sheet")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for point in vertices for v in point),
        face=" ".join(str(v) for face in faces for v in face),
    )
    b.geom(parent, "mesh", (), material=material, mesh=name)


def _transition(b, parent, x1, x2, width1, width2, height1, height2, z, material):
    """Four joined trapezoid panels form an open rectangular duct."""
    for side in (-1, 1):
        _sheet(
            b,
            parent,
            [
                (x1, side * width1, z - height1),
                (x2, side * width2, z - height2),
                (x2, side * width2, z + height2),
                (x1, side * width1, z + height1),
            ],
            material,
        )
        _sheet(
            b,
            parent,
            [
                (x1, -width1, z + side * height1),
                (x2, -width2, z + side * height2),
                (x2, width2, z + side * height2),
                (x1, width1, z + side * height1),
            ],
            material,
        )


def _lswt(b, base, params):
    _material(b, "tunnel_blue", (0.09, 0.22, 0.34, 1))
    # Published test-section envelope is 3.5 x 1.2 x 1.2 m; shell and bay are estimates.
    for x in (-1.65, 0, 1.65):
        for y in (-0.59, 0.59):
            b.box(base, (x, y, 0.04), (0.18, 0.16, 0.04), "dark", collision=True)
            b.box(base, (x, y, 0.53), (0.08, 0.08, 0.49), "tunnel_blue", collision=True)
        _frame(b, base, x, 0.65, 0.65, 1.66, "tunnel_blue", 0.055)
    b.box(base, (0, 0, 1.035), (1.75, 0.62, 0.035), "metal", collision=True)
    b.box(base, (0, 0.625, 1.67), (1.75, 0.025, 0.60), "tunnel_blue", collision=True)
    b.box(base, (0, 0, 2.295), (1.75, 0.70, 0.025), "tunnel_blue", collision=True)
    # Front access glazing with wide perimeter, hinges and hand wheels.
    b.box(base, (0.40, -0.641, 1.67), (0.67, 0.009, 0.48), "glass")
    for x in (-1.00, 1.46):
        b.box(
            base,
            (x, -0.64, 1.67),
            (0.28 if x > 0 else 0.75, 0.028, 0.60),
            "tunnel_blue",
            collision=True,
        )
    for x in (-0.34, 1.14):
        b.box(base, (x, -0.673, 1.67), (0.045, 0.055, 0.59), "tunnel_blue")
    for z in (1.125, 2.215):
        b.box(base, (0.40, -0.67, z), (0.78, 0.06, 0.045), "tunnel_blue")
    for z in (1.36, 1.98):
        b.cylinder(
            base, (1.16, -0.728, z), 0.045, 0.018, "metal", euler=(math.pi / 2, 0, 0)
        )
        b.rod(base, (-0.36, -0.74, z - 0.05), (-0.36, -0.74, z + 0.05), 0.012, "bright")
    _transition(b, base, -3.5, -1.75, 1.15, 0.65, 1.25, 0.65, 1.66, "tunnel_blue")
    for x, w, h in ((-3.5, 1.15, 1.25), (-2.90, 0.98, 1.04), (-2.30, 0.81, 0.85)):
        _frame(b, base, x, w, h, 1.66, "tunnel_blue", 0.065)
    for y in (-0.94, 0.94):
        b.box(base, (-3.15, y, 0.22), (0.34, 0.10, 0.22), "tunnel_blue", collision=True)
    _transition(b, base, 1.75, 3.05, 0.65, 0.80, 0.65, 0.85, 1.66, "tunnel_blue")
    _frame(b, base, 3.05, 0.80, 0.85, 1.66, "tunnel_blue", 0.065)
    for y in (-0.68, 0.68):
        b.box(base, (2.8, y, 0.40), (0.10, 0.10, 0.40), "tunnel_blue", collision=True)
    # Roof walkway is physically attached to the tunnel structure.
    for x in (-1.55, -0.75, 0.05, 0.85, 1.65):
        for y in (-0.70, 0.70):
            b.rod(base, (x, y, 2.30), (x, y, 3.28), 0.021, "tunnel_blue")
    for y in (-0.70, 0.70):
        for z in (2.80, 3.28):
            b.rod(base, (-1.55, y, z), (1.65, y, z), 0.022, "tunnel_blue")
    for x in (-1.52, -1.02):
        b.rod(base, (x, -0.87, 0.02), (x, -0.87, 3.05), 0.021, "metal")
        for z in (0.35, 1.2, 2.2):
            b.rod(base, (x, -0.87, z), (x, -0.66, z), 0.016, "metal")
    for index in range(10):
        b.rod(
            base,
            (-1.52, -0.87, 0.2 + index * 0.24),
            (-1.02, -0.87, 0.2 + index * 0.24),
            0.018,
            "metal",
        )
    b.cylinder(
        base, (0.4, 0, 1.205), 0.105, 0.135, "metal"
    )  # ideal telescopic bearing sleeve
    lift = b.moving(
        base,
        "sting_height",
        (0.4, 0, 1.32),
        (0, 0, 1),
        (-0.04, 0.14),
        mass=0.20,
        kp=300,
    )
    b.cylinder(lift, (0, 0, 0), 0.04, 0.15, "bright", collision=True)
    turn = b.moving(
        lift,
        "model_yaw",
        (0, 0, 0.17),
        (0, 0, 1),
        (-0.65, 0.65),
        kind="hinge",
        mass=0.15,
        kp=230,
    )
    b.cylinder(turn, (0, 0, 0), 0.065, 0.02, "dark", collision=True)
    b.geom(
        turn, "ellipsoid", (0.25, 0.04, 0.035), (0, 0, 0.055), "sample", collision=True
    )
    b.box(turn, (0, 0, 0.055), (0.085, 0.27, 0.008), "metal")
    b.box(turn, (-0.18, 0, 0.063), (0.04, 0.11, 0.006), "metal")
    b.site(turn, "wind_model", (0, 0, 0.055), 0.002)
    b.metadata["capabilities"] = [
        "sting_height",
        "model_yaw",
        "visible_inert_wind_model",
    ]
    b.metadata["limitations"].append(
        "Original inspection sting and inert target; no airflow, turbulence, forces or full tunnel-loop reconstruction. Only test-section dimensions are published."
    )


def _zju_features(world, definition):
    # High structural bay seen in the photo; column spacing is not surveyed.
    for x in (-3.8, 0.1, 3.8):
        box(
            world,
            f"lswt_bay_column_{x}",
            (0.15, 0.16, 2.10),
            (x, 2.59, 2.10),
            (0.83, 0.83, 0.80, 1),
        )
    box(
        world,
        "lswt_bay_beam",
        (4.65, 0.18, 0.18),
        (0, 2.57, 4.0),
        (0.80, 0.81, 0.78, 1),
    )


BUILDERS = {"lswt_model_inspection_workspace": _lswt}
SOURCES = {
    "lswt_model_inspection_workspace": dict(
        reference="ZJU LSWT installed blue wind tunnel",
        url=ZJU,
        dimensions_m=[6.70, 2.45, 3.32],
        dimension_basis="Published test section 3.5 L x 1.2 W x 1.2 H m; external shell, walkway and original sting are estimates.",
    )
}
SAMPLE_INTERFACES = {
    "lswt_model_inspection_workspace": (
        "wind_model",
        (0.50, 0.54, 0.07),
        "clamped",
        "Original inert wind-model geometry on a supported inspection sting",
    )
}
FEATURES = {"zju_low_turbulence_tunnel": _zju_features}


def _boundary_layer(b, base, params):
    _material(b, "plywood", (0.42, 0.25, 0.12, 1))
    _material(b, "roughness", (0.60, 0.50, 0.31, 1))
    # Published 12 x 3 x 1.8 m working section. The front wall is cut away for viewing.
    b.box(base, (0, 0, 0.075), (6.0, 1.50, 0.075), "plywood", collision=True)
    b.box(base, (0, 1.515, 1.05), (6.0, 0.025, 0.90), "plywood", collision=True)
    for x in (-5.9, -3.9, -1.9, 0.1, 2.1, 4.1, 5.9):
        b.box(base, (x, 1.47, 1.05), (0.025, 0.018, 0.90), "roughness")
    for z in (0.18, 1.92):
        b.box(base, (0, 1.46, z), (6.0, 0.025, 0.025), "roughness")
    # Sparse visual grille and roughness elements follow the photographed upstream region.
    for index in range(16):
        y = -1.45 + index * 2.90 / 15
        b.box(base, (-5.85, y, 1.03), (0.018, 0.012, 0.88), "metal")
    for index in range(10):
        b.box(base, (-5.85, 0, 0.16 + index * 0.195), (0.018, 1.47, 0.012), "metal")
    for i in range(14):
        for j in range(9):
            b.box(
                base,
                (-5.50 + i * 0.30, -1.20 + j * 0.30, 0.183),
                (0.027, 0.027, 0.033),
                "roughness",
            )
    b.cylinder(base, (2.0, 0, 0.158), 0.62, 0.008, "dark")
    turn = b.moving(
        base,
        "model_yaw",
        (2.0, 0, 0.178),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.35,
        kp=300,
    )
    b.cylinder(turn, (0, 0, 0), 0.59, 0.010, "plywood", collision=True)
    b.box(turn, (0, 0, 0.022), (0.13, 0.13, 0.012), "metal", collision=True)
    b.box(turn, (0, 0, 0.25), (0.075, 0.075, 0.216), "glass", collision=True)
    for x in (-0.075, 0.075):
        for y in (-0.075, 0.075):
            b.rod(turn, (x, y, 0.034), (x, y, 0.466), 0.003, "bright")
    b.box(turn, (0, 0, 0.036), (0.065, 0.065, 0.002), "sample")
    b.site(turn, "building_coupon", (0, 0, 0.25), 0.002)
    for angle in range(12):
        a = angle * math.tau / 12
        b.cylinder(
            turn, (0.55 * math.cos(a), 0.55 * math.sin(a), 0.012), 0.008, 0.002, "metal"
        )
    # Rear linear carriage rests on short feet, separate from the model turntable.
    for x in (0.50, 1.9):
        b.box(base, (x, 1.12, 0.185), (0.07, 0.17, 0.035), "metal")
    b.box(base, (1.2, 1.12, 0.245), (0.78, 0.18, 0.025), "metal", collision=True)
    for y in (1.04, 1.20):
        b.rod(base, (0.45, y, 0.29), (1.95, y, 0.29), 0.012, "bright")
    traverse = b.moving(
        base,
        "optical_traverse",
        (1.2, 1.12, 0.335),
        (1, 0, 0),
        (-0.40, 0.40),
        mass=0.18,
        kp=280,
    )
    b.box(traverse, (0, 0, 0), (0.12, 0.15, 0.038), "dark", collision=True)
    b.rod(traverse, (0, 0, 0.04), (0, 0, 0.45), 0.018, "dark")
    b.rod(traverse, (0, 0, 0.45), (-0.4, 0, 0.45), 0.020, "bright")
    b.box(traverse, (-0.4, -0.04, 0.45), (0.065, 0.05, 0.04), "dark")
    # Tripod reference geometry, with feet on the tunnel floor and no emitted beam.
    for a in (0, math.tau / 3, 2 * math.tau / 3):
        b.rod(
            base,
            (0.65 + 0.36 * math.cos(a), -0.62 + 0.36 * math.sin(a), 0.16),
            (0.65, -0.62, 0.91),
            0.014,
            "dark",
        )
    b.box(base, (0.65, -0.62, 0.97), (0.10, 0.08, 0.06), "dark")
    b.rod(base, (0.67, -0.61, 1.02), (1.11, -0.38, 1.10), 0.024, "metal")
    b.cylinder(
        base, (1.13, -0.37, 1.10), 0.041, 0.05, "dark", euler=(0, math.pi / 2, 0)
    )
    b.metadata["capabilities"] = [
        "model_yaw",
        "optical_carriage_translation",
        "visible_inert_building_coupon",
    ]
    b.metadata["limitations"].append(
        "Working-section cutaway with original turntable and carriage mechanics. No airflow, optical sheet, PIV, pressure or structural response simulation; no full tunnel loop."
    )


def _hku_features(world, definition):
    box(
        world,
        "hku_tunnel_access_mark",
        (5.8, 0.025, 0.002),
        (0, -1.48, 0.003),
        (0.73, 0.62, 0.31, 1),
        collision=False,
    )


BUILDERS["boundary_layer_inspection_workspace"] = _boundary_layer
SOURCES["boundary_layer_inspection_workspace"] = dict(
    reference="HKU Civil Engineering actual boundary-layer working-section photographs",
    url=HKU,
    dimensions_m=[12.0, 3.08, 1.95],
    dimension_basis="Published working section 12 L x 3 W x 1.8 H m above its floor. Turntable, roughness and optics dimensions are estimated; exterior bay is not surveyed.",
)
SAMPLE_INTERFACES["boundary_layer_inspection_workspace"] = (
    "building_coupon",
    (0.15, 0.15, 0.432),
    "clamped",
    "Transparent inert building model fixed to the yawable floor disc",
)
FEATURES["hku_civil_boundary_layer_tunnel"] = _hku_features


def _teaching_tunnel(b, base, params):
    _material(b, "yellow", (0.91, 0.69, 0.025, 1))
    # Frames carry the inlet, test section and circular downstream shell separately.
    for x, w in ((-2.3, 0.62), (-1.3, 0.52), (0, 0.34), (1.4, 0.39), (2.2, 0.39)):
        for y in (-w, w):
            b.cylinder(
                base, (x, y, 0.08), 0.08, 0.025, "rubber", euler=(math.pi / 2, 0, 0)
            )
            b.box(base, (x, y, 0.135), (0.03, 0.04, 0.04), "metal")
            b.box(base, (x, y, 0.52), (0.035, 0.035, 0.355), "yellow", collision=True)
        for z in (0.19, 0.86):
            b.box(base, (x, 0, z), (0.04, w + 0.04, 0.04), "yellow")
    for y in (-0.40, 0.40):
        b.box(base, (0, y, 0.20), (2.38, 0.035, 0.035), "yellow")
    for x in (-2.55, -2.30, -2.05, -1.8):
        _frame(b, base, x, 0.69, 0.75, 1.28, "yellow", 0.040)
        for j in range(9):
            for side in (-1, 1):
                b.cylinder(
                    base,
                    (x - 0.045, side * 0.70, 0.59 + j * 0.173),
                    0.008,
                    0.005,
                    "metal",
                    euler=(0, math.pi / 2, 0),
                )
    _transition(b, base, -2.55, -1.80, 0.65, 0.65, 0.71, 0.71, 1.28, "yellow")
    b.box(base, (-2.45, 0, 1.28), (0.008, 0.64, 0.70), "dark")
    _transition(b, base, -1.80, -0.55, 0.65, 0.24, 0.71, 0.24, 1.28, "yellow")
    for x in (-0.55, 0.55):
        _frame(b, base, x, 0.24, 0.24, 1.28, "yellow", 0.025)
    b.box(base, (0, 0, 1.04), (0.56, 0.25, 0.018), "metal", collision=True)
    for y in (-0.248, 0.248):
        b.box(base, (0, y, 1.28), (0.55, 0.008, 0.22), "glass")
    b.box(base, (-0.25, 0, 1.522), (0.28, 0.24, 0.007), "glass")
    b.box(base, (0.47, 0, 1.522), (0.06, 0.24, 0.007), "glass")
    # Circular exit sections: closed exterior, with no simulated fan or flow volume.
    for x, radius, half_length in (
        (0.79, 0.27, 0.24),
        (1.32, 0.34, 0.30),
        (1.98, 0.36, 0.36),
    ):
        b.cylinder(
            base, (x, 0, 1.28), radius, half_length, "yellow", euler=(0, math.pi / 2, 0)
        )
        for end in (-1, 1):
            b.cylinder(
                base,
                (x + end * half_length, 0, 1.28),
                radius + 0.035,
                0.025,
                "yellow",
                euler=(0, math.pi / 2, 0),
            )
    b.cylinder(base, (2.36, 0, 1.28), 0.32, 0.006, "dark", euler=(0, math.pi / 2, 0))
    b.box(base, (0, 0, 1.077), (0.10, 0.10, 0.019), "metal")
    turn = b.moving(
        base,
        "cylinder_yaw",
        (0, 0, 1.113),
        (0, 0, 1),
        (-0.80, 0.80),
        kind="hinge",
        mass=0.07,
        kp=180,
    )
    b.cylinder(turn, (0, 0, 0), 0.078, 0.016, "dark", collision=True)
    b.cylinder(turn, (0, 0, 0.12), 0.028, 0.104, "metal", collision=True)
    b.box(turn, (0.027, 0, 0.12), (0.002, 0.004, 0.08), "sample")
    b.site(turn, "cylinder_coupon", (0, 0, 0.12), 0.0006)
    for y in (-0.29, 0.29):
        b.rod(base, (0.23, y, 0.87), (0.23, y, 1.95), 0.012, "bright")
    b.box(base, (0.23, 0, 1.95), (0.05, 0.32, 0.025), "yellow")
    probe = b.moving(
        base,
        "probe_height",
        (0.23, 0, 1.70),
        (0, 0, 1),
        (-0.06, 0.06),
        mass=0.065,
        kp=220,
    )
    b.box(probe, (0, 0, 0), (0.035, 0.035, 0.055), "dark", collision=True)
    b.rod(probe, (0, 0, -0.055), (0, 0, -0.28), 0.0035, "bright")
    b.rod(probe, (0, 0, 0.055), (0, 0, 0.23), 0.010, "metal")
    # Thin tubing terminates at the exterior acquisition side, never drives a flow solver.
    for index in range(9):
        x = -0.38 + index * 0.085
        b.rod(
            base, (x, -0.26, 1.14), (0.45 + index * 0.022, -0.55, 0.79), 0.002, "cream"
        )
        b.rod(
            base,
            (0.45 + index * 0.022, -0.55, 0.79),
            (0.90 + index * 0.022, -0.85, 0.92),
            0.002,
            "cream",
        )
    b.metadata["capabilities"] = [
        "inert_cylinder_yaw",
        "probe_height",
        "visible_inert_cylinder",
    ]
    b.metadata["limitations"].append(
        "Estimated teaching-tunnel exterior and original two-axis inspection fixture. Upper probe slot is an authored access opening; no pressure, flow, fan performance or real device control."
    )


def _ntu_features(world, definition):
    # The acquisition computer and instrument stand directly on the separate bench.
    box(
        world,
        "iam_acquisition",
        (0.25, 0.18, 0.10),
        (1.12, -1.0, 0.93),
        (0.79, 0.79, 0.69, 1),
    )
    box(
        world,
        "iam_front_panel",
        (0.22, 0.005, 0.072),
        (1.12, -1.185, 0.93),
        (0.56, 0.58, 0.51, 1),
        collision=False,
    )
    for i in range(8):
        box(
            world,
            f"iam_port_{i}",
            (0.009, 0.005, 0.013),
            (0.96 + i * 0.045, -1.195, 0.92),
            (0.16, 0.18, 0.18, 1),
            collision=False,
        )
    box(
        world,
        "iam_display_foot",
        (0.15, 0.09, 0.015),
        (1.64, -0.98, 0.845),
        (0.12, 0.13, 0.14, 1),
    )
    box(
        world,
        "iam_display_stem",
        (0.027, 0.025, 0.14),
        (1.64, -0.98, 0.99),
        (0.12, 0.13, 0.14, 1),
    )
    box(
        world,
        "iam_display",
        (0.23, 0.025, 0.16),
        (1.64, -0.98, 1.18),
        (0.07, 0.11, 0.13, 1),
    )


BUILDERS["iam_teaching_tunnel_workspace"] = _teaching_tunnel
SOURCES["iam_teaching_tunnel_workspace"] = dict(
    reference="NTU Taiwan IAM installed yellow teaching wind tunnel",
    url=NTU,
    dimensions_m=[5.03, 1.47, 2.10],
    dimension_basis="All dimensions estimated from the installed photograph; yellow bolted inlet, wheeled supports, glazed test area and round exit are observed.",
)
SAMPLE_INTERFACES["iam_teaching_tunnel_workspace"] = (
    "cylinder_coupon",
    (0.056, 0.056, 0.208),
    "clamped",
    "Visible inert cylinder mounted on the supported inspection disc",
)
FEATURES["ntu_taiwan_iam_fluids"] = _ntu_features


def _osaka_channel(b, base, params):
    _material(b, "turquoise", (0.08, 0.43, 0.48, 1))
    _material(b, "still_water", (0.24, 0.58, 0.67, 0.20))
    # Published internal tank dimensions 14.0 x 0.30 x 0.45 m.
    b.box(base, (0, 0, 0.71), (7.0, 0.15, 0.015), "metal", collision=True)
    for y in (-0.16, 0.16):
        b.box(base, (0, y, 0.95), (7.0, 0.01, 0.225), "glass", collision=True)
        for z in (0.70, 1.20):
            b.box(base, (0, y, z), (7.10, 0.035, 0.03), "turquoise")
    for x in (-7.01, 7.01):
        b.box(base, (x, 0, 0.95), (0.01, 0.18, 0.24), "turquoise", collision=True)
    for x in (-6.8, -4.8, -2.8, -0.8, 1.2, 3.2, 5.2, 6.8):
        b.box(base, (x, 0, 0.035), (0.11, 0.34, 0.035), "turquoise", collision=True)
        for y in (-0.20, 0.20):
            b.box(
                base, (x, y, 0.385), (0.035, 0.035, 0.315), "turquoise", collision=True
            )
            b.box(base, (x, y, 0.95), (0.025, 0.025, 0.25), "turquoise")
        b.box(base, (x, 0, 0.68), (0.05, 0.25, 0.03), "turquoise")
    b.box(base, (0, 0, 1.045), (6.98, 0.147, 0.006), "still_water")
    # End profiles are grounded; only a short local traverse is actuated.
    for y in (-0.40, 0.40):
        b.box(base, (-6.45, y, 0.025), (0.17, 0.14, 0.025), "metal", collision=True)
        b.box(base, (-6.45, y, 1.08), (0.035, 0.035, 1.03), "metal", collision=True)
        b.box(base, (-6.473, y - 0.036, 1.08), (0.004, 0.002, 1.0), "dark")
    b.box(base, (-6.45, 0, 2.13), (0.045, 0.45, 0.035), "metal")
    for y in (-0.23, 0.23):
        b.box(base, (-4.2, y, 1.255), (1.0, 0.032, 0.025), "metal")
        for x in (-5.1, -3.3):
            b.box(base, (x, y, 1.217), (0.055, 0.055, 0.013), "dark")
    carriage = b.moving(
        base,
        "carriage_x",
        (-4.2, 0, 1.315),
        (1, 0, 0),
        (-0.65, 0.65),
        mass=0.22,
        kp=340,
    )
    b.box(carriage, (0, 0, 0), (0.14, 0.30, 0.03), "metal", collision=True)
    for y in (-0.23, 0.23):
        b.box(carriage, (0, y, -0.023), (0.12, 0.035, 0.012), "dark")
    b.box(carriage, (0, 0.265, 0.25), (0.034, 0.03, 0.235), "metal")
    b.box(carriage, (0, 0.13, 0.48), (0.055, 0.165, 0.025), "dark")
    lift = b.moving(
        carriage,
        "specimen_height",
        (0, 0, 0.09),
        (0, 0, 1),
        (-0.06, 0.06),
        mass=0.07,
        kp=210,
    )
    b.cylinder(lift, (0, 0, 0.13), 0.013, 0.25, "bright")
    b.cylinder(lift, (0, 0, -0.175), 0.007, 0.09, "metal")
    b.cylinder(lift, (0, 0, -0.285), 0.034, 0.05, "sample", collision=True)
    b.site(lift, "channel_coupon", (0, 0, -0.285), 0.0008)
    # Distant stationary cross-fixture and drainage hose are observed layout cues.
    b.box(base, (4.5, 0, 1.25), (0.09, 0.27, 0.035), "metal")
    b.rod(base, (6.8, -0.2, 0.75), (6.8, -0.45, 0.08), 0.018, "cyan")
    b.rod(base, (6.8, -0.45, 0.08), (5.8, -0.62, 0.04), 0.018, "cyan")
    b.metadata["capabilities"] = [
        "local_carriage_translation",
        "inert_specimen_height",
        "visible_clamped_channel_coupon",
    ]
    b.metadata["limitations"].append(
        "Static water visualization only: no waves, buoyancy, hydrodynamic load or towing performance. Low-resolution source supports coarse layout; the local two-axis fixture is original."
    )


def _osaka_features(world, definition):
    for index in range(4):
        x = -4.8 + index * 3.2
        box(
            world,
            f"osaka_window_sill_{index}",
            (1.14, 0.08, 0.035),
            (x, 1.75, 0.89),
            (0.64, 0.65, 0.62, 1),
        )
    # Estimated rear instrument cabinet, kept outside the clear front access aisle.
    box(
        world,
        "osaka_end_cabinet",
        (0.45, 0.28, 0.48),
        (7.85, 1.18, 0.48),
        (0.64, 0.70, 0.70, 1),
    )
    box(
        world,
        "osaka_end_panel",
        (0.38, 0.01, 0.15),
        (7.85, 0.89, 0.72),
        (0.17, 0.23, 0.24, 1),
    )


BUILDERS["osaka_narrow_channel_workspace"] = _osaka_channel
SOURCES["osaka_narrow_channel_workspace"] = dict(
    reference="University of Osaka Ocean Space Development two-dimensional wave tank",
    url=OSAKA,
    dimensions_m=[14.25, 0.92, 2.18],
    dimension_basis="Published tank clear dimensions 14 L x 0.30 W x 0.45 depth m. Frame height, carriage and room estimated from the coarse installed photograph; other departmental basins are excluded.",
)
SAMPLE_INTERFACES["osaka_narrow_channel_workspace"] = (
    "channel_coupon",
    (0.068, 0.068, 0.10),
    "clamped",
    "Inert cylindrical test coupon visibly fixed to a supported local traverse",
)
FEATURES["osaka_ocean_wave_lab"] = _osaka_features
