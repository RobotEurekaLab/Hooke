"""Original room-520 furniture and services informed by tour photograph 69.

The image establishes visible object types and adjacency, not dimensions,
electrical connectivity or hidden room boundaries. Furniture is static contact
geometry; fabric and cable dressing are presentation surfaces, not cloth or
electrical simulation. The enclosure's articulated mechanism is unchanged.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from real_labs.architecture import numbers

PREFIX = "covered_optical_table__bay_"
WOOD = (0.34, 0.19, 0.085, 1)


def _body(b, key, position):
    return ET.SubElement(
        b.root.find("worldbody"), "body", name=PREFIX + key, pos=numbers(position)
    )


def _box(b, parent, key, pos, half, material="dark", **kwargs):
    return b.box(
        parent, pos, half, material, name=PREFIX + key, collision=True, **kwargs
    )


def _lead(b, parent, key, points):
    # Fixed routed leads end at visible sockets; no cable force is implied.
    for index, (start, end) in enumerate(zip(points, points[1:])):
        b.rod(parent, start, end, 0.004, "rubber", name=PREFIX + f"{key}_{index}")


def _chair(b):
    chair = _body(b, "wood_chair", (-1.28, -1.64, 0))
    for x in (-0.25, 0.25):
        for y in (-0.23, 0.23):
            _box(
                b,
                chair,
                f"chair_leg_{x}_{y}",
                (x, y, 0.22),
                (0.025, 0.025, 0.22),
                "cream",
                rgba=WOOD,
            )
        _box(
            b,
            chair,
            f"chair_seat_rail_{x}",
            (x, 0, 0.42),
            (0.028, 0.26, 0.034),
            "cream",
            rgba=WOOD,
        )
        _box(
            b,
            chair,
            f"chair_back_post_{x}",
            (x, -0.23, 0.67),
            (0.025, 0.025, 0.25),
            "cream",
            rgba=WOOD,
        )
        _box(
            b,
            chair,
            f"chair_arm_post_{x}",
            (x, 0.23, 0.52),
            (0.021, 0.021, 0.10),
            "cream",
            rgba=WOOD,
        )
        _box(
            b,
            chair,
            f"chair_arm_{x}",
            (x, 0, 0.64),
            (0.035, 0.29, 0.028),
            "cream",
            rgba=WOOD,
        )
    _box(b, chair, "chair_seat", (0, 0, 0.465), (0.23, 0.24, 0.035), "rubber")
    _box(b, chair, "chair_back", (0, -0.23, 0.765), (0.23, 0.028, 0.14), "rubber")
    _box(
        b,
        chair,
        "chair_back_top",
        (0, -0.23, 0.92),
        (0.275, 0.03, 0.025),
        "cream",
        rgba=WOOD,
    )


def _stool(b, key, position, height):
    stool = _body(b, key, position)
    b.cylinder(
        stool,
        (0, 0, height),
        0.22,
        0.03,
        "rubber",
        collision=True,
        name=PREFIX + key + "_seat",
    )
    # Four fixed, splayed tubular legs match the photographed stool family.
    for i in range(4):
        angle = math.pi / 4 + i * math.pi / 2
        lower = (0.22 * math.cos(angle), 0.22 * math.sin(angle), 0.016)
        upper = (0.15 * math.cos(angle), 0.15 * math.sin(angle), height - 0.025)
        b.rod(
            stool,
            lower,
            upper,
            0.016,
            "dark",
            collision=True,
            name=PREFIX + f"{key}_leg_{i}",
        )
    b.ring(stool, (0, 0, 0.24), 0.195, 0.011, "metal", segments=40)


def _controller(b, parent, key, x, y, bottom):
    # Paired white desk units are visible, but their model and function are
    # unresolved. Knobs/displays are visual controls with no invented output.
    for sign in (-1, 1):
        for rear in (-1, 1):
            _box(
                b,
                parent,
                f"{key}_foot_{sign}_{rear}",
                (x + sign * 0.125, y + rear * 0.105, bottom + 0.008),
                (0.014, 0.014, 0.008),
                "rubber",
            )
    _box(
        b, parent, key + "_case", (x, y, bottom + 0.078), (0.165, 0.145, 0.062), "cream"
    )
    _box(
        b,
        parent,
        key + "_panel",
        (x, y - 0.147, bottom + 0.078),
        (0.155, 0.002, 0.052),
        "shell",
    )
    _box(
        b,
        parent,
        key + "_display",
        (x - 0.065, y - 0.151, bottom + 0.082),
        (0.045, 0.002, 0.024),
        "screen",
    )
    for dx in (0.035, 0.10):
        b.cylinder(
            parent,
            (x + dx, y - 0.157, bottom + 0.076),
            0.014,
            0.008,
            "dark",
            euler=(math.pi / 2, 0, 0),
            name=PREFIX + f"{key}_knob_{dx}",
        )
    _box(
        b,
        parent,
        key + "_socket",
        (x, y + 0.149, bottom + 0.07),
        (0.013, 0.004, 0.01),
        "rubber",
    )


def _desk_services(b):
    desk = _body(b, "desk_services", (-2.22, -0.4, 0.82))
    for index in range(2):
        _controller(b, desk, f"desk_controller_{index}", 0.39, 0.30, index * 0.14)
        _lead(
            b,
            desk,
            f"controller_lead_{index}",
            [
                (0.39, 0.453, index * 0.14 + 0.07),
                (0.53, 0.51, 0.025),
                (0.58, 0.55, -0.18),
                (0.36, 0.507, -0.58),
            ],
        )
    _box(b, desk, "mouse", (0.34, -0.32, 0.02), (0.030, 0.050, 0.020), "rubber")
    _lead(
        b,
        desk,
        "mouse_lead",
        [
            (0.34, -0.27, 0.02),
            (0.34, -0.16, 0.016),
            (0.46, -0.08, 0.016),
            (0.56, 0.15, -0.10),
            (0.36, 0.507, -0.58),
        ],
    )
    tower = _body(b, "desktop_tower", (-1.93, -0.13, 0))
    _box(b, tower, "tower_case", (0, 0, 0.31), (0.13, 0.23, 0.27))
    for x in (-0.105, 0.105):
        for y in (-0.19, 0.19):
            _box(
                b,
                tower,
                f"tower_foot_{x}_{y}",
                (x, y, 0.02),
                (0.025, 0.03, 0.02),
                "rubber",
            )
    for z in (0.43, 0.49):
        _box(
            b, tower, f"tower_drive_{z}", (0, -0.233, z), (0.105, 0.003, 0.022), "metal"
        )
    for i in range(10):
        _box(
            b,
            tower,
            f"tower_vent_{i}",
            (0, -0.232, 0.12 + i * 0.018),
            (0.08, 0.002, 0.003),
            "rubber",
        )
    _box(
        b,
        tower,
        "tower_rear_socket",
        (0.07, 0.237, 0.24),
        (0.012, 0.012, 0.008),
        "rubber",
    )


def _curtain(b):
    # Thin double-sided mesh, with varying fold spacing/amplitude and a hung
    # hem. A fixed visual surface replaces the forty rigid-looking slats.
    nx, nz = 160, 24
    xs = np.linspace(-1.61, 2.20, nx + 1)
    zs = np.linspace(0.06, 2.855, nz + 1)
    vertices = []
    for side in (-1, 1):
        for z in zs:
            t = (2.855 - z) / 2.795
            for x in xs:
                phase = 2 * math.pi * ((x + 1.61) / 0.235 + 0.10 * math.sin(1.7 * x))
                depth = (0.025 + 0.020 * t) * math.sin(phase + 0.11 * t)
                depth += 0.014 * math.sin(2.1 * x + 1.4 * t) * t
                vertices.append(
                    (x, 1.65 + depth + side * 0.001, z + 0.009 * math.sin(3.4 * x) * t)
                )
    faces = []
    layer = (nx + 1) * (nz + 1)
    for side in range(2):
        for j in range(nz):
            for i in range(nx):
                a = side * layer + j * (nx + 1) + i
                b0, c, d = a + 1, a + nx + 2, a + nx + 1
                triangles = ((a, b0, c), (a, c, d))
                faces.extend(
                    triangles if side == 0 else (tuple(reversed(v)) for v in triangles)
                )
    # Close all four hems so portable visual formats retain a thin sheet.
    boundary = list(range(nx + 1)) + [j * (nx + 1) + nx for j in range(1, nz + 1)]
    boundary += list(range(layer - 2, layer - nx - 2, -1))
    boundary += [j * (nx + 1) for j in range(nz - 1, 0, -1)]
    for a, c in zip(boundary, boundary[1:] + boundary[:1]):
        faces.extend(((a, a + layer, c + layer), (a, c + layer, c)))
    name = PREFIX + "curtain_fabric"
    asset = b.root.find("asset")
    ET.SubElement(
        asset,
        "material",
        name=name + "_material",
        rgba=".025 .027 .031 1",
        specular=".03",
        shininess=".02",
    )
    ET.SubElement(
        asset,
        "mesh",
        name=name + "_mesh",
        vertex=numbers(np.asarray(vertices).ravel()),
        face=" ".join(str(v) for f in faces for v in f),
    )
    world = b.root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="mesh",
        mesh=name + "_mesh",
        material=name + "_material",
        contype="0",
        conaffinity="0",
    )
    for i, x in enumerate(np.linspace(-1.58, 2.17, 18)):
        b.rod(
            world,
            (x, 1.65, 2.845),
            (x, 1.65, 2.89),
            0.003,
            "metal",
            name=PREFIX + f"curtain_clip_{i}",
        )


def add_context(b):
    _chair(b)
    _stool(b, "front_stool", (0.79, -1.13, 0), 0.63)
    _stool(b, "side_stool", (1.93, -0.56, 0), 0.72)
    _desk_services(b)
    _curtain(b)
    b.metadata["visible_bay_context"] = dict(
        source="University of Chicago room 520 tour photograph 69",
        observed="Wood-frame upholstered chair, round fixed stools, paired white desk controllers, desktop tower and hanging black curtain",
        dimensions="Authored estimates; no surveyed furniture or electrical specifications",
        behavior="Static contact furniture and controller housings; uncalibrated visual fabric, controls and cable dressing",
    )
