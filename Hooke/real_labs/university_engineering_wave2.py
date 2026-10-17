"""Photo-informed haptic gantry and wedge-lined acoustic inspection room."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

COVR = "https://www.isir.upmc.fr/projets/covr-plateforme-haptique-pour-la-realite-virtuelle/"
EPFL = "https://www.epfl.ch/labs/lwe/facilities/anechoic-room-acoustics/"
WOOD = (0.60, 0.37, 0.16, 1)
ABSORBER = (0.69, 0.66, 0.55, 1)
STEEL = (0.53, 0.57, 0.58, 1)


def _material(b, key, rgba):
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=b.name + "__mat_" + key,
        rgba=numbers(rgba),
        specular="0.15",
        shininess="0.15",
    )


def _flat_mesh(b, name, vertices, faces):
    """Explicit split normals preserve flat sheet surfaces across renderers."""

    def cross(u, v):
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    volume6 = sum(
        sum(vertices[a][k] * cross(vertices[bb], vertices[c])[k] for k in range(3))
        for a, bb, c in faces
    )
    if volume6 < 0:
        faces = [(a, c, bb) for a, bb, c in faces]
    split_vertices, split_normals = [], []
    for face in faces:
        a, bb, c = (vertices[i] for i in face)
        n = cross([bb[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)])
        length = math.sqrt(sum(x * x for x in n))
        for vertex in (a, bb, c):
            split_vertices.extend(vertex)
            split_normals.extend(x / length for x in n)
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=numbers(split_vertices),
        face=" ".join(str(i) for i in range(len(split_vertices) // 3)),
        normal=numbers(split_normals),
        smoothnormal="false",
    )


def _panel_prism(b, parent, outline):
    """A thin plywood section with flat faces and collision geometry."""
    vertices = [(x, y, z) for y in (-0.015, 0.015) for x, z in outline]
    cx = sum(v[0] for v in outline) / 4
    cz = sum(v[1] for v in outline) / 4
    vertices.extend(((cx, -0.015, cz), (cx, 0.015, cz)))
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend(((8, j, i), (9, i + 4, j + 4), (i, j, j + 4), (i, j + 4, i + 4)))
    name = b.unique("plywood_mesh")
    _flat_mesh(b, name, vertices, faces)
    b.geom(parent, "mesh", (), material="plywood", mesh=name, collision=True)


def _panel_hole(b, parent, z, radius):
    """Convex ring sectors retain rectangular corners and a true circular port."""
    corner = math.atan2(0.15, 0.42)
    angles = sorted(
        set(
            [i * math.tau / 32 for i in range(33)]
            + [corner, math.pi - corner, math.pi + corner, math.tau - corner]
        )
    )
    for a, other in zip(angles, angles[1:]):
        points = []
        for angle in (a, other):
            x, dz = math.cos(angle), math.sin(angle)
            scale = min(0.42 / max(abs(x), 1e-12), 0.15 / max(abs(dz), 1e-12))
            points.append(((radius * x, z + radius * dz), (scale * x, z + scale * dz)))
        _panel_prism(
            b, parent, [points[0][0], points[0][1], points[1][1], points[1][0]]
        )


def _covr(b, base, params):
    _material(b, "plywood", WOOD)
    # Floor posts and longitudinal rails continuously carry the overhead bridge.
    for x in (-1.8, 1.8):
        for y in (-1.65, 1.65):
            b.box(base, (x, y, 0.03), (0.19, 0.19, 0.03), "metal", collision=True)
            b.box(base, (x, y, 1.55), (0.055, 0.055, 1.52), "bright", collision=True)
            for dx in (-0.037, 0.037):
                b.box(base, (x + dx, y - 0.057, 1.60), (0.004, 0.003, 1.45), "dark")
        b.box(base, (x, 0, 3.08), (0.10, 1.85, 0.075), "dark", collision=True)
        b.box(base, (x, 0, 3.164), (0.035, 1.83, 0.009), "bright")
    bridge = b.moving(
        base,
        "bridge_y",
        (0, 0.25, 3.25),
        (0, 1, 0),
        (-0.70, 0.70),
        mass=12,
        kp=2400,
        force=4000,
    )
    for y in (-0.11, 0.11):
        b.box(bridge, (0, y, 0), (1.94, 0.04, 0.065), "dark", collision=True)
    for x in (-1.8, 1.8):
        b.box(bridge, (x, 0, -0.02), (0.13, 0.22, 0.045), "metal")
        for y in (-0.15, 0.15):
            b.cylinder(
                bridge, (x, y, -0.04), 0.048, 0.035, "rubber", euler=(0, math.pi / 2, 0)
            )
    carriage = b.moving(
        bridge,
        "panel_x",
        (0.50, 0, -0.04),
        (1, 0, 0),
        (-0.60, 0.60),
        mass=5,
        kp=1600,
        force=2000,
    )
    b.box(carriage, (0, 0, 0), (0.20, 0.17, 0.025), "metal")
    for x in (-0.22, 0.22):
        b.rod(carriage, (x, 0, -0.02), (x, 0, -0.39), 0.024, "bright")
    # Solid strips connect the two open circular port bands into one retained panel.
    for lo, hi in [(-0.34, -0.89), (-1.19, -1.54), (-1.84, -2.30)]:
        _panel_prism(b, carriage, [(-0.42, lo), (-0.42, hi), (0.42, hi), (0.42, lo)])
    _panel_hole(b, carriage, -1.04, 0.085)
    _panel_hole(b, carriage, -1.69, 0.065)
    b.box(carriage, (0, -0.020, -0.41), (0.18, 0.012, 0.13), "plywood")
    for x in (-0.36, 0.36):
        for z in (-0.47, -0.77, -1.35, -1.98, -2.20):
            b.cylinder(
                carriage,
                (x, -0.020, z),
                0.010,
                0.006,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
    b.site(carriage, "interaction_panel", (0.25, 0, -1.34))
    b.metadata["capabilities"] = [
        "overhead_bridge_translation",
        "retained_panel_translation",
        "two_true_open_panel_ports",
    ]
    b.metadata["limitations"].append(
        "Original two-axis retained-prop mechanism based on the photographed CoVR arena. Gantry dimensions, motion limits and load capacity are estimates; no force feedback, VR rendering, human contact or haptic-perception model is implemented."
    )


def _covr_room(world, definition):
    for x in (-1.3, 1.3):
        box(
            world,
            f"covr_floor_tape_x_{x}",
            (0.007, 1.42, 0.0007),
            (x, -0.35, 0.0012),
            (0.05, 0.28, 0.24, 1),
            collision=False,
        )
    for y in (-1.77, 1.07):
        box(
            world,
            f"covr_floor_tape_y_{y}",
            (1.3, 0.007, 0.0007),
            (0, y, 0.0012),
            (0.05, 0.28, 0.24, 1),
            collision=False,
        )
    for x in (-2.5, 2.5):
        box(
            world,
            f"arch_roof_covr_cable_tray_{x}",
            (0.15, 2.75, 0.045),
            (x, 0, 3.68),
            (0.38, 0.40, 0.40, 1),
        )
        for y in (-2, 0, 2):
            box(
                world,
                f"arch_roof_covr_hanger_{x}_{y}",
                (0.02, 0.02, 0.17),
                (x, y, 3.87),
                STEEL,
            )


def _acoustic_station(b, base, params):
    _material(b, "absorber", ABSORBER)
    # One-metre depth is sourced. Wedge widths, chamber size and grate are estimated.
    vertices = [
        (-0.115, -0.34, 0),
        (0.115, -0.34, 0),
        (0, -0.34, 1),
        (-0.115, 0.34, 0),
        (0.115, 0.34, 0),
        (0, 0.34, 1),
    ]
    faces = [
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4),
        (0, 4, 3),
        (1, 2, 5),
        (1, 5, 4),
        (2, 0, 3),
        (2, 3, 5),
    ]
    _flat_mesh(b, b.name + "__wedge", vertices, faces)
    # The two stands emerge from the lower structural slab through open grating.
    for x, y in ((-0.7, 0.6), (0.9, -0.1)):
        b.cylinder(base, (x, y, -1.02), 0.18, 0.06, "metal", collision=True)
        b.cylinder(base, (x, y, 0.025), 0.030, 0.985, "metal", collision=True)
    b.cylinder(base, (-0.7, 0.6, 1.04), 0.19, 0.025, "metal", collision=True)
    turn = b.moving(
        base,
        "reference_yaw",
        (-0.7, 0.6, 1.10),
        (0, 0, 1),
        (-0.7, 0.7),
        kind="hinge",
        mass=0.4,
        kp=200,
        force=300,
    )
    b.cylinder(turn, (0, 0, 0), 0.16, 0.025, "dark", collision=True)
    b.box(turn, (0, 0, 0.075), (0.08, 0.055, 0.045), "cream", collision=True)
    for x in (-0.092, 0.092):
        b.box(turn, (x, 0, 0.065), (0.012, 0.06, 0.035), "bright")
    b.site(turn, "inert_acoustic_reference", (0, 0, 0.075))
    b.cylinder(base, (0.9, -0.1, 1.04), 0.13, 0.03, "metal", collision=True)
    b.box(base, (0.9, -0.1, 1.14), (0.27, 0.15, 0.04), "dark", collision=True)
    for y in (-0.22, 0.02):
        b.rod(base, (0.63, y, 1.205), (1.17, y, 1.205), 0.01, "bright")
    mic = b.moving(
        base,
        "microphone_x",
        (0.9, -0.1, 1.25),
        (1, 0, 0),
        (-0.12, 0.12),
        mass=0.15,
        kp=200,
        force=300,
    )
    b.box(mic, (0, 0, 0), (0.12, 0.14, 0.025), "metal", collision=True)
    b.rod(mic, (0, 0, 0.03), (-0.18, 0.12, 0.17), 0.009, "metal", collision=True)
    b.cylinder(mic, (-0.18, 0.12, 0.20), 0.010, 0.026, "dark", collision=True)
    b.metadata["capabilities"] = [
        "retained_reference_yaw",
        "microphone_fixture_translation",
        "visible_inert_reference",
    ]
    b.metadata["limitations"].append(
        "The turntable/reference and microphone traverse are original mechanical task aids. No acoustic wave propagation, attenuation, microphone signal or chamber calibration; only the one-metre absorber depth is a published dimension."
    )


def _acoustic_room(world, definition):
    # Actual floor void: remove the generic continuous floor before adding a grate.
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_floor"):
            world.remove(geom)
    box(
        world,
        "arch_floor_epfl_lower_slab",
        (3, 3.5, 0.08),
        (0, 0, -1.16),
        (0.35, 0.35, 0.31, 1),
    )
    # Mesh bars cross above open space. Leave openings around the two apparatus stems.
    for i in range(81):
        x = -2.96 + i * 0.074
        if abs(x + 0.7) < 0.04 or abs(x - 0.9) < 0.04:
            continue
        box(
            world,
            f"epfl_floor_grate_x_{i}",
            (0.003, 3.48, 0.006),
            (x, 0, -0.006),
            STEEL,
        )
    for j in range(95):
        y = -3.478 + j * 0.074
        if abs(y - 0.6) < 0.04 or abs(y + 0.1) < 0.04:
            continue
        box(
            world,
            f"epfl_floor_grate_y_{j}",
            (2.98, 0.003, 0.006),
            (0, y, -0.006),
            STEEL,
        )
    for x in (-2.8, 0, 2.8):
        box(
            world,
            f"epfl_floor_support_beam_{x}",
            (0.045, 3.47, 0.075),
            (x, 0, -0.087),
            STEEL,
        )
        for y in (-3.2, 0, 3.2):
            box(
                world,
                f"epfl_floor_support_leg_{x}_{y}",
                (0.045, 0.045, 0.459),
                (x, y, -0.621),
                STEEL,
            )
    mesh = definition["equipment"][0]["id"] + "__wedge"
    # 0.72 m square tiles of three alternating triangular wedges.
    for wall in ("left", "right", "front", "back", "roof"):
        na, nb = (
            (9, 6)
            if wall in ("left", "right")
            else ((8, 9) if wall == "roof" else (8, 6))
        )
        for i in range(na):
            for j in range(nb):
                a = (i - (na - 1) / 2) * 0.72
                c = 0.36 + j * 0.72
                # Real door path is left-front; don't obstruct the opening with foam.
                if wall == "left" and abs(a) < 1.2 and c < 2.90:
                    continue
                for k in (-1, 0, 1):
                    along, vertical = (
                        (a + k * 0.235, c) if (i + j) % 2 == 0 else (a, c + k * 0.235)
                    )
                    spin = 0 if (i + j) % 2 == 0 else math.pi / 2
                    if wall == "left":
                        pos, u, v = (-2.99, along, vertical), (0, 1, 0), (0, 0, 1)
                    elif wall == "right":
                        pos, u, v = (2.99, along, vertical), (0, -1, 0), (0, 0, 1)
                    elif wall == "front":
                        pos, u, v = (along, -3.49, vertical), (-1, 0, 0), (0, 0, 1)
                    elif wall == "back":
                        pos, u, v = (along, 3.49, vertical), (1, 0, 0), (0, 0, 1)
                    else:
                        roof_y = -2.88 + j * 0.72 + (k * 0.235 if spin else 0)
                        pos, u, v = (along, roof_y, 4.59), (1, 0, 0), (0, -1, 0)
                    if spin:
                        u, v = v, tuple(-c for c in u)
                    prefix = (
                        "arch_roof_"
                        if wall == "roof"
                        else (
                            "arch_cutaway_wall_"
                            if wall in ("front", "right")
                            else "epfl_"
                        )
                    )
                    ET.SubElement(
                        world,
                        "geom",
                        name=f"{prefix}wedge_{wall}_{i}_{j}_{k}",
                        type="mesh",
                        mesh=mesh,
                        pos=numbers(pos),
                        xyaxes=numbers((*u, *v)),
                        rgba=numbers(ABSORBER),
                        contype="1",
                        conaffinity="1",
                    )
    # The source shows small suspended lamps below the absorbers, not buried panels.
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_luminaire"):
            world.remove(geom)
    for i, (x, y) in enumerate(((-1.2, -1.8), (1.2, 0), (-1.2, 1.8))):
        box(
            world,
            f"arch_roof_epfl_lamp_hanger_{i}",
            (0.010, 0.010, 0.56),
            (x, y, 4.02),
            STEEL,
            collision=False,
        )
        box(
            world,
            f"arch_luminaire_epfl_{i}",
            (0.08, 0.08, 0.018),
            (x, y, 3.46),
            (1, 0.98, 0.90, 1),
            collision=False,
        )


BUILDERS = {
    "covr_retained_prop_gantry": _covr,
    "epfl_anechoic_inspection_stands": _acoustic_station,
}
SOURCES = {
    "covr_retained_prop_gantry": dict(
        reference="Sorbonne ISIR installed CoVR arena photograph",
        url=COVR,
        dimensions_m=[4.1, 3.8, 3.4],
        dimension_basis="Frame, panel dimensions, rail travel and room footprint estimated from one image; no manufacturer load or haptic fidelity claim.",
    ),
    "epfl_anechoic_inspection_stands": dict(
        reference="EPFL LWE photographed anechoic room and support fixtures",
        url=EPFL,
        dimensions_m=[6.0, 7.0, 4.6],
        dimension_basis="One-metre absorber depth published; room dimensions, grate construction and mechanical inspection fixtures estimated.",
    ),
}
SAMPLE_INTERFACES = {
    "covr_retained_prop_gantry": (
        "interaction_panel",
        (0.84, 0.03, 1.96),
        "clamped",
        "Retained plywood interaction panel with two physically open ports",
    ),
    "epfl_anechoic_inspection_stands": (
        "inert_acoustic_reference",
        (0.16, 0.11, 0.09),
        "clamped",
        "Inert calibration shape mechanically retained on a rotation fixture",
    ),
}
FEATURES = {
    "sorbonne_isir_covr": _covr_room,
    "epfl_lwe_anechoic_acoustics": _acoustic_room,
}
