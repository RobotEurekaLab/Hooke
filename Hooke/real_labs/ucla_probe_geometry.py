"""Photo-informed LAPD access segment with an authored dry probe interface.

This local, open-ended excerpt is not an as-built vacuum vessel or a plasma
model. Chamber, port, coil, guide and drive dimensions are estimates. The
original diagnostic joint frames/ranges remain available.
"""

import math
import xml.etree.ElementTree as ET

PORT_Y = -2.40
PORT_Z = 1.298
VESSEL_CENTER_Z = 1.25
VESSEL_INNER_RADIUS = 0.43
VESSEL_OUTER_RADIUS = 0.46
PORT_INNER_RADIUS = 0.075
PORT_OUTER_RADIUS = 0.092
PROBE_HEAD_CENTER_X = -1.30


def _name(b, key):
    return f"{b.name}__lapd_{key}"


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(parent, pos, half, material, collision=True, name=_name(b, key))


def _cylinder(b, parent, key, pos, radius, half, material="metal", axis="z"):
    rotation = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[
        axis
    ]
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        euler=rotation,
        collision=True,
        name=_name(b, key),
    )


def _rod(b, parent, key, a, c, radius=0.008, material="metal"):
    return b.rod(parent, a, c, radius, material, collision=True, name=_name(b, key))


def _mesh(b, parent, key, vertices, faces, material, *, contact=False):
    # The Y-axis coordinate permutation reverses the canonical prism winding.
    volume = 0.0
    for i, j, k in faces:
        a, c, d = vertices[i], vertices[j], vertices[k]
        volume += sum(
            a[q] * (c[(q + 1) % 3] * d[(q + 2) % 3] - c[(q + 2) % 3] * d[(q + 1) % 3])
            for q in range(3)
        )
    if volume < 0:
        faces = [tuple(reversed(face)) for face in faces]
    mesh = _name(b, key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=" ".join(f"{x:.12g}" for p in vertices for x in p),
        face=" ".join(str(i) for f in faces for i in f),
    )
    return b.geom(
        parent,
        "mesh",
        (),
        material=material,
        mesh=mesh,
        collision=contact,
        group=3 if contact else 0,
        name=_name(b, key),
    )


_QUADS = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def _tube(
    b,
    parent,
    key,
    center,
    inner,
    outer,
    cuts,
    material="metal",
    *,
    axis="x",
    count=48,
    opening=None,
    contacts=True,
):
    """Watertight annular wall with a genuine optional rectangular side aperture.

    Surface cells share vertices. Convex contact sectors are independent hidden
    shapes, so no convex hull fills the bore or the side opening.
    """
    angles = [math.tau * i / count for i in range(count)]
    if opening:
        low, high, angle_low, angle_high = opening
        angles += [angle_low % math.tau, angle_high % math.tau]
    angles = sorted(set(angles))
    n = len(angles)
    cells = set()
    for j in range(len(cuts) - 1):
        for i, a in enumerate(angles):
            c = angles[(i + 1) % n]
            if c <= a:
                c += math.tau
            mid = ((a + c) / 2 + math.pi) % math.tau - math.pi
            if (
                opening
                and low <= (cuts[j] + cuts[j + 1]) / 2 <= high
                and angle_low <= mid <= angle_high
            ):
                continue
            cells.add((j, i))
    vertices, faces, lookup = [], [], {}

    def point(t, r, a):
        if axis == "x":
            p = (t, r * math.cos(a), r * math.sin(a))
        else:
            p = (r * math.cos(a), t, r * math.sin(a))
        return tuple(x + offset for x, offset in zip(p, center))

    def vertex(p):
        p = tuple(round(x, 12) for x in p)
        if p not in lookup:
            lookup[p] = len(vertices)
            vertices.append(p)
        return lookup[p]

    for j, i in sorted(cells):
        a, c = angles[i], angles[(i + 1) % n]
        pts = [
            point(t, r, theta)
            for t in (cuts[j], cuts[j + 1])
            for r, theta in ((inner, a), (outer, a), (outer, c), (inner, c))
        ]
        ids = [vertex(p) for p in pts]
        # Inner and outer surfaces, then only exposed cell boundaries.
        visible = [3, 5]
        if (j - 1, i) not in cells:
            visible.append(0)
        if (j + 1, i) not in cells:
            visible.append(1)
        if (j, (i - 1) % n) not in cells:
            visible.append(2)
        if (j, (i + 1) % n) not in cells:
            visible.append(4)
        for qi in visible:
            q = [ids[k] for k in _QUADS[qi]]
            faces.extend(((q[0], q[1], q[2]), (q[0], q[2], q[3])))
        if contacts:
            triangles = [
                t for q in _QUADS for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))
            ]
            _mesh(
                b,
                parent,
                f"{key}_contact_{j}_{i}",
                pts,
                triangles,
                material,
                contact=True,
            )
    return _mesh(b, parent, key + "_surface", vertices, faces, material)


def _sleeve(b, parent, key, center, axis, half_length, inner, wall=0.006):
    a, c = [i for i in range(3) if i != axis]
    for sign in (-1, 1):
        pos, half = list(center), [0.0] * 3
        pos[a] += sign * (inner[0] + wall / 2)
        half[axis], half[a], half[c] = half_length, wall / 2, inner[1] + wall
        _box(b, parent, f"{key}_side_{sign}", pos, half)
        pos, half = list(center), [0.0] * 3
        pos[c] += sign * (inner[1] + wall / 2)
        half[axis], half[a], half[c] = half_length, inner[0], wall / 2
        _box(b, parent, f"{key}_cap_{sign}", pos, half)


def _vessel_and_coils(b, base):
    low, high = PORT_Y - 0.082, PORT_Y + 0.082
    angle_low = math.asin((PORT_Z - 0.081 - VESSEL_CENTER_Z) / VESSEL_INNER_RADIUS)
    angle_high = math.asin((PORT_Z + 0.081 - VESSEL_CENTER_Z) / VESSEL_INNER_RADIUS)
    _tube(
        b,
        base,
        "vessel",
        (0, 0, VESSEL_CENTER_Z),
        VESSEL_INNER_RADIUS,
        VESSEL_OUTER_RADIUS,
        [-3.9, low, high, 3.9],
        axis="y",
        count=64,
        opening=(low, high, angle_low, angle_high),
    )
    for y in (-3.5, -1.75, 0, 1.75, 3.5):
        for x in (-0.60, 0.60):
            _box(b, base, f"vessel_leg_{x}_{y}", (x, y, 0.37), (0.05, 0.10, 0.37))
        _box(b, base, f"vessel_support_{y}", (0, y, 0.75), (0.65, 0.11, 0.04), "bright")
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_lapd_magenta",
        rgba=".48 .022 .245 1",
        specular=".45",
        shininess=".42",
    )
    for i in range(26):
        y = -3.75 + i * 0.30
        # Flat winding bands, not toroidal chains. Pitch/number are authored;
        # the active port lies between bands rather than passing through one.
        _tube(
            b,
            base,
            f"coil_{i}",
            (0, y, 1.25),
            0.485,
            0.615,
            [-0.055, 0.055],
            "lapd_magenta",
            axis="y",
            count=48,
            contacts=False,
        )
        _tube(
            b,
            base,
            f"coil_proxy_{i}",
            (0, y, 1.25),
            0.485,
            0.615,
            [-0.055, 0.055],
            "lapd_magenta",
            axis="y",
            count=16,
            contacts=True,
        ).set("group", "3")
        _box(
            b, base, f"coil_pedestal_{i}", (0, y, 0.3175), (0.20, 0.07, 0.3175), "dark"
        )
        for side in (-1, 1):
            _rod(
                b,
                base,
                f"coil_terminal_{i}_{side}",
                (-0.43, y + side * 0.025, 1.66),
                (-0.49, y + side * 0.025, 1.72),
                0.013,
                "bright",
            )
            points = [
                (-0.49, y + side * 0.025, 1.72),
                (-0.76, y + side * 0.04, 1.91),
                (-1.08, y + side * 0.045, 2.12),
                (-1.45, y + side * 0.045, 2.29),
                (-1.48, y + side * 0.045, 2.35),
            ]
            for k, (a, c) in enumerate(zip(points, points[1:])):
                _rod(b, base, f"coolant_cue_{i}_{side}_{k}", a, c, 0.007, "cream")
    _tube(
        b,
        base,
        "access_neck",
        (0, PORT_Y, PORT_Z),
        PORT_INNER_RADIUS,
        PORT_OUTER_RADIUS,
        [0.410, 0.735],
        count=48,
    )
    _tube(
        b,
        base,
        "access_flange",
        (0, PORT_Y, PORT_Z),
        PORT_INNER_RADIUS,
        0.125,
        [0.735, 0.755],
        count=48,
        material="bright",
    )
    for i in range(8):
        a = math.tau * i / 8
        _cylinder(
            b,
            base,
            f"access_bolt_{i}",
            (0.759, PORT_Y + 0.106 * math.cos(a), PORT_Z + 0.106 * math.sin(a)),
            0.006,
            0.005,
            "dark",
            "x",
        )
    b.site(base, "access_axis", (0.46, PORT_Y, PORT_Z), 0.003)


def _probe_drive(b, base):
    for y in (-2.58, -2.22):
        _box(b, base, f"drive_foot_{y}", (2.20, y, 0.03), (0.15, 0.08, 0.03), "dark")
        _box(
            b,
            base,
            f"vertical_rail_{y}",
            (2.20, y, 0.84),
            (0.020, 0.020, 0.78),
            "bright",
        )
    _box(b, base, "drive_crossbeam", (2.20, PORT_Y, 1.59), (0.08, 0.23, 0.04))
    _box(b, base, "z_motor", (2.20, PORT_Y, 1.695), (0.055, 0.065, 0.065), "dark")
    carriage = b.moving(
        base,
        "probe_vertical_alignment",
        (2.20, PORT_Y, 1.25),
        (0, 0, 1),
        (-0.04, 0.04),
        mass=0.3,
        kp=380,
    )
    _box(b, carriage, "vertical_bridge", (0.0416, 0, 0), (0.011, 0.23, 0.06))
    for y in (-0.18, 0.18):
        _sleeve(b, carriage, f"z_shoe_{y}", (0, y, 0), 2, 0.045, (0.0206, 0.0206), 0.01)
    for y in (-0.09, 0.09):
        _box(
            b,
            carriage,
            f"diagnostic_x_guide_{y}",
            (-0.30, y, -0.045),
            (0.35, 0.010, 0.010),
            "bright",
        )
    _box(b, carriage, "x_drive_mount", (0.066, 0, -0.01), (0.025, 0.060, 0.035))
    _box(b, carriage, "x_motor", (0.135, 0, -0.01), (0.045, 0.050, 0.030), "dark")
    stage = b.moving(
        carriage,
        "probe_approach",
        (-0.30, 0, -0.017),
        (1, 0, 0),
        (-0.20, 0),
        mass=0.15,
        kp=340,
    )
    _box(b, stage, "x_carriage", (0, 0, 0.0026), (0.095, 0.12, 0.018), "dark")
    for x in (-0.055, 0.055):
        for y in (-0.09, 0.09):
            _sleeve(
                b, stage, f"x_shoe_{x}_{y}", (x, y, -0.028), 0, 0.023, (0.0106, 0.0106)
            )
    _box(b, stage, "probe_clamp", (-0.065, 0, 0.065), (0.030, 0.040, 0.050))
    for x in (-0.080, -0.050):
        _cylinder(
            b, stage, f"probe_clamp_screw_{x}", (x, 0, 0.123), 0.008, 0.010, "dark"
        )
    # The head is a retained inert alignment block, not an energized diagnostic.
    # Its extended shaft reaches a real port while preserving the original axes.
    front, rear = PROBE_HEAD_CENTER_X + 0.010, -0.095
    _cylinder(
        b,
        stage,
        "probe_shaft",
        ((front + rear) / 2, 0, 0.065),
        0.010,
        (rear - front) / 2,
        "bright",
        "x",
    )
    _box(
        b,
        stage,
        "inert_probe_head",
        (PROBE_HEAD_CENTER_X, 0, 0.065),
        (0.010, 0.025, 0.025),
        "orange",
    )
    b.site(stage, "probe_target", (PROBE_HEAD_CENTER_X + 0.010, 0.018, 0.081), 0.003)
    b.site(stage, "probe_tip", (PROBE_HEAD_CENTER_X - 0.010, 0, 0.065), 0.003)
    b.site(stage, "probe_center", (PROBE_HEAD_CENTER_X, 0, 0.065), 0.003)
    _cylinder(
        b,
        stage,
        "capped_probe_connection",
        (-0.035, 0.046, 0.065),
        0.008,
        0.008,
        "dark",
        "y",
    )
    b.metadata["inspection_interface"] = dict(
        specimen="Fixed inert 20 x 50 x 50 mm probe head on a retained translating shaft",
        bore_radius_m=PORT_INNER_RADIUS,
        vessel_inner_radius_m=VESSEL_INNER_RADIUS,
        port_axis_local_m=[0.46, PORT_Y, PORT_Z],
        head_half_extents_m=[0.01, 0.025, 0.025],
        external_tip_x_m=0.59,
        inserted_target_tip_x_m=0.40,
        probe_approach_target_m=-0.19,
        vertical_alignment_target_m=0,
        guide_running_clearance_m=0.0006,
        withdrawal="Returns to the external neck; the retained probe is not fully extracted through the flange",
        basis="Authored geometric inspection interface, not manufacturer dimensions, a seal or a plasma diagnostic",
    )


def _static_ports_and_services(b, base):
    for i, y in enumerate((-3.6, -3.0, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3.3)):
        # Static closed feedthrough cues have no simulated insertion capability.
        _cylinder(
            b, base, f"closed_neck_{i}", (0.675, y, 1.25), 0.036, 0.225, "bright", "x"
        )
        _cylinder(
            b, base, f"closed_flange_{i}", (0.914, y, 1.25), 0.080, 0.018, "metal", "x"
        )
        for j in range(6):
            a = math.tau * j / 6
            _cylinder(
                b,
                base,
                f"closed_flange_bolt_{i}_{j}",
                (0.936, y + 0.060 * math.cos(a), 1.25 + 0.060 * math.sin(a)),
                0.005,
                0.006,
                "dark",
                "x",
            )
        _cylinder(
            b,
            base,
            f"fixed_drive_rod_{i}",
            (1.48, y, 1.25),
            0.008,
            0.548,
            "bright",
            "x",
        )
        _box(b, base, f"fixed_stand_foot_{i}", (2.03, y, 0.035), (0.13, 0.13, 0.035))
        _cylinder(
            b, base, f"fixed_stand_post_{i}", (2.03, y, 0.6425), 0.020, 0.5725, "dark"
        )
        _box(b, base, f"fixed_stand_top_{i}", (2.03, y, 1.235), (0.09, 0.11, 0.020))
        _box(b, base, f"fixed_drive_clamp_{i}", (2.02, y, 1.284), (0.065, 0.040, 0.030))
        _cylinder(
            b,
            base,
            f"fixed_drive_end_{i}",
            (2.091, y, 1.25),
            0.020,
            0.015,
            "guard_red",
            "x",
        )
    for y in (-3.7, 0, 3.7):
        _box(b, base, f"manifold_column_{y}", (-1.48, y, 1.18), (0.045, 0.045, 1.18))
    _cylinder(b, base, "back_manifold", (-1.48, 0, 2.35), 0.045, 3.8, "copper", "y")
    for i, y in enumerate((-2.8, 0.6, 2.8)):
        _rod(b, base, f"junction_hanger_{i}", (-1.48, y, 2.35), (-1.48, y, 2.05), 0.023)
        _box(b, base, f"junction_box_{i}", (-1.48, y, 1.9), (0.16, 0.20, 0.29), "shell")
        for z in (1.80, 2.02):
            _box(
                b,
                base,
                f"junction_handle_{i}_{z}",
                (-1.306, y, z),
                (0.016, 0.03, 0.014),
                "dark",
            )
    for x in (2.4, 2.85):
        for y in (0.20, 0.70):
            _box(
                b,
                base,
                f"rack_foot_{x}_{y}",
                (x, y, 0.03),
                (0.06, 0.06, 0.03),
                "rubber",
            )
    _box(b, base, "service_rack", (2.625, 0.45, 0.87), (0.285, 0.32, 0.81), "shell")
    for z in (0.50, 0.86, 1.22):
        _box(b, base, f"rack_front_{z}", (2.625, 0.123, z), (0.23, 0.007, 0.11), "dark")
        _box(
            b,
            base,
            f"rack_display_{z}",
            (2.56, 0.114, z),
            (0.11, 0.002, 0.05),
            "screen",
        )
    # One supported local service cabinet and asymmetric side pump block are
    # exterior cues visible in the official photograph, with estimated sizes.
    _box(b, base, "local_cart_shelf", (2.45, -0.8, 0.52), (0.29, 0.27, 0.035))
    for x in (2.20, 2.70):
        for y in (-1.02, -0.58):
            _box(
                b,
                base,
                f"cart_leg_{x}_{y}",
                (x, y, 0.2425),
                (0.025, 0.025, 0.2425),
                "dark",
            )
    _box(b, base, "cart_instrument", (2.45, -0.8, 0.675), (0.24, 0.22, 0.12), "cream")
    _box(
        b, base, "cart_control_panel", (2.45, -1.027, 0.70), (0.20, 0.010, 0.07), "dark"
    )
    for i, x in enumerate((2.32, 2.57)):
        _cylinder(
            b, base, f"cart_control_{i}", (x, -1.044, 0.70), 0.02, 0.01, "bright", "y"
        )


def build_segment(b, base, params):
    _vessel_and_coils(b, base)
    _probe_drive(b, base)
    _static_ports_and_services(b, base)
    b.metadata["capabilities"] = [
        "Retained inert probe vertical alignment through open guide shoes",
        "Dry physical insertion through a genuine side port into an empty hollow local chamber",
        "Measured interface-relative inspection and retraction to the external neck",
    ]
    b.metadata["limitations"] += [
        "Local 7.8 m open-ended excerpt with estimated 26 coil bands, not historical or upgraded full LAPD geometry",
        "Published 18 m by 0.75 m dimensions describe upgraded plasma volume, not this apparatus envelope",
        "Probe length, port bore, motorized fixture and clearances are authored; masses/inertias and ideal joint drives are uncalibrated",
        "Other fixed diagnostic port/drive cues remain static closed assemblies",
        "Cooling-like lines and cabinets are supported geometric cues, with no electrical, flow, vacuum or seal model",
        "No plasma, magnetic field, radiation, fusion or calibrated diagnostic response is simulated",
    ]
