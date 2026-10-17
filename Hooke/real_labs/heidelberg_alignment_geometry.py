"""Photo-informed HQA exterior and original dry two-axis alignment fixture.

The source establishes exterior cues, not room dimensions, hidden mount design,
quantum operation, vacuum performance or calibrated optical alignment.
"""

import math
import xml.etree.ElementTree as ET
from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

CENTER = (0.37, -0.57, 1.13)
REGISTERED_ANGLES = (0.10, -0.12)


def _name(b, key):
    return f"{b.name}__hqa_{key}"


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(parent, pos, half, material, collision=True, name=_name(b, key))


def _cylinder(b, parent, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent,
        pos,
        radius,
        half,
        material,
        collision=True,
        name=_name(b, key),
        **kwargs,
    )


def _rod(b, parent, key, start, end, radius=0.006, material="dark"):
    return b.rod(
        parent, start, end, radius, material, collision=True, name=_name(b, key)
    )


def build_alignment(b, base, params):
    ET.SubElement(b.root.find("asset"), "material", name=f"{b.name}__mat_hqa_red_optic",
                  rgba=".45 .008 .018 1", emission=".5", specular=".35",
                  shininess=".55")
    for x in (-1.04, 1.04):
        for y in (-0.66, 0.66):
            b.box(base, (x, y, 1.50), (0.025, 0.025, 0.74), "metal", collision=True)
    for y in (-0.66, 0.66):
        b.box(base, (0, y, 2.245), (1.065, 0.025, 0.025), "metal", collision=True)
    for x in (-1.04, 1.04):
        b.box(base, (x, 0, 2.245), (0.025, 0.66, 0.025), "metal", collision=True)
    for i in range(23):
        b.box(base, (-1.01 + i * 0.091, 0, 2.272), (0.013, 0.66, 0.005), "metal")
    for j in range(14):
        b.box(base, (0, -0.63 + j * 0.097, 2.282), (1.04, 0.013, 0.005), "metal")
    b.cylinder(base, (-0.40, 0.14, 0.795), 0.20, 0.035, "dark", collision=True)
    b.cylinder(base, (-0.40, 0.14, 1.145), 0.16, 0.315, "metal", collision=True)
    chamber = ET.SubElement(base, "body", pos="-.40 .14 1.15")
    _port(b, chamber, (math.pi / 2, 0, 0), 0.30, True)
    _port(b, chamber, (math.pi / 2 - 0.62, 0, 0.28), 0.28, True)
    _port(b, chamber, (math.pi / 2 + 0.62, 0, -0.22), 0.28, True)
    _port(b, chamber, (0, math.pi / 2, 0), 0.34, False)
    _port(b, chamber, (0, -math.pi / 2, 0), 0.32, True)
    for index, x in enumerate((-0.67, -0.04)):
        b.box(base, (x, 0.33, 0.97), (0.13, 0.16, 0.21), "dark", collision=True)
        b.cylinder(
            base,
            (x, 0.33, 1.33 + index * 0.02),
            0.080,
            0.15 + index * 0.02,
            "metal",
            collision=True,
        )
        b.box(
            base,
            (x, 0.33, 1.61 + index * 0.06),
            (0.14 - index * 0.01, 0.14 - index * 0.02, 0.13 + index * 0.02),
            "guard_red",
            collision=True,
        )
    # Source-visible black horizontal optics levels have continuous metal posts.
    for x in (0.30, 0.89):
        for y in (-0.16, 0.47):
            b.cylinder(base, (x, y, 1.365), 0.018, 0.605, "bright", collision=True)
    for z in (1.04, 1.45, 1.965):
        b.box(base, (0.595, 0.155, z), (0.335, 0.355, 0.018), "dark", collision=True)
    for z in (1.12, 1.53):
        for x in (0.43, 0.76):
            b.cylinder(base, (x, 0.18, z), 0.055, 0.062, "dark", collision=True)
            b.cylinder(base, (x, 0.18, z + 0.067), 0.042, 0.005, "lens")
    _gimbal(b, base)
    _reference(b, base)
    _services(b, base)
    b.metadata["capabilities"] = [
        "dry inert-disc tip and tilt",
        "model-measured surface-normal registration",
        "visible reference aperture",
    ]
    b.metadata["limitations"].append(
        "Original open gimbal and dry alignment reference, with ideal joints and unchanged surrogate inertia/gains. Source-informed fixed vacuum/optical exterior only; dimensions, bearings and hidden service routes are estimates. No vacuum, optical propagation, atom trapping, quantum evolution or hardware calibration."
    )


def _port(b, parent, euler, extension, cage):
    """Closed static viewport exterior; cage frames retain their actual openings."""
    port = ET.SubElement(parent, "body", euler=numbers(euler))
    key = b.unique("port").split("__")[-1]
    _cylinder(b, port, key + "_neck", (0, 0, extension / 2), 0.070, extension / 2)
    _cylinder(b, port, key + "_flange", (0, 0, extension), 0.105, 0.017, "bright")
    _cylinder(b, port, key + "_window", (0, 0, extension + 0.020), 0.071, 0.003, "hqa_red_optic")
    for i in range(12):
        a = i * math.tau / 12
        _cylinder(
            b,
            port,
            key + f"_bolt_{i}",
            (0.090 * math.cos(a), 0.090 * math.sin(a), extension + 0.023),
            0.006,
            0.007,
        )
    if cage:
        for x in (-0.055, 0.055):
            for y in (-0.055, 0.055):
                _cylinder(
                    b,
                    port,
                    key + f"_rail_{x}_{y}",
                    (x, y, extension + 0.15),
                    0.004,
                    0.126,
                    "bright",
                )
        for level, z in enumerate((extension + 0.045, extension + 0.265)):
            for sign in (-1, 1):
                _box(
                    b,
                    port,
                    key + f"_frame_x_{level}_{sign}",
                    (sign * 0.055, 0, z),
                    (0.013, 0.068, 0.010),
                    "dark",
                )
                _box(
                    b,
                    port,
                    key + f"_frame_y_{level}_{sign}",
                    (0, sign * 0.055, z),
                    (0.042, 0.013, 0.010),
                    "dark",
                )
            # Retained annular optic cell, joined to the four-sided plate.
            annular_mount(
                b,
                port,
                key + f"_cell_{level}",
                (0, 0, z),
                0.034,
                0.044,
                0.010,
                axis="z",
                proxies=True,
                material="dark",
                namespace="hqa",
            )
            _cylinder(b, port, key + f"_optic_{level}", (0, 0, z), 0.034, 0.004, "hqa_red_optic")
            # Three offset adjusters and corner fasteners match the visible
            # retained optical plates; they are fixed external hardware.
            for j, (x, y) in enumerate(((-.052, -.052), (.052, -.052), (.052, .052))):
                _cylinder(b, port, key + f"_adjuster_{level}_{j}",
                          (x, y, z + .022), .006, .014, "metal")
                _cylinder(b, port, key + f"_knob_{level}_{j}",
                          (x, y, z + .039), .009, .005, "dark")
        if abs(euler[0] - math.pi / 2) < 1e-6:
            # Source shows a camera-like barrel on the left-facing cage. It is
            # static exterior context, not a sensor simulation or OEM model.
            for i, (rad, length) in enumerate(
                ((0.034, 0.035), (0.030, 0.032), (0.037, 0.045))
            ):
                z = extension + 0.275 + sum((0.035, 0.032, 0.045)[:i]) + length / 2
                _cylinder(
                    b,
                    port,
                    key + f"_camera_barrel_{i}",
                    (0, 0, z),
                    rad,
                    length / 2,
                    "dark",
                )


def _crescent(b, parent, key, start, end):
    """Watertight partial annulus; cut ends leave real top/bottom spindle slots."""
    count = 40
    inner, outer, half = 0.061, 0.082, 0.010
    angles = [start + (end - start) * i / count for i in range(count + 1)]
    point = lambda r, a, y: (r * math.cos(a), y, r * math.sin(a))
    verts = [
        point(r, a, y) for y in (-half, half) for r in (inner, outer) for a in angles
    ]
    stride = count + 1
    faces = []
    for i in range(count):
        for first, second, reverse in (
            (0, stride, False),
            (2 * stride, 3 * stride, True),
            (0, 2 * stride, True),
            (stride, 3 * stride, False),
        ):
            a, c, d, e = first + i, first + i + 1, second + i + 1, second + i
            for triangle in ((a, c, d), (a, d, e)):
                faces.append(tuple(reversed(triangle)) if reverse else triangle)
    for i in (0, count):
        for triangle in (
            (i, stride + i, 3 * stride + i),
            (i, 3 * stride + i, 2 * stride + i),
        ):
            faces.append(tuple(reversed(triangle)) if i else triangle)
    mesh = _name(b, key + "_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([v for p in verts for v in p]),
        face=" ".join(str(i) for f in faces for i in f),
    )
    b.geom(
        parent, "mesh", (), material="metal", mesh=mesh, name=_name(b, key + "_shell")
    )
    for i in range(count):
        # Each eight-vertex cell is convex; their union preserves both openings.
        vs = [
            point(r, a, y)
            for y in (-half, half)
            for r in (inner, outer)
            for a in angles[i : i + 2]
        ]
        cm = _name(b, key + f"_proxy_mesh_{i}")
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=cm,
            vertex=numbers([v for p in vs for v in p]),
        )
        b.geom(
            parent,
            "mesh",
            (),
            material="metal",
            mesh=cm,
            collision=True,
            group=3,
            rgba=(0.5, 0.5, 0.5, 0),
            name=_name(b, key + f"_proxy_{i}"),
        )


def _gimbal(b, base):
    fixed = ET.SubElement(
        base, "body", name=_name(b, "fixed_mount"), pos=numbers(CENTER)
    )
    _box(b, fixed, "mount_foot", (0, 0, -0.35), (0.14, 0.13, 0.02), "dark")
    _cylinder(b, fixed, "mount_post", (0, 0, -0.228), 0.020, 0.102, "bright")
    _box(b, fixed, "fork_bridge", (0, 0, -0.113), (0.120, 0.022, 0.013), "dark")
    for sign in (-1, 1):
        _box(
            b,
            fixed,
            f"fork_cheek_{sign}",
            (sign * 0.100, 0, -0.059),
            (0.012, 0.016, 0.041),
            "dark",
        )
        annular_mount(
            b,
            fixed,
            f"x_bearing_{sign}",
            (sign * 0.100, 0, 0),
            0.0086,
            0.018,
            0.012,
            axis="x",
            proxies=True,
            material="dark",
            namespace="hqa",
        )
    tip = b.moving(
        base,
        "alignment_tip",
        CENTER,
        (1, 0, 0),
        (-0.16, 0.16),
        kind="hinge",
        mass=0.06,
        kp=100,
    )
    slot = math.asin(0.011 / 0.061)
    _crescent(b, tip, "ring_right", -math.pi / 2 + slot, math.pi / 2 - slot)
    _crescent(b, tip, "ring_left", math.pi / 2 + slot, 3 * math.pi / 2 - slot)
    for sign in (-1, 1):
        _cylinder(
            b,
            tip,
            f"x_spindle_{sign}",
            (sign * 0.095, 0, 0),
            0.008,
            0.021,
            "bright",
            euler=(0, math.pi / 2, 0),
        )
        _cylinder(
            b,
            tip,
            f"x_retainer_{sign}",
            (sign * 0.122, 0, 0),
            0.013,
            0.006,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
        annular_mount(
            b,
            tip,
            f"z_bearing_{sign}",
            (0, 0, sign * 0.070),
            0.0066,
            0.020,
            0.012,
            axis="z",
            proxies=True,
            material="dark",
            namespace="hqa",
        )
    tilt = b.moving(
        tip,
        "alignment_tilt",
        (0, 0, 0),
        (0, 0, 1),
        (-0.20, 0.20),
        kind="hinge",
        mass=0.035,
        kp=85,
    )
    _cylinder(
        b,
        tilt,
        "inert_alignment_disc",
        (0, 0, 0),
        0.046,
        0.008,
        "orange",
        euler=(math.pi / 2, 0, 0),
    )
    for sign in (-1, 1):
        _cylinder(
            b, tilt, f"z_spindle_{sign}", (0, 0, sign * 0.063), 0.006, 0.023, "bright"
        )
    # The original target site is retained at y=-11 mm and now lies on a real
    # contact-capable 3 mm thick witness button bonded to the 16 mm thick disc.
    _cylinder(
        b,
        tilt,
        "witness_button",
        (0, -0.0095, 0),
        0.006,
        0.0015,
        "cream",
        euler=(math.pi / 2, 0, 0),
    )
    b.site(tilt, "hqa_target", (0, -0.011, 0))
    b.site(tilt, "hqa_center", (0, 0, 0), 0.001)
    b.metadata["alignment_interface"] = {
        "target_site": b.metadata["sites"]["hqa_target"],
        "target_geom": _name(b, "witness_button"),
        "disc_geom": _name(b, "inert_alignment_disc"),
        "target_diameter_m": 0.012,
        "disc_size_m": [0.092, 0.016, 0.092],
        "normal_local": [0, -1, 0],
        "registered_angles_rad": list(REGISTERED_ANGLES),
        "bearing_running_gap_m": 0.0006,
        "scope": "Authored rigid surface-normal registration toward a fixed open aperture; not laser, quantum or vacuum science",
    }


def _reference(b, base):
    tip, tilt = REGISTERED_ANGLES
    normal = (
        math.sin(tilt),
        -math.cos(tilt) * math.cos(tip),
        -math.cos(tilt) * math.sin(tip),
    )
    center = tuple(c + 0.180 * n for c, n in zip(CENTER, normal))
    _box(
        b,
        base,
        "reference_foot",
        (center[0], center[1], 0.770),
        (0.027, 0.027, 0.010),
        "dark",
    )
    top = center[2] - 0.017
    _cylinder(
        b,
        base,
        "reference_post",
        (center[0], center[1], (0.78 + top) / 2),
        0.006,
        (top - 0.78) / 2,
        "bright",
    )
    denominator = math.sqrt(2 * (1 + normal[2]))
    quat = (
        math.sqrt((1 + normal[2]) / 2),
        -normal[1] / denominator,
        normal[0] / denominator,
        0,
    )
    holder = ET.SubElement(
        base,
        "body",
        name=_name(b, "reference_holder"),
        pos=numbers(center),
        quat=numbers(quat),
    )
    annular_mount(
        b,
        holder,
        "reference_aperture",
        (0, 0, 0),
        0.006,
        0.019,
        0.005,
        axis="z",
        proxies=True,
        material="dark",
        namespace="hqa",
    )
    b.site(holder, "hqa_reference", (0, 0, 0), 0.001)
    interface = b.metadata["alignment_interface"]
    interface.update(
        reference_site=b.metadata["sites"]["hqa_reference"],
        reference_aperture_radius_m=0.006,
        normal_registration_tolerance_rad=0.001,
        ray_center_tolerance_m=0.0002,
        reference_center_local_m=list(center),
    )


def _services(b, base):
    # Source-informed installed service routes. These are static supported
    # exteriors with explicit terminal attachments, not flexible cable physics.
    for index, x in enumerate((-0.67, -0.04)):
        z = 1.61 + index * 0.06
        _box(
            b,
            base,
            f"red_front_panel_{index}",
            (x, 0.188 + index * 0.02, z),
            (0.120 - index * 0.01, 0.002, 0.110 + index * 0.02),
            "guard_red",
        )
        for side in (-1, 1):
            _cylinder(
                b,
                base,
                f"red_front_fastener_{index}_{side}",
                (x + side * 0.11, 0.183 + index * 0.02, z - 0.09),
                0.004,
                0.005,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
        for n in range(3):
            tx = x + (n - 1) * 0.025
            rear = 0.47 - index * 0.02
            _cylinder(
                b,
                base,
                f"terminal_{index}_{n}",
                (tx, rear + 0.006, z + 0.03),
                0.006,
                0.008,
                "dark",
                euler=(math.pi / 2, 0, 0),
            )
            points = (
                (tx - 0.10, 0.660, 2.245),
                (tx - 0.08, 0.60, 2.10),
                (tx + 0.035, 0.57, 1.88),
                (tx, rear + 0.014, z + 0.03),
            )
            for k, (a, c) in enumerate(zip(points, points[1:])):
                _rod(
                    b,
                    base,
                    f"rear_service_{index}_{n}_{k}",
                    a,
                    c,
                    0.0035,
                    "blue" if n == 0 else "dark",
                )
    # Foreground pale loops remain to the left of the dry alignment fixture.
    for index, dx in enumerate((0, 0.035, 0.07)):
        pts = (
            (-0.53 + dx, 0.10, 1.34),
            (-0.66 + dx, -0.05, 1.30),
            (-0.75 + dx, -0.13, 1.05),
            (-0.70 + dx, -0.06, 0.86),
            (-0.53 + dx, 0.04, 0.82),
            (-0.40 + dx, 0.13, 0.86),
        )
        for k, (a, c) in enumerate(zip(pts, pts[1:])):
            _rod(b, base, f"pale_service_{index}_{k}", a, c, 0.005, "cream")
    # Source-visible dense optical-stack fasteners and retained lens cells.
    for level, z in enumerate((1.04, 1.45, 1.965)):
        for index in range(8):
            _cylinder(
                b,
                base,
                f"deck_fastener_{level}_{index}",
                (0.31 + index * 0.080, -0.177, z + 0.020),
                0.0035,
                0.002,
                "bright",
            )
    # Source-visible heterogeneous optics at the two black deck levels.
    # All are supported on the deck, outside the independently checked gimbal.
    for i, (x, y, z, height) in enumerate(((.34, .35, 1.04, .12),
            (.66, -.07, 1.04, .16), (.87, .35, 1.04, .20),
            (.38, .35, 1.45, .19), (.67, -.06, 1.45, .12),
            (.87, .36, 1.45, .15))):
        _box(b, base, f"optic_foot_{i}", (x, y, z + .028), (.035, .035, .010), "dark")
        _cylinder(b, base, f"optic_post_{i}", (x, y, z + .038 + height / 2),
                  .009, height / 2, "bright")
        _box(b, base, f"optic_cell_{i}", (x, y, z + height + .058),
             (.026, .012, .028), "dark")
        _cylinder(b, base, f"optic_face_{i}", (x, y - .014, z + height + .058),
                  .019, .002, "hqa_red_optic", euler=(math.pi / 2, 0, 0))
    b.metadata["limitations"].append(
        "Red viewport appearance reproduces the photographed illumination cue only; "
        "emissive surfaces do not simulate laser propagation or optical power.")
