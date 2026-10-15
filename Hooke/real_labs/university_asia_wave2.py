"""Additional university instruments with explicit mechanical-only contracts."""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers


COEB = "https://www.ntu.edu.sg/coeb/research-capabilities/equipment"
AJA = "https://personal.ntu.edu.sg/wensiang/equipment.html"
MA8 = "https://cde.nus.edu.sg/e6nanofab/mask-aligner-ma8-class-100/"
GCF = "https://gcf.hkust.edu.hk/"

SOURCES = {
    "nearfield_optical_station": {
        "reference": "NTU COEB actual installed SNOM workstation photograph",
        "url": COEB,
        "dimensions_m": [1.18, 0.74, 0.84],
        "dimension_basis": "Estimated source-informed microscope body, supported dark probe stack and inactive adjacent display. Open guides and all three original coarse positioning drives are authored; no OEM control, optical result or nanometre performance claim.",
    },
    "multiport_vacuum_inspection_station": {
        "reference": "NTU Intelligent Materials actual AJA sputtering chamber photograph",
        "url": AJA,
        "dimensions_m": [1.55, 1.48, 2.25],
        "dimension_basis": "Estimated chamber exterior and authored supported inspection pose. Port count is representative; internal targets and vacuum circuit are not reconstructed.",
    },
    "e6_mask_alignment_station": {
        "reference": "NUS E6NanoFab installed SUSS MA8 yellow-room photograph",
        "url": MA8,
        "dimensions_m": [1.85, 1.45, 2.05],
        "dimension_basis": "Estimated installed frame and optics; published motorized XY/theta capability informs the axis types, not claimed resolution or travel. A 150 mm inert wafer is an authored sample.",
    },
    "geotechnical_robotic_testbed": {
        "reference": "HKUST GCF actual four-axis robot and stiffened model-container photograph",
        "url": GCF,
        "dimensions_m": [2.90, 1.92, 1.80],
        "dimension_basis": "Estimated standalone model-container inspection station; axes are original mechanical proxies. Not a complete or spinning centrifuge, surveyed apparatus or soil mechanics model.",
    },
}


def _supported_display(b, parent, x, y, z, width=0.45):
    b.box(parent, (x, y, z + 0.014), (width * 0.27, 0.12, 0.014), "dark")
    b.box(parent, (x, y + 0.045, z + 0.17), (0.025, 0.025, 0.145), "metal")
    b.screen(parent, (x, y + 0.045, z + 0.35), width, width * 0.67)


def _nearfield(b, base, params):
    from real_labs.ntu_coeb_geometry import build_station

    build_station(b, base, params)


def _coeb_features(world, definition):
    # Local light-screening context is visible; its complete room extent is unknown.
    dark = (0.05, 0.055, 0.065, 1)
    # Optical-table furniture has a generic silver top. This photographed bay
    # explicitly uses a dark breadboard, as recorded in its bench definition.
    bench = next(item for item in definition["benches"] if item["id"] == "snom_isolation_bench")
    top = world.find(".//geom[@name='snom_isolation_bench_top']")
    top.set("rgba", numbers(bench["color"]))
    for x in (-1.75, 1.10):
        box(
            world,
            f"coeb_screen_post_{x}",
            (0.025, 0.025, 1.15),
            (x, 1.20, 1.15),
            (0.42, 0.45, 0.47, 1),
        )
    box(
        world,
        "coeb_light_screen",
        (1.44, 0.028, 0.97),
        (-0.325, 1.20, 1.32),
        dark,
        collision=False,
    )
    box(world, "coeb_screen_crossrail", (1.45, 0.03, 0.022), (-0.325, 1.20, 2.30), dark)


def _vacuum_station(b, base, params):
    for x in (-0.48, 0.48):
        for y in (-0.42, 0.42):
            b.box(base, (x, y, 0.42), (0.045, 0.045, 0.42), "metal", collision=True)
            b.cylinder(base, (x, y, 0.032), 0.065, 0.032, "dark")
    b.box(base, (0, 0, 0.82), (0.55, 0.49, 0.035), "metal", collision=True)
    b.cylinder(base, (0, 0, 1.11), 0.35, 0.255, "bright", collision=True)
    # Metallic bands suggest the photographed foil jacket without image textures.
    for index in range(14):
        b.ring(
            base,
            (0, 0, 0.895 + index * 0.032),
            0.353 + 0.005 * math.sin(index * 1.7),
            0.012,
            "metal",
            segments=20,
        )
    for angle in (-math.pi / 2, 0, math.pi / 2, math.pi):
        dx, dy = math.cos(angle), math.sin(angle)
        b.rod(
            base,
            (dx * 0.28, dy * 0.28, 1.12),
            (dx * 0.55, dy * 0.55, 1.12),
            0.094,
            "bright",
        )
        quat = (math.sqrt(0.5), -dy * math.sqrt(0.5), dx * math.sqrt(0.5), 0)
        b.cylinder(base, (dx * 0.56, dy * 0.56, 1.12), 0.13, 0.027, "metal", quat=quat)
        b.cylinder(
            base,
            (dx * 0.590, dy * 0.590, 1.12),
            0.090,
            0.007,
            "lens" if dy < -0.5 else "dark",
            quat=quat,
        )
        for index in range(8):
            phi = index * math.pi / 4
            b.cylinder(
                base,
                (
                    dx * 0.593 - dy * 0.111 * math.cos(phi),
                    dy * 0.593 + dx * 0.111 * math.cos(phi),
                    1.12 + 0.111 * math.sin(phi),
                ),
                0.010,
                0.008,
                "bright",
                quat=quat,
            )
    b.cylinder(base, (0, 0, 1.39), 0.38, 0.024, "metal")
    for x in (-0.27, 0.27):
        b.rod(base, (x, 0.22, 0.85), (x, 0.22, 2.16), 0.023, "bright", collision=True)
    b.box(base, (0, 0.22, 2.17), (0.32, 0.06, 0.035), "metal")
    lid = b.moving(
        base,
        "inspection_lid_lift",
        (0, 0, 1.72),
        (0, 0, 1),
        (-0.035, 0.18),
        mass=3.0,
        kp=850,
        force=1000,
    )
    b.cylinder(lid, (0, 0, 0), 0.38, 0.025, "bright")
    # The collision core leaves the two guide-bearing passages free.
    b.cylinder(lid, (0, 0, 0), 0.30, 0.024, "bright", collision=True)
    for x in (-0.27, 0.27):
        b.cylinder(lid, (x, 0.22, 0.04), 0.039, 0.055, "metal")
    b.cylinder(lid, (0, 0.07, 0.18), 0.055, 0.15, "metal")
    for index in range(10):
        b.ring(lid, (0, 0.07, 0.055 + index * 0.025), 0.060, 0.006, "dark", segments=16)
    # Visible inert witness coupon sits on an original top inspection fixture.
    b.cylinder(base, (0, 0, 1.435), 0.064, 0.021, "dark")
    carrier = b.moving(
        base,
        "witness_height",
        (0, 0, 1.475),
        (0, 0, 1),
        (-0.008, 0.018),
        mass=0.08,
        kp=200,
    )
    b.cylinder(carrier, (0, 0, 0), 0.06, 0.020, "metal", collision=True)
    b.cylinder(carrier, (0, 0, 0.021), 0.022, 0.001, "sample")
    b.site(carrier, "witness_coupon", (0, 0, 0.022), 0.001)
    b.box(base, (0.37, -0.32, 0.43), (0.12, 0.13, 0.34), "dark")
    b.screen(base, (0.37, -0.452, 0.55), 0.16, 0.12)
    b.rod(base, (-0.08, 0.33, 0.93), (-0.33, 0.42, 0.55), 0.048, "metal")
    b.box(base, (-0.29, 0.32, 0.25), (0.16, 0.14, 0.16), "dark")
    b.metadata["capabilities"] = [
        "guided_inspection_lid_lift",
        "witness_coupon_height",
        "visible_inert_coupon",
    ]
    b.metadata["limitations"].append(
        "Default pose is an authored open maintenance vignette. The witness fixture is an original mechanical proxy above a solid exterior body, not a reconstructed vacuum chamber interior. No vacuum, sputtering, plasma, deposition, bakeout or process controls."
    )


def _vacuum_features(world, definition):
    metal = (0.38, 0.41, 0.42, 1)
    for x in (1.18, 1.83):
        for y in (0.5, 1.2):
            box(
                world,
                f"aja_rack_post_{x}_{y}",
                (0.025, 0.025, 0.99),
                (x, y, 0.99),
                metal,
            )
    for index in range(5):
        z = 0.15 + index * 0.36
        box(
            world,
            f"aja_rack_shelf_{index}",
            (0.35, 0.36, 0.02),
            (1.505, 0.85, z),
            metal,
        )
        box(
            world,
            f"aja_controller_{index}",
            (0.29, 0.28, 0.10),
            (1.505, 0.82, z + 0.12),
            (0.72, 0.74, 0.72, 1),
        )
        box(
            world,
            f"aja_controller_screen_{index}",
            (0.085, 0.003, 0.035),
            (1.42, 0.537, z + 0.12),
            (0.03, 0.14, 0.17, 1),
            collision=False,
        )


def _mask_aligner(b, base, params):
    for x in (-0.53, 0.53):
        for y in (-0.45, 0.45):
            b.box(base, (x, y, 0.44), (0.034, 0.034, 0.44), "metal", collision=True)
            b.cylinder(base, (x, y, 0.025), 0.055, 0.025, "dark")
    for z in (0.17, 0.84):
        b.box(base, (0, 0, z), (0.58, 0.5, 0.024), "metal", collision=True)
    for x in (-0.37, 0.37):
        b.box(base, (x, 0.25, 1.37), (0.095, 0.115, 0.505), "cream", collision=True)
    b.box(base, (0, 0.25, 1.92), (0.49, 0.15, 0.045), "cream", collision=True)
    b.box(base, (0, 0.05, 1.70), (0.23, 0.33, 0.10), "cream")
    for x in (-0.04, 0.04):
        b.rod(base, (x, -0.19, 1.73), (x * 1.2, -0.35, 1.85), 0.028, "dark")
    b.cylinder(base, (0, -0.11, 1.53), 0.06, 0.095, "dark")
    b.box(base, (0, -0.13, 0.93), (0.22, 0.20, 0.06), "metal", collision=True, condim=1)
    xstage = b.moving(
        base, "wafer_x", (0, -0.13, 1.013), (1, 0, 0), (-0.04, 0.04), mass=0.25, kp=300
    )
    # Ideal linear-bearing contact has normal support without tangential friction.
    b.box(xstage, (0, 0, 0), (0.20, 0.18, 0.023), "dark", collision=True, condim=1)
    ystage = b.moving(
        xstage, "wafer_y", (0, 0, 0.037), (0, 1, 0), (-0.035, 0.035), mass=0.14, kp=280
    )
    b.cylinder(ystage, (0, 0, 0), 0.125, 0.013, "metal", collision=True)
    theta = b.moving(
        ystage,
        "wafer_theta",
        (0, 0, 0.021),
        (0, 0, 1),
        (-0.18, 0.18),
        kind="hinge",
        mass=0.08,
        kp=160,
    )
    b.cylinder(theta, (0, 0, 0), 0.105, 0.008, "dark", collision=True)
    b.cylinder(theta, (0, 0, 0.0085), 0.075, 0.0005, "sample")
    b.box(theta, (0.044, 0, 0.0091), (0.013, 0.002, 0.0001), "bright")
    b.site(theta, "alignment_wafer", (0, 0, 0.009), 0.003)
    # Raised mask holder is carried by the two main columns, leaving target visible.
    for x in (-0.25, 0.25):
        b.box(base, (x, 0.10, 1.24), (0.055, 0.15, 0.024), "metal")
    for y in (-0.23, 0.13):
        b.box(base, (0, y, 1.24), (0.25, 0.025, 0.024), "dark")
    b.box(base, (0, -0.05, 1.24), (0.22, 0.16, 0.002), "glass")
    # Bulky rear lamp housing and closed flexible exhaust path are distinctive.
    b.box(base, (0.42, 0.33, 1.62), (0.19, 0.20, 0.25), "dark")
    points = (
        (0.58, 0.34, 1.75),
        (0.76, 0.34, 1.73),
        (0.90, 0.34, 1.58),
        (0.92, 0.34, 1.29),
        (0.85, 0.34, 1.07),
        (0.60, 0.34, 0.96),
    )
    for start, end in zip(points, points[1:]):
        b.rod(base, start, end, 0.088, "metal")
    for index in range(13):
        b.ring(
            base,
            (0.916, 0.34, 1.23 + index * 0.025),
            0.091,
            0.004,
            "bright",
            segments=16,
        )
    b.box(base, (0.42, 0.08, 0.62), (0.12, 0.29, 0.19), "cream", collision=True)
    b.screen(base, (0.42, -0.217, 0.68), 0.16, 0.11)
    for index in range(5):
        b.cylinder(
            base,
            (0.35 + (index % 2) * 0.11, -0.23, 0.55 + (index // 2) * 0.07),
            0.020,
            0.012,
            "dark",
            euler=(math.pi / 2, 0, 0),
        )
    b.metadata["capabilities"] = [
        "wafer_xy_position",
        "wafer_rotation",
        "visible_150mm_inert_wafer",
    ]
    b.metadata["limitations"].append(
        "No UV exposure, photochemistry, imprinting, resist development, mask contact or calibrated overlay accuracy. Three axes are bounded coarse mechanical proxies; amber room appearance does not establish cleanroom certification."
    )


def _nanofab_features(world, definition):
    # Perforated raised-floor tile appearance, with a continuous solid floor below.
    for x in range(-8, 9):
        for y in range(-7, 8):
            if (x + y) % 2 == 0:
                box(
                    world,
                    f"ma8_floor_slot_{x}_{y}",
                    (0.055, 0.008, 0.0007),
                    (x * 0.27, y * 0.27, 0.001),
                    (0.25, 0.24, 0.18, 1),
                    collision=False,
                )
    box(
        world,
        "ma8_service_strip",
        (2.25, 0.04, 0.08),
        (0, 2.12, 1.32),
        (0.55, 0.54, 0.43, 1),
    )
    for index in range(6):
        box(
            world,
            f"ma8_wall_socket_{index}",
            (0.055, 0.008, 0.045),
            (-1.8 + index * 0.65, 2.074, 1.32),
            (0.88, 0.84, 0.66, 1),
            collision=False,
        )


def _gcf_web_cells():
    """Estimated hole layout: large upper bores and smaller staggered bores."""
    for low, high, center, radii in (
        (0.010, 0.160, 0.086, (0.030, 0.042, 0.026, 0.035, 0.042, 0.028)),
        (0.160, 0.365, 0.258, (0.071, 0.082, 0.068, 0.084, 0.074, 0.082, 0.071)),
    ):
        width = 2.42 / len(radii)
        for index, radius in enumerate(radii):
            left = -1.21 + width * index
            yield (left, left + width, low, high), (left + width / 2, center, radius)


def _gcf_web_sectors(bounds, hole):
    """Convex quadrilaterals fill a rectangle around a circular polygon bore.

    Include corner rays so the outer boundary remains exactly rectangular.
    Each quadrilateral becomes a convex contact prism; a single convex hull
    of the complete plate would incorrectly close the lightening apertures.
    """
    left, right, low, high = bounds
    cx, cz, radius = hole
    if radius >= min(cx - left, right - cx, cz - low, high - cz):
        raise ValueError("GCF aperture must leave material around its full perimeter")
    angles = {math.tau * i / 32 for i in range(32)}
    angles.update(
        math.atan2(z - cz, x - cx) % math.tau
        for x in (left, right)
        for z in (low, high)
    )
    perimeter = []
    for angle in sorted(angles):
        dx, dz = math.cos(angle), math.sin(angle)
        tx = ((right if dx > 0 else left) - cx) / dx if abs(dx) > 1e-12 else math.inf
        tz = ((high if dz > 0 else low) - cz) / dz if abs(dz) > 1e-12 else math.inf
        distance = min(tx, tz)
        perimeter.append(
            (
                (cx + radius * dx, cz + radius * dz),
                (cx + distance * dx, cz + distance * dz),
            )
        )
    for index, (inner, outer) in enumerate(perimeter):
        next_inner, next_outer = perimeter[(index + 1) % len(perimeter)]
        yield (inner, outer, next_outer, next_inner)


def _gcf_bridge_webs(b, bridge):
    """Finite-thickness webs with the same open geometry for view and contact."""
    from real_labs.perforated_web import web_surface

    sectors, contact_meshes = [], []
    faces = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)) + tuple(
        face
        for i in range(4)
        for face in ((i, i + 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, (i + 1) % 4))
    )
    asset = b.root.find("asset")
    for bounds, hole in _gcf_web_cells():
        for quad in _gcf_web_sectors(bounds, hole):
            sectors.append(quad)
            vertices = [(x, y, z) for y in (-0.010, 0.010) for x, z in quad]
            mesh = b.unique("gcf_web_contact_mesh")
            ET.SubElement(
                asset,
                "mesh",
                name=mesh,
                vertex=numbers([v for point in vertices for v in point]),
                face=" ".join(str(v) for face in faces for v in face),
            )
            contact_meshes.append(mesh)
    visual_vertices, visual_faces = web_surface(sectors, half_thickness=0.010)
    visual = b.unique("gcf_perforated_web_mesh")
    ET.SubElement(
        asset,
        "mesh",
        name=visual,
        vertex=numbers([v for point in visual_vertices for v in point]),
        face=" ".join(str(v) for face in visual_faces for v in face),
    )
    for side in (-1, 1):
        b.geom(
            bridge,
            "mesh",
            (),
            (0, side * 0.15, 0),
            material="cream",
            mesh=visual,
            name=b.unique("gcf_perforated_web"),
            density=0,
        )
        for mesh in contact_meshes:
            b.geom(
                bridge,
                "mesh",
                (),
                (0, side * 0.15, 0),
                mesh=mesh,
                collision=True,
                group=3,
                rgba=(0, 0, 0, 0),
                density=0,
                name=b.unique("gcf_web_contact"),
            )


def _gcf_diamond_wall(b, parent, half_length, fixed, plane):
    """Flat welded stiffeners clipped to the backed container wall perimeter."""
    low, high, pitch = 0.305, 0.935, 0.16
    for slope in (-1, 1):
        first = math.floor((low - half_length) / pitch)
        last = math.ceil((high + half_length) / pitch)
        for index in range(first, last + 1):
            offset = index * pitch
            candidates = []
            for u in (-half_length, half_length):
                z = slope * u + offset
                if low <= z <= high:
                    candidates.append((u, z))
            for z in (low, high):
                u = (z - offset) / slope
                if -half_length <= u <= half_length:
                    candidates.append((u, z))
            points = sorted(set(candidates))
            if len(points) < 2:
                continue
            (a, za), (c, zc) = points[0], points[-1]
            length = math.hypot(c - a, zc - za)
            if length < 0.025:
                continue
            angle = math.atan2(zc - za, c - a)
            if plane == "xz":
                pos, size, euler = (
                    ((a + c) / 2, fixed, (za + zc) / 2),
                    (length / 2, 0.016, 0.010),
                    (0, -angle, 0),
                )
            else:
                pos, size, euler = (
                    (fixed, (a + c) / 2, (za + zc) / 2),
                    (0.016, length / 2, 0.010),
                    (angle, 0, 0),
                )
            b.box(parent, pos, size, "metal", euler=euler, collision=True).set(
                "name", b.unique("gcf_diamond_stiffener")
            )


def _geotechnical_testbed(b, base, params):
    # The reference shows a neutral interior, not a polished copper fill.
    # Composition is unknown; this remains a rigid, authored inert model bed.
    ET.SubElement(
        b.root.find("asset"),
        "material",
        name=f"{b.name}__mat_gcf_inert_bed",
        rgba=".42 .43 .42 1",
        specular=".12",
        shininess=".04",
        reflectance="0",
        metallic="0",
        roughness=".8",
    )
    for x in (-1.17, 1.17):
        for y in (-0.72, 0.72):
            b.box(base, (x, y, 0.12), (0.10, 0.10, 0.12), "metal", collision=True)
            b.box(base, (x, y, 0.028), (0.16, 0.15, 0.028), "dark")
    b.box(base, (0, 0, 0.25), (1.25, 0.79, 0.04), "metal", collision=True)
    for side in (-1, 1):
        b.box(
            base, (side * 1.22, 0, 0.62), (0.065, 0.79, 0.33), "bright", collision=True
        )
        b.box(
            base, (0, side * 0.76, 0.62), (1.155, 0.055, 0.33), "bright", collision=True
        )
        _gcf_diamond_wall(b, base, 1.15, side * 0.825, "xz")
        _gcf_diamond_wall(b, base, 0.76, side * 1.294, "yz")
        b.box(
            base, (side * 1.30, 0, 1.025), (0.10, 0.89, 0.045), "metal", collision=True
        )
        b.box(base, (side * 1.30, 0, 1.088), (0.035, 0.84, 0.018), "bright")
        for y in (-0.69, 0.69):
            b.box(base, (side * 1.30, y, 0.94), (0.075, 0.065, 0.044), "metal")
    b.box(base, (0, 0, 0.585), (1.15, 0.69, 0.295), "gcf_inert_bed", collision=True)
    b.box(base, (0.04, 0.18, 0.96), (0.06, 0.06, 0.08), "sample", collision=True)
    b.site(base, "inert_pier", (0.04, 0.18, 1.04), 0.005)
    bridge = b.moving(
        base,
        "bridge_y",
        (0, 0, 1.18),
        (0, 1, 0),
        (-0.48, 0.48),
        mass=12,
        kp=1000,
        force=2500,
    )
    for side in (-1, 1):
        b.box(
            bridge,
            (side * 1.30, 0, 0.12),
            (0.09, 0.18, 0.21),
            "guard_red",
            collision=True,
        )
        b.box(
            bridge,
            (side * 1.30, 0, 0.355),
            (0.09, 0.18, 0.025),
            "guard_red",
            collision=True,
        )
        b.cylinder(
            bridge,
            (side * 1.30, 0.11, -0.045),
            0.045,
            0.038,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
    for y in (-0.15, 0.15):
        for z in (0.035, 0.34):
            b.box(bridge, (0, y, z), (1.23, 0.018, 0.025), "bright", collision=True)
    _gcf_bridge_webs(b, bridge)
    b.box(bridge, (0, 0, 0.38), (1.28, 0.17, 0.025), "metal", collision=True)
    trolley = b.moving(
        bridge, "carriage_x", (0, 0, 0.245), (1, 0, 0), (-0.76, 0.76), mass=1.4, kp=420
    )
    # The carriage rides above the solid web rather than intersecting it.
    # Its lower face meets the deck at bridge-local z=.405; the overhanging
    # bracket supports the unchanged external probe column and axis origins.
    b.box(trolley, (0, 0, 0.215), (0.13, 0.16, 0.055), "dark", collision=True).set(
        "name", b.unique("gcf_top_carriage")
    )
    b.box(trolley, (0, -0.22, 0.17), (0.10, 0.16, 0.01), "metal", collision=True)
    from real_labs.hkust_probe_geometry import add_probe_head

    add_probe_head(b, trolley)
    b.metadata["capabilities"] = [
        "bridge_translation",
        "transverse_carriage",
        "probe_height",
        "probe_rotation",
        "visible_inert_model_pier",
    ]
    b.metadata["limitations"].append(
        "Standalone mechanical positioning rig only, with a rigid inert bed and pier. It does not spin, generate centrifuge acceleration, deform soil, apply calibrated loads, model earthquakes or reproduce the GCF control system."
    )
    b.metadata["limitations"].append(
        "Flat perforated bridge webs, staggered hole sizes and diamond wall stiffeners follow the official robot photograph. Hole dimensions, weld sections and the top-carriage bracket are estimated exterior reconstruction, not measured CAD or structural validation. Web contact is decomposed around each true bore; slide joints idealize captive guide reactions."
    )


def _gcf_features(world, definition):
    # A service bay around the separately photographed model container.
    for index in range(7):
        box(
            world,
            f"gcf_wall_panel_{index}",
            (0.37, 0.02, 0.45),
            (-2.6 + index * 0.87, 2.77, 1.72),
            (0.67, 0.70, 0.67, 1),
            collision=False,
        )
    box(
        world,
        "gcf_bay_mark_front",
        (1.60, 0.022, 0.001),
        (0, -1.35, 0.002),
        (0.90, 0.70, 0.13, 1),
        collision=False,
    )
    for side in (-1, 1):
        box(
            world,
            f"gcf_bay_mark_side_{side}",
            (0.022, 1.25, 0.001),
            (side * 1.68, -0.1, 0.002),
            (0.90, 0.70, 0.13, 1),
            collision=False,
        )


BUILDERS = {
    "nearfield_optical_station": _nearfield,
    "multiport_vacuum_inspection_station": _vacuum_station,
    "e6_mask_alignment_station": _mask_aligner,
    "geotechnical_robotic_testbed": _geotechnical_testbed,
}
SAMPLE_INTERFACES = {
    "nearfield_optical_station": (
        "optical_coupon",
        (0.012, 0.012, 0.001),
        "clamped",
        "Visible inert optical coupon on a supported XY carrier",
    ),
    "multiport_vacuum_inspection_station": (
        "witness_coupon",
        (0.044, 0.044, 0.002),
        "clamped",
        "Visible inert witness coupon on an authored inspection fixture",
    ),
    "e6_mask_alignment_station": (
        "alignment_wafer",
        (0.15, 0.15, 0.001),
        "clamped",
        "Visible inert 150 mm alignment wafer with an asymmetric fiducial",
    ),
    "geotechnical_robotic_testbed": (
        "inert_pier",
        (0.12, 0.12, 0.16),
        "clamped",
        "Visible rigid model pier seated on an inert test bed",
    ),
}
FEATURES = {
    "ntu_coeb_nanophotonics": _coeb_features,
    "ntu_intelligent_materials": _vacuum_features,
    "nus_e6nanofab_l1": _nanofab_features,
    "hkust_geotechnical_centrifuge": _gcf_features,
}
