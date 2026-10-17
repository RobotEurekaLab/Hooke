"""Turner exterior topology with a bounded, authored maintenance fixture.

Official historical drawings support the four hollow members, end bridges,
parked two-arm carriage and continuous shaft with two bearings. Dimensions not
specified there remain estimates. The two existing inspection controls retain
their original frames/drives; they do not operate the OEM swinging carriage.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.cambridge_structure import (
    parked_swing_carriages, principal_members, upper_bearing_mount,
)


def _box(b, parent, key, pos, half, material="metal"):
    return b.box(
        parent, pos, half, material, collision=True, name=f"{b.name}__turner_{key}"
    )


def _mesh(b, key, vertices, faces):
    name = f"{b.name}__turner_{key}_mesh"
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=name,
        vertex=" ".join(str(v) for p in vertices for v in p),
        face=" ".join(str(i) for f in faces for i in f),
    )
    return name


def _bearing(b, base, key="bearing", low=0.576, high=0.750):
    """Open fixed race: smooth closed visual annulus and convex contact sectors.

    The 2 mm nominal radial gap is an authored running clearance. The existing
    ideal hinge carries load; rolling elements/friction/preload are not solved.
    """
    n, inner, outer = 48, 0.252, 0.52
    vertices = [
        (r * math.cos(math.tau * i / n), r * math.sin(math.tau * i / n), z)
        for z in (low, high)
        for r in (inner, outer)
        for i in range(n)
    ]
    faces = []
    quads = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    for i in range(n):
        j = (i + 1) % n
        ids = [i, n + i, n + j, j, 2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j]
        local = [(q[0], q[1], q[2]) for q in quads] + [
            (q[0], q[2], q[3]) for q in quads
        ]
        mesh = _mesh(b, f"{key}_contact_{i}", [vertices[k] for k in ids], local)
        b.geom(
            base,
            "mesh",
            (),
            mesh=mesh,
            collision=True,
            group="3",
            material="metal",
            name=f"{b.name}__turner_{key}_contact_{i}",
        )
        # Omit radial cell interfaces from the one continuous visible annulus.
        for q in (quads[0], quads[1], quads[3], quads[5]):
            idsq = [ids[k] for k in q]
            faces.extend(((idsq[0], idsq[1], idsq[2]), (idsq[0], idsq[2], idsq[3])))
    mesh = _mesh(b, key + "_surface", vertices, faces)
    b.geom(
        base,
        "mesh",
        (),
        mesh=mesh,
        material="bright",
        name=f"{b.name}__turner_{key}_surface",
    )


def build_turner(b, base, params):
    """Preserve both original frames, ranges, drives and the rigid sample."""
    b.cylinder(
        base,
        (0, 0, 0.11),
        0.93,
        0.11,
        "metal",
        collision=True,
        name=f"{b.name}__turner_foundation",
    )
    b.cylinder(
        base,
        (0, 0, 0.398),
        0.52,
        0.178,
        "dark",
        collision=True,
        name=f"{b.name}__turner_pedestal",
    )
    _bearing(b, base)
    upper_bearing_mount(b, base)
    _bearing(b, base, "upper_bearing", low=1.720, high=1.894)
    rotor = b.moving(
        base,
        "maintenance_index",
        (0, 0, 1.02),
        (0, 0, 1),
        (-0.2, 0.2),
        kind="hinge",
        mass=8,
        kp=500,
        force=350,
    )
    b.cylinder(
        rotor,
        (0, 0, 0.469),
        0.25,
        0.911,
        "bright",
        collision=True,
        name=f"{b.name}__turner_rotating_hub",
    )
    principal_members(b, rotor)
    parked_swing_carriages(b, rotor)
    for index in range(6):
        b.cylinder(rotor, (0, 0, 0.925 + index * 0.065), 0.30, 0.022,
                   "dark" if index % 2 else "bright", collision=True,
                   name=f"{b.name}__turner_slip_ring_{index}")
    for x in (-1.12, 1.12):
        _box(b, rotor, f"platform_bracket_{x}", (x, 0, 0.2025), (0.045, 0.65, 0.0075))
    for side, y in (("front", -0.47), ("rear", 0.47)):
        _box(b, rotor, "platform_" + side, (0, y, 0.27), (1.39, 0.20, 0.06), "shell")
    for side, x in (("left", -0.83), ("right", 0.83)):
        _box(b, rotor, "platform_" + side, (x, 0, 0.27), (0.56, 0.27, 0.06), "shell")
    for y in (-0.66, 0.66):
        _box(b, rotor, f"platform_rim_y_{y}", (0, y, 0.405), (1.42, 0.025, 0.075))
    # The source shows tapered cheeks at the central tray, not square blocks.
    # Their section is inferred; each welded cheek overlaps the platform edge.
    for sign in (-1, 1):
        cross = [(sign * 1.28, 0.329), (sign * 1.42, 0.329), (sign * 1.28, 0.48)]
        if sign < 0:
            cross.reverse()
        vertices = [(x, y, z) for y in (-0.67, 0.67) for x, z in cross]
        faces = [
            (0, 1, 2),
            (3, 5, 4),
            (0, 3, 4),
            (0, 4, 1),
            (1, 4, 5),
            (1, 5, 2),
            (2, 5, 3),
            (2, 3, 0),
        ]
        mesh = _mesh(b, f"platform_cheek_{sign}", vertices, faces)
        b.geom(
            rotor,
            "mesh",
            (),
            mesh=mesh,
            material="shell",
            collision=True,
            name=f"{b.name}__turner_platform_cheek_{sign}",
        )
    for index, (x, w, h) in enumerate(
        ((-0.86, 0.19, 0.09), (-0.40, 0.16, 0.12), (0.69, 0.23, 0.07))
    ):
        _box(
            b,
            rotor,
            f"platform_instrument_{index}",
            (x, 0.43 if index == 1 else 0, 0.33 + h),
            (w, 0.21, h),
            "shell",
        )
        _box(
            b,
            rotor,
            f"instrument_panel_{index}",
            (x, 0.213 if index == 1 else -0.217, 0.33 + h),
            (w * 0.72, 0.007, h * 0.65),
            "dark",
        )
    # Rectangular rails and genuinely open shoes: 0.5 mm nominal running gap.
    for y in (-0.27, 0.27):
        _box(b, rotor, f"rail_{y}", (4.04, y, 0.217), (0.56, 0.012, 0.012), "bright")
        for x in (3.5, 4.58):
            _box(
                b, rotor, f"rail_pedestal_{x}_{y}", (x, y, 0.200), (0.02, 0.026, 0.005)
            )
    carriage = b.moving(
        rotor,
        "specimen_carriage",
        (4.04, 0, 0.272),
        (1, 0, 0),
        (-0.15, 0.15),
        mass=0.8,
        kp=220,
    )
    _box(b, carriage, "carrier_plate", (0, 0, 0), (0.32, 0.32, 0.025), "bright")
    for x in (-0.20, 0.20):
        for y in (-0.27, 0.27):
            key = f"shoe_{x}_{y}"
            _box(b, carriage, key + "_top", (x, y, -0.03375), (0.05, 0.0275, 0.00875))
            _box(b, carriage, key + "_bottom", (x, y, -0.072), (0.05, 0.0275, 0.0045))
            for side in (-1, 1):
                _box(
                    b,
                    carriage,
                    key + f"_side_{side}",
                    (x, y + side * 0.020, -0.055),
                    (0.05, 0.0075, 0.0125),
                )
    for y in (-0.30, 0.30):
        _box(b, carriage, f"tray_side_y_{y}", (0, y, 0.11), (0.32, 0.02, 0.085))
    for x in (-0.30, 0.30):
        _box(b, carriage, f"tray_side_x_{x}", (x, 0, 0.11), (0.02, 0.28, 0.085))
    _box(b, carriage, "inert_sample", (0, 0, 0.105), (0.25, 0.25, 0.08), "copper")
    for x in (-0.16, 0, 0.16):
        _box(
            b,
            carriage,
            f"sample_retainer_{x}",
            (x, 0, 0.189),
            (0.008, 0.20, 0.004),
            "cream",
        )
    b.site(carriage, "inert_soil_model", (0.06, 0.10, 0.185), size=0.015)
    b.site(carriage, "sample_center", (0, 0, 0.105), size=0.015)
    b.metadata["capabilities"] = [
        "Bounded geometric beam indexing",
        "Retained inert specimen access and physical return",
    ]
    b.metadata["limitations"].append(
        "Source-shaped four-member bridge, two bearing housings and a continuous shaft. "
        "Both source-style swing platforms are rigidly parked: no swing or torsion dynamics. "
        "The translating retained-sample fixture is an authored maintenance overlay, "
        "not the OEM specimen carriage. Heights, clearances and wall thickness are estimated. "
        "Original mass, inertia and gravity compensation are inspection surrogates. "
        "Running clearances are carried by ideal joints, not rolling/contact bearing mechanics. "
        "Use cambridge_motion for measured 0.025 rad/s indexing and 0.035 m/s carriage "
        "speed checks; raw position commands are not speed-limited. No geotechnical "
        "similitude, high-speed dynamics, real operating procedure or safety validation."
    )
    b.metadata["turner_access_contract"] = {
        "rotor_speed_limit_rad_s": 0.025,
        "carriage_speed_limit_m_s": 0.035,
        "outer_radius_bound_m": 4.80,
        "coupled_point_speed_bound_m_s": 0.155,
        "surface_site": "inert_soil_model",
        "scope": "authored inspection task only",
    }

    b.metadata["turner_exterior_scope"] = {
        "source": "https://www-g.eng.cam.ac.uk/125/achievements/centrifuges/index.htm",
        "principal_members": 4,
        "member_outer_section_m": [0.4, 0.2],
        "member_length_m": 9.44,
        "member_length_scope": "approximately 10 m in source; inspection chamber fit estimated",
        "member_wall_thickness_m": 0.02,
        "wall_thickness_scope": "estimated, not a structural design specification",
        "bearing_housings": 2,
        "shaft_elevations_m": [0.578, 2.4],
        "shaft_scope": "continuous exterior, estimated fit to retained inspection datum",
        "swing_platforms": "two rigidly parked exterior assemblies, no swing actuator",
        "translation_fixture": "authored inspection overlay, not the OEM swing mechanism",
        "validation": "bounded low-speed inspection only; no high-g or bearing-load validation",
    }
