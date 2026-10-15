"""Source-informed Leeds exterior and an explicitly authored contact fixture.

Photographs establish the lever, collars and chamber arrangement, not dimensions,
powered axes, load calibration or environmental/process behavior.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers
from real_labs.perforated_web import web_surface


def named(b, key):
    return f"{b.name}__leeds_{key}"


def _bearing_carriage(b, parent):
    """One continuous two-bore shell with open convex collision sectors.

    The prismatic joint supplies ideal bearing guidance; 0.5 mm nominal radial
    clearance keeps the explicitly collidable columns out of the carriage solid.
    """
    sectors = []
    for cx, left, right in ((-0.03, -0.045, 0), (0.03, 0, 0.045)):
        cy, radius, low, high = 0.01, 0.0085, -0.025, 0.045
        angles = sorted(
            set(
                [math.tau * i / 32 for i in range(32)]
                + [
                    math.atan2(y - cy, x - cx) % math.tau
                    for x in (left, right)
                    for y in (low, high)
                ]
            )
        )
        inside, outside = [], []
        for angle in angles:
            dx, dy = math.cos(angle), math.sin(angle)
            length = min(
                (
                    ((right - cx) if dx > 0 else (left - cx)) / dx
                    if abs(dx) > 1e-12
                    else math.inf
                ),
                (
                    ((high - cy) if dy > 0 else (low - cy)) / dy
                    if abs(dy) > 1e-12
                    else math.inf
                ),
            )
            inside.append((cx + radius * dx, cy + radius * dy))
            outside.append((cx + length * dx, cy + length * dy))
        sectors.extend(
            (
                inside[i],
                outside[i],
                outside[(i + 1) % len(angles)],
                inside[(i + 1) % len(angles)],
            )
            for i in range(len(angles))
        )
    vertices, faces = web_surface(sectors, 0.035)
    # web_surface extrudes X/Z along Y; swap its Y/Z and reverse winding.
    mesh = named(b, "carriage_surface_mesh")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        maxhullvert="4",
        vertex=numbers([v for x, y, z in vertices for v in (x, z, y)]),
        face=numbers([v for a, c, d in faces for v in (a, d, c)]),
    )
    b.geom(
        parent, "mesh", (), material="bright", mesh=mesh, name=named(b, "load_carriage")
    )
    contacts = []
    for index, sector in enumerate(sectors):
        mesh = named(b, f"carriage_contact_mesh_{index}")
        ET.SubElement(
            b.root.find("asset"),
            "mesh",
            name=mesh,
            vertex=numbers(
                [v for z in (-0.035, 0.035) for x, y in sector for v in (x, y, z)]
            ),
        )
        name = named(b, f"carriage_contact_{index}")
        b.geom(
            parent,
            "mesh",
            (),
            material="bright",
            mesh=mesh,
            collision=True,
            group="3",
            name=name,
        )
        contacts.append(name)
    b.metadata["bearing_contact_geoms"] = contacts


def viewport_door(b, base):
    """A continuous perforated plate with matching convex contact sectors."""
    door = b.moving(
        base,
        "access_door",
        (-0.46, -0.341, 1.15),
        (0, 0, -1),
        (0, 1.2),
        kind="hinge",
        mass=0.7,
    )
    door.set("euler", "0 0 -.78")
    door.find("joint").set("ref", ".78")
    cx, radius, thickness = 0.49, 0.072, 0.018
    angles = sorted(
        set(
            [math.tau * i / 64 for i in range(64)]
            + [
                math.atan2(z, x - cx) % math.tau
                for x in (0.015, 0.885)
                for z in (-0.277, 0.277)
            ]
        )
    )
    inner, outer = [], []
    for angle in angles:
        dx, dz = math.cos(angle), math.sin(angle)
        length = min(
            (
                ((0.885 - cx) if dx > 0 else (0.015 - cx)) / dx
                if abs(dx) > 1e-12
                else math.inf
            ),
            (0.277 if dz > 0 else -0.277) / dz if abs(dz) > 1e-12 else math.inf,
        )
        inner.append((cx + radius * dx, radius * dz))
        outer.append((cx + length * dx, length * dz))
    sectors = [
        (inner[i], outer[i], outer[(i + 1) % len(angles)], inner[(i + 1) % len(angles)])
        for i in range(len(angles))
    ]
    vertices, faces = web_surface(sectors, thickness)
    asset = b.root.find("asset")
    mesh = named(b, "door_surface_mesh")
    ET.SubElement(
        asset,
        "mesh",
        name=mesh,
        maxhullvert="4",
        vertex=numbers([v for p in vertices for v in p]),
        face=numbers([v for p in faces for v in p]),
    )
    b.geom(door, "mesh", (), material="metal", mesh=mesh, name=named(b, "door_surface"))
    for i, sector in enumerate(sectors):
        mesh = named(b, f"door_contact_mesh_{i}")
        ET.SubElement(
            asset,
            "mesh",
            name=mesh,
            vertex=numbers(
                [
                    v
                    for y in (-thickness, thickness)
                    for x, z in sector
                    for v in (x, y, z)
                ]
            ),
        )
        b.geom(
            door,
            "mesh",
            (),
            material="metal",
            mesh=mesh,
            collision=True,
            group="3",
            name=named(b, f"door_contact_{i}"),
        )
    for y in (-0.021, 0.021):
        b.ring(door, (cx, y, 0), 0.079, 0.007, "bright", "xz", 48)
    b.geom(
        door,
        "cylinder",
        (0.0715, 0.005),
        (cx, 0, 0),
        "lens",
        euler=(math.pi / 2, 0, 0),
        collision=True,
        name=named(b, "viewport_glass"),
    )
    for z in (-0.10, 0.10):
        b.rod(
            door,
            (0.80, -0.018, z),
            (0.80, -0.055, z),
            0.008,
            "bright",
            collision=True,
            name=named(b, f"handle_mount_{z}"),
        )
    b.rod(
        door,
        (0.80, -0.055, -0.10),
        (0.80, -0.055, 0.10),
        0.011,
        "dark",
        collision=True,
        name=named(b, "door_handle"),
    )
    b.box(door, (0.18, -0.019, 0.13), (0.065, 0.001, 0.039), "warning")
    # Visible closure pads meet the plate at q=0. They model a mechanical
    # interface only, with no pressure/sealing claim.
    for x in (-0.414, 0.414):
        b.box(
            base,
            (x, -0.3215, 1.15),
            (0.012, 0.0015, 0.255),
            "rubber",
            name=named(b, f"closure_strip_{x}"),
        )
    for z in (0.888, 1.405):
        b.box(base, (0, -0.3215, z), (0.400, 0.0015, 0.006), "rubber")
    # Knuckles are split along the same hinge axis; no imaginary pressure seal.
    for z in (-0.20, 0.20):
        b.cylinder(base, (-0.46, -0.341, 1.15 + z), 0.013, 0.025, "bright")
        b.box(base, (-0.44, -0.317, 1.15 + z), (0.02, 0.018, 0.022), "metal")
        b.rod(door, (0, 0, z), (0.035, 0, z), 0.010, "bright")
    return door


def contact_fixture(b, base):
    """Generic compliant normal approach and ±12 mm specimen translation.

    The coupon home datum is retained. A force-limited finite-stiffness servo acts as
    the authored compliant load train; no manufacturer force calibration is implied.
    """
    # The chamber photograph resolves repeated brass fasteners, silver backing
    # plates and the fixture's machined base. Absolute sizes are inferred.
    for i, x in enumerate((-.31, -.16, .04, .24)):
        for j, z in enumerate((1.00, 1.18, 1.34)):
            b.cylinder(base, (x, .271, z), .012, .004, "copper",
                       euler=(math.pi / 2, 0, 0),
                       name=named(b, f"rear_fastener_{i}_{j}"))
            b.cylinder(base, (x, .266, z), .005, .002, "dark",
                       euler=(math.pi / 2, 0, 0),
                       name=named(b, f"rear_socket_{i}_{j}"))
    for x in (-.325, -.175):
        for y in (-.005, .11):
            b.cylinder(base, (x, y, .936), .007, .003, "dark",
                       name=named(b, f"base_anchor_{x}_{y}"))
    b.cylinder(
        base,
        (0.15, -0.06, 0.923),
        0.135,
        0.038,
        "copper",
        collision=True,
        name=named(b, "stage_foundation"),
    )
    b.cylinder(
        base,
        (0.15, -0.06, 0.993),
        0.118 * math.cos(math.pi / 48) - 0.009,
        0.032,
        "metal",
        collision=True,
        name=named(b, "stage_core"),
    )
    for z in (0.981, 1.003, 1.025):
        b.ring(base, (0.15, -0.06, z), 0.118, 0.009, "copper", "xy", 48)
    # A rear header joins all three retained loops and meets the foundation.
    # Their inner tangent surfaces also touch the supporting core. No flow model.
    b.cylinder(
        base,
        (0.15, 0.058, 0.993),
        0.009,
        0.032,
        "copper",
        collision=True,
        name=named(b, "coil_support_header"),
    )
    for y in (-0.126, 0.006):
        b.box(
            base,
            (0.15, y, 1.036),
            (0.072, 0.009, 0.011),
            "bright",
            collision=True,
            name=named(b, f"slide_rail_{y}"),
        )
    slide = b.moving(
        base,
        "specimen_slide",
        (0.15, -0.06, 1.0635),
        (1, 0, 0),
        (-0.012, 0.012),
        mass=0.35,
        kp=6000,
        force=40,
    )
    b.cylinder(
        slide,
        (0, 0, 0),
        0.105,
        0.0165,
        "bright",
        collision=True,
        name=named(b, "specimen_platen"),
    )
    b.cylinder(
        slide,
        (0, 0, 0.0205),
        0.064,
        0.004,
        "metal",
        collision=True,
        name=named(b, "coupon"),
        friction=".35 .001 .0001",
        solref=".04 1",
        solimp=".95 .99 .001",
    )
    for i in range(3):
        angle = math.tau * i / 3
        # Pads touch the upper coupon face; outside posts meet the platen.
        for key, radius, z, half in (
            ("clip_post", 0.085, 0.0225, (0.004, 0.007, 0.006)),
            ("clip_bridge", 0.073, 0.0305, (0.016, 0.007, 0.002)),
            ("clip_pad", 0.060, 0.0265, (0.004, 0.007, 0.002)),
        ):
            b.box(
                slide,
                (radius * math.cos(angle), radius * math.sin(angle), z),
                half,
                "bright",
                collision=True,
                euler=(0, 0, angle),
                name=named(b, f"{key}_{i}"),
            )
        b.cylinder(
            slide,
            (0.085 * math.cos(angle), 0.085 * math.sin(angle), 0.0275),
            0.003,
            0.005,
            "dark",
            name=named(b, f"clip_screw_{i}"),
        )
    b.site(slide, "dry_tribology_coupon", (0, 0, 0.0245), 0.025)
    # Broad machined base and a low bearing carriage match visible source cues.
    b.box(
        base,
        (-0.25, 0.055, 0.909),
        (0.092, 0.075, 0.024),
        "bright",
        collision=True,
        name=named(b, "load_base"),
    )
    for y in (0.012, 0.098):
        b.box(base, (-0.25, y, 0.942), (0.083, 0.010, 0.009), "metal")
    b.box(
        base,
        (-0.25, 0.055, 1.023),
        (0.052, 0.047, 0.09),
        "metal",
        collision=True,
        name=named(b, "bearing_pedestal"),
    )
    for x in (-0.28, -0.22):
        b.cylinder(
            base,
            (x, 0.065, 1.159),
            0.008,
            0.070,
            "bright",
            collision=True,
            name=named(b, f"approach_guide_{x}"),
        )
    b.geom(
        base,
        "cylinder",
        (0.025, 0.028),
        (-0.25, -0.018, 1.021),
        "copper",
        euler=(math.pi / 2, 0, 0),
        name=named(b, "bearing_cover"),
    )
    b.rod(base, (-0.35, -0.015, 0.946), (-0.15, -0.015, 0.946), 0.009, "copper")
    for x in (-0.35, -0.15):
        b.box(base, (x, -0.015, 0.941), (0.012, 0.021, 0.008), "bright")
        b.geom(
            base,
            "cylinder",
            (0.017, 0.012),
            (x, -0.015, 0.946),
            "metal",
            euler=(0, math.pi / 2, 0),
        )
    arm = b.moving(
        base,
        "contact_approach",
        (-0.25, 0.055, 1.19),
        (0, 0, -1),
        (0, 0.025),
        mass=0.4,
        kp=1000,
        force=10,
    )
    _bearing_carriage(b, arm)
    angle = math.atan2(-0.115, 0.40)
    b.box(
        arm,
        (0.2225, -0.06396875, 0),
        (math.hypot(0.355, 0.1020625) / 2, 0.014, 0.009),
        "bright",
        collision=True,
        euler=(0, 0, angle),
        name=named(b, "flat_lever"),
    )
    b.cylinder(
        arm,
        (0.40, -0.115, 0.008),
        0.006,
        0.085,
        "bright",
        collision=True,
        name=named(b, "load_shaft"),
    )
    for key, z, radius, half in (
        ("lower_collar", -0.011, 0.023, 0.012),
        ("upper_collar", 0.033, 0.030, 0.007),
    ):
        b.cylinder(
            arm,
            (0.40, -0.115, z),
            radius,
            half,
            "bright",
            collision=True,
            name=named(b, key),
        )
    b.cylinder(
        arm,
        (0.40, -0.115, -0.0735),
        0.003,
        0.0055,
        "bright",
        collision=True,
        name=named(b, "counterface_stem"),
    )
    b.geom(
        arm,
        "sphere",
        (0.003,),
        (0.40, -0.115, -0.079),
        "bright",
        collision=True,
        name=named(b, "counterface"),
        friction=".35 .001 .0001",
        solref=".04 1",
        solimp=".95 .99 .001",
    )
    b.site(arm, "counterface_tip", (0.40, -0.115, -0.082), 0.001)
    b.metadata["capabilities"].append("Measured generic loaded sliding and retraction")
    b.metadata["inspection_targets"] = {"contact_approach": 0.012}
    b.metadata["contact_operation"] = dict(
        kind="authored_generic_loaded_sliding",
        approach_control="contact_approach",
        slide_control="specimen_slide",
        held_control="access_door",
        counterface=named(b, "counterface"),
        coupon=named(b, "coupon"),
        normal_stiffness_n_per_m=1000,
        actuator_force_limit_n=10,
        contact_time_constant_s=0.04,
        contact_damping_ratio=1,
        commanded_load_position_m=0.024,
        home_tip_gap_m=0.020,
        slide_amplitude_m=0.010,
        generic_friction=(0.35, 0.001, 0.0001),
        interpretation="Finite-stiffness servo contact with a retained rigid coupon; normal force and relative motion are measured in MuJoCo. Parameters are authored, not fitted to the photographed instrument. No validated friction, wear, thermal, gas or vacuum model.",
    )


def roof_services(b, base):
    """Connected estimated flanges and corrugated loops, all passive."""
    ports = ((-0.28, 0.10), (-0.04, 0.10), (0.19, 0.10), (0.30, 0.20))
    for i, (x, y) in enumerate(ports):
        b.cylinder(
            base,
            (x, y, 1.49),
            0.035,
            0.04,
            "bright",
            collision=True,
            name=named(b, f"roof_port_{i}"),
        )
        b.cylinder(base, (x, y, 1.534), 0.046, 0.004, "bright", collision=True)
    for pair, height in (((0, 1), 0.23), ((2, 3), 0.16)):
        first, last = [ports[i] for i in pair]
        points = []
        for i in range(25):
            t = i / 24
            points.append(
                (
                    first[0] * (1 - t) + last[0] * t,
                    first[1] * (1 - t) + last[1] * t,
                    1.538 + height * math.sin(math.pi * t),
                )
            )
        for i, (a, c) in enumerate(zip(points, points[1:])):
            b.rod(
                base,
                a,
                c,
                0.021 if pair[0] == 0 else 0.013,
                "metal",
                name=named(b, f"hose_{pair[0]}_{i}"),
            )
        for i in range(1, 24):
            direction = [c - a for a, c in zip(points[i - 1], points[i + 1])]
            length = math.sqrt(sum(v * v for v in direction))
            ends = [
                tuple(
                    p + sign * 0.002 * v / length for p, v in zip(points[i], direction)
                )
                for sign in (-1, 1)
            ]
            b.geom(
                base,
                "cylinder",
                (0.024 if pair[0] == 0 else 0.016, 0.002),
                material="bright",
                fromto=(*ends[0], *ends[1]),
                name=named(b, f"hose_rib_{pair[0]}_{i}"),
            )
    for i, (x, y) in enumerate(ports[1::2]):
        b.box(base, (x, y, 1.56), (0.030, 0.025, 0.022), "metal")
        b.cylinder(base, (x, y, 1.599), 0.009, 0.017, "bright")
        b.rod(base, (x - 0.035, y, 1.617), (x + 0.035, y, 1.617), 0.007, "dark")
