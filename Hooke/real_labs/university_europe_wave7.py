"""Reference-informed dry tribology, ice, cryostat and X-ray workcells.

Only access mechanisms and retained inert specimens are simulated. Geometry is
original and estimated from official photographs; no scientific solver or OEM
internal mechanism is implied by the apparatus exterior.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

LEEDS = "https://www.leeds.ac.uk/experimental-facilities/dir-record/profiles/23792/advanced-coatings-system-and-environmental-tribology-suite"
COPENHAGEN = "https://nbi.ku.dk/english/research/pice/centre-for-ice-and-climate/ice-core-facilities/"
UPPSALA = "https://www.uu.se/en/department/physics-and-astronomy/infrastructure/freia-laboratory/gersemi-vertical-cryostat"
LUND = (
    "https://www.maxiv.lu.se/beamlines-accelerators/beamlines/nanomax/imaging-station/"
)
STEEL = (0.49, 0.53, 0.56, 1)
WHITE = (0.84, 0.85, 0.83, 1)
DARK = (0.055, 0.065, 0.075, 1)
BLUE = (0.07, 0.25, 0.38, 1)
WOOD = (0.63, 0.46, 0.28, 1)


class _Room:
    def __init__(self, world, name):
        self.world, self.name, self.index = world, name, 0

    def box(self, key, pos, half, color, collision=False, **kwargs):
        self.index += 1
        return box(
            self.world,
            f"europe7_{self.name}_{key}_{self.index}",
            half,
            pos,
            color,
            collision=collision,
            **{
                k: numbers(v) if isinstance(v, (tuple, list)) else str(v)
                for k, v in kwargs.items()
            },
        )

    def rod(self, key, start, end, radius, color):
        self.index += 1
        return ET.SubElement(
            self.world,
            "geom",
            name=f"europe7_{self.name}_{key}_{self.index}",
            type="capsule",
            size=str(radius),
            fromto=numbers((*start, *end)),
            rgba=numbers(color),
            contype="0",
            conaffinity="0",
        )


def _flange(b, parent, center, radius, plane="xy"):
    b.ring(parent, center, radius, 0.009, "bright", plane, 28)
    a, c = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    for i in range(12):
        p = list(center)
        p[a] += radius * math.cos(math.tau * i / 12)
        p[c] += radius * math.sin(math.tau * i / 12)
        b.geom(parent, "sphere", (0.006,), p, "dark")


def _stand(b, base, center, half, height, material="metal"):
    x, y = center
    hx, hy = half
    for dx in (-hx + 0.04, hx - 0.04):
        for dy in (-hy + 0.04, hy - 0.04):
            b.box(
                base,
                (x + dx, y + dy, height / 2),
                (0.024, 0.024, height / 2),
                material,
                collision=True,
            )
            b.cylinder(
                base, (x + dx, y + dy, 0.022), 0.04, 0.022, "rubber", collision=True
            )
    b.box(base, (x, y, height), (hx, hy, 0.025), material, collision=True)
    b.box(base, (x, y, 0.18), (hx, hy, 0.015), material, collision=True)


def _screen(b, base, x, y, z):
    b.box(base, (x, y, z), (0.24, 0.025, 0.15), "dark")
    b.box(base, (x, y - 0.029, z), (0.217, 0.003, 0.127), "screen")
    for i in range(5):
        b.box(
            base,
            (x - 0.02, y - 0.034, z - 0.07 + i * 0.032),
            (0.14 + i * 0.01, 0.002, 0.002),
            "cyan",
        )
    b.rod(base, (x, y, z - 0.15), (x, y, z - 0.32), 0.016, "dark")
    b.box(base, (x, y - 0.02, z - 0.325), (0.13, 0.10, 0.014), "dark")


def _tribology(b, base, params):
    _stand(b, base, (0, 0), (0.57, 0.40), 0.78)
    # Thick hollow chamber with a clear swinging door.
    b.box(base, (0, 0, 0.845), (0.43, 0.32, 0.040), "metal", collision=True)
    for x in (-0.414, 0.414):
        b.box(base, (x, 0, 1.15), (0.026, 0.32, 0.27), "bright", collision=True)
    b.box(base, (0, 0.30, 1.15), (0.41, 0.026, 0.27), "metal", collision=True)
    b.box(base, (0, 0, 1.425), (0.44, 0.32, 0.025), "bright", collision=True)
    for x in (-0.30, -0.10, 0.10, 0.30):
        b.cylinder(base, (x, 0.15, 1.49), 0.042, 0.05, "bright")
        _flange(b, base, (x, 0.15, 1.53), 0.05)
    for i in range(3):
        b.rod(
            base,
            (-0.30 + i * 0.12, 0.15, 1.55),
            (-0.30 + i * 0.12, 0.40, 1.74 + i * 0.05),
            0.012,
            "bright",
        )
    door = b.moving(
        base,
        "access_door",
        (-0.46, -0.359, 1.15),
        (0, 0, -1),
        (0, 1.2),
        kind="hinge",
        mass=0.7,
    )
    door.set("euler", numbers((0, 0, -0.78)))
    door.find("joint").set("ref", ".78")
    b.box(door, (0.45, 0, 0), (0.435, 0.018, 0.277), "metal", collision=True)
    _flange(b, door, (0.49, -0.023, 0), 0.085, "xz")
    b.geom(
        door,
        "cylinder",
        (0.07, 0.006),
        (0.49, -0.026, 0),
        "lens",
        euler=(math.pi / 2, 0, 0),
    )
    b.rod(door, (0.80, -0.055, -0.10), (0.80, -0.055, 0.10), 0.011, "dark")
    b.box(door, (0.18, -0.023, 0.13), (0.065, 0.004, 0.039), "warning")
    # Concentric copper stage and permanently retained dry specimen.
    b.cylinder(base, (0.15, -0.06, 0.93), 0.135, 0.038, "copper", collision=True)
    for z in (0.977, 1.001, 1.025):
        b.cylinder(base, (0.15, -0.06, z), 0.116, 0.010, "copper", collision=True)
        b.ring(base, (0.15, -0.06, z + 0.011), 0.112, 0.004, "bright")
    b.cylinder(base, (0.15, -0.06, 1.058), 0.105, 0.022, "bright", collision=True)
    b.cylinder(base, (0.15, -0.06, 1.084), 0.064, 0.004, "lens", collision=True)
    for a in (0, 2.1, 4.2):
        b.box(
            base,
            (0.15 + 0.08 * math.cos(a), -0.06 + 0.08 * math.sin(a), 1.089),
            (0.020, 0.009, 0.005),
            "bright",
            euler=(0, 0, a),
        )
    b.site(base, "dry_tribology_coupon", (0.15, -0.06, 1.088), 0.025)
    # Original non-contact approach: stroke stops above the inert coupon.
    b.box(base, (-0.25, 0.12, 1.035), (0.05, 0.08, 0.15), "metal", collision=True)
    for x in (-0.276, -0.224):
        b.rod(base, (x, 0.07, 1.11), (x, 0.07, 1.39), 0.009, "bright")
    arm = b.moving(
        base, "contact_approach", (-0.25, 0.055, 1.235), (0, 0, 1), (0, 0.10), mass=0.18
    )
    b.rod(arm, (0, 0, 0), (0.40, -0.115, 0.025), 0.017, "bright", collision=True)
    b.cylinder(arm, (0.40, -0.115, -0.021), 0.012, 0.038, "metal", collision=True)
    b.cylinder(arm, (0.40, -0.115, -0.064), 0.004, 0.008, "bright", collision=True)
    b.box(arm, (0, 0.02, 0), (0.044, 0.026, 0.035), "dark", collision=True)
    # Source-specific adjacent controller and monitor, on supported tables.
    _stand(b, base, (-0.88, 0.06), (0.25, 0.30), 0.78, "dark")
    b.box(base, (-0.88, 0.07, 1.075), (0.235, 0.26, 0.265), "cream", collision=True)
    b.box(base, (-0.88, -0.198, 1.23), (0.18, 0.004, 0.055), "dark")
    for i in range(8):
        b.box(base, (-0.88, -0.198, 0.91 + i * 0.025), (0.18, 0.003, 0.003), "dark")
    _stand(b, base, (0.91, -0.04), (0.31, 0.32), 0.79)
    _screen(b, base, 0.91, 0.08, 1.16)
    b.box(base, (0.91, -0.21, 0.828), (0.225, 0.08, 0.012), "cream")
    for i in range(12):
        for j in range(4):
            b.box(
                base,
                (0.70 + i * 0.034, -0.255 + j * 0.027, 0.845),
                (0.012, 0.008, 0.003),
                "dark",
            )


def _tribology_room(world, definition):
    p = _Room(world, "leeds")
    for z in (1.50, 2.0):
        p.rod(
            "copper_wall_service",
            (-2.1, 2.15, z),
            (2.1, 2.15, z),
            0.012,
            (0.56, 0.30, 0.17, 1),
        )
    p.rod("service_drop", (1.3, 2.15, 1.50), (1.3, 2.15, 0.9), 0.014, STEEL)
    # Secured passive cylinder, not a functioning gas supply.
    ET.SubElement(
        world,
        "geom",
        name="leeds_passive_cylinder",
        type="cylinder",
        pos="1.45 1.72 .7",
        size=".13 .66",
        rgba=numbers(DARK),
        contype="1",
        conaffinity="1",
    )
    for z in (0.5, 1.0):
        p.rod("cylinder_restraint", (1.3, 1.72, z), (1.60, 1.72, z), 0.018, STEEL)
        p.rod("wall_bracket", (1.45, 1.75, z), (1.45, 2.15, z), 0.018, STEEL)
    p.rod("cylinder_neck", (1.45, 1.72, 1.3), (1.45, 1.72, 1.48), 0.033, STEEL)


BUILDERS = {"leeds_dry_tribology_station": _tribology}
SOURCES = {
    "leeds_dry_tribology_station": dict(
        reference="Leeds official environmental tribology suite photographs",
        url=LEEDS,
        dimensions_m=[2.5, 1.5, 1.9],
        dimension_basis="Photographic exterior estimates; original unpowered access and approach fixture, no manufacturer CAD or calibrated dimensions",
    )
}
SAMPLE_INTERFACES = {
    "leeds_dry_tribology_station": (
        "dry_tribology_coupon",
        (0.128, 0.128, 0.008),
        "clamped",
        "Visible inert solid disk retained on the concentric copper specimen stage; approach stops clear of its surface",
    )
}
FEATURES = {"leeds_environmental_tribology": _tribology_room}


def _ice_workcell(b, base, params):
    _stand(b, base, (0, 0), (0.37, 0.35), 0.48, "metal")
    for x in (-0.22, 0.22):
        b.cylinder(base, (x, 0, 0.540), 0.033, 0.035, "rubber", collision=True)
    b.box(base, (0, 0.10, 0.620), (0.25, 0.18, 0.045), "cream", collision=True)
    # The photographed unpowered saw is represented by a hollow upper guard.
    b.box(base, (-0.15, 0.12, 1.085), (0.10, 0.09, 0.43), "cream", collision=True)
    b.box(base, (0, 0.16, 1.48), (0.27, 0.026, 0.29), "cream", collision=True)
    for x in (-0.253, 0.253):
        b.box(base, (x, 0.08, 1.48), (0.018, 0.075, 0.29), "cream", collision=True)
    b.box(base, (0, 0.08, 1.78), (0.27, 0.075, 0.017), "cream", collision=True)
    b.box(base, (0, 0.08, 1.18), (0.27, 0.075, 0.017), "cream", collision=True)
    b.geom(
        base,
        "cylinder",
        (0.205, 0.027),
        (0, 0.087, 1.48),
        "bright",
        euler=(math.pi / 2, 0, 0),
    )
    b.geom(
        base,
        "cylinder",
        (0.066, 0.035),
        (0, 0.05, 1.48),
        "dark",
        euler=(math.pi / 2, 0, 0),
    )
    for a in range(8):
        angle = math.tau * a / 8
        b.rod(
            base,
            (0.06 * math.cos(angle), 0.023, 1.48 + 0.06 * math.sin(angle)),
            (0.19 * math.cos(angle), 0.023, 1.48 + 0.19 * math.sin(angle)),
            0.008,
            "metal",
        )
    cover = b.moving(
        base,
        "wheel_guard",
        (-0.285, -0.023, 1.48),
        (0, 0, -1),
        (0, 1.1),
        kind="hinge",
        mass=0.28,
    )
    cover.set("euler", numbers((0, 0, -0.75)))
    cover.find("joint").set("ref", ".75")
    b.box(cover, (0.278, 0, 0), (0.268, 0.015, 0.295), "shell", collision=True)
    for a, c in (
        ((0.04, -0.021, -0.26), (0.51, -0.021, 0.26)),
        ((0.04, -0.021, 0.26), (0.51, -0.021, -0.26)),
    ):
        b.rod(cover, a, c, 0.008, "metal")
    b.cylinder(
        cover, (0.48, -0.047, 0), 0.025, 0.018, "dark", euler=(math.pi / 2, 0, 0)
    )
    # Passive red blade guide; the dry carriage is separated from its envelope.
    b.box(base, (0.17, 0.02, 1.06), (0.045, 0.07, 0.105), "guard_red", collision=True)
    b.box(base, (0.205, -0.034, 0.99), (0.013, 0.016, 0.17), "bright", collision=True)
    b.box(base, (0, -0.085, 0.86), (0.35, 0.29, 0.016), "metal", collision=True)
    for x in (-0.22, -0.02):
        b.rod(base, (x, -0.31, 0.891), (x, -0.09, 0.891), 0.007, "bright")
    carrier = b.moving(
        base,
        "inert_core_carriage",
        (-0.115, -0.22, 0.92),
        (0, 1, 0),
        (-0.075, 0.075),
        mass=0.18,
    )
    b.box(carrier, (0, 0, 0), (0.102, 0.064, 0.015), "bright", collision=True)
    b.geom(
        carrier,
        "cylinder",
        (0.034, 0.066),
        (0, 0, 0.048),
        "glass",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.geom(
        carrier,
        "cylinder",
        (0.026, 0.052),
        (0, 0, 0.048),
        "shell",
        euler=(0, math.pi / 2, 0),
    )
    for x in (-0.069, 0.069):
        b.box(carrier, (x, 0, 0.038), (0.012, 0.049, 0.019), "blue", collision=True)
    b.site(carrier, "inert_ice_look_core", (0, 0, 0.048), 0.02)
    b.rod(base, (0.25, 0.10, 1.58), (0.30, 0.10, 1.58), 0.020, "metal")
    b.box(base, (0.34, 0.10, 1.58), (0.057, 0.06, 0.115), "cream", collision=True)
    b.box(base, (0.34, 0.033, 1.60), (0.033, 0.005, 0.039), "warning")
    b.geom(
        base,
        "cylinder",
        (0.018, 0.008),
        (0.34, 0.02, 1.60),
        "guard_red",
        euler=(math.pi / 2, 0, 0),
    )
    b.rod(base, (0.27, 0.20, 0.60), (0.27, 0.28, 0.20), 0.007, "dark")
    b.rod(base, (0.27, 0.28, 0.20), (0.24, -0.43, 0.045), 0.007, "dark")
    b.box(base, (0.24, -0.43, 0.025), (0.065, 0.09, 0.025), "dark", collision=True)


def _coldroom(world, definition):
    p = _Room(world, "copenhagen")
    # Only photographed bench/bandsaw bay; storage room is not merged into it.
    for x, width in ((-1.4, 0.82), (1.32, 0.82)):
        p.box("wood_worktop", (x, 1.30, 0.88), (width, 0.43, 0.028), WOOD, True)
        p.box("blue_front_rail", (x, 0.915, 0.80), (width, 0.04, 0.05), BLUE, True)
        for xx in (x - width + 0.06, x + width - 0.06):
            for y in (0.94, 1.64):
                p.box("blue_leg", (xx, y, 0.41), (0.036, 0.036, 0.41), BLUE, True)
                for z in (0.30, 0.43, 0.56, 0.69):
                    p.box("leg_hole", (xx, y - 0.038, z), (0.009, 0.002, 0.013), DARK)
        p.box(
            "underbench_bin",
            (x, 1.28, 0.20),
            (0.29, 0.25, 0.19),
            (0.41, 0.43, 0.44, 1),
            True,
        )
        p.box("bin_dark_opening", (x, 1.28, 0.394), (0.25, 0.21, 0.006), DARK)
        for xx in (x - 0.20, x - 0.07, x + 0.07, x + 0.20):
            p.box(
                "bin_rib",
                (xx, 1.023, 0.22),
                (0.010, 0.008, 0.16),
                (0.52, 0.54, 0.54, 1),
            )
    for x in (-2.35, -1.55, -0.75, 0.05, 0.85, 1.65, 2.35):
        p.box(
            "insulated_panel_seam",
            (x, 2.21, 1.40),
            (0.006, 0.006, 1.40),
            (0.63, 0.65, 0.64, 1),
        )
    # Freezer door is an interior photographed element, not the room entrance.
    p.box("freezer_door", (1.68, 2.17, 1.08), (0.56, 0.035, 1.02), WHITE)
    for x in (1.07, 2.29):
        p.box("freezer_frame", (x, 2.13, 1.08), (0.035, 0.035, 1.07), STEEL)
    p.box("freezer_head", (1.68, 2.13, 2.17), (0.64, 0.035, 0.035), STEEL)
    p.rod("freezer_handle", (1.25, 2.07, 0.89), (1.25, 2.07, 1.20), 0.015, DARK)
    # Small foreground table remains separate, preserving an aisle.
    p.box("near_table", (-1.51, -0.40, 0.86), (0.72, 0.34, 0.028), WOOD, True)
    for x in (-2.17, -0.85):
        for y in (-0.67, -0.13):
            p.box("near_table_leg", (x, y, 0.415), (0.03, 0.03, 0.415), BLUE, True)


BUILDERS["copenhagen_dry_ice_workcell"] = _ice_workcell
SOURCES["copenhagen_dry_ice_workcell"] = dict(
    reference="Copenhagen PICE official ice-core workroom photographs",
    url=COPENHAGEN,
    dimensions_m=[0.94, 0.87, 1.80],
    dimension_basis="Photographic estimates for a small unpowered saw; original dry retained-core carriage, no cutting or refrigeration mechanism",
)
SAMPLE_INTERFACES["copenhagen_dry_ice_workcell"] = (
    "inert_ice_look_core",
    (0.132, 0.068, 0.068),
    "clamped",
    "Visible rigid translucent ice-look cylinder retained on a dry carriage separated from the passive blade guide",
)
FEATURES["copenhagen_pice_ice_workroom"] = _coldroom


def _hollow_wall(b, parent, center, outer, inner, half, material, segments=48):
    vertices, faces = [], []
    for z in (-half, half):
        for radius in (outer, inner):
            for i in range(segments):
                a = 2 * math.pi * i / segments
                vertices.append(
                    (
                        center[0] + radius * math.cos(a),
                        center[1] + radius * math.sin(a),
                        center[2] + z,
                    )
                )
    for i in range(segments):
        j = (i + 1) % segments
        for a, c, d, e in (
            (i, j, 2 * segments + j, 2 * segments + i),
            (segments + j, segments + i, 3 * segments + i, 3 * segments + j),
            (j, i, segments + i, segments + j),
            (2 * segments + i, 2 * segments + j, 3 * segments + j, 3 * segments + i),
        ):
            faces.extend(((a, c, d), (a, d, e)))
    mesh = b.unique("hollow_wall")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([x for v in vertices for x in v]),
        face=" ".join(str(x) for f in faces for x in f),
    )
    b.geom(parent, "mesh", (), material=material, mesh=mesh)
    for i in range(segments):
        a = 2 * math.pi * i / segments
        r = (outer + inner) / 2
        b.box(
            parent,
            (center[0] + r * math.cos(a), center[1] + r * math.sin(a), center[2]),
            ((outer - inner) / 2, outer * math.tan(math.pi / segments), half),
            material,
            collision=True,
            euler=(0, 0, a),
            rgba=(0.5, 0.5, 0.5, 0),
        )


def _gersemi(b, base, params):
    # Floor elevation is authored: the cryostat top sits below surrounding walks.
    b.cylinder(base, (0, 0, 0.02), 0.55, 0.02, "dark", collision=True)
    b.cylinder(base, (0, 0, 0.4475), 0.72, 0.4075, "metal", collision=True)
    _hollow_wall(b, base, (0, 0, 0.89), 0.81, 0.31, 0.035, "bright")
    _flange(b, base, (0, 0, 0.929), 0.76)
    for i in range(12):
        a = math.tau * i / 12
        x, y = 0.63 * math.cos(a), 0.63 * math.sin(a)
        h = 0.11 + (i % 3) * 0.035
        b.cylinder(base, (x, y, 0.94 + h / 2), 0.032, h / 2, "bright", collision=True)
        _flange(b, base, (x, y, 0.94 + h), 0.042)
        if i % 2:
            b.rod(base, (x, y, 0.96 + h), (x * 1.27, y * 1.27, 0.77), 0.008, "metal")
        else:
            b.rod(base, (x, y, 0.95 + h), (x, y, 0.995 + h), 0.014, "bright")
            b.ring(base, (x, y, 1.01 + h), 0.054, 0.008, "blue")
    lid = b.moving(
        base,
        "inspection_port_cover",
        (-0.35, 0, 0.967),
        (0, -1, 0),
        (0, 1.15),
        kind="hinge",
        mass=0.45,
    )
    lid.set("euler", numbers((0, -0.70, 0)))
    lid.find("joint").set("ref", ".70")
    b.cylinder(lid, (0.35, 0, 0), 0.335, 0.019, "bright", collision=True)
    _flange(b, lid, (0.35, 0, 0.023), 0.305)
    b.rod(lid, (0.29, -0.09, 0.055), (0.29, 0.09, 0.055), 0.011, "dark")
    # Dry retained inspection insert beside the vessel, not a cryogenic loading sequence.
    for x in (0.88, 1.05):
        b.rod(base, (x, -0.69, 0.08), (x, -0.69, 1.30), 0.016, "bright", collision=True)
        b.box(base, (x, -0.69, 0.035), (0.055, 0.06, 0.035), "metal", collision=True)
    b.box(base, (0.965, -0.69, 0.13), (0.125, 0.09, 0.045), "dark", collision=True)
    carrier = b.moving(
        base, "dry_insert_lift", (0.965, -0.78, 0.92), (0, 0, 1), (0, 0.22), mass=0.25
    )
    b.box(carrier, (0, 0.09, 0), (0.055, 0.025, 0.038), "blue", collision=True)
    b.rod(carrier, (0, 0.055, 0), (0, -0.09, 0), 0.018, "bright", collision=True)
    b.cylinder(carrier, (0, -0.10, 0.025), 0.07, 0.017, "metal", collision=True)
    b.cylinder(carrier, (0, -0.10, 0.081), 0.033, 0.038, "cream", collision=True)
    b.ring(carrier, (0, -0.10, 0.065), 0.039, 0.008, "blue")
    b.site(carrier, "inert_cryostat_insert", (0, -0.10, 0.081), 0.020)
    # Continuous raised platforms and guardrails preserve the photographed pit.
    for x in (-1.39, 1.39):
        b.box(base, (x, 0.10, 0.53), (0.31, 1.37, 0.53), "metal", collision=True)
        b.box(base, (x, 0.10, 1.07), (0.32, 1.38, 0.015), "bright", collision=True)
    b.box(base, (0, 1.59, 0.53), (1.70, 0.18, 0.53), "metal", collision=True)
    b.box(base, (0, 1.59, 1.07), (1.72, 0.19, 0.015), "bright", collision=True)
    for x in (-1.075, 1.075):
        for y in (-0.45, 0.50, 1.4):
            b.rod(base, (x, y, 1.09), (x, y, 2.04), 0.023, "bright", collision=True)
        for z in (1.56, 2.04):
            b.rod(base, (x, -0.45, z), (x, 1.4, z), 0.024, "bright", collision=True)
    # The foreground rail separates the pit from the viewing floor.
    for x in (-0.89, 0, 0.89):
        b.rod(base, (x, -1.16, 0), (x, -1.16, 1.37), 0.026, "bright", collision=True)
    for z in (0.88, 1.37):
        b.rod(
            base, (-0.89, -1.16, z), (0.89, -1.16, z), 0.026, "bright", collision=True
        )
    # Taller cylindrical service vessel on the left platform.
    b.cylinder(base, (-1.40, 0.92, 1.93), 0.22, 0.53, "metal", collision=True)
    b.geom(base, "ellipsoid", (0.22, 0.22, 0.10), (-1.4, 0.92, 2.46), "bright")
    for dx in (-0.14, 0.14):
        b.rod(base, (-1.4 + dx, 0.92, 1.08), (-1.4 + dx, 0.92, 1.50), 0.023, "bright")
    for z in (1.50, 2.45):
        _flange(b, base, (-1.4, 0.92, z), 0.229)
    b.rod(base, (-1.4, 0.92, 1.50), (-1.4, 0.92, 0.80), 0.058, "bright")
    b.rod(base, (-1.4, 0.92, 0.80), (-0.68, 0.45, 1.02), 0.058, "bright")
    b.rod(base, (-1.4, 0.92, 2.52), (-1.4, 1.75, 2.76), 0.036, "bright")
    # Raised rear control cabinets follow the source; no live cryogen controls.
    for x in (-0.54, 0.29, 1.12):
        b.box(base, (x, 1.66, 1.735), (0.30, 0.25, 0.65), "cream", collision=True)
        b.box(base, (x, 1.403, 2.12), (0.24, 0.005, 0.19), "dark")
        b.rod(base, (x + 0.23, 1.37, 1.55), (x + 0.23, 1.37, 1.82), 0.009, "dark")
    # Steps into the raised left walkway, outside the apparatus and room door.
    for i in range(4):
        b.box(
            base,
            (-1.39, -1.77 + i * 0.125, 0.135 * (i + 1)),
            (0.30, 0.0625, 0.135 * (i + 1)),
            "dark",
            collision=True,
        )


def _gersemi_room(world, definition):
    p = _Room(world, "gersemi")
    for i in range(5):
        x = -2.22 + i * 0.24
        p.rod("overhead_service", (x, -1.7, 2.87), (x, 2.9, 2.87), 0.032, STEEL).set(
            "name", f"arch_roof_gersemi_pipe_{i}"
        )
    for y in (-1.5, 0, 1.5, 2.8):
        p.rod("service_support", (-2.36, y, 2.81), (-1.15, y, 2.81), 0.016, STEEL).set(
            "name", f"arch_roof_gersemi_support_{y}"
        )
        for x in (-2.30, -1.25):
            p.rod("bracket", (x, y, 2.81), (x, y, 3.19), 0.012, STEEL).set(
                "name", f"arch_roof_gersemi_bracket_{x}_{y}"
            )
    p.box("rear_panel", (0, 3.16, 1.50), (2.75, 0.022, 1.45), WHITE)
    for x in (-2, -1, 0, 1, 2):
        p.box("panel_joint", (x, 3.12, 1.5), (0.006, 0.005, 1.45), STEEL)


BUILDERS["uppsala_gersemi_dry_bay"] = _gersemi
SOURCES["uppsala_gersemi_dry_bay"] = dict(
    reference="Uppsala FREIA official Gersemi pit and top-plate photographs",
    url=UPPSALA,
    dimensions_m=[3.50, 3.90, 2.80],
    dimension_basis="Photo-estimated relative pit levels, support and cabinet geometry; original dry side insert fixture and simplified access cover",
)
SAMPLE_INTERFACES["uppsala_gersemi_dry_bay"] = (
    "inert_cryostat_insert",
    (0.066, 0.066, 0.076),
    "clamped",
    "Visible inert cylindrical insert clamped to a supported external dry inspection lift, outside the cryostat",
)
FEATURES["uppsala_freia_gersemi"] = _gersemi_room


def _nanomax(b, base, params):
    # Two dense granite supports carry a long rail bed and the flight tube.
    for y in (-1.07, 1.28):
        b.box(base, (0, y, 0.43), (0.52, 0.47, 0.43), "dark", collision=True)
        for x in (-0.42, 0.42):
            for yy in (y - 0.36, y + 0.36):
                b.cylinder(base, (x, yy, 0.022), 0.05, 0.022, "rubber", collision=True)
        for z in (0.17, 0.46, 0.74):
            b.geom(
                base,
                "cylinder",
                (0.018, 0.005),
                (0.525, y, z),
                "metal",
                euler=(0, math.pi / 2, 0),
            )
    for x in (-0.40, 0.40):
        b.box(base, (x, 0.16, 0.94), (0.052, 1.72, 0.075), "metal", collision=True)
        for dx in (-0.039, 0.039):
            b.rod(base, (x + dx, -1.54, 0.97), (x + dx, 1.86, 0.97), 0.007, "bright")
        for y in (-1.3, -0.9, -0.5, -0.1, 0.3, 0.7, 1.1, 1.5):
            b.box(base, (x, y, 1.019), (0.06, 0.012, 0.012), "bright")
    b.box(base, (0, -1.10, 1.06), (0.54, 0.49, 0.03), "bright", collision=True)
    # Hollow front imaging chamber. The front access mechanism is authored.
    for x in (-0.485, 0.485):
        b.box(base, (x, -1.1, 1.40), (0.03, 0.43, 0.30), "metal", collision=True)
    b.box(base, (0, -0.695, 1.40), (0.49, 0.025, 0.30), "metal", collision=True)
    b.box(base, (0, -1.1, 1.715), (0.535, 0.47, 0.025), "bright", collision=True)
    b.box(base, (0, -1.1, 1.095), (0.50, 0.44, 0.025), "bright", collision=True)
    for x in (-0.32, 0, 0.32):
        b.cylinder(base, (x, -1.12, 1.79), 0.045, 0.045, "bright")
        _flange(b, base, (x, -1.12, 1.837), 0.060)
    for side in (-1, 1):
        for y, z, r in ((-1.34, 1.50, 0.07), (-0.92, 1.43, 0.12), (-1.12, 1.25, 0.09)):
            b.geom(
                base,
                "cylinder",
                (r, 0.043),
                (side * 0.547, y, z),
                "bright",
                euler=(0, math.pi / 2, 0),
            )
            _flange(b, base, (side * 0.593, y, z), r * 1.10, "yz")
            b.geom(
                base,
                "cylinder",
                (r * 0.70, 0.008),
                (side * 0.601, y, z),
                "lens",
                euler=(0, math.pi / 2, 0),
            )
    door = b.moving(
        base,
        "sample_access_door",
        (-0.545, -1.584, 1.40),
        (0, 0, -1),
        (0, 1.18),
        kind="hinge",
        mass=0.65,
    )
    door.set("euler", numbers((0, 0, -0.84)))
    door.find("joint").set("ref", ".84")
    b.box(door, (0.53, 0, 0), (0.515, 0.021, 0.278), "bright", collision=True)
    for x, z, r in ((0.35, 0.05, 0.18), (0.77, 0.15, 0.065)):
        _flange(b, door, (x, -0.026, z), r, "xz")
        b.geom(
            door,
            "cylinder",
            (r * 0.78, 0.008),
            (x, -0.033, z),
            "lens",
            euler=(math.pi / 2, 0, 0),
        )
    b.rod(door, (0.92, -0.07, -0.16), (0.92, -0.07, 0.12), 0.012, "dark")
    for x in (-0.20, 0.20):
        b.rod(base, (x, -1.40, 1.158), (x, -0.82, 1.158), 0.012, "bright")
    tray = b.moving(
        base,
        "dry_specimen_carriage",
        (0, -1.12, 1.20),
        (0, 1, 0),
        (-0.10, 0.10),
        mass=0.23,
    )
    b.box(tray, (0, 0, 0), (0.23, 0.135, 0.024), "blue", collision=True)
    b.box(tray, (0, 0, 0.034), (0.11, 0.09, 0.010), "bright", collision=True)
    b.cylinder(tray, (0, 0, 0.055), 0.052, 0.010, "copper", collision=True)
    b.cylinder(tray, (0, 0, 0.068), 0.031, 0.004, "lens", collision=True)
    for x in (-0.075, 0.075):
        b.box(tray, (x, 0, 0.060), (0.018, 0.045, 0.012), "bright", collision=True)
    b.site(tray, "inert_nanomax_coupon", (0, 0, 0.069), 0.021)
    # Main cylindrical flight tube with segmented bolted joints.
    b.geom(
        base,
        "cylinder",
        (0.245,),
        material="bright",
        fromto=(0, -0.67, 1.49, 0, 1.79, 1.49),
        collision=True,
    )
    for y in (-0.64, -0.06, 0.55, 1.16, 1.79):
        b.geom(
            base,
            "cylinder",
            (0.266, 0.028),
            (0, y, 1.49),
            "metal",
            euler=(math.pi / 2, 0, 0),
        )
        _flange(b, base, (0, y - 0.031, 1.49), 0.258, "xz")
    for y in (-0.22, 0.44, 1.1, 1.73):
        b.box(base, (0, y, 1.13), (0.23, 0.065, 0.105), "metal", collision=True)
        for x in (-0.22, 0.22):
            b.rod(base, (x, y, 1.22), (x * 0.75, y, 1.34), 0.027, "metal")
    b.geom(
        base,
        "cylinder",
        (0.29, 0.07),
        (0, 1.88, 1.49),
        "bright",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    _flange(b, base, (0, 1.96, 1.49), 0.282, "xz")
    for x in (-0.13, 0.13):
        b.cylinder(base, (x, 1.85, 1.82), 0.025, 0.08, "bright")
        _flange(b, base, (x, 1.85, 1.905), 0.036)
    # Long source-specific side cantilever and red-ended passive actuator.
    b.box(base, (0.835, -0.30, 0.90), (0.385, 1.05, 0.045), "metal", collision=True)
    for y in (-1.1, 0.50):
        b.box(base, (0.62, y, 0.76), (0.17, 0.04, 0.12), "metal", collision=True)
    b.box(base, (0.60, -0.98, 1.015), (0.055, 0.040, 0.070), "metal", collision=True)
    b.rod(base, (0.60, -0.98, 1.10), (1.17, -0.55, 1.10), 0.025, "bright")
    b.geom(
        base,
        "cylinder",
        (0.055,),
        material="guard_red",
        fromto=(1.10, -0.603, 1.10, 1.21, -0.52, 1.10),
    )
    b.box(base, (1.20, -0.51, 1.00), (0.057, 0.05, 0.055), "bright", collision=True)
    for i in range(4):
        x = -0.56 - i * 0.024
        b.rod(
            base,
            (x, -1.3, 1.37),
            (x, -1.58, 0.33),
            0.005,
            "warning" if i % 2 else "blue",
        )
        b.rod(
            base,
            (x, -1.58, 0.33),
            (x, 1.42, 0.33),
            0.005,
            "warning" if i % 2 else "blue",
        )
    b.box(base, (-0.66, 0.20, 0.25), (0.035, 1.60, 0.035), "metal")


def _nanomax_hutch(world, definition):
    p = _Room(world, "nanomax")
    p.box(
        "red_rear_hutch", (0, 3.44, 1.60), (2.61, 0.035, 1.60), (0.55, 0.07, 0.055, 1)
    )
    for x in (-1.68, 1.30):
        p.box("back_cable_tray", (x, 3.34, 1.68), (0.10, 0.035, 1.28), STEEL)
        for z in (0.5, 0.8, 1.1, 1.4, 1.7, 2, 2.3, 2.6):
            p.box("tray_rung", (x, 3.29, z), (0.095, 0.018, 0.007), DARK)
    p.rod("rear_vent", (1.25, 2.63, 0), (1.25, 2.63, 3.20), 0.12, STEEL)
    for z in [i * 0.08 for i in range(8, 38)]:
        # Ring-like ribs are modest geometry rather than a copied texture.
        for dx in (-0.125, 0.125):
            p.rod("duct_rib", (1.25 + dx, 2.55, z), (1.25 + dx, 2.71, z), 0.006, STEEL)
    p.box("left_service_chassis", (-1.12, 2.29, 0.62), (0.26, 0.29, 0.62), WHITE, True)
    p.box(
        "orange_control_panel",
        (-1.10, 2.30, 1.42),
        (0.26, 0.03, 0.19),
        (0.90, 0.34, 0.05, 1),
    )
    for x in (-1.32, -1.04):
        p.rod("control_support", (x, 2.3, 1.24), (x, 2.3, 0.62), 0.012, STEEL)
    for x in (-2.24, -1.90):
        p.rod("wall_line", (x, 3.25, 2.87), (x, -1.0, 2.87), 0.017, BLUE).set(
            "name", f"arch_roof_nanomax_line_{x}"
        )
        for y in (-0.8, 1.1, 3.1):
            p.rod(
                "wall_line_bracket", (x, y, 2.87), (-2.72, y, 2.87), 0.012, STEEL
            ).set("name", f"arch_roof_nanomax_bracket_{x}_{y}")


BUILDERS["lund_nanomax_dry_endstation"] = _nanomax
SOURCES["lund_nanomax_dry_endstation"] = dict(
    reference="MAX IV NanoMAX EH1 official installed imaging station photograph",
    url=LUND,
    dimensions_m=[2.0, 3.60, 1.95],
    dimension_basis="Source-proportioned chamber, rail, flight tube and granite supports; estimated dimensions and original dry access/carriage mechanisms",
)
SAMPLE_INTERFACES["lund_nanomax_dry_endstation"] = (
    "inert_nanomax_coupon",
    (0.062, 0.062, 0.008),
    "clamped",
    "Visible inert circular coupon mechanically retained on a dry carrier inside the representative access chamber",
)
FEATURES["lund_maxiv_nanomax_eh1"] = _nanomax_hutch
