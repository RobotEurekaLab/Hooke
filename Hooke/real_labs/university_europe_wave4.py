"""Optics, structural loading, space hardware and beamline reference workcells.

Original geometry with limited mechanical controls; source photographs establish
apparatus and layout cues, not measured room plans or scientific transfer models.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

LEUVEN = "https://fys.kuleuven.be/iks/ns/experimental-facilities/iglis/IGLIS-1"
SACLAY = "https://lmps.ens-paris-saclay.fr/fr/moyens-sur-le-site-de-lens-paris-saclay"
GLASGOW = "https://www.gla.ac.uk/research/az/space-exploration-technology/facilities/"
UPPSALA = "https://www.uu.se/en/centre/tandemlab/infrastructure/accelerators"

WHITE = (0.84, 0.85, 0.81, 1)
STEEL = (0.46, 0.49, 0.50, 1)
DARK = (0.065, 0.085, 0.095, 1)
BLUE = (0.06, 0.32, 0.59, 1)


class _Room:
    def __init__(self, world, name):
        self.world, self.name, self.index = world, name, 0

    def box(self, key, pos, half, color, collision=False, **kwargs):
        self.index += 1
        return box(
            self.world,
            f"europe4_{self.name}_{key}_{self.index}",
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
            name=f"europe4_{self.name}_{key}_{self.index}",
            type="capsule",
            size=str(radius),
            fromto=numbers((*start, *end)),
            rgba=numbers(color),
            contype="0",
            conaffinity="0",
        )


def _optical_table(b, base, half=(1.22, 0.65), height=0.87):
    for x in (-half[0] + 0.16, half[0] - 0.16):
        for y in (-half[1] + 0.15, half[1] - 0.15):
            b.cylinder(base, (x, y, 0.075), 0.085, 0.075, "rubber", collision=True)
            b.cylinder(
                base,
                (x, y, (height - 0.06) / 2),
                0.065,
                (height - 0.06) / 2,
                "metal",
                collision=True,
            )
    b.box(base, (0, 0, height), (*half, 0.065), "dark", collision=True)
    b.box(
        base, (0, 0, height + 0.070), (half[0] - 0.01, half[1] - 0.01, 0.005), "metal"
    )
    for i in range(15):
        for j in range(8):
            b.cylinder(
                base,
                (
                    -half[0] + 0.08 + i * (2 * half[0] - 0.16) / 14,
                    -half[1] + 0.07 + j * (2 * half[1] - 0.14) / 7,
                    height + 0.077,
                ),
                0.003,
                0.0008,
                "dark",
            )


def _laser_enclosure(b, base, pos, length, width, height):
    x, y, z = pos
    b.box(
        base,
        (x, y, z + height / 2),
        (length / 2, width / 2, height / 2),
        "blue",
        collision=True,
    )
    b.box(
        base,
        (x - length / 2 - 0.009, y, z + height / 2),
        (0.008, width / 2, height / 2),
        "dark",
    )
    b.box(
        base,
        (x + length / 2 + 0.009, y, z + height / 2),
        (0.008, width / 2, height / 2),
        "dark",
    )
    for side in (-1, 1):
        for yy in (-width * 0.37, width * 0.37):
            b.cylinder(
                base,
                (x + side * (length / 2 + 0.019), y + yy, z + height * 0.38),
                0.012,
                0.007,
                "bright",
                euler=(0, math.pi / 2, 0),
            )
    for i in range(8):
        b.box(
            base,
            (
                x - length * 0.38 + i * length * 0.105,
                y + width / 2 + 0.001,
                z + height * 0.32,
            ),
            (0.018, 0.001, 0.006),
            "dark",
        )
    for xx in (-length * 0.38, length * 0.38):
        for yy in (-width * 0.35, width * 0.35):
            b.cylinder(
                base, (x + xx, y + yy, z + height + 0.002), 0.007, 0.002, "bright"
            )


def _iglis_alignment_table(b, base, params):
    """Blue laser housings flank an explicitly authored inert target XY mount."""
    _optical_table(b, base)
    _laser_enclosure(b, base, (0, 0.25, 0.95), 1.91, 0.35, 0.21)
    # Silver pump enclosure at the front follows the photographed blue/silver pair.
    b.housing(
        base,
        ((0.952, 0.50, 0.15, -0.25), (1.11, 0.48, 0.14, -0.25)),
        radius=0.022,
        material="bright",
    )
    b.box(base, (0, -0.25, 1.025), (0.48, 0.14, 0.070), "bright", collision=True)
    for x in (-0.38, 0.38):
        b.box(base, (x, -0.398, 1.037), (0.055, 0.004, 0.035), "dark")
        for y in (-0.30, -0.20):
            b.cylinder(base, (x, y, 1.114), 0.008, 0.002, "dark")
    # Alignment stage beside the housings; target is visible and mechanically retained.
    b.box(base, (0.84, -0.26, 0.974), (0.25, 0.22, 0.025), "dark", collision=True)
    for y in (-0.41, -0.12):
        b.rod(base, (0.60, y, 1.012), (1.08, y, 1.012), 0.007, "bright")
    xstage = b.moving(
        base, "target_x", (0.84, -0.26, 1.042), (1, 0, 0), (-0.045, 0.045), mass=0.22
    )
    b.box(xstage, (0, 0, 0), (0.16, 0.18, 0.022), "blue", collision=True)
    for x in (-0.115, 0.115):
        b.rod(xstage, (x, -0.14, 0.035), (x, 0.14, 0.035), 0.006, "bright")
    ystage = b.moving(
        xstage, "target_y", (0, 0, 0.065), (0, 1, 0), (-0.035, 0.035), mass=0.13
    )
    b.box(ystage, (0, 0, 0), (0.13, 0.095, 0.019), "dark", collision=True)
    b.rod(ystage, (0, 0, 0.02), (0, 0, 0.16), 0.012, "bright", collision=True)
    b.box(ystage, (0, 0, 0.213), (0.052, 0.012, 0.05), "cream", collision=True)
    b.box(ystage, (0, -0.014, 0.213), (0.038, 0.002, 0.038), "bright")
    b.box(ystage, (0, -0.017, 0.213), (0.025, 0.001, 0.0015), "dark")
    b.box(ystage, (0, -0.017, 0.213), (0.0015, 0.001, 0.025), "dark")
    b.site(ystage, "inert_alignment_target", (0, 0, 0.213), size=0.004)
    for x in (-1.02, -0.66):
        b.cylinder(base, (x, -0.16, 0.975), 0.042, 0.024, "dark")
        b.rod(base, (x, -0.16, 1.0), (x, -0.16, 1.20), 0.01, "bright")
        b.ring(base, (x, -0.16, 1.24), 0.045, 0.011, "dark", plane="xz")
        b.cylinder(
            base, (x, -0.16, 1.24), 0.034, 0.007, "lens", euler=(math.pi / 2, 0, 0)
        )
    for x in (-0.81, 0.57):
        b.rod(base, (x, 0.12, 0.76), (x, 0.19, 0.31), 0.008, "rubber")
        b.box(base, (x, 0.23, 0.42), (0.14, 0.25, 0.27), "blue")
        for dx in (-0.10, 0.10):
            for y in (0.06, 0.40):
                b.box(
                    base,
                    (x + dx, y, 0.075),
                    (0.022, 0.022, 0.075),
                    "dark",
                    collision=True,
                )
    b.metadata["capabilities"] = [
        "Retained alignment-target X translation",
        "Retained alignment-target Y translation",
    ]
    b.metadata["limitations"].append(
        "Original geometry from the IGLIS-1 blue laser/pump table photograph. "
        "The XY target is an authored mechanical fixture, not a verified original "
        "laser component. No emitted laser, optical propagation, gain medium, "
        "spectroscopy, ionization or separate Jetlab apparatus is simulated."
    )


def _iglis_room(world, definition):
    p = _Room(world, "iglis")
    # A perpendicular blue amplifier table and two monitors distinguish the compact bay.
    p.box("amplifier_table", (-1.73, -0.15, 0.87), (0.49, 1.07, 0.065), DARK, True)
    for x in (-2.10, -1.36):
        for y in (-0.99, 0.69):
            p.box(
                "amplifier_table_leg", (x, y, 0.40), (0.044, 0.044, 0.40), STEEL, True
            )
    p.box("blue_amplifier", (-1.73, -0.15, 1.076), (0.36, 0.91, 0.139), BLUE, True)
    p.box("amplifier_front", (-1.73, -1.071, 1.076), (0.36, 0.013, 0.139), DARK)
    for x in (-1.96, -1.76, -1.53):
        p.box("amplifier_port", (x, -1.087, 1.07), (0.025, 0.003, 0.033), STEEL)
    p.box("monitor_shelf", (-0.65, 1.93, 1.01), (0.81, 0.32, 0.032), STEEL, True)
    for x in (-1.28, -0.02):
        p.box("rack_leg", (x, 1.93, 0.50), (0.021, 0.24, 0.50), STEEL, True)
    for x in (-1.02, -0.36):
        p.rod("screen_post", (x, 1.93, 1.045), (x, 1.93, 1.36), 0.019, DARK)
        p.box("screen_back", (x, 1.93, 1.49), (0.28, 0.022, 0.18), DARK)
        p.box(
            "screen_face",
            (x, 1.903, 1.49),
            (0.257, 0.003, 0.155),
            (0.035, 0.095, 0.11, 1),
        )
    p.box("rear_electronics_rack", (0.05, 2.04, 0.71), (0.32, 0.28, 0.70), STEEL, True)
    for z in (0.30, 0.55, 0.80, 1.05):
        p.box("rack_drawer", (0.05, 1.753, z), (0.28, 0.01, 0.092), WHITE)
    # Selected visible service trunk; it connects to the rear wall.
    p.box("wall_service_trunk", (-1.17, 2.34, 2.1), (1.5, 0.05, 0.07), WHITE)
    p.box("blue_drop", (-0.80, 2.30, 1.93), (0.045, 0.055, 0.16), BLUE)
    p.box("rear_laser_table", (1.53, 1.85, 0.88), (0.79, 0.35, 0.055), DARK, True)
    for x in (0.87, 2.19):
        for y in (1.61, 2.09):
            p.box("rear_laser_leg", (x, y, 0.414), (0.027, 0.027, 0.414), STEEL, True)
    p.box("rear_blue_laser", (1.53, 1.87, 1.09), (0.70, 0.20, 0.15), BLUE, True)
    p.box("rear_laser_front", (0.82, 1.87, 1.09), (0.009, 0.20, 0.15), DARK)


def _sleeve_colliders(b, parent, position, orientation, outer, bore, half):
    """Physical hollow shell so a piston can enter its supported housing."""
    body = ET.SubElement(
        parent,
        "body",
        name=b.unique("sleeve"),
        pos=numbers(position),
        euler=numbers(orientation),
    )
    for i in range(24):
        a = 2 * math.pi * i / 24
        radius = (outer + bore) / 2
        b.box(
            body,
            (radius * math.cos(a), radius * math.sin(a), 0),
            ((outer - bore) / 2, outer * math.tan(math.pi / 24), half),
            "metal",
            collision=True,
            euler=(0, 0, a),
            rgba=(0.5, 0.5, 0.5, 0),
        )


def _astree_triaxial_frame(b, base, params):
    """Six visible loading directions, with two limited approach mechanisms."""
    b.box(base, (0, 0, 0.12), (1.40, 1.40, 0.12), "dark", collision=True)
    for x in (-0.87, 0.87):
        for y in (-0.87, 0.87):
            b.cylinder(base, (x, y, 0.30), 0.15, 0.09, "metal", collision=True)
            b.cylinder(base, (x, y, 1.69), 0.075, 1.34, "bright", collision=True)
            b.cylinder(base, (x, y, 3.04), 0.12, 0.065, "metal", collision=True)
    for y in (-0.87, 0.87):
        b.box(base, (0, y, 2.94), (1.02, 0.14, 0.12), "blue", collision=True)
        b.box(base, (0, y, 0.39), (1.02, 0.14, 0.11), "blue", collision=True)
    for x in (-0.87, 0.87):
        b.box(base, (x, 0, 2.94), (0.14, 0.87, 0.12), "blue", collision=True)
        b.box(base, (x, 0, 0.39), (0.14, 0.87, 0.11), "blue", collision=True)
    # Horizontal actuator housings and their supported crossheads.
    for axis in (0, 1):
        for sign in (-1, 1):
            at = [0, 0, 1.35]
            at[axis] = sign * 1.05
            orientation = (0, math.pi / 2, 0) if axis == 0 else (math.pi / 2, 0, 0)
            b.cylinder(
                base,
                at,
                0.26,
                0.25,
                "metal",
                euler=orientation,
                collision=not (axis == 0 and sign == -1),
            )
            if axis == 0 and sign == -1:
                _sleeve_colliders(b, base, at, orientation, 0.26, 0.085, 0.25)
            foot = at.copy()
            foot[2] = 0.80
            b.box(base, foot, (0.18, 0.18, 0.38), "dark", collision=True)
            flange = at.copy()
            flange[axis] = sign * 0.79
            b.cylinder(base, flange, 0.28, 0.03, "bright", euler=orientation)
            for i in range(8):
                a = 2 * math.pi * i / 8
                bolt = flange.copy()
                bolt[1 - axis] += 0.231 * math.cos(a)
                bolt[2] += 0.231 * math.sin(a)
                b.cylinder(base, bolt, 0.012, 0.035, "dark", euler=orientation)
            if axis == 0 and sign == -1:
                continue
            start = [0, 0, 1.35]
            end = start.copy()
            start[axis] = sign * 0.80
            end[axis] = sign * 0.29
            b.rod(base, start, end, 0.061, "bright", collision=True)
            jaw = end.copy()
            jaw[axis] = sign * 0.26
            half = [0.10, 0.10, 0.10]
            half[axis] = 0.05
            b.box(base, jaw, half, "metal", collision=True)
    xram = b.moving(
        base,
        "horizontal_approach",
        (-0.80, 0, 1.35),
        (1, 0, 0),
        (0, 0.20),
        mass=1.2,
        kp=280,
    )
    b.rod(xram, (0, 0, 0), (0.30, 0, 0), 0.059, "bright", collision=True)
    b.box(xram, (0.345, 0, 0), (0.045, 0.095, 0.095), "metal", collision=True)
    # Upper and lower loading assemblies retain an open central working region.
    b.cylinder(base, (0, 0, 2.70), 0.28, 0.22, "metal")
    _sleeve_colliders(b, base, (0, 0, 2.70), (0, 0, 0), 0.28, 0.092, 0.22)
    b.cylinder(base, (0, 0, 2.48), 0.31, 0.04, "bright")
    _sleeve_colliders(b, base, (0, 0, 2.48), (0, 0, 0), 0.31, 0.092, 0.04)
    zram = b.moving(
        base, "vertical_approach", (0, 0, 2.46), (0, 0, -1), (0, 0.20), mass=1.3, kp=280
    )
    b.cylinder(zram, (0, 0, -0.29), 0.072, 0.30, "bright", collision=True)
    b.cylinder(zram, (0, 0, -0.64), 0.145, 0.05, "metal", collision=True)
    b.cylinder(zram, (0, 0, -0.73), 0.072, 0.04, "bright", collision=True)
    b.cylinder(base, (0, 0, 0.55), 0.24, 0.28, "dark", collision=True)
    b.cylinder(base, (0, 0, 0.87), 0.46, 0.068, "metal", collision=True)
    for i in range(16):
        a = 2 * math.pi * i / 16
        b.cylinder(
            base,
            (0.366 * math.cos(a), 0.366 * math.sin(a), 0.941),
            0.024,
            0.002,
            "dark",
        )
    b.cylinder(base, (0, 0, 1.036), 0.09, 0.10, "bright", collision=True)
    b.box(base, (0, 0, 1.157), (0.15, 0.15, 0.02), "metal", collision=True)
    for x in (-0.10, 0.10):
        b.box(base, (x, 0, 1.259), (0.017, 0.12, 0.079), "dark", collision=True)
    b.box(base, (0, 0, 1.29), (0.078, 0.078, 0.09), "cream", collision=True)
    for y in (-0.10, 0.10):
        b.box(base, (0, y, 1.24), (0.075, 0.016, 0.065), "copper")
    b.site(base, "inert_structural_coupon", (0, 0, 1.29), size=0.009)
    for x in (-1.20, 1.20):
        b.rod(base, (x, 0.18, 1.35), (x, 0.42, 0.81), 0.022, "rubber")
        b.rod(base, (x, 0.42, 0.81), (x, 0.90, 0.48), 0.022, "rubber")
    b.metadata["capabilities"] = [
        "Restricted horizontal actuator approach",
        "Restricted vertical actuator approach",
    ]
    b.metadata["limitations"].append(
        "Official ASTREE dimensions describe a 0.650 x 0.650 x 1.500 m test space "
        "and 0.250 m cylinder strokes. Only two 0.20 m unloaded approach controls "
        "are represented; frame proportions and central fixture are estimates. "
        "No hydraulic circuit, force calibration, stress/strain, stereo-correlation, "
        "material deformation or full six-axis loading process."
    )


def _astree_bay(world, definition):
    p = _Room(world, "astree")
    p.box("controller_cabinet", (1.93, 1.82, 0.74), (0.28, 0.38, 0.74), WHITE, True)
    p.box("controller_screen", (1.93, 1.429, 1.04), (0.20, 0.008, 0.15), DARK)
    for z in (0.22, 0.32, 0.42, 0.52):
        p.box("cabinet_vent", (1.93, 1.427, z), (0.20, 0.003, 0.012), DARK)
    # Cabinet position and conduit are explicit local service-space estimates.
    p.rod("floor_conduit", (1.28, 1.12, 0.07), (1.91, 1.46, 0.07), 0.035, DARK)


def _iset_prototype_frame(b, base, params):
    """Dry positioning in a source-inspired red/aluminium prototyping frame."""
    for x in (-0.54, 0.54):
        for y in (-0.43, 0.43):
            b.cylinder(base, (x, y, 0.026), 0.047, 0.026, "metal", collision=True)
            b.box(base, (x, y, 0.87), (0.022, 0.022, 0.82), "bright", collision=True)
            b.box(base, (x + 0.024, y, 0.87), (0.004, 0.011, 0.79), "dark")
    for z in (0.22, 0.64, 1.69):
        for y in (-0.43, 0.43):
            b.box(base, (0, y, z), (0.54, 0.021, 0.021), "bright", collision=True)
        for x in (-0.54, 0.54):
            b.box(base, (x, 0, z), (0.021, 0.43, 0.021), "bright", collision=True)
    b.box(base, (0, 0.0, 0.39), (0.48, 0.38, 0.22), "cream", collision=True)
    b.box(base, (0, -0.39, 0.39), (0.435, 0.009, 0.18), "shell")
    b.box(base, (-0.22, -0.407, 0.45), (0.06, 0.003, 0.027), "dark")
    b.cylinder(
        base, (0.19, -0.412, 0.47), 0.026, 0.008, "bright", euler=(math.pi / 2, 0, 0)
    )
    for y in (-0.43, 0.43):
        b.box(base, (0, y, 1.744), (0.59, 0.028, 0.018), "guard_red", collision=True)
    for x in (-0.56, 0.56):
        b.box(base, (x, 0, 1.735), (0.022, 0.43, 0.022), "guard_red", collision=True)
        b.box(base, (x, 0, 1.13), (0.004, 0.39, 0.42), "glass", collision=True)
    b.box(base, (0, 0.432, 1.13), (0.50, 0.004, 0.42), "glass", collision=True)
    # The source does not expose original internal motion; this fixture is authored.
    for x in (-0.35, 0.35):
        b.rod(base, (x, -0.33, 0.754), (x, 0.34, 0.754), 0.012, "bright")
    bed = b.moving(base, "specimen_y", (0, 0, 0.80), (0, 1, 0), (-0.11, 0.11), mass=0.7)
    b.box(bed, (0, 0, 0), (0.39, 0.22, 0.025), "metal", collision=True)
    b.box(bed, (0, 0, 0.038), (0.24, 0.16, 0.010), "dark", collision=True)
    # Original retained rigid payload-shaped test coupon, no spacecraft dynamics.
    for x in (-0.105, 0.105):
        for y in (-0.10, 0.10):
            b.box(bed, (x, y, 0.162), (0.012, 0.012, 0.115), "bright", collision=True)
    for z in (0.061, 0.273):
        for y in (-0.10, 0.10):
            b.box(bed, (0, y, z), (0.105, 0.012, 0.012), "bright", collision=True)
        for x in (-0.105, 0.105):
            b.box(bed, (x, 0, z), (0.012, 0.10, 0.012), "bright", collision=True)
    b.box(bed, (0, 0.09, 0.16), (0.082, 0.007, 0.084), "blue")
    b.site(bed, "inert_payload_coupon", (0, 0, 0.162), size=0.007)
    for y in (-0.075, 0.075):
        b.rod(base, (-0.50, y, 1.49), (0.50, y, 1.49), 0.011, "bright")
    for x in (-0.50, 0.50):
        b.box(base, (x, 0, 1.49), (0.024, 0.14, 0.075), "metal", collision=True)
    head = b.moving(
        base, "inspection_head_x", (0, 0, 1.465), (1, 0, 0), (-0.30, 0.30), mass=0.30
    )
    b.box(head, (0, 0, 0), (0.067, 0.095, 0.058), "dark", collision=True)
    b.cylinder(head, (0, 0, -0.098), 0.028, 0.04, "metal", collision=True)
    b.cylinder(head, (0, 0, -0.143), 0.017, 0.005, "lens")
    b.screen(base, (-0.28, -0.447, 0.57), width=0.15, height=0.09)
    b.metadata["capabilities"] = [
        "Retained inert prototype translation",
        "Dry inspection-head traverse",
    ]
    b.metadata["limitations"].append(
        "Glasgow's official I-SET page establishes a real space-hardware lab and "
        "prototyping capability. The red/aluminium frame is source-informed; its "
        "exact equipment identity is unconfirmed and the internal two-axis fixture "
        "is an authored surrogate. No 3D printing, vacuum, vibration, space "
        "environment or spacecraft qualification result is computed."
    )


def _iset_room(world, definition):
    p = _Room(world, "iset")
    # Central white instrument island and tall repeating cream/black fixture.
    p.box("central_island", (-0.70, 0.22, 0.85), (1.29, 0.68, 0.035), WHITE, True)
    for x in (-1.84, 0.44):
        for y in (-0.31, 0.75):
            p.box("island_leg", (x, y, 0.412), (0.026, 0.026, 0.412), WHITE, True)
    p.box("upright_fixture_base", (-0.96, 0.28, 0.93), (0.29, 0.27, 0.045), DARK)
    for x in (-1.19, -0.73):
        p.rod("upright_fixture_rail", (x, 0.30, 0.98), (x, 0.30, 1.92), 0.021, STEEL)
    for z in (1.12, 1.43, 1.74):
        p.box(
            "cream_fixture_module",
            (-0.96, 0.30, z),
            (0.27, 0.18, 0.115),
            (0.84, 0.82, 0.69, 1),
        )
        p.box("fixture_access", (-0.96, 0.11, z), (0.11, 0.009, 0.071), DARK)
    # Foreground mobile grey instrument cabinet, identity intentionally unspecified.
    p.box(
        "foreground_cabinet",
        (0.99, -1.15, 0.43),
        (0.42, 0.36, 0.35),
        (0.59, 0.62, 0.59, 1),
        True,
    )
    p.box("cabinet_door", (0.99, -1.518, 0.43), (0.38, 0.008, 0.31), WHITE)
    for x in (0.68, 1.30):
        for y in (-1.40, -0.90):
            p.box("cabinet_castor", (x, y, 0.053), (0.041, 0.037, 0.053), DARK, True)
    p.box(
        "foreground_upright",
        (0.99, -1.15, 1.15),
        (0.17, 0.15, 0.37),
        (0.62, 0.65, 0.62, 1),
        True,
    )
    p.box("upright_panel", (0.99, -1.307, 1.16), (0.12, 0.004, 0.16), DARK)
    p.rod("instrument_cable", (1.12, -1.32, 1.35), (1.24, -1.34, 1.03), 0.012, DARK)
    # Window-side bench, densely stacked small instruments and service shelves.
    p.box("window_bench", (-2.89, 0.84, 0.87), (0.49, 1.71, 0.033), WHITE, True)
    for x in (-3.23, -2.57):
        for y in (-0.70, 2.38):
            p.box("window_bench_leg", (x, y, 0.419), (0.027, 0.027, 0.419), STEEL, True)
    for y in (-0.45, 0.19, 0.83, 1.47, 2.11):
        p.box("bench_instrument", (-2.95, y, 1.08), (0.25, 0.24, 0.18), WHITE)
        p.box("instrument_face", (-2.693, y, 1.09), (0.004, 0.18, 0.12), DARK)
        p.box(
            "instrument_readout",
            (-2.686, y + 0.035, 1.12),
            (0.002, 0.095, 0.056),
            (0.03, 0.16, 0.19, 1),
        )
    for x in (-3.25, -2.58):
        for y in (-0.79, 2.46):
            p.box("shelf_support", (x, y, 1.33), (0.02, 0.02, 0.43), STEEL)
    p.box("rear_utility_shelf", (-2.91, 0.84, 1.75), (0.40, 1.70, 0.025), STEEL)
    for y in (-0.44, 0.15, 0.74, 1.33, 1.92):
        p.box(
            "stored_instrument",
            (-2.92, y, 1.91),
            (0.25, 0.22, 0.14),
            (0.72, 0.74, 0.69, 1),
        )
    # Source orange circulation marking is retained; dimensions remain estimates.
    orange = (0.88, 0.35, 0.08, 1)
    p.box("orange_long_aisle", (2.72, 0.10, 0.003), (0.023, 2.34, 0.002), orange)
    p.box("orange_front_aisle", (1.39, -2.24, 0.003), (1.33, 0.023, 0.002), orange)
    p.box("back_workbench", (-0.02, 2.55, 0.86), (1.19, 0.35, 0.033), WHITE, True)
    for x in (-1.02, 0.98):
        for y in (2.31, 2.79):
            p.box("back_bench_leg", (x, y, 0.413), (0.025, 0.025, 0.413), WHITE, True)
    for x in (-0.73, -0.02, 0.69):
        p.box("back_instrument", (x, 2.55, 1.13), (0.24, 0.26, 0.23), DARK)


def _pelletron_beamline(b, base, params):
    """Cream accelerator envelope and supported beamline; mechanical loading only."""
    # Large horizontal cream tank is the dominant source feature.
    for x in (-2.92, -1.47):
        b.box(base, (x, 0.55, 0.19), (0.27, 0.88, 0.19), "metal", collision=True)
        b.box(base, (x, 0.55, 0.47), (0.20, 0.67, 0.12), "blue", collision=True)
    b.cylinder(
        base,
        (-2.18, 0.55, 1.49),
        1.035,
        1.10,
        "cream",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for x in (-3.28, -1.08):
        b.geom(
            base,
            "ellipsoid",
            (0.12, 1.035, 1.035),
            (x, 0.55, 1.49),
            "cream",
            collision=True,
        )
    # Bolted circular flange and access fittings around the front beam port.
    b.cylinder(
        base, (-0.948, 0.55, 1.49), 0.48, 0.024, "cream", euler=(0, math.pi / 2, 0)
    )
    for i in range(20):
        a = 2 * math.pi * i / 20
        b.cylinder(
            base,
            (-0.916, 0.55 + 0.425 * math.cos(a), 1.49 + 0.425 * math.sin(a)),
            0.016,
            0.011,
            "dark",
            euler=(0, math.pi / 2, 0),
        )
    b.cylinder(
        base,
        (-0.897, 0.55, 1.49),
        0.18,
        0.041,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for y, z in ((0.83, 1.68), (0.35, 1.79), (0.72, 1.26)):
        b.cylinder(
            base, (-0.94, y, z), 0.032, 0.065, "bright", euler=(0, math.pi / 2, 0)
        )
    # Blue floor portals support the narrow continuous flanged beam tube.
    for x in (-0.60, 0.75, 2.34):
        for y in (0.15, 0.95):
            b.box(base, (x, y, 0.59), (0.055, 0.055, 0.57), "blue", collision=True)
            b.box(base, (x, y, 0.029), (0.10, 0.10, 0.029), "bright", collision=True)
        b.box(base, (x, 0.55, 1.145), (0.064, 0.45, 0.055), "blue", collision=True)
        b.box(base, (x, 0.55, 1.31), (0.055, 0.085, 0.115), "metal", collision=True)
        b.cylinder(
            base, (x, 0.55, 1.49), 0.134, 0.055, "blue", euler=(0, math.pi / 2, 0)
        )
    b.rod(
        base, (-0.857, 0.55, 1.49), (1.94, 0.55, 1.49), 0.084, "bright", collision=True
    )
    for x in (-0.45, 0.22, 0.86, 1.49, 1.86):
        b.cylinder(
            base, (x, 0.55, 1.49), 0.125, 0.021, "metal", euler=(0, math.pi / 2, 0)
        )
        for i in range(8):
            a = 2 * math.pi * i / 8
            b.cylinder(
                base,
                (x + 0.025, 0.55 + 0.109 * math.cos(a), 1.49 + 0.109 * math.sin(a)),
                0.007,
                0.006,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
    for y in (0.24, 0.87):
        b.rod(base, (-0.60, y, 1.08), (2.41, y, 1.08), 0.018, "bright")
        b.rod(base, (-0.58, y, 1.105), (2.41, y, 1.105), 0.004, "rubber")
    # Representative accessible end station; source does not reveal exact internals.
    chamber = ET.SubElement(
        base,
        "body",
        name=b.unique("end_station"),
        pos="2.18 .55 1.49",
        euler=f"{math.pi/2} 0 0",
    )
    b.cylinder(chamber, (0, 0, 0), 0.253, 0.265, "metal", rgba=(0.5, 0.5, 0.5, 0))
    # Actual hollow chamber body, made of supported radial wall segments.
    for i in range(40):
        a = 2 * math.pi * i / 40
        b.box(
            chamber,
            (0.233 * math.cos(a), 0.233 * math.sin(a), 0),
            (0.023, 0.020, 0.265),
            "metal",
            collision=True,
            euler=(0, 0, a),
        )
    b.ring(chamber, (0, 0, 0.276), 0.250, 0.022, "bright")
    b.ring(chamber, (0, 0, -0.276), 0.250, 0.022, "bright")
    b.cylinder(chamber, (0, 0, -0.275), 0.225, 0.015, "metal")
    # Door swings outward, and the retained coupon can rotate with the door closed.
    door = b.moving(
        chamber,
        "access_door",
        (-0.272, 0, 0.301),
        (0, -1, 0),
        (0, 1.35),
        kind="hinge",
        mass=0.7,
        kp=240,
    )
    b.cylinder(door, (0.272, 0, 0), 0.245, 0.017, "glass", collision=True)
    b.ring(door, (0.272, 0, 0.022), 0.233, 0.016, "metal")
    b.rod(door, (0.42, -0.04, 0.055), (0.42, 0.04, 0.055), 0.012, "bright")
    for yy in (-0.10, 0.10):
        b.cylinder(
            chamber, (-0.27, yy, 0.30), 0.017, 0.035, "metal", euler=(math.pi / 2, 0, 0)
        )
    b.rod(chamber, (0, 0, -0.272), (0, 0, -0.045), 0.023, "bright")
    rotor = b.moving(
        chamber,
        "coupon_rotation",
        (0, 0, -0.045),
        (0, 0, 1),
        (-1.05, 1.05),
        kind="hinge",
        mass=0.20,
    )
    b.cylinder(rotor, (0, 0, 0), 0.066, 0.025, "dark", collision=True)
    b.box(rotor, (0, 0, 0.038), (0.070, 0.041, 0.010), "cream", collision=True)
    b.box(rotor, (-0.032, 0, 0.053), (0.007, 0.030, 0.004), "copper")
    for x in (-0.062, 0.062):
        b.box(rotor, (x, 0, 0.057), (0.009, 0.043, 0.012), "metal")
    b.site(rotor, "inert_beamline_coupon", (0, 0, 0.038), size=0.005)
    b.box(base, (2.18, 0.55, 1.18), (0.22, 0.21, 0.053), "metal", collision=True)
    b.rod(base, (2.18, 0.55, 1.20), (2.18, 0.55, 1.25), 0.08, "bright")
    b.metadata["capabilities"] = [
        "Mechanical end-station door access",
        "Retained inert coupon orientation",
    ]
    b.metadata["limitations"].append(
        "The official Uppsala image establishes a cream horizontal tank, steel "
        "beam tube and blue support portals. Exact dimensions and this end-station "
        "loading mechanism are authored estimates. No accelerator operation, "
        "voltage, ion optics, radiation, vacuum dynamics, beam-material interaction "
        "or scientific measurement is simulated."
    )


def _tandem_bay(world, definition):
    p = _Room(world, "tandem")
    # Selected bay, not all six beamlines or the separately pictured accelerators.
    p.box(
        "front_console",
        (-0.83, -0.42, 0.71),
        (0.28, 0.24, 0.71),
        (0.79, 0.81, 0.71, 1),
        True,
    )
    for z in (0.36, 0.63, 0.90, 1.17):
        p.box("console_module", (-0.83, -0.668, z), (0.245, 0.006, 0.096), WHITE)
        for x in (-0.95, -0.76):
            p.box(
                "console_display", (x, -0.677, z + 0.025), (0.055, 0.002, 0.034), DARK
            )
    for x in (0.62, 1.70):
        p.box(
            "yellow_floor_unit",
            (x, 1.64, 0.32),
            (0.37, 0.34, 0.32),
            (0.86, 0.66, 0.12, 1),
            True,
        )
        p.rod("unit_service_hose", (x, 1.27, 0.51), (x, 0.88, 1.08), 0.014, DARK)
    p.box("back_service_trunk", (-0.13, 2.62, 2.78), (3.53, 0.045, 0.07), STEEL)
    for x in (-3.42, -1.68, 0.12, 1.98, 3.18):
        p.box("wall_pipe", (x, 2.62, 1.64), (0.016, 0.023, 1.09), WHITE)
    # A source-inspired tall diagnostic support remains fixed and unpowered.
    p.box("diagnostic_foot", (2.70, 0.94, 0.07), (0.16, 0.17, 0.07), STEEL, True)
    p.rod("diagnostic_post", (2.70, 0.94, 0.14), (2.70, 0.94, 2.29), 0.023, STEEL)
    p.box("diagnostic_head", (2.70, 0.94, 2.34), (0.10, 0.13, 0.10), DARK)


BUILDERS = {
    "uppsala_pelletron_beamline": _pelletron_beamline,
    "iset_framed_prototype_station": _iset_prototype_frame,
    "astree_triaxial_loading_frame": _astree_triaxial_frame,
    "iglis_blue_laser_alignment_table": _iglis_alignment_table,
}
SOURCES = {
    "uppsala_pelletron_beamline": dict(
        reference="Uppsala Tandem Laboratory official Pelletron/beamline installed photograph",
        url=UPPSALA,
        dimensions_m=[6.05, 2.07, 2.54],
        dimension_basis="Estimated operator-scale tank, beamline and support dimensions; end station and mechanical travel are representative original fixtures",
    ),
    "iset_framed_prototype_station": dict(
        reference="Glasgow I-SET installed laboratory photograph and facility description",
        url=GLASGOW,
        dimensions_m=[1.18, 0.89, 1.77],
        dimension_basis="Estimated floor-frame proportions; exact source equipment identity unknown; inspection fixture and travel authored",
    ),
    "astree_triaxial_loading_frame": dict(
        reference="Paris-Saclay LMPS ASTREE official installed apparatus image and dimensions",
        url=SACLAY,
        dimensions_m=[2.80, 2.80, 3.12],
        dimension_basis="Published 0.650 x 0.650 x 1.500 m test space and 0.250 m stroke; exterior frame estimated; preview uses 0.20 m unloaded approach",
    ),
    "iglis_blue_laser_alignment_table": dict(
        reference="KU Leuven IGLIS-1 installed laser-room photograph",
        url=LEUVEN,
        dimensions_m=[2.48, 1.30, 1.34],
        dimension_basis="Estimated optical-table and enclosure proportions from official image; XY mount and travel authored",
    ),
}
SAMPLE_INTERFACES = {
    "uppsala_pelletron_beamline": (
        "inert_beamline_coupon",
        (0.14, 0.082, 0.02),
        "clamped",
        "Visible inert asymmetric coupon retained in an accessible unpowered end-station rotation carrier",
    ),
    "iset_framed_prototype_station": (
        "inert_payload_coupon",
        (0.234, 0.224, 0.236),
        "clamped",
        "Visible original rigid payload-shaped coupon retained on a supported carriage",
    ),
    "astree_triaxial_loading_frame": (
        "inert_structural_coupon",
        (0.156, 0.156, 0.18),
        "clamped",
        "Visible rigid inert specimen retained on the fixed lower central fixture; no material deformation",
    ),
    "iglis_blue_laser_alignment_table": (
        "inert_alignment_target",
        (0.104, 0.024, 0.10),
        "clamped",
        "Visible inert crosshair target retained on a supported XY mount",
    ),
}
FEATURES = {
    "uppsala_tandem_pelletron": _tandem_bay,
    "glasgow_iset_exploration": _iset_room,
    "ku_leuven_iglis1_laser": _iglis_room,
    "paris_saclay_lmps_astree": _astree_bay,
}
