"""European university laboratories reconstructed from inspected official stills.

Authored mechanisms expose specimen positioning and access, not scientific
processes. Except FloWave's published basin diameter/depth, room and apparatus
geometry is estimated. Source photographs are never embedded as textures.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers


DELFT = "https://www.tudelft.nl/en/ceg/about-faculty/departments/engineering-structures/sections-labs/macrolab-stevinlaboratory/structures-laboratory"
ETH_FAB = "https://arch.ethz.ch/en/forschung.html"
FLOWAVE = "https://flowave.eng.ed.ac.uk/"
MOUGEL = "https://mougel.ethz.ch/lab-tour.html"

WHITE = (0.83, 0.85, 0.84, 1)
STEEL = (0.43, 0.48, 0.51, 1)
DARK = (0.075, 0.095, 0.11, 1)
BLUE = (0.055, 0.20, 0.43, 1)
YELLOW = (0.94, 0.70, 0.07, 1)
RED = (0.56, 0.11, 0.09, 1)
GLASS = (0.57, 0.72, 0.76, 0.18)


class _Details:
    """Named static architecture, separate from articulated instruments."""

    def __init__(self, world, name):
        self.world, self.name, self.count = world, name, 0

    def geom(self, key, shape, pos, size, color, *, collision=False, **extra):
        self.count += 1
        attrs = dict(
            name=f"europe_{self.name}_{key}_{self.count}",
            type=shape,
            size=numbers(size),
            rgba=numbers(color),
            contype="1" if collision else "0",
            conaffinity="1" if collision else "0",
        )
        if "fromto" not in extra:
            attrs["pos"] = numbers(pos)
        attrs.update(
            {
                k: numbers(v) if isinstance(v, (tuple, list)) else str(v)
                for k, v in extra.items()
            }
        )
        return ET.SubElement(self.world, "geom", attrs)

    def box(self, key, pos, half, color, **extra):
        return self.geom(key, "box", pos, half, color, **extra)

    def rod(self, key, start, end, radius, color, **extra):
        return self.geom(
            key, "capsule", start, (radius,), color, fromto=(*start, *end), **extra
        )

    def cylinder(self, key, pos, radius, half_height, color, **extra):
        return self.geom(key, "cylinder", pos, (radius, half_height), color, **extra)


def _column(b, parent, x, y, height, material="blue", width=0.32):
    """Three plates form an I-section with a continuous anchored foot."""
    b.box(
        parent,
        (x, y, 0.06),
        (width * 0.95, width * 0.85, 0.06),
        material,
        collision=True,
    )
    b.box(
        parent,
        (x, y, height / 2),
        (0.025, width / 2, height / 2),
        material,
        collision=True,
    )
    for dx in (-width / 2, width / 2):
        b.box(
            parent,
            (x + dx, y, height / 2),
            (0.025, width / 2, height / 2),
            material,
            collision=True,
        )
    for dx in (-width * 0.66, width * 0.66):
        for dy in (-width * 0.55, width * 0.55):
            b.cylinder(parent, (x + dx, y + dy, 0.132), 0.022, 0.012, "bright")


def _beam_test_rig(b, base, params):
    """Large inert beam specimens, approaching load head and sensor carriage."""
    b.box(base, (0, 0, 0.11), (5.45, 2.36, 0.11), "metal", collision=True)
    for x in (-3.6, 3.6):
        for y in (-1.72, 1.72):
            _column(b, base, x, y, 4.05, width=0.38)
        b.box(base, (x, 0, 3.78), (0.21, 1.97, 0.31), "blue", collision=True)
        for y in (-1.45, -0.65, 0.65, 1.45):
            b.box(base, (x - 0.23, y, 3.78), (0.015, 0.025, 0.31), "bright")
    # Two visible I-section specimens on bearing pedestals.
    for y in (-0.77, 0.77):
        for x in (-4.08, 4.08):
            b.box(base, (x, y, 0.48), (0.34, 0.45, 0.26), "metal", collision=True)
            b.cylinder(
                base, (x, y, 0.765), 0.065, 0.34, "bright", euler=(math.pi / 2, 0, 0)
            )
        b.box(base, (0, y, 0.875), (4.72, 0.43, 0.07), "cream", collision=True)
        b.box(base, (0, y, 1.105), (4.72, 0.16, 0.16), "cream", collision=True)
        b.box(base, (0, y, 1.335), (4.72, 0.43, 0.07), "cream", collision=True)
        for x in (-3.8, -2.4, -1.1, 0.4, 1.8, 3.2):
            b.box(base, (x, y - 0.447, 1.29), (0.024, 0.018, 0.025), "dark")
            b.rod(base, (x, y - 0.45, 1.29), (x - 0.2, y - 0.48, 0.44), 0.006, "dark")
    b.site(base, "beam_specimen", (0, -0.77, 1.105), size=0.02)
    # Fixed ram barrel meets the portal; its moving rod remains in the barrel
    # over the full 220 mm preview stroke and stops above the inert specimen.
    b.cylinder(base, (-3.6, -0.77, 3.06), 0.15, 0.54, "metal")
    for angle in (i * math.tau / 16 for i in range(16)):
        x, y = -3.6 + 0.128 * math.cos(angle), -0.77 + 0.128 * math.sin(angle)
        b.rod(base, (x, y, 2.54), (x, y, 3.58), 0.022, "metal", collision=True)
    b.cylinder(base, (-3.6, -0.77, 2.51), 0.17, 0.025, "bright")
    head = b.moving(
        base,
        "load_head_approach",
        (-3.6, -0.77, 2.49),
        (0, 0, -1),
        (0, 0.22),
        mass=3,
        kp=450,
        force=220,
    )
    b.cylinder(head, (0, 0, -0.32), 0.063, 0.41, "bright", collision=True)
    b.box(head, (0, 0, -0.76), (0.24, 0.30, 0.035), "metal", collision=True)
    b.site(head, "load_contact_plane", (0, 0, -0.795))
    for x in (-3.6, 3.6):
        b.rod(base, (x, -0.91, 3.34), (x + 0.47, -1.43, 3.38), 0.026, "dark")
        b.rod(base, (x + 0.47, -1.43, 3.38), (x + 0.47, -1.43, 0.27), 0.026, "dark")
    # A supported rail carries a measurement camera alongside the specimen.
    for x in (-1.65, 1.65):
        b.box(base, (x, -2.16, 0.63), (0.065, 0.10, 0.41), "metal", collision=True)
    for y in (-2.24, -2.06):
        b.rod(base, (-1.72, y, 1.08), (1.72, y, 1.08), 0.024, "bright")
    sensor = b.moving(
        base,
        "sensor_traverse",
        (0, -2.15, 1.14),
        (1, 0, 0),
        (-1.25, 1.25),
        mass=0.8,
        kp=220,
    )
    b.box(sensor, (0, 0, 0), (0.20, 0.15, 0.035), "blue", collision=True)
    b.rod(sensor, (0, 0, 0.035), (0, 0, 0.57), 0.027, "bright", collision=True)
    b.box(sensor, (0, 0.02, 0.65), (0.09, 0.115, 0.075), "dark", collision=True)
    b.cylinder(
        sensor, (0, 0.158, 0.65), 0.047, 0.026, "lens", euler=(math.pi / 2, 0, 0)
    )
    b.metadata["capabilities"] = [
        "Loading-head approach above an inert beam",
        "Measurement-camera traverse alongside the specimen",
    ]
    b.metadata["limitations"].append(
        "Delft reference-informed rig, not OEM CAD. Head retains clearance above rigid beam; "
        "no applied-load calibration, concrete deformation, fracture or sensor transfer model."
    )


def _delft_hall(world, definition):
    p = _Details(world, "delft")
    # Gallery and glazing are visible in the official hall photograph.
    p.box("gallery_deck", (0.2, 5.15, 3.75), (6.0, 0.66, 0.12), WHITE, collision=True)
    for x in (-5.5, -2.7, 0.2, 3.1, 5.9):
        p.box(
            "gallery_post", (x, 5.15, 1.84), (0.10, 0.10, 1.84), WHITE, collision=True
        )
        p.rod("gallery_rail_post", (x, 4.50, 3.88), (x, 4.50, 4.9), 0.022, STEEL)
    for z in (4.35, 4.90):
        p.rod("gallery_rail", (-5.8, 4.50, z), (6.2, 4.50, z), 0.022, STEEL)
    p.box("crane_beam", (-0.4, 1.35, 6.2), (7.5, 0.21, 0.25), RED)
    for y in (-4.1, 4.65):
        p.box("crane_track", (0, y, 6.3), (7.8, 0.12, 0.16), STEEL)
    p.rod("crane_cable", (-3.7, 1.35, 6.0), (-3.7, 1.35, 4.25), 0.02, DARK)
    p.cylinder(
        "crane_block", (-3.7, 1.35, 4.15), 0.18, 0.10, YELLOW, euler=(math.pi / 2, 0, 0)
    )
    for x in (-5.2, -3.3, -1.4, 0.5):
        for dx in (-0.58, 0.58):
            p.box(
                "stored_reaction_post",
                (x + dx, 4.1, 1.25),
                (0.085, 0.27, 1.25),
                RED,
                collision=True,
            )
        for z in (0.16, 1.2, 2.45):
            p.box("stored_reaction_crossbar", (x, 4.1, z), (0.67, 0.27, 0.10), RED)
    for x in (-3.35, -2.45):
        p.box("monitor_foot", (x, -4.38, 0.802), (0.17, 0.13, 0.020), STEEL)
        p.rod("monitor_stand", (x, -4.34, 0.82), (x, -4.34, 1.09), 0.025, STEEL)
        p.box("acquisition_monitor", (x, -4.35, 1.18), (0.28, 0.045, 0.18), DARK)
        p.box(
            "acquisition_display",
            (x, -4.399, 1.18),
            (0.253, 0.006, 0.153),
            (0.16, 0.30, 0.36, 1),
        )
        p.box("acquisition_keyboard", (x, -4.64, 0.81), (0.23, 0.085, 0.018), DARK)
    p.box("acquisition_unit", (-4.10, -4.37, 0.96), (0.17, 0.21, 0.17), WHITE)
    for z in (0.88, 0.97, 1.06):
        p.box("daq_channel", (-4.10, -4.584, z), (0.12, 0.007, 0.020), STEEL)
    # Separate mobile barriers leave the front circulation route continuous.
    for x in (-3.6, -1.6, 0.4, 2.4):
        for dx in (-0.86, 0.86):
            p.box("barrier_foot", (x + dx, -2.95, 0.065), (0.15, 0.28, 0.065), DARK)
            p.rod(
                "barrier_post",
                (x + dx, -2.95, 0.1),
                (x + dx, -2.95, 1.0),
                0.032,
                YELLOW,
            )
        for z in (0.34, 0.93):
            p.rod(
                "barrier_rail",
                (x - 0.86, -2.95, z),
                (x + 0.86, -2.95, z),
                0.035,
                YELLOW,
            )
        for dx in (-0.50, 0, 0.50):
            p.rod(
                "barrier_infill",
                (x + dx, -2.95, 0.34),
                (x + dx, -2.95, 0.93),
                0.022,
                YELLOW,
            )
    for x in (-2.6, 0.0, 2.2):
        for dx, dy in ((-0.30, -0.23), (0.30, -0.23), (0, 0.30)):
            p.rod(
                "camera_tripod_leg",
                (x + dx, -2.45 + dy, 0.02),
                (x, -2.45, 1.28),
                0.016,
                DARK,
            )
        p.box("measurement_camera", (x, -2.45, 1.39), (0.11, 0.095, 0.07), DARK)
        p.cylinder(
            "measurement_lens",
            (x, -2.325, 1.39),
            0.04,
            0.03,
            STEEL,
            euler=(math.pi / 2, 0, 0),
        )


def _suspended_fabricator(b, base, params):
    """Anchored overhead gantry with a locked-pose industrial-arm surrogate."""
    for x in (-3.55, 3.55):
        for y in (-2.15, 2.15):
            _column(b, base, x, y, 6.05, "shell", width=0.30)
    for y in (-2.15, 2.15):
        b.box(base, (0, y, 5.96), (3.82, 0.14, 0.16), "shell", collision=True)
        b.box(base, (0, y, 5.77), (3.47, 0.065, 0.035), "bright")
    for x in (-3.55, 3.55):
        b.box(base, (x, 0, 5.95), (0.15, 2.16, 0.17), "shell", collision=True)
    bridge = b.moving(
        base,
        "gantry_traverse",
        (0, 0, 5.61),
        (1, 0, 0),
        (-1.35, 1.35),
        mass=8,
        kp=700,
        force=500,
    )
    b.box(bridge, (0, 0, 0), (0.60, 2.22, 0.12), "shell", collision=True)
    for y in (-1.72, 1.72):
        b.box(bridge, (0, y, 0.14), (0.32, 0.18, 0.08), "dark")
    for y in (-0.25, 0.25):
        b.box(bridge, (0, y, -1.315), (0.19, 0.055, 1.215), "shell", collision=True)
    b.box(bridge, (0, 0, -0.30), (0.21, 0.26, 0.07), "shell", collision=True)
    for x in (-0.18, 0.18):
        b.box(bridge, (x, 0, -2.48), (0.03, 0.26, 0.07), "shell", collision=True)
    # Rail-backed carriage stays engaged with the surrounding mast at all heights.
    lift = b.moving(
        bridge,
        "gantry_drop",
        (0, 0, -2.60),
        (0, 0, -1),
        (0, 0.55),
        mass=4,
        kp=500,
        force=300,
    )
    b.box(lift, (0, 0, 0.46), (0.12, 0.13, 0.55), "bright", collision=True)
    b.box(lift, (0, 0, -0.10), (0.28, 0.25, 0.095), "shell", collision=True)
    # White industrial elbow/link architecture follows the photo. These links
    # remain in an authored pose; the exposed controls belong to the carrier.
    b.cylinder(lift, (0, 0, -0.30), 0.21, 0.12, "shell", collision=True)
    b.cylinder(lift, (0.08, 0, -0.48), 0.19, 0.20, "shell", euler=(math.pi / 2, 0, 0))
    b.rod(lift, (0.10, 0, -0.48), (0.36, 0, -1.07), 0.125, "shell", collision=True)
    b.cylinder(lift, (0.37, 0, -1.08), 0.16, 0.16, "shell", euler=(math.pi / 2, 0, 0))
    b.rod(lift, (0.39, 0, -1.08), (1.02, 0, -1.18), 0.092, "shell", collision=True)
    for start, end in (
        ((0.03, -0.23, -0.37), (0.29, -0.20, -1.02)),
        ((0.29, -0.20, -1.02), (0.91, -0.14, -1.13)),
    ):
        b.rod(lift, start, end, 0.024, "dark")
    b.cylinder(lift, (1.04, 0, -1.18), 0.115, 0.15, "metal")
    tool = b.moving(
        lift,
        "tool_rotation",
        (1.04, 0, -1.36),
        (0, 0, 1),
        (-0.65, 0.65),
        kind="hinge",
        mass=0.5,
        kp=100,
    )
    b.cylinder(tool, (0, 0, 0), 0.12, 0.055, "metal", collision=True)
    b.box(tool, (0, 0, -0.08), (0.18, 0.13, 0.03), "dark", collision=True)
    # Opposed pads continuously grip a visible inert construction coupon.
    for y in (-0.106, 0.106):
        b.box(tool, (0, y, -0.19), (0.16, 0.025, 0.085), "metal", collision=True)
        b.box(tool, (0, y * 0.78, -0.19), (0.15, 0.006, 0.060), "rubber")
    b.box(tool, (0, 0, -0.20), (0.30, 0.075, 0.055), "cream", collision=True)
    for x in (-0.15, 0.04, 0.20):
        b.box(tool, (x, -0.076, -0.20), (0.002, 0.002, 0.05), "copper")
    b.site(tool, "construction_coupon", (0, 0, -0.20), size=0.012)
    # Chain segments remain attached to the vertical carrier rather than floating.
    for z in (-0.35 - 0.16 * i for i in range(14)):
        b.box(bridge, (0.23, 0.32, z), (0.055, 0.035, 0.06), "dark")
    b.metadata["capabilities"] = [
        "Overhead carrier travel",
        "Suspended carrier height",
        "Rotation of a clamped construction coupon",
    ]
    b.metadata["limitations"].append(
        "ETH reference-informed mechanical surrogate, not a verified OEM robot or gantry. "
        "The elbow links are locked in an authored pose; only gantry X/Z and tool rotation "
        "are actuated. Coupon is rigidly retained; no autonomous fabrication or release policy."
    )


def _fabrication_hall(world, definition):
    p = _Details(world, "fabrication")
    p.box("gallery_deck", (-1.6, 5.45, 3.60), (6.9, 0.85, 0.13), WHITE, collision=True)
    for x in (-8.2, -4.5, -0.7, 3.4):
        p.box(
            "gallery_column", (x, 5.50, 1.74), (0.13, 0.13, 1.74), WHITE, collision=True
        )
    for x in (-7.6, -5.9, -4.2, -2.5, -0.8, 0.9, 2.6, 4.3):
        p.box("gallery_glazing", (x, 4.61, 4.76), (0.81, 0.018, 1.00), GLASS)
        p.box("gallery_mullion", (x - 0.85, 4.61, 4.76), (0.025, 0.035, 1.05), STEEL)
    for z in (3.73, 5.79):
        p.box("gallery_header", (-1.65, 4.61, z), (6.82, 0.06, 0.04), WHITE)
    p.box(
        "stair_landing", (6.85, 5.30, 3.60), (1.55, 0.85, 0.13), WHITE, collision=True
    )
    p.box("landing_column", (8.0, 5.1, 1.74), (0.13, 0.13, 1.74), WHITE, collision=True)
    # Stair is backed by a solid stringer ramp; individual treads read at human scale.
    for i in range(18):
        x, z = 5.45 + 0.165 * i, 0.10 + 0.20 * i
        p.box("stair_tread", (x, 4.50, z), (0.093, 0.66, 0.055), WHITE, collision=True)
        if i % 3 == 0:
            p.rod("stair_guard_post", (x, 3.86, z), (x, 3.86, z + 0.92), 0.021, STEEL)
    p.rod("stair_stringer", (5.42, 4.05, 0.03), (8.30, 4.05, 3.53), 0.12, WHITE)
    p.rod("stair_stringer", (5.42, 4.95, 0.03), (8.30, 4.95, 3.53), 0.12, WHITE)
    p.rod("stair_handrail", (5.42, 3.86, 0.97), (8.32, 3.86, 4.47), 0.025, STEEL)
    for x in (-8, -5.5, -2.8, 0, 2.8, 5.5, 8):
        p.cylinder("safety_post", (x, -2.95, 0.67), 0.045, 0.67, YELLOW, collision=True)
        p.cylinder("safety_foot", (x, -2.95, 0.04), 0.14, 0.04, STEEL)
    # Authored fixtures sit below the tool workspace, with clearance at full drop.
    for x in (-4.2, 4.2):
        p.box(
            "fixture_pedestal",
            (x + 1.04, 0.45, 0.32),
            (0.85, 0.36, 0.32),
            DARK,
            collision=True,
        )
        p.box(
            "fixture_platen",
            (x + 1.04, 0.45, 0.68),
            (0.94, 0.43, 0.04),
            STEEL,
            collision=True,
        )
        for dx in (-0.6, -0.3, 0, 0.3, 0.6):
            p.cylinder(
                "fixture_pin", (x + 1.04 + dx, 0.45, 0.765), 0.025, 0.045, YELLOW
            )


def _circular_wave_basin(b, base, params):
    """25 m diameter, 2 m nominal water depth, with two local paddle drives."""
    # Equipment datum remains at the hall floor; its supported basin body is recessed.
    base = ET.SubElement(base, "body", name=f"{b.name}__recessed_basin", pos="0 0 -2.2")
    b.cylinder(base, (0, 0, 0.10), 12.85, 0.10, "metal", collision=True)
    active = {68: "paddle_west", 76: "paddle_east"}
    for i in range(96):
        angle = i * math.tau / 96
        c, sn = math.cos(angle), math.sin(angle)
        b.box(
            base,
            (12.88 * c, 12.88 * sn, 1.15),
            (0.23, 0.435, 1.15),
            "shell",
            euler=(0, 0, angle),
            collision=True,
        )
        if i in active:
            paddle = b.moving(
                base,
                active[i],
                (12.57 * c, 12.57 * sn, 1.32),
                (-c, -sn, 0),
                (0, 0.12),
                mass=1.8,
                kp=360,
                force=200,
            )
            b.box(
                paddle,
                (0, 0, 0),
                (0.045, 0.365, 1.02),
                "metal",
                euler=(0, 0, angle),
                collision=True,
            )
            # Carrier rails and motor housings are visibly fixed to the perimeter.
            b.rod(
                base,
                (12.48 * c, 12.48 * sn, 2.39),
                (13.48 * c, 13.48 * sn, 2.39),
                0.028,
                "bright",
            )
            b.box(
                base,
                (13.24 * c, 13.24 * sn, 2.48),
                (0.25, 0.18, 0.08),
                "dark",
                euler=(0, 0, angle),
            )
            b.rod(paddle, (0, 0, 0.97), (0.58 * c, 0.58 * sn, 1.06), 0.025, "bright")
        else:
            b.box(
                base,
                (12.57 * c, 12.57 * sn, 1.32),
                (0.045, 0.365, 1.02),
                "metal",
                euler=(0, 0, angle),
                collision=True,
            )
        # Annular grate strips bridge shell to outer floor. Their top is hall z=0.
        b.box(
            base,
            (13.65 * c, 13.65 * sn, 2.125),
            (0.93, 0.455, 0.075),
            "bright",
            euler=(0, 0, angle),
            collision=True,
        )
        for dr in (-0.66, -0.30, 0.06, 0.42, 0.78):
            b.box(
                base,
                ((13.65 + dr) * c, (13.65 + dr) * sn, 2.207),
                (0.011, 0.44, 0.007),
                "dark",
                euler=(0, 0, angle),
            )
    b.ring(base, (0, 0, 2.27), 14.60, 0.045, "warning", segments=96)
    b.ring(base, (0, 0, 2.215), 12.64, 0.025, "bright", segments=96)
    # Display water is a thin, non-colliding surface; no hidden solid fills the pool.
    b.cylinder(
        base, (0, 0, 2.196), 12.49, 0.004, "glass", rgba=(0.22, 0.48, 0.57, 0.66)
    )
    # Added inert calibration buoy has a real support rod down to the basin floor.
    b.rod(base, (0, 0, 0.20), (0, 0, 2.24), 0.033, "metal", collision=True)
    b.cylinder(base, (0, 0, 2.30), 0.46, 0.075, "orange", collision=True)
    b.ring(base, (0, 0, 2.39), 0.34, 0.06, "shell")
    b.cylinder(base, (0, 0, 2.60), 0.055, 0.20, "bright", collision=True)
    b.box(base, (0, 0, 2.84), (0.12, 0.08, 0.04), "dark")
    b.site(base, "calibration_buoy", (0, 0, 2.30), size=0.02)
    b.metadata["capabilities"] = [
        "Two independently positioned perimeter wave paddles",
        "Visible supported inert calibration model",
    ]
    b.metadata["limitations"].append(
        "Published 25 m diameter and 2 m depth anchor the basin. Only two representative "
        "paddles have mechanical controls; no wave propagation, current, buoyancy, splash "
        "or hydrodynamic load is simulated. Central buoy and its support are authored fixtures."
    )


def _flowave_hall(world, definition):
    p = _Details(world, "flowave")
    # Cut the floor around the actual recessed basin instead of filling its water
    # volume with the default rectangular collision slab.
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_floor"):
            world.remove(geom)
    p.box("basement_foundation", (0, 0, -2.29), (16, 16, 0.09), STEEL, collision=True)
    for j in range(16):
        lo, hi = -16 + 2 * j, -14 + 2 * j
        near = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        edge = math.sqrt(max(0, 12.72**2 - near**2))
        for sign in (-1, 1):
            p.box(
                "floor_apron",
                (sign * (16 + edge) / 2, (lo + hi) / 2, -0.075),
                ((16 - edge) / 2, 1, 0.075),
                (0.49, 0.53, 0.54, 1),
                collision=True,
            )
    # Dark peripheral truss and yellow bridge crane are visible source features.
    for y in (13.85,):
        for z in (0.36, 1.66):
            p.rod("perimeter_truss_chord", (-10.8, y, z), (10.8, y, z), 0.065, DARK)
        for i in range(12):
            x = -10.8 + i * 1.8
            p.rod(
                "perimeter_truss_diagonal",
                (x, y, 0.36),
                (x + 0.9, y, 1.66),
                0.055,
                DARK,
            )
            p.rod(
                "perimeter_truss_diagonal",
                (x + 0.9, y, 1.66),
                (x + 1.8, y, 0.36),
                0.055,
                DARK,
            )
    for x in (-15.25, 15.25):
        p.box("crane_runway", (x, 0, 7.9), (0.15, 15.5, 0.22), WHITE)
        for y in (-11, -3, 5, 12):
            p.box(
                "runway_column", (x, y, 3.84), (0.14, 0.18, 3.84), WHITE, collision=True
            )
            p.box(
                "runway_foot", (x, y, 0.06), (0.24, 0.28, 0.06), STEEL, collision=True
            )
    p.box("yellow_bridge_crane", (0, 8.8, 7.5), (15.25, 0.33, 0.36), YELLOW)
    p.box("crane_trolley", (2.7, 8.8, 7.04), (0.65, 0.42, 0.12), STEEL)
    p.rod("crane_cable", (2.7, 8.8, 6.92), (2.7, 8.8, 4.6), 0.022, DARK)
    p.cylinder(
        "crane_hook_block",
        (2.7, 8.8, 4.47),
        0.16,
        0.08,
        YELLOW,
        euler=(math.pi / 2, 0, 0),
    )
    for y in (-11, -5.5, 0, 5.5, 11):
        for z in (8.2, 8.75):
            g = p.rod("roof_chord", (-15.5, y, z), (15.5, y, z), 0.07, WHITE)
            g.set("name", "arch_roof_" + g.get("name"))
        for x in (-15, -10, -5, 0, 5, 10):
            g = p.rod("roof_diagonal", (x, y, 8.2), (x + 2.5, y, 8.75), 0.055, WHITE)
            g.set("name", "arch_roof_" + g.get("name"))
            g = p.rod(
                "roof_diagonal", (x + 2.5, y, 8.75), (x + 5, y, 8.2), 0.055, WHITE
            )
            g.set("name", "arch_roof_" + g.get("name"))
    for x in (-13.7, -10.2, -6.7, -3.2, 0.3, 3.8, 7.3, 10.8):
        p.box(
            "control_cabinet",
            (x, 15.15, 1.04),
            (0.56, 0.32, 1.04),
            WHITE,
            collision=True,
        )
        p.box("cabinet_split", (x, 14.817, 1.12), (0.004, 0.008, 0.9), STEEL)
        p.box("cabinet_panel", (x + 0.22, 14.795, 1.62), (0.12, 0.025, 0.14), DARK)


def _glovebox_module(b, base, params):
    """Two-port glovebox, external pass-through lid and internal vial carriage."""
    for x in (-1.16, 1.16):
        for y in (-0.43, 0.43):
            b.cylinder(base, (x, y, 0.035), 0.055, 0.035, "rubber", collision=True)
            b.box(base, (x, y, 0.49), (0.036, 0.036, 0.42), "shell", collision=True)
    for y in (-0.43, 0.43):
        for z in (0.22, 0.88):
            b.box(base, (0, y, z), (1.22, 0.032, 0.032), "shell", collision=True)
    b.box(base, (0, 0, 0.93), (1.32, 0.55, 0.035), "metal", collision=True)
    b.box(base, (0, 0.535, 1.49), (1.3, 0.025, 0.525), "shell", collision=True)
    b.box(base, (-1.295, 0, 1.49), (0.025, 0.55, 0.525), "shell", collision=True)
    # Right wall has a real opening behind the pass-through rather than a solid
    # plate pretending to allow specimen transfer.
    for y in (-0.40, 0.40):
        b.box(base, (1.295, y, 1.49), (0.025, 0.15, 0.525), "shell", collision=True)
    b.box(base, (1.295, 0, 1.82), (0.025, 0.25, 0.195), "shell", collision=True)
    b.box(base, (1.295, 0, 1.045), (0.025, 0.25, 0.080), "shell", collision=True)
    b.box(base, (0, 0, 2.045), (1.35, 0.57, 0.075), "cream", collision=True)
    b.box(
        base,
        (0, -0.497, 1.50),
        (1.235, 0.009, 0.49),
        "glass",
        euler=(-0.15, 0, 0),
        collision=True,
    )
    for x in (-1.27, 1.27):
        b.box(
            base, (x, -0.50, 1.50), (0.035, 0.035, 0.51), "cream", euler=(-0.15, 0, 0)
        )
    b.box(base, (0, -0.57, 1.0), (1.3, 0.035, 0.035), "cream")
    for x in (-0.60, 0.60):
        b.cylinder(
            base, (x, -0.57, 1.44), 0.145, 0.069, "metal", euler=(math.pi / 2, 0, 0)
        )
        b.ring(base, (x, -0.645, 1.44), 0.14, 0.012, "bright", plane="xz")
        b.rod(base, (x, -0.66, 1.44), (x, -1.035, 1.35), 0.092, "rubber")
        b.geom(base, "ellipsoid", (0.084, 0.13, 0.042), (x, -1.09, 1.33), "rubber")
        for dx in (-0.053, -0.018, 0.018, 0.053):
            b.rod(
                base,
                (x + dx, -1.14, 1.33),
                (x + dx * 0.96, -1.235, 1.325),
                0.013,
                "rubber",
            )
    # Hollow cylindrical pass-through, including collision shell and flange.
    for i in range(32):
        angle = math.tau * i / 32
        y, z = 0.238 * math.cos(angle), 1.36 + 0.238 * math.sin(angle)
        b.rod(base, (1.30, y, z), (2.055, y, z), 0.022, "metal", collision=True)
    for x in (1.34, 2.055):
        b.ring(base, (x, 0, 1.36), 0.249, 0.027, "bright", plane="yz", segments=32)
    lid = b.moving(
        base,
        "pass_through_lid",
        (2.105, -0.23, 1.36),
        (0, 0, -1),
        (0, 1.25),
        kind="hinge",
        mass=0.65,
        kp=100,
        force=80,
    )
    b.cylinder(
        lid,
        (0, 0.23, 0),
        0.225,
        0.022,
        "metal",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    b.cylinder(lid, (0.025, 0.23, 0), 0.19, 0.004, "bright", euler=(0, math.pi / 2, 0))
    b.rod(lid, (0.073, 0.15, -0.05), (0.073, 0.32, -0.05), 0.015, "dark")
    for y in (-0.15, 0.15):
        b.rod(base, (-0.55, y, 0.997), (0.55, y, 0.997), 0.011, "bright")
    tray = b.moving(
        base,
        "vial_carriage",
        (0, 0, 1.035),
        (1, 0, 0),
        (-0.24, 0.24),
        mass=0.35,
        kp=180,
    )
    b.box(tray, (0, 0, 0), (0.24, 0.20, 0.024), "blue", collision=True)
    for x in (-0.14, 0, 0.14):
        b.cylinder(tray, (x, 0, 0.032), 0.042, 0.008, "metal")
        b.cylinder(tray, (x, 0, 0.095), 0.036, 0.055, "glass")
        b.cylinder(tray, (x, 0, 0.156), 0.039, 0.008, "orange")
    b.site(tray, "inert_vial", (0, 0, 0.095))
    for x in (-0.95, 0.94):
        b.box(base, (x, 0.25, 0.44), (0.20, 0.20, 0.18), "shell", collision=True)
        b.vents(base, (x, 0.044, 0.41), 0.25)
    b.box(base, (-0.82, 0.2, 2.235), (0.23, 0.28, 0.11), "shell")
    b.cylinder(base, (0.50, 0.27, 2.275), 0.12, 0.16, "metal")
    b.rod(base, (0.50, 0.27, 2.44), (0.50, 0.27, 2.59), 0.025, "bright")
    b.screen(base, (-0.90, -0.594, 1.87), 0.27, 0.14)
    b.metadata["capabilities"] = [
        "External pass-through lid access",
        "Internal inert-vial carriage positioning",
    ]
    b.metadata["limitations"].append(
        "Source-informed two-port module; model identity and dimensions unverified. "
        "Internal carriage is an authored task fixture. No gas purification, vacuum "
        "cycle, containment, atmospheric chemistry or glove deformation is simulated."
    )


def _mougel_aisle(world, definition):
    p = _Details(world, "mougel")
    for x in (-2.1, 2.1):
        p.box("overhead_service_rail", (x, 0, 2.89), (0.035, 5.7, 0.035), STEEL)
        for y in (-4.7, -1.9, 0.9, 3.7, 5.25):
            p.rod("service_drop", (x, y, 2.90), (x, y, 3.14), 0.022, STEEL)
    for y in (-4.5, -1.5, 1.5, 4.5):
        p.box("ceiling_crossframe", (0, y, 3.08), (2.45, 0.035, 0.035), WHITE)
        for x in (-0.62, 0.62):
            p.box(
                "linear_task_light",
                (x, y, 2.94),
                (0.06, 0.88, 0.025),
                (1, 0.98, 0.89, 1),
            )
    # End desk and window are visible source cues, with an estimated cabinet.
    p.box("end_worktop", (0, 5.42, 0.76), (0.87, 0.38, 0.035), WHITE, collision=True)
    for x in (-0.77, 0.77):
        for y in (5.14, 5.70):
            p.box(
                "end_desk_leg",
                (x, y, 0.36),
                (0.028, 0.028, 0.36),
                STEEL,
                collision=True,
            )
    p.box("end_monitor", (0, 5.60, 1.10), (0.26, 0.045, 0.17), DARK)
    p.rod("end_monitor_stand", (0, 5.59, 0.80), (0, 5.59, 0.95), 0.025, STEEL)
    for x, y in ((-1.05, 0.65), (1.05, 3.35)):
        p.cylinder("wood_stool_seat", (x, y, 0.64), 0.18, 0.028, (0.72, 0.53, 0.30, 1))
        p.rod("stool_column", (x, y, 0.16), (x, y, 0.61), 0.025, DARK)
        for a in (0, 2.094, 4.189):
            p.rod(
                "stool_foot",
                (x, y, 0.20),
                (x + 0.22 * math.cos(a), y + 0.22 * math.sin(a), 0.03),
                0.014,
                DARK,
            )
    for y in (-3, -1, 1, 3):
        p.box(
            "floor_tile_seam",
            (0, y, 0.002),
            (2.76, 0.003, 0.0007),
            (0.42, 0.42, 0.40, 1),
        )


BUILDERS = {
    "mougel_glovebox_module": _glovebox_module,
    "flowave_circular_basin": _circular_wave_basin,
    "stevin_beam_test_rig": _beam_test_rig,
    "suspended_fabrication_gantry": _suspended_fabricator,
}
SOURCES = {
    "mougel_glovebox_module": {
        "reference": "ETH Mougel Group two-sided glovebox aisle official lab-tour photograph",
        "url": MOUGEL,
        "dimensions_m": [3.85, 1.81, 2.61],
        "dimension_basis": "Estimated modules, sleeves and pass-through; no OEM model or measured drawing verified",
    },
    "flowave_circular_basin": {
        "reference": "University of Edinburgh FloWave circular wave/current basin",
        "url": FLOWAVE,
        "dimensions_m": [29.3, 29.3, 3.08],
        "dimension_basis": "Official basin diameter 25 m and water depth 2 m; rim, paddle drives, support and hall estimated",
    },
    "suspended_fabrication_gantry": {
        "reference": "ETH Robotic Fabrication Laboratory gantry/robot photograph; authored locked-pose arm surrogate",
        "url": ETH_FAB,
        "dimensions_m": [7.8, 4.9, 6.15],
        "dimension_basis": "Visual estimate; no OEM identity, measured envelope or manufacturer stroke verified",
    },
    "stevin_beam_test_rig": {
        "reference": "TU Delft Stevinlaboratory beam-testing photograph; authored loading and measurement mechanisms",
        "url": DELFT,
        "dimensions_m": [10.9, 4.72, 4.05],
        "dimension_basis": "Estimated from inspected hall photograph; no surveyed beam or machine dimensions",
    },
}
SAMPLE_INTERFACES = {
    "mougel_glovebox_module": (
        "inert_vial",
        (0.072, 0.072, 0.11),
        "clamped",
        "Visible inert capped vial retained on an internal positioning tray",
    ),
    "flowave_circular_basin": (
        "calibration_buoy",
        (0.92, 0.92, 0.15),
        "clamped",
        "Authored inert calibration buoy fixed on a visible basin-floor support",
    ),
    "suspended_fabrication_gantry": (
        "construction_coupon",
        (0.60, 0.15, 0.11),
        "clamped",
        "Visible inert construction coupon retained between opposing tool pads",
    ),
    "stevin_beam_test_rig": (
        "beam_specimen",
        (9.44, 0.86, 0.60),
        "clamped",
        "Visible inert I-section beam retained on bearing pedestals",
    ),
}
FEATURES = {
    "eth_mougel_inorganic": _mougel_aisle,
    "edinburgh_flowave_basin": _flowave_hall,
    "delft_stevin_structures": _delft_hall,
    "eth_robotic_fabrication": _fabrication_hall,
}
