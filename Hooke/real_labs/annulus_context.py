"""ANU photo-informed control furniture, separate from the rotating instrument.

The official annulus photo resolves a black desk with a monitor shelf, a stool
and a populated open-front electronics rack. Dimensions, internal supports,
panel controls and cable routes are authored estimates. These fixed exteriors
have contact geometry; no electronics, caster dynamics or seat load rating is
implemented. Source media and displayed experimental results are not copied.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers


DARK = (0.025, 0.029, 0.031, 1)
STEEL = (0.49, 0.51, 0.52, 1)
PANEL = (0.65, 0.66, 0.64, 1)
CREAM = (0.83, 0.84, 0.79, 1)


def _box(parent, name, position, half, color=DARK, *, collision=True):
    return box(
        parent, "anu_context_" + name, half, position, color, collision=collision
    )


def _rod(parent, name, first, last, radius, color=STEEL, *, collision=True):
    return ET.SubElement(
        parent,
        "geom",
        name="anu_context_" + name,
        type="cylinder",
        fromto=numbers((*first, *last)),
        size=str(radius),
        rgba=numbers(color),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
    )


def _cable(parent, name, points, color=DARK):
    # A parked cable is visual context, not a physical hose or electrical link.
    for i, (first, last) in enumerate(zip(points, points[1:])):
        _rod(parent, f"{name}_{i}", first, last, 0.002, color, collision=False)


def _desk(world, definition):
    desk = next(
        item for item in definition["benches"] if item["id"] == "acquisition_desk"
    )
    body = world.find("body[@name='acquisition_desk']")
    height = desk["size"][2]
    # Work in the existing desk frame so furniture placement stays data-driven.
    for side in (-1, 1):
        _box(
            body,
            f"hutch_upright_{side}",
            (side * 0.525, 0.18, height + 0.11),
            (0.018, 0.10, 0.11),
        )
    _box(body, "hutch_shelf", (0, 0.16, height + 0.24), (0.60, 0.20, 0.02))
    _box(body, "monitor_foot", (0, 0.16, height + 0.272), (0.12, 0.09, 0.012))
    _box(body, "monitor_post", (0, 0.18, height + 0.365), (0.017, 0.018, 0.09))
    _box(body, "monitor_bezel", (0, 0.18, height + 0.62), (0.30, 0.024, 0.175))
    _box(
        body,
        "monitor_idle_screen",
        (0, 0.154, height + 0.62),
        (0.281, 0.002, 0.154),
        (0.025, 0.039, 0.044, 1),
        collision=False,
    )
    _box(body, "keyboard", (0.20, -0.255, height + 0.013), (0.18, 0.068, 0.013), CREAM)
    for row in range(5):
        for column in range(14):
            _box(
                body,
                f"key_{row}_{column}",
                (0.041 + column * 0.0245, -0.307 + row * 0.026, height + 0.029),
                (0.009, 0.009, 0.003),
                CREAM,
                collision=False,
            )
    _box(body, "daq_case", (-0.34, 0.13, height + 0.049), (0.125, 0.085, 0.049), CREAM)
    _box(
        body,
        "daq_front",
        (-0.34, 0.043, height + 0.049),
        (0.090, 0.002, 0.025),
        DARK,
        collision=False,
    )
    _box(
        body, "computer_case", (0.01, 0.18, height + 0.092), (0.072, 0.13, 0.092), DARK
    )
    for index in range(7):
        _box(
            body,
            f"computer_vent_{index}",
            (-0.052 + index * 0.02, 0.048, height + 0.092),
            (0.002, 0.002, 0.067),
            STEEL,
            collision=False,
        )
    _box(
        body,
        "paper_stack",
        (-0.17, -0.22, height + 0.002),
        (0.074, 0.09, 0.002),
        CREAM,
        collision=False,
    )
    _rod(
        body,
        "rubber_mouse",
        (0.47, -0.275, height + 0.014),
        (0.47, -0.243, height + 0.014),
        0.014,
        DARK,
    )
    _cable(
        body,
        "monitor_lead",
        [
            (0, 0.202, height + 0.46),
            (0.09, 0.27, height + 0.34),
            (0.10, 0.28, height + 0.13),
            (0.071, 0.27, height + 0.10),
        ],
    )
    _cable(
        body,
        "keyboard_lead",
        [
            (0.20, -0.187, height + 0.017),
            (0.22, -0.05, height + 0.012),
            (0.15, 0.01, height + 0.012),
            (0.071, 0.07, height + 0.05),
        ],
    )


def _stool(world):
    body = ET.SubElement(world, "body", name="anu_context_stool", pos="1.24 .14 0")
    _rod(body, "stool_steel_stem", (0, 0, 0.08), (0, 0, 0.49), 0.024)
    _rod(body, "stool_rubber_seat", (0, 0, 0.475), (0, 0, 0.525), 0.145, DARK)
    for index in range(5):
        angle = index * math.tau / 5
        x, y = 0.245 * math.cos(angle), 0.245 * math.sin(angle)
        tangent = (-0.012 * math.sin(angle), 0.012 * math.cos(angle))
        _rod(body, f"stool_spoke_{index}", (0, 0, 0.10), (x, y, 0.053), 0.018, DARK)
        _rod(
            body,
            f"stool_rubber_caster_{index}",
            (x - tangent[0], y - tangent[1], 0.030),
            (x + tangent[0], y + tangent[1], 0.030),
            0.030,
            DARK,
        )


def _rack(world):
    body = ET.SubElement(world, "body", name="anu_context_rack", pos="2.195 .695 0")
    for x in (-0.265, 0.265):
        for y in (-0.295, 0.295):
            key = f"{x}_{y}"
            _rod(body, f"rack_rubber_foot_{key}", (x, y, 0), (x, y, 0.045), 0.026, DARK)
            _box(body, f"rack_upright_{key}", (x, y, 0.92), (0.018, 0.018, 0.875))
    for z in (0.08, 1.77):
        for y in (-0.295, 0.295):
            _box(body, f"rack_crossbar_{z}_{y}", (0, y, z), (0.265, 0.018, 0.022))
        for x in (-0.265, 0.265):
            _box(body, f"rack_side_rail_{z}_{x}", (x, 0, z), (0.018, 0.295, 0.022))
    _box(body, "rack_lower_grille", (0, -0.289, 0.18), (0.265, 0.010, 0.080), PANEL)
    for index in range(12):
        _box(
            body,
            f"grille_slot_{index}",
            (0, -0.300, 0.113 + 0.012 * index),
            (0.24, 0.001, 0.002),
            DARK,
            collision=False,
        )
    # Varied panel heights follow the photo's occupied upper rack and open bay.
    for index, (z, halfheight) in enumerate(
        ((0.75, 0.09), (1.0, 0.13), (1.24, 0.08), (1.53, 0.16))
    ):
        _box(
            body,
            f"rack_chassis_{index}",
            (0, 0, z),
            (0.247, 0.280, halfheight - 0.009),
            DARK,
        )
        _box(
            body,
            f"rack_aluminium_panel_{index}",
            (0, -0.285, z),
            (0.265, 0.012, halfheight),
            PANEL,
        )
        for x in (-0.252, 0.252):
            for dz in (-halfheight + 0.012, halfheight - 0.012):
                _rod(
                    body,
                    f"rack_screw_{index}_{x}_{dz}",
                    (x, -0.301, z + dz),
                    (x, -0.296, z + dz),
                    0.003,
                    DARK,
                    collision=False,
                )
        _box(
            body,
            f"rack_display_{index}",
            (0.12, -0.299, z + halfheight * 0.35),
            (0.052, 0.002, 0.018),
            DARK,
            collision=False,
        )
        for knob in range(3 if index == 3 else 5):
            x = -0.18 + 0.075 * knob
            _rod(
                body,
                f"rack_dial_{index}_{knob}",
                (x, -0.31, z - 0.03),
                (x, -0.297, z - 0.03),
                0.009,
                DARK,
                collision=False,
            )
    # Front patch loops and lower service bundles remain within the fixed rack.
    for index in range(6):
        x = -0.17 + 0.058 * index
        color = (0.52, 0.10, 0.055, 1) if index % 2 else DARK
        first, last = (x, -0.299, 0.99), (x + 0.017, -0.299, 0.73)
        points = [
            (
                (1 - t) * first[0] + t * last[0],
                -0.299 - 0.072 * math.sin(math.pi * t),
                (1 - t) * first[2] + t * last[2],
            )
            for t in [i / 12 for i in range(13)]
        ]
        _cable(body, f"patch_loop_{index}", points, color)
        _cable(
            body,
            f"lower_bundle_{index}",
            [
                (x, 0.20, 0.675),
                (x + 0.03, 0.13, 0.42),
                (x - 0.025, 0.22, 0.28),
                (x + 0.05, 0.27, 0.12),
            ],
        )


def add_control_station(world, definition):
    """Add supported parked furniture without changing instrument articulation."""
    _desk(world, definition)
    _stool(world)
    _rack(world)
