"""Passive source-visible PICE saw and local service details.

Two room photographs establish exterior silhouettes and parked leads. Sizes,
hidden attachments and routing are estimates; none of these parts powers a
band, cuts a sample or simulates an electrical or flexible-cable process.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import box, numbers

MACHINE_PREFIX = "ice_station__source_"
ROOM_PREFIX = "pice_context_"


def add_machine_context(b, base):
    def shape(kind, key, pos, size, material="cream", **kwargs):
        return b.geom(
            base,
            kind,
            size,
            pos,
            material,
            collision=True,
            density=0,
            name=MACHINE_PREFIX + key,
            **kwargs,
        )

    def block(key, pos, half, material="cream", **kwargs):
        return shape("box", key, pos, half, material, **kwargs)

    def rod(key, start, end, radius, material="dark"):
        return shape(
            "capsule", key, (0, 0, 0), (radius,), material, fromto=(*start, *end)
        )

    # Transverse dark motor exterior below the table, connected to the fixed
    # lower arm and base. No shaft rotation or closed powered band is authored.
    block("motor_mount", (0.15, 0.13, 0.680), (0.060, 0.070, 0.025), "metal")
    shape(
        "cylinder",
        "motor_case",
        (0.29, 0.13, 0.728),
        (0.075, 0.12),
        "dark",
        euler=(0, math.pi / 2, 0),
    )
    shape(
        "cylinder",
        "motor_end",
        (0.417, 0.13, 0.728),
        (0.071, 0.014),
        "dark",
        euler=(0, math.pi / 2, 0),
    )
    for i in range(6):
        shape(
            "cylinder",
            f"motor_rib_{i}",
            (0.205 + i * 0.032, 0.13, 0.728),
            (0.078, 0.003),
            "dark",
            euler=(0, math.pi / 2, 0),
        )
    shape("sphere", "motor_cable_gland", (0.29, 0.211, 0.728), (0.013,), "dark")

    # The visible lower cover is parked open. It has fixed supported hinge
    # hardware, so no new control or moving collision surface is introduced.
    block("lower_guard_mount", (-0.262, 0.05, 0.70), (0.023, 0.070, 0.045), "metal")
    rod(
        "lower_guard_pin",
        (-0.285, -0.023, 0.575),
        (-0.285, -0.023, 0.825),
        0.012,
        "metal",
    )
    angle = -0.90

    def panel_point(u, v, z):
        return (
            -0.285 + u * math.cos(angle) - v * math.sin(angle),
            -0.023 + u * math.sin(angle) + v * math.cos(angle),
            z,
        )

    block(
        "lower_guard_panel",
        panel_point(0.17, 0, 0.70),
        (0.17, 0.012, 0.135),
        "shell",
        euler=(0, 0, angle),
    )
    for i, ends in enumerate(
        (((0.025, 0.595), (0.315, 0.805)), ((0.025, 0.805), (0.315, 0.595)))
    ):
        rod(
            f"lower_guard_brace_{i}",
            panel_point(ends[0][0], -0.017, ends[0][1]),
            panel_point(ends[1][0], -0.017, ends[1][1]),
            0.006,
            "metal",
        )
    rod(
        "lower_guard_latch_stem",
        panel_point(0.29, -0.010, 0.70),
        panel_point(0.29, -0.036, 0.70),
        0.006,
    )
    shape(
        "sphere", "lower_guard_latch", panel_point(0.29, -0.038, 0.70), (0.016,), "dark"
    )

    # Original controller case, stop button and mounting rod remain untouched.
    # An internal enabled stem connects the case to its already visible mount.
    rod("control_mount_core", (0.25, 0.10, 1.58), (0.31, 0.10, 1.58), 0.011, "metal")
    block("stop_plate_backing", (0.34, 0.039, 1.60), (0.034, 0.0015, 0.040))
    block("selector_plate", (0.34, 0.036, 1.67), (0.034, 0.004, 0.026))
    shape(
        "cylinder",
        "selector_bezel",
        (0.34, 0.028, 1.67),
        (0.020, 0.006),
        "dark",
        euler=(math.pi / 2, 0, 0),
    )
    shape(
        "cylinder",
        "selector_knob",
        (0.34, 0.016, 1.67),
        (0.011, 0.009),
        "dark",
        euler=(math.pi / 2, 0, 0),
    )
    block("selector_mark", (0.34, 0.0065, 1.676), (0.0015, 0.001, 0.004), "cream")
    block("control_label", (0.34, 0.038, 1.525), (0.036, 0.003, 0.010))
    for i in range(2):
        block(
            f"label_rule_{i}",
            (0.34, 0.034, 1.523 + 0.005 * i),
            (0.024, 0.001, 0.0007),
            "dark",
        )
    shape("cylinder", "top_adjuster_stem", (0, 0.08, 1.825), (0.014, 0.030), "metal")
    shape("cylinder", "top_adjuster_grip", (0, 0.08, 1.858), (0.060, 0.008), "dark")

    # A parked coil rests on the existing lower stand shelf (top z=.195).
    # Its continuation passes behind the upper shelf, not through its solid top.
    points = []
    for i in range(97):
        turn = i / 48
        theta = math.tau * turn
        points.append(
            (
                (0.270 - 0.005 * turn) * math.cos(theta),
                (0.225 - 0.004 * turn) * math.sin(theta),
                0.201 + 0.006 * turn,
            )
        )
    for i, (start, end) in enumerate(zip(points, points[1:])):
        rod(f"parked_coil_{i}", start, end, 0.006)
    block("parked_plug", (0.295, 0, 0.210), (0.029, 0.016, 0.015), "dark")
    lead = [
        points[-1],
        (0.29, 0.20, 0.213),
        (0.29, 0.367, 0.25),
        (0.29, 0.367, 0.55),
        (0.29, 0.267, 0.62),
        (0.29, 0.227, 0.69),
        (0.29, 0.211, 0.728),
    ]
    for i, (start, end) in enumerate(zip(lead, lead[1:])):
        rod(f"parked_motor_lead_{i}", start, end, 0.006)
    b.metadata["source_exterior_context"] = {
        "revision": 1,
        "visible": "Lower motor exterior, parked-open lower cover, static selector/top adjuster and coiled lead on the existing lower stand shelf",
        "estimated": "Dimensions, internal attachments, parked lead routing and control identities",
        "scope": "Fixed exterior geometry only; no powered band, new controls, electrical behavior or cutting",
    }


def add_room_services(world):
    dark = (0.055, 0.065, 0.075, 1)
    white = (0.84, 0.85, 0.83, 1)
    metal = (0.49, 0.53, 0.56, 1)
    blue = (0.07, 0.25, 0.38, 1)

    def block(key, pos, half, color):
        return box(world, ROOM_PREFIX + key, half, pos, color, collision=True)

    def rod(key, start, end, radius, color=dark):
        return ET.SubElement(
            world,
            "geom",
            name=ROOM_PREFIX + key,
            type="capsule",
            fromto=numbers((*start, *end)),
            size=str(radius),
            rgba=numbers(color),
            contype="1",
            conaffinity="1",
        )

    # Wall rear faces meet the existing north-wall interior plane y=2.25.
    block("upper_wall_switch", (1.08, 2.213, 1.86), (0.065, 0.037, 0.070), dark)
    block(
        "wall_switch_tag",
        (1.10, 2.174, 1.835),
        (0.025, 0.002, 0.013),
        (0.86, 0.65, 0.12, 1),
    )
    block("wall_socket", (0.78, 2.220, 1.08), (0.065, 0.030, 0.055), white)
    rod("socket_plug", (0.80, 2.199, 1.08), (0.80, 2.175, 1.08), 0.016)
    pipe = [(1.08, 2.244, 2.87), (1.08, 2.244, 1.08), (0.78, 2.244, 1.08)]
    for i, (start, end) in enumerate(zip(pipe, pipe[1:])):
        rod(f"wall_conduit_{i}", start, end, 0.008, metal)
    for i, z in enumerate((1.33, 1.64, 2.08, 2.55)):
        block(f"conduit_clip_{i}", (1.08, 2.242, z), (0.023, 0.010, 0.012), metal)

    # The source has a small blue box and parked leads below the right bench.
    # A narrow bracket makes its attachment to the timber underside continuous.
    block("bench_box_mount", (1.44, 1.32, 0.851), (0.105, 0.12, 0.004), metal)
    block("bench_service_box", (1.44, 1.32, 0.790), (0.12, 0.15, 0.061), blue)
    rod("bench_cable_gland", (1.46, 1.174, 0.770), (1.46, 1.150, 0.770), 0.014)
    # Static slack route stays beside the retained bin and bench legs. It is a
    # source-informed parked cable, not a simulated deformable or live service.
    lead = [
        (1.46, 1.15, 0.770),
        (1.46, 1.15, 0.68),
        (1.51, 1.14, 0.53),
        (1.47, 1.22, 0.35),
        (1.42, 1.55, 0.12),
        (1.48, 1.75, 0.006),
    ]
    for i, (start, end) in enumerate(zip(lead, lead[1:])):
        rod(f"bench_slack_{i}", start, end, 0.006)
    points = [
        (
            1.48 + 0.14 * math.sin(math.tau * i / 48),
            1.87 - 0.12 * math.cos(math.tau * i / 48),
            0.006,
        )
        for i in range(49)
    ]
    for i, (start, end) in enumerate(zip(points, points[1:])):
        rod(f"floor_lead_loop_{i}", start, end, 0.006)
    lead = [
        points[-1],
        (1.59, 2.02, 0.08),
        (1.19, 2.14, 0.38),
        (0.84, 2.17, 0.70),
        (0.80, 2.175, 1.08),
    ]
    for i, (start, end) in enumerate(zip(lead, lead[1:])):
        rod(f"wall_socket_lead_{i}", start, end, 0.006)
