"""Source-informed FloWave apron, open grating, supports and hall services.

Room dimensions, sections, grate pitch and service routes are estimates. The
visible ring grating and yellow/navy/white structures follow official photographs.
Below-deck supports depict a fixed inspection configuration, not a reconstruction
of the undocumented floor-lifting or pneumatic paddle transmission systems.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

STEEL = (0.50, 0.55, 0.57, 1)
WHITE = (0.82, 0.83, 0.81, 1)
NAVY = (0.075, 0.065, 0.24, 1)
YELLOW = (0.83, 0.70, 0.04, 1)
DARK = (0.07, 0.085, 0.09, 1)
PREFIX = "flowave_context_"
GRATE_INNER = 12.91
GRATE_OUTER = 14.94


class Context:
    def __init__(self, world):
        self.world = world

    def geom(self, name, kind, size, pos=None, color=STEEL, **attrs):
        fields = dict(
            name=PREFIX + name,
            type=kind,
            size=numbers(size),
            rgba=numbers(color),
            contype="1",
            conaffinity="1",
            density="0",
        )
        if pos is not None:
            fields["pos"] = numbers(pos)
        fields.update(
            {
                k: numbers(v) if isinstance(v, (list, tuple)) else str(v)
                for k, v in attrs.items()
            }
        )
        return ET.SubElement(self.world, "geom", fields)

    def box(self, name, pos, half, color=STEEL, **attrs):
        return self.geom(name, "box", half, pos, color, **attrs)

    def rod(self, name, start, end, radius, color=STEEL, **attrs):
        return self.geom(
            name, "capsule", (radius,), color=color, fromto=(*start, *end), **attrs
        )

    def ring(self, name, radius, radial_half, bottom, top, color=STEEL, count=168):
        for i in range(count):
            a = math.tau * (i + 0.5) / count
            self.box(
                f"{name}_{i}",
                (radius * math.cos(a), radius * math.sin(a), (bottom + top) / 2),
                (
                    radial_half,
                    (radius + radial_half) * math.tan(math.pi / count),
                    (top - bottom) / 2,
                ),
                color,
                euler=(0, 0, a),
            )


def _apron(p):
    # The old -2.20 m foundation would cross the published lower shell. The
    # replacement top is -5.20 m, tangent to the concrete shell-floor underside.
    p.box("foundation", (0, 0, -5.30), (16, 16, 0.10), (0.42, 0.45, 0.46, 1))
    # Conservative rectangular tessellation leaves a real 30.2 m pit opening.
    # Solid backfill outside the pit carries the hall and its perimeter cabinets.
    for j in range(128):
        lo, hi = -16 + j * 0.25, -16 + (j + 1) * 0.25
        near = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        edge = math.sqrt(max(0, 15.08**2 - near**2))
        for side in (-1, 1):
            p.box(
                f"apron_{j}_{side}",
                (side * (16 + edge) / 2, (lo + hi) / 2, -2.60),
                ((16 - edge) / 2, 0.125, 2.60),
                (0.50, 0.53, 0.53, 1),
            )


def _grating(p):
    # 2.03 m wide genuine open metal grid. Pitch/sections are estimates from
    # photographs, not manufacturer specifications or a structural load rating.
    for i in range(168 * 8):
        a = math.tau * i / (168 * 8)
        r = (GRATE_INNER + GRATE_OUTER) / 2
        p.box(
            f"grate_radial_{i}",
            (r * math.cos(a), r * math.sin(a), -0.014),
            ((GRATE_OUTER - GRATE_INNER) / 2, 0.0025, 0.014),
            STEEL,
            euler=(0, 0, a),
        )
    for j in range(35):
        r = GRATE_INNER + j * (GRATE_OUTER - GRATE_INNER) / 34
        p.ring(f"grate_cross_{j}", r, 0.0025, -0.028, 0)
    for j, r in enumerate((12.94, 13.70, 14.90)):
        p.ring(f"grate_ring_beam_{j}", r, 0.035, -0.17, -0.028, NAVY)
    for i in range(28):
        a = math.tau * i / 28
        c, s = math.cos(a), math.sin(a)
        p.box(
            f"grate_radial_beam_{i}",
            (13.91 * c, 13.91 * s, -0.11),
            (1.04, 0.035, 0.082),
            NAVY,
            euler=(0, 0, a),
        )
        for j, r in enumerate((13.25, 14.80)):
            p.box(
                f"grate_column_{i}_{j}",
                (r * c, r * s, -2.568),
                (0.065, 0.065, 2.46),
                NAVY,
                euler=(0, 0, a),
            )
            p.box(
                f"grate_foot_{i}_{j}",
                (r * c, r * s, -4.97),
                (0.14, 0.14, 0.03),
                NAVY,
                euler=(0, 0, a),
            )
    # Continuous rim pipes sit on explicit feet reaching the grating bars.
    for i in range(168):
        a, b = math.tau * i / 168, math.tau * (i + 1) / 168
        for j, (r, z, radius, color) in enumerate(
            ((13.00, 0.090, 0.023, YELLOW), (12.94, 0.055, 0.014, STEEL))
        ):
            p.rod(
                f"rim_pipe_{i}_{j}",
                (r * math.cos(a), r * math.sin(a), z),
                (r * math.cos(b), r * math.sin(b), z),
                radius,
                color,
            )
        r = 13.00
        p.box(
            f"pipe_mount_{i}",
            (r * math.cos(a), r * math.sin(a), 0.035),
            (0.095, 0.025, 0.035),
            YELLOW,
            euler=(0, 0, a),
        )


def _services(p):
    # Source-visible stationary pressure vessels and blue/yellow service routes.
    # They are closed exterior context; no pneumatic force path is fabricated.
    p.ring("service_shelf", 14.20, 0.35, -1.58, -1.53, NAVY)
    for i in range(28):
        a = math.tau * i / 28
        c, s = math.cos(a), math.sin(a)
        p.box(
            f"shelf_bracket_{i}",
            (14.39 * c, 14.39 * s, -1.60),
            (0.42, 0.05, 0.07),
            NAVY,
            euler=(0, 0, a),
        )
    for i in range(168):
        a = math.tau * i / 168
        c, s = math.cos(a), math.sin(a)
        r = 14.2
        p.geom(f"pressure_vessel_{i}", "cylinder", (0.11, 0.23), (r * c, r * s, -1.30))
        p.geom(
            f"vessel_top_{i}", "ellipsoid", (0.11, 0.11, 0.065), (r * c, r * s, -1.07)
        )
        p.rod(f"vessel_neck_{i}", (r * c, r * s, -1.05), (r * c, r * s, -0.92), 0.025)
        p.rod(
            f"vessel_branch_{i}",
            (r * c, r * s, -0.94),
            ((r + 0.20) * c, (r + 0.20) * s, -0.94),
            0.022,
        )
        b = math.tau * (i + 1) / 168
        p.rod(
            f"service_header_{i}",
            ((r + 0.20) * c, (r + 0.20) * s, -0.94),
            ((r + 0.20) * math.cos(b), (r + 0.20) * math.sin(b), -0.94),
            0.029,
            (0.18, 0.22, 0.42, 1),
        )
        # Frame braces stay behind the entire ±.12 rad moving-flap envelope.
        p.rod(
            f"rear_frame_{i}",
            (13.03 * c, 13.03 * s, -2.39),
            (13.70 * c, 13.70 * s, -0.17),
            0.016,
            STEEL,
        )


def _hall(p):
    # Source shows navy peripheral lattice, white cabinets and a yellow crane.
    # Leave the original front camera azimuths and door approaches unobstructed.
    for i in range(56):
        a, b = math.pi * i / 56, math.pi * (i + 1) / 56
        r = 14.12
        for z in (0.22, 1.32):
            p.rod(
                f"perimeter_chord_{i}_{z}",
                (r * math.cos(a), r * math.sin(a), z),
                (r * math.cos(b), r * math.sin(b), z),
                0.050,
                NAVY,
            )
        p.rod(
            f"perimeter_diagonal_{i}",
            (r * math.cos(a), r * math.sin(a), 0.22),
            (r * math.cos(b), r * math.sin(b), 1.32),
            0.040,
            NAVY,
        )
        p.rod(
            f"perimeter_post_{i}",
            (r * math.cos(a), r * math.sin(a), -0.10),
            (r * math.cos(a), r * math.sin(a), 1.32),
            0.043,
            NAVY,
        )
        # This foot overlaps a grating cross/radial junction and radial support.
        p.box(
            f"perimeter_foot_{i}",
            (r * math.cos(a), r * math.sin(a), 0.012),
            (0.13, 0.10, 0.020),
            NAVY,
            euler=(0, 0, a),
        )
    for x in (-15.35, 15.35):
        p.box(f"crane_runway_{x}", (x, 0, 7.89), (0.16, 15.5, 0.22), WHITE)
        for y in (-11, -3, 5, 12):
            p.box(f"runway_column_{x}_{y}", (x, y, 3.865), (0.15, 0.18, 3.865), WHITE)
            p.box(f"runway_foot_{x}_{y}", (x, y, 0.06), (0.24, 0.28, 0.06))
    # End trucks touch runway tops; bridge is supported between those trucks.
    for x in (-15.35, 15.35):
        p.box(f"crane_truck_{x}", (x, 8.8, 8.16), (0.35, 0.57, 0.08), DARK)
    p.box("yellow_bridge_crane", (0, 8.8, 8.29), (15.60, 0.33, 0.14), YELLOW)
    p.box("crane_trolley", (2.7, 8.8, 8.08), (0.65, 0.42, 0.12))
    p.rod("crane_cable", (2.7, 8.8, 7.98), (2.7, 8.8, 4.60), 0.022, DARK)
    p.box("crane_hook_block", (2.7, 8.8, 4.47), (0.16, 0.08, 0.14), YELLOW)
    for y in (-11, -3, 5, 12):
        for z in (8.14, 8.75):
            g = p.rod(f"roof_chord_{y}_{z}", (-15.5, y, z), (15.5, y, z), 0.07, WHITE)
            g.set("name", "arch_roof_" + g.get("name"))
        for i, x in enumerate((-15, -10, -5, 0, 5, 10)):
            for j, (a, b) in enumerate(
                (
                    ((x, y, 8.14), (x + 2.5, y, 8.75)),
                    ((x + 2.5, y, 8.75), (x + 5, y, 8.14)),
                )
            ):
                g = p.rod(f"roof_diagonal_{y}_{i}_{j}", a, b, 0.055, WHITE)
                g.set("name", "arch_roof_" + g.get("name"))
    for i, x in enumerate((-13.7, -10.2, -6.7, -3.2, 0.3, 3.8, 7.3, 10.8)):
        p.box(f"control_cabinet_{i}", (x, 15.3, 1.04), (0.56, 0.30, 1.04), WHITE)
        p.box(f"cabinet_divider_{i}", (x, 14.989, 1.12), (0.004, 0.016, 0.90))
        p.box(f"cabinet_panel_{i}", (x + 0.22, 14.972, 1.62), (0.12, 0.031, 0.14), DARK)
        p.box(
            f"cabinet_handle_{i}", (x - 0.11, 14.951, 1.07), (0.015, 0.035, 0.10), DARK
        )


def add_hall(world, definition):
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_floor"):
            world.remove(geom)
    p = Context(world)
    _apron(p)
    _grating(p)
    _services(p)
    _hall(p)
