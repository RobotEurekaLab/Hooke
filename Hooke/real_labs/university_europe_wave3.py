"""Four distinct university workstations with bounded mechanical previews.

Geometry is authored from official installed-apparatus photographs. Hidden
internals, proportions and control travel are estimates, not manufacturer CAD.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

EPFL = "https://www.epfl.ch/labs/lspn/group/lab-tour/"
KTH = "https://www.kth.se/mmk/mechatronics/laboratories/robot-design-lab"
MANCHESTER = "https://www.scieng.manchester.ac.uk/tomorrowlabs/high-voltage-lab/"
BIRMINGHAM = "https://www.birmingham.ac.uk/research/centres-institutes/human-brain-health/faciities/magnetic-resonance-imaging"

WHITE = (0.83, 0.85, 0.83, 1)
STEEL = (0.44, 0.48, 0.49, 1)
DARK = (0.08, 0.10, 0.11, 1)
YELLOW = (0.95, 0.67, 0.05, 1)


class _Room:
    """Small named architectural details, independent of instrument controls."""

    def __init__(self, world, name):
        self.world, self.name, self.count = world, name, 0

    def geom(self, key, shape, pos, size, color, *, collision=False, **extra):
        self.count += 1
        attrs = dict(
            name=f"europe3_{self.name}_{key}_{self.count}",
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
                k: numbers(v) if isinstance(v, (list, tuple)) else str(v)
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

    def cylinder(self, key, pos, radius, half, color, **extra):
        return self.geom(key, "cylinder", pos, (radius, half), color, **extra)


def _synthesis_hood(b, base, params):
    """Service-rich hood with sash access and a supported inert flask lift."""
    b.box(base, (0, 0, 0.07), (0.85, 0.43, 0.07), "dark", collision=True)
    b.box(base, (0, 0.02, 0.46), (0.84, 0.40, 0.32), "shell", collision=True)
    for x in (-0.42, 0.42):
        b.box(base, (x, -0.392, 0.45), (0.40, 0.017, 0.29), "cream")
        b.rod(base, (x - 0.10, -0.425, 0.63), (x + 0.10, -0.425, 0.63), 0.009, "bright")
    b.box(base, (0, -0.01, 0.83), (0.87, 0.46, 0.06), "shell", collision=True)
    b.box(base, (0, 0.0, 0.91), (0.86, 0.44, 0.024), "dark", collision=True)
    # Back and jambs leave a real open work volume; no solid collision box.
    b.box(base, (0, 0.435, 1.70), (0.87, 0.035, 0.79), "cream", collision=True)
    for x in (-0.84, 0.84):
        b.box(base, (x, 0.01, 1.72), (0.027, 0.42, 0.80), "shell", collision=True)
        b.box(base, (x, -0.46, 1.78), (0.025, 0.025, 0.81), "metal", collision=True)
    b.box(base, (0, 0.015, 2.53), (0.87, 0.43, 0.065), "shell", collision=True)
    b.vents(base, (0, -0.43, 2.48), 1.30, 5)
    # The glass slides outside the enclosure; its full stroke clears the roof.
    sash = b.moving(
        base, "sash_height", (0, -0.50, 1.64), (0, 0, 1), (0, 0.48), mass=1.4
    )
    b.box(sash, (0, 0, 0), (0.793, 0.006, 0.35), "glass", collision=True)
    for z in (-0.36, 0.36):
        b.box(sash, (0, 0, z), (0.81, 0.023, 0.015), "bright", collision=True)
    for x in (-0.80, 0.80):
        b.box(sash, (x, 0, 0), (0.012, 0.018, 0.35), "bright", collision=True)
    b.rod(sash, (-0.23, -0.045, -0.34), (0.23, -0.045, -0.34), 0.014, "metal")
    # Service valves, sockets and restrained hoses follow the photographed fascia.
    for i in range(7):
        x = -0.66 + i * 0.22
        b.cylinder(
            base, (x, -0.481, 0.84), 0.024, 0.013, "bright", euler=(math.pi / 2, 0, 0)
        )
        b.box(
            base, (x, -0.507, 0.84), (0.009, 0.013, 0.025), "cyan" if i % 2 else "blue"
        )
        if i in (1, 4):
            b.box(base, (x, -0.482, 0.73), (0.026, 0.006, 0.025), "dark")
    for x in (-0.60, -0.28, 0.16, 0.59):
        b.rod(base, (x, 0.30, 0.97), (x, 0.30, 2.30), 0.008, "bright")
    for z in (1.24, 1.72, 2.13):
        b.rod(base, (-0.65, 0.30, z), (0.65, 0.30, z), 0.008, "bright")
    for x in (-0.48, 0.10):
        b.rod(base, (x, 0.30, 1.74), (x, -0.07, 1.74), 0.006, "metal")
        b.ring(base, (x, -0.07, 1.74), 0.042, 0.004, "metal")
    # Authored linear laboratory lift: sleeves and rods remain connected at all heights.
    b.box(base, (-0.28, -0.02, 0.97), (0.18, 0.16, 0.03), "metal", collision=True)
    for x in (-0.40, -0.16):
        b.cylinder(base, (x, -0.02, 1.03), 0.029, 0.05, "dark", collision=True)
    lift = b.moving(
        base, "flask_lift", (-0.28, -0.02, 1.11), (0, 0, 1), (0, 0.13), mass=0.7
    )
    for x in (-0.12, 0.12):
        b.cylinder(lift, (x, 0, -0.095), 0.012, 0.105, "bright")
    b.box(lift, (0, 0, 0), (0.18, 0.16, 0.019), "metal", collision=True)
    b.housing(
        lift, ((0.022, 0.15, 0.135, 0), (0.087, 0.14, 0.115, 0.01)), material="blue"
    )
    b.box(lift, (0, 0, 0.056), (0.14, 0.105, 0.030), "blue", collision=True)
    for x in (-0.066, 0.065):
        b.cylinder(
            lift, (x, -0.129, 0.059), 0.018, 0.008, "cream", euler=(math.pi / 2, 0, 0)
        )
    b.cylinder(lift, (0, 0.015, 0.104), 0.095, 0.016, "bright", collision=True)
    b.geom(lift, "sphere", (0.077,), (0, 0.015, 0.189), "glass", collision=True)
    b.cylinder(lift, (0, 0.015, 0.277), 0.022, 0.045, "glass", collision=True)
    b.cylinder(lift, (0, 0.015, 0.323), 0.027, 0.011, "copper")
    b.geom(lift, "sphere", (0.052,), (0, 0.015, 0.166), "copper")
    b.site(lift, "inert_flask", (0, 0.015, 0.189), size=0.008)
    # Distinct visible supplies remain inert props, without chemical recipes.
    for i in range(4):
        x = 0.30 + i * 0.095
        b.cylinder(base, (x, -0.17, 0.977), 0.023, 0.039, "glass")
        b.cylinder(base, (x, -0.17, 1.022), 0.025, 0.008, "blue")
    for x in (0.30, 0.58):
        b.cylinder(base, (x, 0.16, 1.018), 0.04, 0.075, "cream")
        b.rod(base, (x, 0.16, 1.095), (x - 0.05, 0.16, 1.145), 0.009, "cream")
    for x in (-0.68, 0.68):
        b.rod(base, (x, 0.37, 1.51), (x + 0.035, 0.25, 1.32), 0.011, "rubber")
        b.rod(base, (x + 0.035, 0.25, 1.32), (x, 0.18, 1.08), 0.011, "rubber")
    b.metadata["capabilities"] = [
        "Vertical sash access",
        "Retained inert flask elevation",
    ]
    b.metadata["limitations"].append(
        "The photographed EPFL bay is historical 2019. Hood size, sash stroke and "
        "guided lifting mechanism are estimates; the latter substitutes for the "
        "observed laboratory jack. No extraction airflow, vacuum, heating or chemistry."
    )


def _epfl_bay(world, definition):
    p = _Room(world, "epfl")
    # Reference supports a two-hood bay; the surrounding room is a minimal backdrop.
    for x in (-1.0, 1.0):
        p.box("rear_utility_trunk", (x, 1.54, 1.1), (0.89, 0.06, 0.05), STEEL)
    for x in (-0.75, 1.15):
        p.cylinder("yellow_stool_seat", (x, -0.42, 0.57), 0.19, 0.035, YELLOW)
        p.cylinder("stool_column", (x, -0.42, 0.31), 0.025, 0.22, STEEL)
        for angle in (0, 2 * math.pi / 3, 4 * math.pi / 3):
            p.rod(
                "stool_foot",
                (x, -0.42, 0.14),
                (x + 0.19 * math.cos(angle), -0.42 + 0.19 * math.sin(angle), 0.025),
                0.018,
                STEEL,
            )


def _prototyping_printer(b, base, params):
    """Enclosed desktop printer, with dry positioning of a retained rigid coupon."""
    for x in (-0.36, 0.36):
        for y in (-0.30, 0.30):
            b.box(base, (x, y, 0.39), (0.026, 0.026, 0.39), "metal", collision=True)
            b.box(base, (x, y, 0.025), (0.033, 0.033, 0.025), "rubber")
    b.box(base, (0, 0, 0.80), (0.45, 0.38, 0.025), "cream", collision=True)
    b.box(base, (0, 0, 0.830), (0.44, 0.37, 0.005), "plant")
    # Continuous steel chassis and clear panels leave the moving volume empty.
    b.box(base, (0, 0, 0.875), (0.26, 0.27, 0.039), "dark", collision=True)
    for x in (-0.248, 0.248):
        for y in (-0.256, 0.256):
            b.box(base, (x, y, 1.205), (0.012, 0.014, 0.29), "dark", collision=True)
    for y in (-0.252, 0.252):
        b.box(base, (0, y, 1.50), (0.26, 0.019, 0.014), "dark", collision=True)
    for x in (-0.252, 0.252):
        b.box(base, (x, 0, 1.50), (0.008, 0.245, 0.014), "dark", collision=True)
        b.box(base, (x, 0, 1.21), (0.004, 0.234, 0.28), "glass", collision=True)
    b.box(base, (0, 0.257, 1.21), (0.236, 0.004, 0.279), "metal", collision=True)
    b.box(base, (0, -0.273, 1.22), (0.23, 0.004, 0.265), "glass", collision=True)
    b.rod(base, (0.196, -0.291, 1.12), (0.196, -0.291, 1.25), 0.009, "metal")
    b.box(base, (0, 0, 1.522), (0.247, 0.253, 0.004), "glass")
    for x in (-0.209, 0.209):
        b.rod(base, (x, 0.190, 0.925), (x, 0.190, 1.45), 0.006, "bright")
    bed = b.moving(
        base, "build_plate_z", (0, -0.018, 1.00), (0, 0, 1), (0, 0.20), mass=0.5
    )
    b.box(bed, (0, 0.035, -0.018), (0.195, 0.173, 0.012), "metal", collision=True)
    b.box(bed, (0, 0, 0), (0.188, 0.18, 0.005), "dark", collision=True)
    for x in (-0.17, 0.17):
        b.rod(bed, (x, 0.145, -0.019), (x, 0.19, -0.06), 0.009, "metal")
    # A bolted rigid bracket demonstrates transport; there is no generated print.
    b.box(bed, (0, -0.025, 0.023), (0.074, 0.045, 0.018), "orange", collision=True)
    for x in (-0.064, 0.064):
        b.box(bed, (x, -0.025, 0.063), (0.011, 0.045, 0.023), "orange", collision=True)
        b.cylinder(bed, (x, -0.025, 0.091), 0.006, 0.004, "bright")
    b.site(bed, "prototype_coupon", (0, -0.025, 0.023), size=0.005)
    for y in (-0.083, 0.08):
        b.rod(base, (-0.227, y, 1.416), (0.227, y, 1.416), 0.006, "bright")
    head = b.moving(
        base, "toolhead_x", (0, 0, 1.409), (1, 0, 0), (-0.13, 0.13), mass=0.15
    )
    b.box(head, (0, 0, 0), (0.046, 0.060, 0.032), "dark", collision=True)
    b.cylinder(head, (0, -0.069, 0), 0.021, 0.009, "metal", euler=(math.pi / 2, 0, 0))
    b.cylinder(head, (0, 0, -0.046), 0.011, 0.014, "bright", collision=True)
    b.geom(
        head,
        "ellipsoid",
        (0.008, 0.008, 0.009),
        (0, 0, -0.067),
        "copper",
        collision=True,
    )
    # Top spool carriage and filament guides are passive source-inspired details.
    b.box(base, (0, 0.04, 1.568), (0.22, 0.20, 0.032), "dark")
    for x in (-0.145, 0, 0.145):
        b.cylinder(
            base, (x, 0.06, 1.692), 0.088, 0.025, "cream", euler=(0, math.pi / 2, 0)
        )
        for dx in (-0.028, 0.028):
            b.cylinder(
                base,
                (x + dx, 0.06, 1.692),
                0.094,
                0.003,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
        b.rod(base, (x, 0.06, 1.692), (x, 0.06, 1.59), 0.010, "metal")
    b.box(base, (0, -0.209, 1.693), (0.226, 0.006, 0.092), "glass")
    b.screen(base, (-0.12, -0.284, 1.482), width=0.11, height=0.065)
    b.metadata["capabilities"] = [
        "Dry build-plate positioning",
        "Dry toolhead traverse",
    ]
    b.metadata["limitations"].append(
        "Source establishes a row of enclosed desktop printers, not verified OEM "
        "dimensions or kinematics. This original surrogate has two positioning axes "
        "and a retained rigid coupon; no extrusion, heating, material deposition or slicing."
    )


def _kth_workroom(world, definition):
    p = _Room(world, "kth")
    wood = (0.61, 0.48, 0.29, 1)
    green = (0.11, 0.35, 0.30, 1)
    # Tool wall and parts storage remain separate from the window-side printer row.
    p.box("tool_bench", (-2.70, 0.70, 0.85), (0.38, 1.48, 0.035), wood, collision=True)
    p.box("tool_cutting_mat", (-2.70, 0.70, 0.89), (0.35, 1.42, 0.006), green)
    for x in (-2.98, -2.41):
        for y in (-0.61, 2.01):
            p.box(
                "tool_bench_leg",
                (x, y, 0.408),
                (0.025, 0.025, 0.408),
                STEEL,
                collision=True,
            )
    p.box("pegboard", (-3.15, 0.70, 1.45), (0.018, 1.48, 0.44), (0.64, 0.68, 0.70, 1))
    for y in (-0.65, -0.35, -0.05, 0.25, 0.55, 0.85, 1.15, 1.45, 1.75, 2.05):
        for z in (1.12, 1.26, 1.40, 1.54, 1.68, 1.82):
            p.box("pegboard_hole", (-3.129, y, z), (0.0015, 0.006, 0.006), DARK)
        p.rod("hanging_tool_shaft", (-3.10, y, 1.32), (-3.10, y, 1.56), 0.007, STEEL)
        p.rod(
            "hanging_tool_handle",
            (-3.10, y, 1.30),
            (-3.10, y, 1.40),
            0.016,
            (0.84, 0.28, 0.08, 1) if y < 0.5 else (0.11, 0.32, 0.52, 1),
        )
    for z in (1.96, 2.30):
        p.box("parts_shelf", (-3.02, 0.70, z), (0.20, 1.49, 0.018), wood)
        for y in (-0.52, -0.12, 0.28, 0.68, 1.08, 1.48, 1.88):
            p.box(
                "parts_bin",
                (-3.00, y, z + 0.115),
                (0.15, 0.17, 0.095),
                (0.14, 0.21, 0.30, 1),
            )
            p.box("bin_label", (-2.846, y, z + 0.11), (0.001, 0.074, 0.035), WHITE)
    # Actual prototyping fixtures make this an assembly room rather than an office.
    p.box(
        "assembly_top", (0.15, -0.87, 0.85), (1.23, 0.45, 0.035), wood, collision=True
    )
    p.box("assembly_mat", (0.15, -0.87, 0.89), (1.19, 0.42, 0.006), green)
    for x in (-0.93, 1.23):
        for y in (-1.18, -0.57):
            p.box(
                "assembly_leg",
                (x, y, 0.408),
                (0.029, 0.029, 0.408),
                STEEL,
                collision=True,
            )
    for x in (-0.50, 0.02):
        p.box("fixture_extrusion", (x, -0.91, 0.932), (0.020, 0.26, 0.024), STEEL)
        p.box("extrusion_slot", (x, -0.91, 0.957), (0.008, 0.24, 0.002), DARK)
    for y in (-1.12, -0.70):
        p.box("fixture_crossrail", (-0.24, y, 0.931), (0.27, 0.020, 0.024), STEEL)
    for x in (-0.53, 0.05):
        p.box(
            "fixture_clamp",
            (x, -1.22, 0.92),
            (0.028, 0.065, 0.037),
            (0.08, 0.18, 0.45, 1),
        )
        p.rod("clamp_screw", (x, -1.22, 0.84), (x, -1.22, 1.005), 0.009, STEEL)
    for x, y in ((0.49, -0.87), (0.85, -0.91)):
        p.box(
            "inert_circuit_board",
            (x, y, 0.91),
            (0.115, 0.065, 0.012),
            (0.10, 0.39, 0.18, 1),
        )
        for dx in (-0.055, 0.02, 0.07):
            p.box("board_component", (x + dx, y, 0.929), (0.02, 0.025, 0.007), DARK)
    # Screen arm and a bench magnifier are mounted, not suspended decorations.
    p.rod("monitor_post", (0.80, -0.54, 0.90), (0.80, -0.54, 1.27), 0.018, STEEL)
    p.rod("monitor_arm", (0.80, -0.54, 1.27), (0.55, -0.54, 1.34), 0.014, STEEL)
    p.box("monitor", (0.55, -0.54, 1.39), (0.27, 0.027, 0.16), DARK)
    p.box(
        "monitor_screen",
        (0.55, -0.57, 1.39),
        (0.247, 0.001, 0.136),
        (0.04, 0.12, 0.17, 1),
    )
    p.box("task_lamp_base", (-0.85, -0.64, 0.91), (0.10, 0.08, 0.015), WHITE)
    p.rod("task_lamp_arm", (-0.85, -0.64, 0.92), (-0.68, -0.74, 1.27), 0.015, WHITE)
    p.cylinder("task_lamp_head", (-0.68, -0.74, 1.27), 0.08, 0.018, WHITE)


def _annular_shell(b, parent, center, outer, inner, half, material, segments=48):
    """Original hollow ring mesh; segments retain an actual central opening."""
    vertices, faces = [], []
    for z in (-half, half):
        for radius in (outer, inner):
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                vertices.append(
                    (
                        center[0] + radius * math.cos(angle),
                        center[1] + radius * math.sin(angle),
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
    mesh = b.unique("annular_shell")
    ET.SubElement(
        b.root.find("asset"),
        "mesh",
        name=mesh,
        vertex=numbers([x for v in vertices for x in v]),
        face=" ".join(str(x) for f in faces for x in f),
    )
    b.geom(parent, "mesh", (), material=material, mesh=mesh)


def _dielectric_cell(b, base, params):
    """Unpowered reference-informed vessel with supported service mechanisms."""
    for x in (-0.43, 0.43):
        for y in (-0.37, 0.37):
            b.box(base, (x, y, 0.96), (0.027, 0.027, 0.94), "dark", collision=True)
            b.box(base, (x, y, 0.025), (0.07, 0.06, 0.025), "metal", collision=True)
    for z in (0.18, 0.66, 1.91):
        for y in (-0.37, 0.37):
            b.box(base, (0, y, z), (0.43, 0.024, 0.022), "dark", collision=True)
        for x in (-0.43, 0.43):
            b.box(base, (x, 0, z), (0.024, 0.37, 0.022), "dark", collision=True)
    b.box(base, (0, 0, 0.69), (0.385, 0.32, 0.025), "metal", collision=True)
    b.cylinder(base, (0, 0, 0.729), 0.327, 0.014, "bright", collision=True)
    # Transparent cylinder is hollow: radial panels provide the physical wall.
    _annular_shell(b, base, (0, 0, 1.046), 0.315, 0.306, 0.298, "glass")
    for i in range(40):
        a = 2 * math.pi * i / 40
        b.box(
            base,
            (0.312 * math.cos(a), 0.312 * math.sin(a), 1.046),
            (0.0045, 0.026, 0.298),
            "glass",
            collision=True,
            euler=(0, 0, a),
            rgba=(0.4, 0.7, 0.72, 0),
        )
    b.ring(base, (0, 0, 0.752), 0.317, 0.009, "bright")
    b.ring(base, (0, 0, 1.347), 0.320, 0.009, "bright")
    # Static appearance of dark insulating liquid only, no dielectric/fluid solver.
    b.cylinder(base, (0, 0, 0.941), 0.301, 0.19, "lens", rgba=(0.55, 0.48, 0.28, 0.08))
    b.ring(base, (0, 0, 1.132), 0.300, 0.003, "copper")
    for x in (-0.39, 0.39):
        b.rod(base, (x, 0.0, 1.30), (x, 0.0, 1.83), 0.009, "bright")
    lid = b.moving(
        base, "service_lid_lift", (0, 0, 1.381), (0, 0, 1), (0, 0.13), mass=1.0
    )
    _annular_shell(b, lid, (0, 0, 0), 0.345, 0.036, 0.015, "glass")
    b.ring(lid, (0, 0, 0), 0.340, 0.010, "bright")
    for i in range(16):
        a = 2 * math.pi * i / 16
        b.box(
            lid,
            (0.19 * math.cos(a), 0.19 * math.sin(a), 0),
            (0.147, 0.035, 0.012),
            "glass",
            collision=True,
            euler=(0, 0, a),
            rgba=(0.4, 0.7, 0.72, 0),
        )
    for i in range(8):
        a = 2 * math.pi * i / 8
        b.cylinder(
            lid,
            (0.326 * math.cos(a), 0.326 * math.sin(a), 0.024),
            0.009,
            0.014,
            "bright",
        )
    for x in (-0.39, 0.39):
        b.box(lid, (x, 0, 0), (0.027, 0.05, 0.045), "metal")
        b.box(lid, (x * 0.84, 0, 0), (0.04, 0.02, 0.014), "bright")
    # Bright yellow central fitting and a real bore for the supported insert rod.
    _annular_shell(b, lid, (0, 0, 0.078), 0.068, 0.027, 0.06, "warning")
    _annular_shell(b, lid, (0, 0, 0.149), 0.047, 0.025, 0.013, "warning")
    insert = b.moving(
        lid, "specimen_height", (0, 0, 0.30), (0, 0, 1), (-0.05, 0.07), mass=0.35
    )
    b.cylinder(insert, (0, 0, -0.323), 0.015, 0.354, "bright", collision=True)
    b.cylinder(insert, (0, 0, 0.045), 0.031, 0.015, "metal", collision=True)
    b.cylinder(insert, (0, 0, -0.688), 0.125, 0.019, "bright", collision=True)
    b.cylinder(insert, (0, 0, -0.731), 0.061, 0.024, "metal", collision=True)
    # Retained inert material coupon, not an energized electrode model.
    b.rod(
        insert, (0, -0.07, -0.731), (0, -0.222, -0.731), 0.008, "bright", collision=True
    )
    b.box(insert, (0, -0.225, -0.738), (0.035, 0.008, 0.025), "cream", collision=True)
    for x in (-0.045, 0.045):
        b.box(insert, (x, -0.226, -0.737), (0.011, 0.008, 0.026), "copper")
    b.site(insert, "inert_dielectric_coupon", (0, -0.225, -0.738), size=0.004)
    b.metadata["capabilities"] = [
        "Supported service-lid elevation",
        "Retained inert coupon positioning",
    ]
    b.metadata["limitations"].append(
        "The official photograph shows the cropped vessel, lid, yellow fitting and "
        "surrounding black frame. Floor stand, travel, lifting hardware and specimen "
        "attachment are authored estimates. All geometry is unpowered: no electric "
        "field, dielectric breakdown, voltage source, fluid flow, heating or experimental procedure."
    )


def _manchester_workstation(world, definition):
    p = _Room(world, "manchester")
    # A cropped workcell backdrop makes no claim about the wider HV hall.
    p.box("local_dark_background", (0, 0.87, 1.07), (0.67, 0.025, 0.78), DARK)
    for x in (-0.64, 0.64):
        p.box("background_support", (x, 0.87, 0.53), (0.018, 0.025, 0.53), STEEL)
    p.box(
        "inert_service_box",
        (0.87, 0.53, 0.20),
        (0.19, 0.22, 0.20),
        WHITE,
        collision=True,
    )
    p.box("service_cover", (0.87, 0.295, 0.20), (0.16, 0.01, 0.15), STEEL)
    for z in (0.12, 0.17, 0.22, 0.27):
        p.box("service_slot", (0.87, 0.283, z), (0.10, 0.004, 0.007), DARK)
    p.box("operator_mat", (0, -0.91, 0.008), (0.64, 0.36, 0.008), (0.19, 0.21, 0.20, 1))


def _mri_phantom_station(b, base, params):
    """Hollow MRI-shaped shell and powered table carrying an inert phantom."""
    b.box(base, (0, 0.76, 0.09), (0.87, 0.73, 0.09), "dark", collision=True)
    # The source has a large circular face with a silver rim. Keep the bore hollow.
    tunnel = ET.SubElement(
        base,
        "body",
        name=b.unique("hollow_scanner"),
        pos="0 .74 1.12",
        euler=f"{math.pi/2} 0 0",
    )
    _annular_shell(b, tunnel, (0, 0, 0), 1.02, 0.365, 0.60, "shell", segments=72)
    _annular_shell(b, tunnel, (0, 0, 0.625), 1.049, 0.960, 0.024, "metal", segments=72)
    _annular_shell(b, tunnel, (0, 0, 0.655), 0.960, 0.360, 0.018, "shell", segments=72)
    _annular_shell(b, tunnel, (0, 0, 0.01), 0.365, 0.351, 0.605, "cream", segments=72)
    # Radial solid sectors approximate the shell without filling the tunnel.
    for i in range(64):
        a = 2 * math.pi * i / 64
        b.box(
            tunnel,
            (0.688 * math.cos(a), 0.688 * math.sin(a), 0),
            (0.323, 0.036, 0.598),
            "shell",
            collision=True,
            euler=(0, 0, a),
            rgba=(0.86, 0.88, 0.88, 0),
        )
    for x in (-0.74, 0.74):
        b.box(base, (x, 0.051, 1.20), (0.075, 0.011, 0.026), "metal")
        for dx in (-0.035, 0, 0.035):
            b.cylinder(
                base,
                (x + dx, 0.032, 1.20),
                0.008,
                0.004,
                "cream",
                euler=(math.pi / 2, 0, 0),
            )
    b.screen(base, (0, 0.043, 1.80), width=0.18, height=0.12)
    # Original table foundation: visible wheels, foot casting and telescopic lift.
    b.housing(base, ((0.09, 0.37, 0.54, -1.48), (0.22, 0.32, 0.48, -1.46)), radius=0.06)
    b.box(base, (0, -1.46, 0.15), (0.31, 0.45, 0.06), "shell", collision=True)
    for x in (-0.31, 0.31):
        for y in (-1.85, -1.10):
            b.cylinder(
                base,
                (x, y, 0.071),
                0.07,
                0.029,
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
            b.cylinder(
                base,
                (x * 1.06, y, 0.071),
                0.037,
                0.011,
                "bright",
                euler=(0, math.pi / 2, 0),
            )
    b.housing(
        base, ((0.20, 0.22, 0.28, -1.48), (0.64, 0.19, 0.24, -1.44)), radius=0.035
    )
    b.box(base, (0, -1.46, 0.43), (0.18, 0.22, 0.21), "shell", collision=True)
    lift = b.moving(
        base, "table_height", (0, 0, 0.72), (0, 0, 1), (0, 0.06), mass=1.5, kp=300
    )
    b.box(lift, (0, -1.46, -0.065), (0.151, 0.19, 0.095), "metal")
    b.box(lift, (0, -1.27, 0.0), (0.28, 0.66, 0.026), "shell", collision=True)
    for x in (-0.21, 0.21):
        b.rod(lift, (x, -1.86, 0.065), (x, -0.18, 0.065), 0.014, "bright")
        b.box(lift, (x, -1.03, 0.030), (0.018, 0.84, 0.018), "metal", collision=True)
    carriage = b.moving(
        lift, "table_insert", (0, -0.92, 0.15), (0, 1, 0), (0, 0.85), mass=1.2, kp=300
    )
    b.box(carriage, (0, -0.37, -0.055), (0.20, 0.24, 0.024), "metal", collision=True)
    b.box(carriage, (0, 0, 0), (0.20, 1.11, 0.025), "shell", collision=True)
    for x in (-0.196, 0.196):
        b.rod(carriage, (x, -1.06, 0.027), (x, 0.99, 0.027), 0.012, "bright")
    b.box(carriage, (0, -0.21, 0.049), (0.18, 0.76, 0.022), "rubber", collision=True)
    # Visible inert water-equivalent-looking phantom in a restrained cradle.
    b.box(carriage, (0, 0.38, 0.050), (0.15, 0.19, 0.024), "cream", collision=True)
    for x in (-0.127, 0.127):
        b.box(carriage, (x, 0.38, 0.106), (0.021, 0.19, 0.055), "shell", collision=True)
    b.cylinder(
        carriage,
        (0, 0.38, 0.150),
        0.093,
        0.147,
        "cyan",
        euler=(math.pi / 2, 0, 0),
        collision=True,
    )
    for y in (0.224, 0.536):
        b.cylinder(
            carriage, (0, y, 0.150), 0.096, 0.006, "cream", euler=(math.pi / 2, 0, 0)
        )
    for y in (0.27, 0.49):
        # Straight retaining straps make attachment visible; not a human head coil.
        b.box(carriage, (0, y, 0.249), (0.14, 0.012, 0.005), "cream")
    b.site(carriage, "inert_mri_phantom", (0, 0.38, 0.150), size=0.009)
    b.metadata["capabilities"] = [
        "Supported table elevation",
        "Retained inert phantom insertion",
    ]
    b.metadata["limitations"].append(
        "Birmingham identifies a Siemens MAGNETOM Prisma, but this geometry is an "
        "original exterior-informed surrogate, not OEM CAD or an approved scanner model. "
        "Bore size, table mechanism and motion ranges are estimated. Only inert phantom "
        "positioning is modeled: no patients, diagnosis, MR image synthesis, magnetic/RF "
        "fields, pulse sequences, shielding or clinical operating procedure."
    )


def _mri_room(world, definition):
    p = _Room(world, "mri")
    # The reference supports one recessed storage wall, not a complete control suite.
    p.box(
        "storage_back", (-1.86, 0.72, 1.20), (0.038, 0.57, 1.12), WHITE, collision=True
    )
    for y in (0.15, 1.29):
        p.box(
            "storage_jamb", (-1.58, y, 1.20), (0.29, 0.023, 1.12), WHITE, collision=True
        )
    for z in (0.12, 0.63, 1.14, 1.65, 2.23):
        p.box(
            "storage_shelf",
            (-1.57, 0.72, z),
            (0.30, 0.57, 0.018),
            WHITE,
            collision=True,
        )
    for z in (0.74, 1.24, 1.74):
        for y in (0.42, 0.83):
            p.box(
                "stored_phantom_pad",
                (-1.53, y, z),
                (0.19, 0.12, 0.075),
                (0.76, 0.83, 0.82, 1),
            )
    p.box("wall_service_cabinet", (0, 2.72, 2.40), (0.79, 0.18, 0.22), WHITE)


BUILDERS = {
    "birmingham_mri_phantom_station": _mri_phantom_station,
    "manchester_unpowered_dielectric_cell": _dielectric_cell,
    "epfl_synthesis_hood": _synthesis_hood,
    "kth_enclosed_prototyping_printer": _prototyping_printer,
}
SOURCES = {
    "birmingham_mri_phantom_station": dict(
        reference="Birmingham CHBH scanner-room photograph; page identifies MAGNETOM Prisma",
        url=BIRMINGHAM,
        dimensions_m=[2.10, 3.53, 2.16],
        dimension_basis="Estimated scanner silhouette, bore, table and motion ranges from operator-scale official photograph; not OEM drawings",
    ),
    "manchester_unpowered_dielectric_cell": dict(
        reference="University of Manchester insulating-liquid vessel official photograph",
        url=MANCHESTER,
        dimensions_m=[1.00, 0.86, 1.94],
        dimension_basis="Estimated from cropped fixture photograph; vessel and support dimensions, travel and internal coupon are unmeasured authored surrogates",
    ),
    "kth_enclosed_prototyping_printer": dict(
        reference="KTH Robot Design Lab installed printer row and prototyping workroom",
        url=KTH,
        dimensions_m=[0.90, 0.76, 1.80],
        dimension_basis="Estimated bench and printer proportions from official photograph; internal guides and travel are authored",
    ),
    "epfl_synthesis_hood": dict(
        reference="EPFL LSPN installed twin-hood photograph (2019)",
        url=EPFL,
        dimensions_m=[1.74, 1.10, 2.60],
        dimension_basis="Estimated human-scale hood dimensions; no measured plan or OEM CAD; motion ranges authored",
    ),
}
SAMPLE_INTERFACES = {
    "birmingham_mri_phantom_station": (
        "inert_mri_phantom",
        (0.192, 0.32, 0.192),
        "clamped",
        "Visible inert cylindrical phantom retained in a cradle on the moving table",
    ),
    "manchester_unpowered_dielectric_cell": (
        "inert_dielectric_coupon",
        (0.07, 0.016, 0.05),
        "clamped",
        "Visible inert coupon retained in a submerged-looking unpowered mechanical carrier",
    ),
    "kth_enclosed_prototyping_printer": (
        "prototype_coupon",
        (0.15, 0.09, 0.09),
        "clamped",
        "Visible rigid prototype bracket retained on the moving build plate",
    ),
    "epfl_synthesis_hood": (
        "inert_flask",
        (0.154, 0.154, 0.25),
        "clamped",
        "Visible sealed inert flask retained on the lift-mounted hotplate surrogate",
    ),
}
FEATURES = {
    "birmingham_chbh_mri": _mri_room,
    "manchester_hv_dielectric_fluids": _manchester_workstation,
    "epfl_lspn_chemistry": _epfl_bay,
    "kth_robot_design_prototyping": _kth_workroom,
}
