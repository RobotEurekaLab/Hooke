"""Reference-informed centrifuge, shake-table, wind-tunnel and yarn workstations.

Original geometry and restricted mechanical preview controls. Published device
sizes anchor scale where available; hidden room boundaries and internals remain
estimates. No source photographs, OEM CAD or validated scientific solvers.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

CAMBRIDGE = "https://www.eng.cam.ac.uk/node/374"
BRISTOL = "https://www.bristol.ac.uk/research/groups/earthquakegeo/equip/"
TUM = "https://www.epc.ed.tum.de/en/aer/wind-tunnels/wind-tunnel-a/"
LEEDS = "https://www.leeds.ac.uk/leeds-institute-textiles-colour-1/doc/facilities"

WHITE = (0.83, 0.84, 0.81, 1)
STEEL = (0.43, 0.47, 0.48, 1)
DARK = (0.07, 0.09, 0.10, 1)
YELLOW = (0.96, 0.66, 0.045, 1)
BLUE = (0.06, 0.36, 0.58, 1)


class _Details:
    """Named static architecture, separate from articulated instruments."""

    def __init__(self, world, name):
        self.world, self.name, self.count = world, name, 0

    def geom(self, key, shape, pos, size, color, *, collision=False, **extra):
        self.count += 1
        attrs = dict(
            name=f"europe2_{self.name}_{key}_{self.count}",
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


def _omit_unverified_ceiling_fixtures(world):
    """Selected-bay models omit generic lights with no modeled roof support."""
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("arch_luminaire_"):
            world.remove(geom)


def _turner_centrifuge(b, base, params):
    """Restricted maintenance indexing and a retained inert specimen carrier."""
    b.cylinder(base, (0, 0, 0.11), 0.93, 0.11, "metal", collision=True)
    b.cylinder(base, (0, 0, 0.48), 0.52, 0.37, "dark", collision=True)
    b.cylinder(base, (0, 0, 0.825), 0.67, 0.055, "bright")
    rotor = b.moving(
        base,
        "maintenance_index",
        (0, 0, 1.02),
        (0, 0, 1),
        (-0.20, 0.20),
        kind="hinge",
        mass=8,
        kp=500,
        force=350,
    )
    for z in (-0.12, 0.15):
        b.box(rotor, (0, 0, z), (4.72, 0.42, 0.045), "metal", collision=True)
    b.box(rotor, (0, 0, 0.01), (4.68, 0.065, 0.095), "metal", collision=True)
    for x in (-4.62, -3, -1.5, 1.5, 3, 4.62):
        b.box(rotor, (x, 0, 0.01), (0.035, 0.40, 0.12), "metal")
    b.box(rotor, (0, 0, 0.27), (1.39, 0.67, 0.06), "shell", collision=True)
    for y in (-0.66, 0.66):
        b.box(rotor, (0, y, 0.405), (1.42, 0.025, 0.075), "metal")
    for x in (-1.37, 1.37):
        b.box(rotor, (x, 0, 0.405), (0.025, 0.67, 0.075), "metal")
    for x in (-0.86, -0.16, 0.69):
        b.box(rotor, (x, 0, 0.42), (0.19, 0.21, 0.09), "shell")
        b.box(rotor, (x, -0.22, 0.44), (0.13, 0.009, 0.055), "dark")
    # End assemblies remain above the floor and below the low concrete roof.
    for x in (-4.35, 4.35):
        b.box(rotor, (x, 0, -0.34), (0.35, 0.38, 0.16), "metal", collision=True)
        for y in (-0.26, 0.26):
            b.rod(rotor, (x, y, -0.17), (x, y, 0.15), 0.033, "bright")
    for y in (-0.27, 0.27):
        b.rod(rotor, (3.48, y, 0.217), (4.60, y, 0.217), 0.015, "bright")
    carriage = b.moving(
        rotor,
        "specimen_carriage",
        (4.04, 0, 0.272),
        (1, 0, 0),
        (-0.15, 0.15),
        mass=0.8,
        kp=220,
    )
    b.box(carriage, (0, 0, 0), (0.32, 0.32, 0.025), "bright", collision=True)
    for y in (-0.30, 0.30):
        b.box(carriage, (0, y, 0.11), (0.32, 0.02, 0.085), "metal", collision=True)
    for x in (-0.30, 0.30):
        b.box(carriage, (x, 0, 0.11), (0.02, 0.28, 0.085), "metal", collision=True)
    b.box(carriage, (0, 0, 0.105), (0.25, 0.25, 0.08), "copper", collision=True)
    for x in (-0.16, 0, 0.16):
        b.box(carriage, (x, 0, 0.189), (0.008, 0.20, 0.004), "cream")
    b.site(carriage, "inert_soil_model", (0, 0, 0.105), size=0.015)
    b.metadata["capabilities"] = [
        "Restricted low-speed maintenance indexing",
        "Retained inert specimen-carriage positioning",
    ]
    b.metadata["limitations"].append(
        "Historical Cambridge reference and published 10 m machine diameter; authored "
        "carrier and restricted 0.4 rad indexing envelope. This is not a high-speed "
        "centrifuge simulation: no soil response, acceleration scaling or validated rotor dynamics."
    )


def _turner_chamber(world, definition):
    _omit_unverified_ceiling_fixtures(world)
    p = _Details(world, "turner")
    # Low ceiling coffers and the foreground service grate are observed cues.
    for y in (-4.6, -2.3, 0, 2.3, 4.6):
        g = p.box(
            "ceiling_crossbeam", (0, y, 2.20), (6.1, 0.16, 0.14), (0.61, 0.60, 0.56, 1)
        )
        g.set("name", "arch_roof_" + g.get("name"))
    for x in (-4.6, -2.3, 0, 2.3, 4.6):
        g = p.box("ceiling_rib", (x, 0, 2.22), (0.12, 6.1, 0.12), (0.61, 0.60, 0.56, 1))
        g.set("name", "arch_roof_" + g.get("name"))
    p.box("grate_recess", (0, -4.15, 0.003), (1.34, 1.1, 0.004), DARK)
    for i in range(25):
        x = -1.25 + 0.105 * i
        p.box("service_grate", (x, -4.15, 0.014), (0.013, 1.05, 0.009), STEEL)
    for y in (-5.2, -4.65, -4.1, -3.55, -3.1):
        p.box("grate_crossbar", (0, y, 0.020), (1.28, 0.016, 0.012), STEEL)
    for x in (-3.5, 0, 3.5):
        p.box("wall_service_trunk", (x, 6.14, 0.66), (1.64, 0.055, 0.08), STEEL)
    for y in (-3.8, 0, 3.8):
        light = p.box(
            "roof_linear_light", (0, y, 2.05), (0.10, 0.55, 0.025), (1, 0.96, 0.81, 1)
        )
        light.set("name", "arch_luminaire_turner_" + light.get("name"))
        mount = p.box("light_mount", (0, y, 2.09), (0.025, 0.08, 0.04), STEEL)
        mount.set("name", "arch_roof_" + mount.get("name"))
    for x in (-5.50, 5.50):
        p.box(
            "end_service_cabinet",
            (x, 3.4, 0.65),
            (0.30, 0.45, 0.65),
            WHITE,
            collision=True,
        )


def _shake_table(b, base, params):
    """Photographed three-metre platform with a supported XY preview carriage."""
    base = ET.SubElement(base, "body", name=f"{b.name}__recessed_pit", pos="0 0 -.8")
    b.box(base, (0, 0, 0.08), (2.35, 2.35, 0.08), "metal", collision=True)
    for y in (-1.35, 1.35):
        for x in (-1.42, 1.42):
            b.box(base, (x, y, 0.31), (0.12, 0.13, 0.15), "dark", collision=True)
        b.rod(base, (-1.83, y, 0.49), (1.83, y, 0.49), 0.045, "bright")
    xstage = b.moving(
        base,
        "table_x",
        (0, 0, 0.57),
        (1, 0, 0),
        (-0.08, 0.08),
        mass=2,
        kp=500,
        force=300,
    )
    b.box(xstage, (0, 0, 0), (1.55, 1.52, 0.035), "dark", collision=True)
    for x in (-1.26, 1.26):
        b.rod(xstage, (x, -1.78, 0.068), (x, 1.78, 0.068), 0.025, "bright")
    platform = b.moving(
        xstage,
        "table_y",
        (0, 0, 0.165),
        (0, 1, 0),
        (-0.08, 0.08),
        mass=3,
        kp=650,
        force=350,
    )
    b.box(platform, (0, 0, 0), (1.50, 1.50, 0.065), "bright", collision=True)
    for x in (-1.3, -0.78, -0.26, 0.26, 0.78, 1.3):
        for y in (-1.3, -0.78, -0.26, 0.26, 0.78, 1.3):
            b.cylinder(platform, (x, y, 0.067), 0.014, 0.002, "dark")
    # Sleeve walls leave physical clearances for their supported sliding rods.
    for y in (-1.06, 1.06):
        for k in range(12):
            a = math.tau * k / 12
            yy, zz = y + 0.088 * math.cos(a), 0.54 + 0.088 * math.sin(a)
            b.rod(
                base, (-2.21, yy, zz), (-1.77, yy, zz), 0.018, "metal", collision=True
            )
        b.rod(
            xstage,
            (-2.03, y, -0.03),
            (-1.48, y, -0.03),
            0.036,
            "bright",
            collision=True,
        )
    for x in (-1.06, 1.06):
        for k in range(12):
            a = math.tau * k / 12
            xx, zz = x + 0.076 * math.cos(a), 0.138 + 0.076 * math.sin(a)
            b.rod(
                xstage, (xx, -2.18, zz), (xx, -1.77, zz), 0.014, "metal", collision=True
            )
        b.rod(
            platform,
            (x, -2.03, -0.027),
            (x, -1.47, -0.027),
            0.030,
            "bright",
            collision=True,
        )
    # The yellow bolted test frame is the inert specimen, not a building model.
    for x in (-1.05, 1.05):
        for y in (-1.05, 1.05):
            b.box(
                platform, (x, y, 0.092), (0.17, 0.17, 0.027), "warning", collision=True
            )
            b.box(
                platform,
                (x, y, 1.475),
                (0.065, 0.065, 1.355),
                "warning",
                collision=True,
            )
            for dx in (-0.115, 0.115):
                b.cylinder(platform, (x + dx, y, 0.126), 0.018, 0.008, "metal")
    for z in (1.43, 2.78):
        for y in (-1.05, 1.05):
            b.box(platform, (0, y, z), (1.13, 0.07, 0.075), "warning", collision=True)
        for x in (-1.05, 1.05):
            b.box(platform, (x, 0, z), (0.07, 1.13, 0.075), "warning", collision=True)
    for y in (-1.05, 1.05):
        b.rod(
            platform,
            (-1.04, y, 0.17),
            (1.04, y, 1.39),
            0.027,
            "warning",
            collision=True,
        )
    b.box(platform, (0.94, -1.14, 1.48), (0.06, 0.025, 0.035), "dark")
    b.rod(platform, (0.94, -1.16, 1.48), (0.93, -1.20, 0.15), 0.006, "dark")
    b.site(platform, "structural_frame", (0, 0, 1.475), size=0.025)
    b.metadata["capabilities"] = [
        "Two-axis supported table displacement with retained test frame"
    ]
    b.metadata["limitations"].append(
        "Official 3 x 3 m platform size; XY preview slides approximate a mechanically "
        "supported table, not Bristol's complete eight-actuator hydraulic system. No "
        "earthquake waveform, hydraulic response, material damage or structural validation."
    )


def _equals_bay(world, definition):
    _omit_unverified_ceiling_fixtures(world)
    p = _Details(world, "equals")
    for g in list(world.findall("geom")):
        if g.get("name", "").startswith("arch_floor"):
            world.remove(g)
    # Four aprons leave a real square service pit below the flush platform.
    for x in (-3.85, 3.85):
        p.box(
            "floor_side",
            (x, 0, -0.08),
            (1.40, 4.4, 0.08),
            (0.55, 0.56, 0.53, 1),
            collision=True,
        )
    for y in (-3.42, 3.42):
        p.box(
            "floor_end",
            (0, y, -0.08),
            (2.45, 0.98, 0.08),
            (0.55, 0.56, 0.53, 1),
            collision=True,
        )
    p.box("pit_foundation", (0, 0, -0.90), (2.45, 2.45, 0.10), STEEL, collision=True)
    for y in (-2.38, 2.38):
        p.box("pit_edge", (0, y, 0.015), (2.4, 0.065, 0.03), STEEL)
    for x in (-2.38, 2.38):
        p.box("pit_edge", (x, 0, 0.015), (0.065, 2.4, 0.03), STEEL)
    for x in (-2.72, 2.72):
        for y in (-2.45, 0, 2.45):
            p.rod("guard_post", (x, y, 0), (x, y, 1.10), 0.025, STEEL)
        for z in (0.53, 1.07):
            p.rod("guard_rail", (x, -2.45, z), (x, 2.45, z), 0.025, STEEL)
    for i in range(20):
        x = -2.9 + i * 0.30
        p.box(
            "hazard_dash",
            (x, -2.95, 0.004),
            (0.09, 0.07, 0.003),
            YELLOW,
            euler=(0, 0, 0.65),
        )
    for y in (0.0, 0.75, 1.5, 2.25, 3):
        for z in (0.40, 0.95):
            p.rod("stored_actuator", (-4.60, y, z), (-3.30, y, z), 0.08, STEEL)
            p.rod("stored_piston", (-3.30, y, z), (-3.03, y, z), 0.035, WHITE)
    for x in (-4.7, -3.0):
        p.box(
            "actuator_rack", (x, 1.5, 0.62), (0.045, 1.85, 0.62), DARK, collision=True
        )


def _wind_tunnel_station(b, base, params):
    """Supported yaw model and traverse; no aerodynamic field is generated."""
    for x in (-1.8, 1.8):
        for y in (-2.05, 2.05):
            b.box(base, (x, y, 0.045), (0.23, 0.23, 0.045), "metal", collision=True)
            b.box(base, (x, y, 0.31), (0.09, 0.09, 0.22), "metal", collision=True)
    b.box(base, (0, 0, 0.575), (2.05, 2.40, 0.045), "copper", collision=True)
    for x in (-1.45, -0.7, 0, 0.7, 1.45):
        b.box(base, (x, 0, 0.622), (0.003, 2.39, 0.002), "dark")
    for y in (-1.6, -0.8, 0, 0.8, 1.6):
        b.box(base, (0, y, 0.622), (2.04, 0.003, 0.002), "dark")
    b.cylinder(base, (0, -0.45, 0.66), 1.05, 0.035, "metal", collision=True)
    stage = b.moving(
        base,
        "model_yaw",
        (0, -0.45, 0.705),
        (0, 0, 1),
        (-0.40, 0.40),
        kind="hinge",
        mass=1.8,
        kp=260,
    )
    b.cylinder(stage, (0, 0, 0), 1.00, 0.010, "cream", collision=True)
    # Original unbranded rigid vehicle-like test body with visible wheel support.
    for x in (-0.34, 0.34):
        for y in (-0.56, 0.56):
            b.cylinder(
                stage,
                (x, y, 0.125),
                0.115,
                0.045,
                "rubber",
                euler=(0, math.pi / 2, 0),
                collision=True,
            )
            b.cylinder(
                stage,
                (x + (0.047 if x > 0 else -0.047), y, 0.125),
                0.062,
                0.007,
                "bright",
                euler=(0, math.pi / 2, 0),
            )
    vehicle = stage
    b.housing(
        vehicle,
        [(0.15, 0.34, 0.77, 0), (0.28, 0.37, 0.83, 0), (0.39, 0.29, 0.54, 0.02)],
        radius=0.035,
        material="shell",
    )
    b.box(vehicle, (0, 0, 0.245), (0.29, 0.73, 0.08), "shell", collision=True)
    b.housing(
        vehicle,
        [(0.365, 0.265, 0.43, 0.06), (0.54, 0.20, 0.27, 0.08)],
        radius=0.025,
        material="dark",
    )
    for x in (-0.23, 0.23):
        b.box(vehicle, (x, -0.813, 0.29), (0.065, 0.012, 0.027), "cream")
        b.box(vehicle, (x, 0.813, 0.29), (0.065, 0.012, 0.023), "guard_red")
    for x in (-0.46, 0.46):
        b.box(stage, (x, 0, 0.048), (0.07, 0.09, 0.038), "metal", collision=True)
        b.rod(
            stage, (x, 0, 0.065), (0.30 if x > 0 else -0.30, 0, 0.18), 0.018, "bright"
        )
    b.site(stage, "vehicle_specimen", (0, 0, 0.29), size=0.014)
    # Official clear opening is 2.4 m wide by 1.8 m high. Rear duct is a short
    # visible throat only; hidden return circuit and fan are deliberately absent.
    for x in (-1.45, 1.45):
        b.box(base, (x, 2.62, 1.52), (0.25, 0.24, 0.90), "copper", collision=True)
        b.box(base, (x, 2.62, 0.31), (0.25, 0.24, 0.31), "metal", collision=True)
    for z in (0.55, 2.49):
        b.box(base, (0, 2.62, z), (1.70, 0.24, 0.07), "copper", collision=True)
    for x in (-1.225, 1.225):
        b.box(base, (x, 3.15, 1.52), (0.025, 0.55, 0.9), "dark", collision=True)
    for z in (0.60, 2.44):
        b.box(base, (0, 3.15, z), (1.25, 0.55, 0.02), "dark", collision=True)
    b.box(base, (0, 3.69, 1.52), (1.25, 0.025, 0.90), "dark")
    for x in (-1.45, 1.45):
        b.box(base, (x, 1.25, 1.45), (0.035, 0.05, 0.825), "bright", collision=True)
        b.box(base, (x, 1.25, 0.665), (0.12, 0.12, 0.045), "metal", collision=True)
    for z in (1.95, 2.20):
        b.rod(base, (-1.44, 1.25, z), (1.44, 1.25, z), 0.022, "bright")
    b.box(base, (-1.53, 1.25, 2.07), (0.09, 0.085, 0.12), "blue")
    carriage = b.moving(
        base,
        "probe_traverse",
        (0, 1.25, 2.075),
        (1, 0, 0),
        (-0.70, 0.70),
        mass=0.25,
        kp=150,
    )
    b.box(carriage, (0, 0, 0), (0.11, 0.075, 0.075), "metal", collision=True)
    b.rod(
        carriage, (0, -0.03, -0.07), (0, -0.03, -0.74), 0.016, "bright", collision=True
    )
    b.rod(
        carriage, (0, -0.03, -0.74), (0, -0.26, -0.74), 0.010, "metal", collision=True
    )
    b.site(carriage, "probe_tip", (0, -0.26, -0.74), size=0.005)
    b.metadata["capabilities"] = [
        "Clamped rigid-model yaw positioning",
        "Supported measurement-probe traverse",
    ]
    b.metadata["limitations"].append(
        "Published 2.4 x 1.8 m clear opening and 4.8 m test length anchor scale. Original rigid vehicle specimen and yaw/traverse mechanisms are previews, not verified OEM assemblies. No airflow, aerodynamic forces, rolling-road dynamics or calibrated measurements."
    )


def _wind_tunnel_bay(world, definition):
    _omit_unverified_ceiling_fixtures(world)
    p = _Details(world, "wind_tunnel")
    # Source establishes the selected test bay, not the unseen entire loop.
    for x in (-3.6, 3.6):
        p.box("crane_column", (x, 1.9, 2.4), (0.12, 0.14, 2.4), WHITE, collision=True)
        p.box("crane_foot", (x, 1.9, 0.06), (0.25, 0.27, 0.06), STEEL, collision=True)
    p.box("overhead_beam", (0, 1.9, 4.65), (3.72, 0.17, 0.18), YELLOW)
    p.box("lifting_carriage", (-1.5, 1.9, 4.42), (0.28, 0.25, 0.06), DARK)
    p.rod("lifting_chain", (-1.5, 1.9, 4.36), (-1.5, 1.9, 3.0), 0.008, STEEL)
    # Low approach barrier is kept outside the door and operator approach.
    for x in (-2.2, 2.2):
        p.cylinder("barrier_post", (x, -2.7, 0.51), 0.025, 0.51, STEEL)
        p.cylinder("barrier_foot", (x, -2.7, 0.025), 0.11, 0.025, STEEL)
    for i in range(18):
        x = -2.2 + i * (4.4 / 18)
        y = -2.7
        z = 0.92 - 0.18 * (1 - (x / 2.2) ** 2)
        xx = x + 4.4 / 18
        zz = 0.92 - 0.18 * (1 - (xx / 2.2) ** 2)
        p.rod("chain", (x, y, z), (xx, y, zz), 0.008, DARK)
    p.box(
        "control_base", (3.05, -0.25, 0.10), (0.34, 0.29, 0.10), STEEL, collision=True
    )
    p.box(
        "control_pedestal",
        (3.05, -0.25, 0.75),
        (0.05, 0.05, 0.55),
        STEEL,
        collision=True,
    )
    p.box("control_panel", (3.05, -0.25, 1.38), (0.26, 0.08, 0.19), WHITE)
    p.box("control_display", (3.05, -0.335, 1.41), (0.19, 0.006, 0.11), DARK)
    for z in (1.45, 3.15):
        p.box("panel_seam", (0, 4.68, z), (4.4, 0.004, 0.007), STEEL)
    for x in (-3.3, -1.65, 0, 1.65, 3.3):
        p.box("panel_seam", (x, 4.68, 2.7), (0.007, 0.004, 2.05), STEEL)


def _yarn_spinning_station(b, base, params):
    """Sloped laboratory yarn machine with feed roller and retained bobbin slide."""
    for x in (-0.69, 0.69):
        for y in (-0.34, 0.34):
            b.cylinder(base, (x, y, 0.045), 0.055, 0.045, "rubber", collision=True)
            b.box(base, (x, y, 0.29), (0.045, 0.045, 0.20), "metal", collision=True)
    b.box(base, (0, 0, 0.53), (0.76, 0.43, 0.05), "shell", collision=True)
    b.box(base, (0, 0.32, 0.81), (0.70, 0.08, 0.24), "metal", collision=True)
    # The side shells and sloped roller plane are visible in the LITAC photo.
    for x in (-0.755, 0.755):
        side = ET.SubElement(
            base,
            "body",
            name=f"{b.name}__side_{'left' if x<0 else 'right'}",
            pos=f"{x} 0 0",
        )
        b.housing(
            side,
            [
                (0.53, 0.055, 0.43, 0),
                (0.83, 0.055, 0.45, 0),
                (1.74, 0.055, 0.16, 0.29),
                (1.87, 0.055, 0.13, 0.32),
            ],
            radius=0.025,
            material="blue",
        )
        b.box(side, (0, 0.36, 1.23), (0.046, 0.035, 0.58), "shell", collision=True)
        b.rod(side, (0, -0.40, 0.78), (0, 0.16, 1.77), 0.035, "shell", collision=True)
        for y, z in [(-0.33, 0.80), (0.23, 1.76)]:
            b.cylinder(
                side,
                (-0.062 if x < 0 else 0.062, y, z),
                0.026,
                0.006,
                "dark",
                euler=(0, math.pi / 2, 0),
            )
    for y, z in [(-0.40, 0.88), (-0.25, 1.13), (-0.10, 1.38), (0.055, 1.63)]:
        b.rod(
            base,
            (-0.69, y + 0.07, z),
            (0.69, y + 0.07, z),
            0.035,
            "metal",
            collision=True,
        )
        for x in (-0.48, -0.16, 0.16, 0.48):
            b.box(base, (x, y, z), (0.125, 0.05, 0.075), "cream", collision=True)
            b.cylinder(
                base,
                (x, y - 0.060, z),
                0.040,
                0.046,
                "bright",
                euler=(math.pi / 2, 0, 0),
            )
            b.cylinder(
                base, (x, y - 0.110, z), 0.027, 0.008, "blue", euler=(math.pi / 2, 0, 0)
            )
    # A single supported feed shaft rotates visibly via indexed end markers.
    shaft = b.moving(
        base,
        "feed_roller",
        (0, -0.165, 1.53),
        (1, 0, 0),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.15,
        kp=80,
    )
    b.cylinder(
        shaft,
        (0, 0, 0),
        0.043,
        0.64,
        "bright",
        euler=(0, math.pi / 2, 0),
        collision=True,
    )
    for x in (-0.48, -0.16, 0.16, 0.48):
        b.cylinder(
            shaft,
            (x, 0, 0),
            0.059,
            0.11,
            "cream",
            euler=(0, math.pi / 2, 0),
            collision=True,
        )
        b.rod(shaft, (x - 0.112, 0, 0), (x - 0.112, 0.057, 0), 0.006, "blue")
    for x in (-0.69, 0.69):
        b.box(base, (x, -0.16, 1.53), (0.035, 0.087, 0.083), "shell", collision=True)
    # Exposed lower guides are fixed; bobbin motion does not claim to simulate
    # yarn feed, changing tension, contact or fibre formation.
    for x in (-0.48, -0.16, 0.16, 0.48):
        b.rod(base, (x, -0.47, 0.64), (x, -0.50, 1.04), 0.005, "bright")
        for z in (0.70, 0.84, 0.97):
            b.rod(base, (x, -0.50, z), (x + 0.07, -0.50, z), 0.004, "bright")
            b.ring(
                base,
                (x + 0.075, -0.50, z),
                0.012,
                0.002,
                "bright",
                plane="xz",
                segments=12,
            )
        b.cylinder(base, (x, -0.48, 0.63), 0.025, 0.025, "blue")
    for y in (-0.38, -0.19):
        b.rod(base, (-0.50, y, 0.595), (0.50, y, 0.595), 0.012, "bright")
    carriage = b.moving(
        base,
        "bobbin_carriage",
        (0, -0.285, 0.636),
        (1, 0, 0),
        (-0.16, 0.16),
        mass=0.25,
        kp=140,
    )
    b.box(carriage, (0, 0, 0), (0.18, 0.11, 0.021), "metal", collision=True)
    b.cylinder(carriage, (0, 0, 0.083), 0.021, 0.061, "bright", collision=True)
    for z in (0.027, 0.149):
        b.cylinder(carriage, (0, 0, z), 0.066, 0.006, "blue", collision=True)
    b.cylinder(carriage, (0, 0, 0.088), 0.046, 0.054, "cream", collision=True)
    for z in (0.044, 0.06, 0.076, 0.092, 0.108, 0.124):
        b.ring(carriage, (0, 0, z), 0.047, 0.0015, "copper", segments=20)
    b.site(carriage, "inert_yarn_bobbin", (0, 0, 0.088), size=0.005)
    # Standing-height control screen has a continuous mount back to the frame.
    b.rod(base, (0.72, 0.20, 0.73), (1.08, 0.20, 0.73), 0.035, "metal", collision=True)
    b.rod(base, (1.08, 0.20, 0.73), (1.08, 0.20, 1.49), 0.032, "metal", collision=True)
    b.box(base, (1.08, 0.20, 1.52), (0.22, 0.055, 0.18), "dark", collision=True)
    b.screen(base, (1.08, 0.14, 1.53), width=0.35, height=0.23)
    b.cylinder(
        base, (1.29, 0.126, 1.38), 0.022, 0.009, "guard_red", euler=(math.pi / 2, 0, 0)
    )
    b.metadata["capabilities"] = [
        "Supported indexed feed-roller rotation",
        "Retained inert bobbin carriage positioning",
    ]
    b.metadata["limitations"].append(
        "Reference-informed sloped blue/white laboratory yarn machine, with original estimated proportions and mechanisms. Not an OEM CAD replica. No yarn formation, changing thread path, textile constitutive model or calibrated tension control; lower guides are static geometry."
    )


def _yarn_workstation(world, definition):
    _omit_unverified_ceiling_fixtures(world)
    p = _Details(world, "yarn")
    # A local workstation backdrop, not a claim about the entire LITAC room.
    p.box("service_trunk", (0, 1.84, 1.12), (2.45, 0.035, 0.065), WHITE)
    for x in (-1.5, -0.8, 0, 0.8, 1.5):
        p.box("socket", (x, 1.80, 1.12), (0.040, 0.012, 0.042), STEEL)
    for x in (-1.9, -1.1):
        for y in (0.97, 1.53):
            p.box(
                "rear_surface_leg",
                (x, y, 0.42),
                (0.035, 0.035, 0.42),
                STEEL,
                collision=True,
            )
    p.box("rear_surface", (-1.5, 1.25, 0.87), (0.55, 0.40, 0.03), WHITE, collision=True)
    p.box("coupon_tray", (-1.5, 1.25, 0.913), (0.21, 0.14, 0.013), BLUE)
    for x in (-1.60, -1.48, -1.36):
        p.cylinder(
            "stored_bobbin", (x, 1.25, 0.965), 0.033, 0.04, (0.88, 0.84, 0.72, 1)
        )
    p.box(
        "operator_mat", (0.2, -1.12, 0.008), (1.02, 0.40, 0.008), (0.19, 0.23, 0.24, 1)
    )


BUILDERS = {
    "equals_xy_shake_table": _shake_table,
    "turner_geotechnical_centrifuge": _turner_centrifuge,
    "tum_model_test_station": _wind_tunnel_station,
    "litac_yarn_spinning_station": _yarn_spinning_station,
}

SOURCES = {
    "equals_xy_shake_table": {
        "reference": "Bristol EQUALS installed shaking-table and " "frame photograph",
        "url": BRISTOL,
        "dimensions_m": [4.7, 4.7, 3.65],
        "dimension_basis": "Official 3 x 3 m platform; selected "
        "pit, frame and authored XY mechanisms "
        "estimated",
    },
    "turner_geotechnical_centrifuge": {
        "reference": "Cambridge Turner centrifuge " "historical chamber photograph",
        "url": CAMBRIDGE,
        "dimensions_m": [10, 1.5, 1.6],
        "dimension_basis": "Official 10 m centrifuge "
        "diameter; structural section, "
        "carrier and stroke estimated",
    },
    "tum_model_test_station": {
        "reference": "TUM Wind Tunnel A official test-bay "
        "photographs and dimensions",
        "url": TUM,
        "dimensions_m": [4.1, 6.14, 2.62],
        "dimension_basis": "Official 2.4 m width x 1.8 m height "
        "opening and 4.8 m test-section length; "
        "throat, supports and mechanical travel "
        "estimated",
    },
    "litac_yarn_spinning_station": {
        "reference": "University of Leeds LITAC yarn-spinning "
        "workstation photograph",
        "url": LEEDS,
        "dimensions_m": [2.12, 1.07, 1.91],
        "dimension_basis": "Estimated from operator-scale "
        "official photograph; neither "
        "measured geometry nor OEM CAD",
    },
}

SAMPLE_INTERFACES = {
    "equals_xy_shake_table": (
        "structural_frame",
        (2.34, 2.34, 2.76),
        "clamped",
        "Visible rigid yellow test frame bolted to the moving " "platform",
    ),
    "turner_geotechnical_centrifuge": (
        "inert_soil_model",
        (0.5, 0.5, 0.16),
        "clamped",
        "Rigid inert geotechnical specimen retained in a " "carrier",
    ),
    "tum_model_test_station": (
        "vehicle_specimen",
        (0.77, 1.68, 0.55),
        "clamped",
        "Original unbranded rigid vehicle specimen retained on yaw " "turntable",
    ),
    "litac_yarn_spinning_station": (
        "inert_yarn_bobbin",
        (0.132, 0.132, 0.134),
        "clamped",
        "Visible rigid inert yarn bobbin retained on carriage " "spindle",
    ),
}

FEATURES = {
    "bristol_equals_shaking_table": _equals_bay,
    "cambridge_schofield_turner": _turner_chamber,
    "tum_wind_tunnel_a": _wind_tunnel_bay,
    "leeds_litac_yarn_spinning": _yarn_workstation,
}
