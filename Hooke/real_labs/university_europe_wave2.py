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
    from real_labs.cambridge_geometry import build_turner

    build_turner(b, base, params)


def _turner_chamber(world, definition):
    from real_labs.cambridge_chamber import build_chamber

    return build_chamber(world, definition)


def _shake_table(b, base, params):
    from real_labs.bristol_table_geometry import build_table

    build_table(b, base, params)


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
    p.box("pit_foundation", (0, 0, -1.98), (2.45, 2.45, 0.10), STEEL, collision=True)
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
    # The source shows stored actuator hardware on an open metal rack.
    for x in (-4.55, -3.28):
        for y in (-0.35, 3.35):
            p.box("rack_post", (x, y, 0.62), (0.035, 0.035, 0.62), STEEL, collision=True)
    for z in (0.295, 0.845):
        for x in (-4.55, -3.28):
            p.box("rack_shelf_rail", (x, 1.5, z), (0.055, 1.89, 0.025), STEEL, collision=True)
        for y in (-0.35, 3.35):
            p.box("rack_cross_member", (-3.915, y, z), (0.67, 0.04, 0.025), STEEL, collision=True)
    for y in (0.0, 0.75, 1.5, 2.25, 3):
        for z in (0.40, 0.95):
            p.rod("stored_actuator", (-4.60, y, z), (-3.30, y, z), 0.08, STEEL, collision=True)
            p.rod("stored_piston", (-3.30, y, z), (-3.03, y, z), 0.035, WHITE, collision=True)


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
    from real_labs.yarn_details import (
        add_carriage_bearings,
        add_feed_bearings,
        add_pressure_modules,
    )

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
    add_pressure_modules(b, base)
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
    add_feed_bearings(b, base, shaft)
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
        b.rod(
            base,
            (-0.50, y, 0.595),
            (0.50, y, 0.595),
            0.012,
            "bright",
            collision=True,
        )
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
    add_carriage_bearings(b, base, carriage)
    b.metadata["assembly_adaptations"] = {
        "reference": LEEDS,
        "observed": "Layered roller/pressure-arm face, rounded pale guards, small adjusters and sloped blue sides",
        "authored": "Journal/bearing mounts and split linear-guide shoes, with 0.5 mm nominal running clearance",
        "limits": "Fixed pressure-arm/roller exterior details; no bearing-load, changing thread path or yarn-process model",
    }
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
        "dimensions_m": [4.86, 4.86, 4.68],
        "dimension_basis": "Official 3 x 3 m platform; selected "
        "pit, frame and eight articulated actuator anchor positions estimated; "
        "300 mm actuator stroke and 70 kN capacity published; no calibrated hydraulics",
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
