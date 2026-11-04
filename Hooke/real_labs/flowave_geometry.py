"""FloWave's physical basin and independently hinged, uncalibrated flap array.

Primary sources establish 168 flaps, a 25 m / 2 m test region, a 30 m / 5 m
concrete shell and hinges 1.7 m below the water surface. Freeboard, structural
sections, the bounded angular range and drive properties are authored estimates.
The raised test floor is represented in a fixed supported inspection state;
its real lifting/buoyancy system and all fluid response remain unmodeled.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.architecture import numbers

COUNT = 168
RADIUS = 12.53
WATER_Z = -0.35
TEST_FLOOR_Z = WATER_Z - 2.0
HINGE_Z = WATER_Z - 1.7
SHELL_FLOOR_Z = -5.0
FLAP_HEIGHT = 2.20
FLAP_HALF_WIDTH = 0.2025
LIMIT = 0.12
PREFIX = "flowave_v2_"


def key(index):
    return f"flap_{index:03d}_angle_v2"


class Parts:
    """Explicit fixed/moving ownership, finite contacts and zero added inertia."""

    def __init__(self, builder):
        self.b = builder

    def box(self, parent, name, pos, half, material="metal", **attrs):
        return self.b.box(
            parent,
            pos,
            half,
            material,
            name=PREFIX + name,
            collision=True,
            density=0,
            **attrs,
        )

    def rod(self, parent, name, start, end, radius, material="metal", **attrs):
        return self.b.rod(
            parent,
            start,
            end,
            radius,
            material,
            name=PREFIX + name,
            collision=True,
            density=0,
            **attrs,
        )

    def ring(
        self, parent, name, inner, outer, bottom, top, count=168, material="metal"
    ):
        """Tangent contact sectors with finite radial width and real open bore."""
        radius = (inner + outer) / 2
        for i in range(count):
            angle = math.tau * (i + 0.5) / count
            self.box(
                parent,
                f"{name}_{i}",
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    (top + bottom) / 2,
                ),
                (
                    (outer - inner) / 2,
                    outer * math.tan(math.pi / count),
                    (top - bottom) / 2,
                ),
                material,
                euler=(0, 0, angle),
            )

    def bearing(self, parent, name, y):
        """Open Y bearing with 0.6 mm radial and 2 mm thrust running gaps."""
        inner, outer, half = 0.0206, 0.036, 0.006
        for i in range(12):
            angle = math.tau * (i + 0.5) / 12
            radius = (inner + outer) / 2
            self.box(
                parent,
                f"{name}_sector_{i}",
                (radius * math.cos(angle), y, radius * math.sin(angle)),
                ((outer - inner) / 2, half, outer * math.tan(math.pi / 12)),
                "dark",
                euler=(0, -angle, 0),
            )
        self.box(parent, name + "_pedestal", (0.062, y, -0.132), (0.029, 0.010, 0.171))


def _flap(b, p, parent, index):
    angle = index * math.tau / COUNT
    c, s = math.cos(angle), math.sin(angle)
    mount = ET.SubElement(
        parent,
        "body",
        name=PREFIX + f"station_{index}",
        pos=numbers((RADIUS * c, RADIUS * s, HINGE_Z)),
        euler=numbers((0, 0, angle)),
    )
    ET.SubElement(mount, "inertial", pos="0 0 0", mass="3", diaginertia=".03 .03 .03")
    p.box(mount, f"mount_base_{index}", (0.062, 0, -0.292), (0.082, 0.231, 0.016))
    for side in (-1, 1):
        p.bearing(mount, f"bearing_{index}_{side}", side * 0.213)
    flap = b.moving(
        mount,
        key(index),
        (0, 0, 0),
        (0, -1, 0),
        (-LIMIT, LIMIT),
        kind="hinge",
        mass=18,
        kp=1500,
        force=800,
    )
    # Estimated aluminum/frame equivalent: correct metre-scale COM/inertia,
    # rather than the generic small-instrument inertia. Ideal gravity support
    # is explicit; no OEM motor, counterweight or hydrodynamic load is fitted.
    inertia = flap.find("inertial")
    inertia.set("pos", ".02 0 1.10")
    inertia.set("diaginertia", "7.506 7.264 .250")
    actuator = b.actuator.find(f"position[@name='{b.name}__actuator_{key(index)}']")
    actuator.set("kv", str(2 * math.sqrt(1500 * (7.264 + 18 * 1.10**2))))
    p.box(
        flap, f"panel_{index}", (0, 0, 1.105), (0.006, FLAP_HALF_WIDTH, 1.075), "bright"
    )
    for side in (-1, 1):
        p.box(
            flap,
            f"panel_rib_{index}_{side}",
            (0.018, side * 0.188, 1.105),
            (0.018, 0.011, 1.075),
            "dark",
        )
    for z in (0.04, 2.17):
        p.box(
            flap,
            f"panel_cross_{index}_{z}",
            (0.018, 0, z),
            (0.018, 0.199, 0.014),
            "dark",
        )
    b.cylinder(
        flap,
        (0, 0, 0),
        0.020,
        0.225,
        "bright",
        collision=True,
        density=0,
        name=PREFIX + f"shaft_{index}",
        euler=(math.pi / 2, 0, 0),
    )
    p.box(flap, f"shaft_web_{index}", (0.012, 0, 0.036), (0.024, 0.187, 0.035))
    for side in (-1, 1):
        # Finite collars stop 2 mm short of the fixed bearing's inner face.
        b.cylinder(
            flap,
            (0, side * 0.202, 0),
            0.029,
            0.002,
            "metal",
            collision=True,
            density=0,
            name=PREFIX + f"shaft_collar_{index}_{side}",
            euler=(math.pi / 2, 0, 0),
        )
    b.site(flap, f"flap_{index:03d}_top", (0, 0, 2.18), size=0.007)
    return flap


def add_basin(b, parent, params):
    p = Parts(b)
    base = ET.SubElement(parent, "body", name=PREFIX + "basin")
    ET.SubElement(
        base, "inertial", pos="0 0 -2.5", mass="1000", diaginertia="1000 1000 1000"
    )
    # Deep outer shell is distinct from the upper two-metre test region.
    b.cylinder(
        base,
        (0, 0, SHELL_FLOOR_Z - 0.10),
        15.0,
        0.10,
        "shell",
        collision=True,
        density=0,
        name=PREFIX + "shell_floor",
    )
    p.ring(base, "shell_wall", 14.90, 15.10, SHELL_FLOOR_Z, 0, material="shell")
    # Representative locked test-floor state. The real floor lifts; no lifting
    # system or buoyancy simulation is implied by these disclosed support locks.
    b.cylinder(
        base,
        (0, 0, TEST_FLOOR_Z - 0.10),
        12.48,
        0.10,
        "shell",
        collision=True,
        density=0,
        name=PREFIX + "locked_test_floor",
    )
    p.ring(
        base,
        "hinge_support_ring",
        12.43,
        13.16,
        HINGE_Z - 0.34,
        HINGE_Z - 0.292,
        material="metal",
    )
    for i in range(28):
        a = math.tau * i / 28
        c, s = math.cos(a), math.sin(a)
        p.box(
            base,
            f"support_column_{i}",
            (13.06 * c, 13.06 * s, -3.695),
            (0.08, 0.08, 1.305),
            "metal",
            euler=(0, 0, a),
        )
        p.box(
            base,
            f"floor_lock_{i}",
            (12.43 * c, 12.43 * s, TEST_FLOOR_Z - 0.08),
            (0.21, 0.10, 0.08),
            "metal",
            euler=(0, 0, a),
        )
    for i in range(COUNT):
        _flap(b, p, base, i)
    # Water remains visible and optical, without mass, contact, wave or current
    # response. The outer return plenum is a geometry/context reconstruction.
    asset = b.root.find("asset")
    ET.SubElement(
        asset, "material", name=f"{b.name}__mat_clear_water", rgba="1 1 1 .15"
    )
    b.cylinder(
        base,
        (0, 0, (WATER_Z + TEST_FLOOR_Z) / 2),
        12.525,
        1.0,
        "clear_water",
        density=0,
        name=PREFIX + "water_display_volume",
    )
    # Existing inert reference retains its size and water-relative elevation.
    p.rod(
        base, "reference_support", (0, 0, TEST_FLOOR_Z), (0, 0, WATER_Z + 0.04), 0.033
    )
    b.cylinder(
        base,
        (0, 0, WATER_Z + 0.10),
        0.46,
        0.075,
        "orange",
        collision=True,
        density=0,
        name=PREFIX + "inert_reference",
    )
    p.rod(
        base,
        "reference_mast",
        (0, 0, WATER_Z + 0.175),
        (0, 0, WATER_Z + 0.60),
        0.055,
        "bright",
    )
    p.box(base, "reference_head", (0, 0, WATER_Z + 0.64), (0.12, 0.08, 0.04), "dark")
    b.site(base, "calibration_buoy", (0, 0, WATER_Z + 0.10), size=0.02)
    b.metadata["coordinated_inspection_targets"] = {
        key(i): (0.055 if i % 2 else -0.055) for i in range(COUNT)
    }
    b.metadata["capabilities"] = [
        "168 individually hinged flap positions",
        "Full-array bounded mechanical inspection",
    ]
    b.metadata["control_contract"] = {
        "version": 2,
        "units": "radians",
        "old_controls_rejected": ["paddle_west", "paddle_east"],
        "mapping": None,
        "hinge_depth_below_water_m": 1.7,
        "range_rad": [-LIMIT, LIMIT],
        "limits_basis": "Authored bounded mechanical inspection, not OEM maximum stroke",
        "drive_basis": "Estimated 18 kg equivalent, 1500 Nm/rad PD, 800 Nm torque cap, ideal gravity support",
    }
    b.metadata["limitations"].extend(
        [
            "No hydrodynamic wave/current, active absorption, fluid mass or scientific transfer function.",
            "Raised floor is fixed in an authored supported inspection state; actual floor lifting and buoyancy are not modeled.",
            "Only source-supported basin dimensions, paddle count and hinge depth are verified; sections, freeboard, mass and drive parameters are estimates.",
            "Panels leave finite unsealed gaps; this is not a watertight wave boundary. Fixed structural inertias are numerical placeholders; the optical water has zero mass.",
        ]
    )

    # The canonical scene retains its historical names exactly. Alternate
    # equipment instances receive their own stable namespace; actuator and
    # site names are already generated from b.name by the equipment builder.
    if b.name != "wave_basin":
        for element in base.iter():
            value = element.get("name", "")
            if value.startswith(PREFIX):
                element.set("name", b.name + "__" + value)
