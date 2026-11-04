"""Authored dry-access hardware for the representative UVA chamber.

This is a mechanically inspectable surrogate, not a dimensional replica or a
vacuum/pressure seal. Bearing clearances, internal supports and service routing
are authored engineering assumptions. Existing mechanism frames stay fixed.
"""

import math

from real_labs.mounting import annular_mount


def name(b, key):
    return f"{b.name}__uva_{key}"


def box(b, parent, key, pos, half, material="metal"):
    return b.box(parent, pos, half, material, collision=True, name=name(b, key))


def cylinder(b, parent, key, pos, radius, half, material="metal", **kwargs):
    return b.cylinder(
        parent, pos, radius, half, material, collision=True, name=name(b, key), **kwargs
    )


def ring(b, parent, key, center, inner, outer, half, *, axis="z"):
    annular_mount(
        b,
        parent,
        key,
        center,
        inner,
        outer,
        half,
        axis=axis,
        proxies=True,
        namespace="uva",
    )


def rod(b, parent, key, start, end, radius, material="metal"):
    return b.rod(
        parent, start, end, radius, material, collision=True, name=name(b, key)
    )


def _foundation(b, base):
    b.feet(base, 0.78, 0.73, height=0.025)
    box(b, base, "foundation", (0, 0, 0.065), (0.39, 0.365, 0.04), "dark")
    for side in (-1, 1):
        box(
            b,
            base,
            f"vessel_saddle_{side}",
            (side * 0.19, 0, 0.1855),
            (0.022, 0.18, 0.0805),
        )
    cylinder(b, base, "floor", (0, 0, 0.278), 0.25, 0.012)
    # An actual open top passage; no convex collider fills the loading cavity.
    ring(b, base, "wall", (0, 0, 0.43), 0.226, 0.254, 0.155)
    ring(b, base, "lower_flange", (0, 0, 0.285), 0.226, 0.270, 0.014)
    ring(b, base, "upper_flange", (0, 0, 0.593), 0.226, 0.270, 0.008)
    for row, z in enumerate((0.305, 0.597)):
        for index in range(12):
            angle = index * math.tau / 12
            cylinder(
                b,
                base,
                f"flange_bolt_{row}_{index}",
                (0.263 * math.cos(angle), 0.263 * math.sin(angle), z),
                0.0045,
                0.004,
                "dark",
            )
    # The front port remains an opaque inspection cover, explicitly not a
    # transparent viewing aperture or a seal simulation.
    cylinder(
        b,
        base,
        "front_cover_flange",
        (0, -0.264, 0.445),
        0.064,
        0.025,
        quat=(0.707106781, 0.707106781, 0, 0),
    )
    cylinder(
        b,
        base,
        "front_cover",
        (0, -0.291, 0.445),
        0.048,
        0.002,
        "lens",
        quat=(0.707106781, 0.707106781, 0, 0),
    )


def _lid(b, base):
    lid = b.moving(
        base,
        "chamber_lid",
        (0, 0.285, 0.617),
        (1, 0, 0),
        (-1.7, 0),
        kind="hinge",
        mass=3,
    )
    cylinder(b, lid, "lid_disc", (0, -0.285, 0), 0.26, 0.014, "bright")
    for side in (-1, 1):
        cylinder(
            b,
            lid,
            f"handle_standoff_{side}",
            (side * 0.07, -0.46, 0.0285),
            0.01,
            0.0145,
            "dark",
        )
    rod(b, lid, "handle", (-0.07, -0.46, 0.045), (0.07, -0.46, 0.045), 0.012, "dark")
    b.site(lid, "lid_handle", (0, -0.46, 0.045))
    # The moving sleeve has a real open bore around a fixed, grounded pin.
    ring(b, lid, "hinge_sleeve", (0, 0, 0), 0.0126, 0.018, 0.035, axis="x")
    box(b, lid, "hinge_leaf", (0, -0.0355, 0), (0.035, 0.0225, 0.012))
    cylinder(
        b,
        base,
        "hinge_pin",
        (0, 0.285, 0.617),
        0.012,
        0.110,
        quat=(0.707106781, 0, 0.707106781, 0),
    )
    for side in (-1, 1):
        box(
            b,
            base,
            f"hinge_post_{side}",
            (side * 0.088, 0.321, 0.360),
            (0.020, 0.016, 0.255),
        )
        cylinder(
            b,
            base,
            f"hinge_fixed_knuckle_{side}",
            (side * 0.088, 0.285, 0.617),
            0.024,
            0.020,
            quat=(0.707106781, 0, 0.707106781, 0),
        )
    return lid


def _platen(b, base):
    # Bearing housing meets the vessel floor. The 0.6 mm radial / 1 mm axial
    # running spaces are constrained-joint assumptions, not simulated rollers.
    ring(b, base, "platen_bearing", (0, 0, 0.305), 0.0186, 0.055, 0.015)
    platen = b.moving(
        base,
        "platen",
        (0, 0, 0.341),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.4,
    )
    cylinder(b, platen, "platen_spindle", (0, 0, -0.026), 0.018, 0.024)
    cylinder(b, platen, "platen_disc", (0, 0, 0), 0.11, 0.009, "dark")
    b.cylinder(
        platen, (0, 0, 0.013), 0.05, 0.002, "lens", name=name(b, "platen_top_finish")
    )
    # Preserve the support surface used by the existing free-coupon pocket.
    box(b, platen, "platen_core", (0, 0, -0.0025), (0.042, 0.042, 0.0175), "dark")
    b.site(platen, "substrate", (0, 0, 0.019))
    return platen


def _services(b, base):
    cylinder(b, base, "pump_foot", (0.34, 0.14, 0.1155), 0.059, 0.0105, "dark")
    cylinder(b, base, "pump_body", (0.34, 0.14, 0.286), 0.059, 0.16)
    rod(b, base, "pump_vessel_neck", (0.215, 0.14, 0.4), (0.34, 0.14, 0.4), 0.030)
    for index, z in enumerate((0.18, 0.205, 0.23, 0.255)):
        ring(b, base, f"pump_band_{index}", (0.34, 0.14, z), 0.059, 0.064, 0.005)
    # These endpoints explicitly match the representative process-bay routing.
    # They do not establish a flow path through the solid pump surrogate.
    rod(b, base, "roughing_connector", (0.34, 0.19, 0.29), (0.33, 0.32, 0.29), 0.03)
    rod(b, base, "coolant_connector", (0.39, 0.14, 0.20), (0.40, 0.30, 0.20), 0.012)
    rod(b, base, "gas_connector", (-0.226, 0.05, 0.5), (-0.25, 0.07, 0.5), 0.009)
    cylinder(b, base, "gas_coupling", (-0.245, 0.065, 0.5), 0.015, 0.013)
    b.site(base, "gas_inlet", (-0.24, 0.05, 0.5))
    box(b, base, "controller_pedestal", (-0.34, 0.24, 0.230), (0.045, 0.065, 0.125))
    box(b, base, "controller", (-0.36, 0.24, 0.470), (0.075, 0.095, 0.115), "shell")
    b.screen(base, (-0.36, 0.139, 0.49), 0.115, 0.09)
    # A stationary hollow gland surrounds the original vertical valve spindle.
    box(b, base, "valve_stand", (0.29, -0.08, 0.250), (0.030, 0.032, 0.145))
    cylinder(b, base, "valve_body", (0.29, -0.08, 0.401), 0.027, 0.006)
    ring(b, base, "valve_gland", (0.29, -0.08, 0.4415), 0.0066, 0.020, 0.0345)
    valve = b.moving(
        base,
        "vacuum_valve",
        (0.29, -0.08, 0.485),
        (0, 0, 1),
        (0, math.pi / 2),
        kind="hinge",
        mass=0.1,
    )
    cylinder(b, valve, "valve_spindle", (0, 0, -0.0385), 0.006, 0.0385)
    box(b, valve, "valve_handle", (0, 0, 0), (0.042, 0.009, 0.007), "blue")
    # Fixed manifold from body to vessel, below the handle sweep.
    rod(b, base, "valve_manifold", (0.226, -0.08, 0.390), (0.29, -0.08, 0.390), 0.016)
    return valve


def _source_visible_ports(b, base):
    """Sputter1-inspired exterior cues; all dimensions and routing are authored.

    Capped branches are solid exterior context. Only the separate open top is
    a functional geometric access path; this is not a load-lock simulator.
    """
    import numpy as np

    def pipe(key, start, end, radius, material="metal"):
        return b.geom(
            base,
            "cylinder",
            (radius,),
            material=material,
            fromto=tuple(start) + tuple(end),
            collision=True,
            name=name(b, key),
        )

    for side in (-1, 1):
        start = np.array([side * 0.18, -0.17, 0.49])
        end = np.array([side * 0.29, -0.274, 0.637])
        axis = (end - start) / np.linalg.norm(end - start)
        pipe(f"angled_port_{side}", start, end, 0.027, "bright")
        center = start + 0.80 * (end - start)
        pipe(
            f"angled_flange_{side}",
            center - axis * 0.010,
            center + axis * 0.010,
            0.056,
            "bright",
        )
        u = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        for i in range(10):
            angle = i * math.tau / 10
            position = center + 0.046 * (u * math.cos(angle) + v * math.sin(angle))
            pipe(
                f"angled_bolt_{side}_{i}",
                position - axis * 0.011,
                position + axis * 0.014,
                0.0035,
                "dark",
            )
        pipe(
            f"angled_cap_{side}", end - axis * 0.008, end + axis * 0.008, 0.034, "dark"
        )
    pipe("lateral_manifold", (-0.244, 0, 0.42), (-0.55, 0, 0.42), 0.066, "bright")
    for i, x in enumerate((-0.285, -0.530)):
        pipe(
            f"lateral_flange_{i}",
            (x - 0.011, 0, 0.42),
            (x + 0.011, 0, 0.42),
            0.084,
            "bright",
        )
        for index in range(12):
            angle = index * math.tau / 12
            y, z = 0.074 * math.cos(angle), 0.42 + 0.074 * math.sin(angle)
            pipe(
                f"lateral_bolt_{i}_{index}",
                (x - 0.014, y, z),
                (x + 0.014, y, z),
                0.004,
                "dark",
            )
    pipe("lateral_end_cover", (-0.558, 0, 0.42), (-0.550, 0, 0.42), 0.066)
    box(
        b,
        base,
        "lateral_support_foot",
        (-0.49, 0, 0.0155),
        (0.047, 0.070, 0.0155),
        "dark",
    )
    box(b, base, "lateral_support", (-0.49, 0, 0.1925), (0.025, 0.045, 0.1615))


def add_chamber(b, base, params):
    _foundation(b, base)
    _lid(b, base)
    _platen(b, base)
    _services(b, base)
    _source_visible_ports(b, base)
    b.metadata["capabilities"] = [
        "measured hinged-lid access to a real open cavity",
        "rotating platen with an existing physically supported free coupon",
        "bounded dry valve-handle rotation",
    ]
    b.metadata["uva_access"] = dict(
        version=1,
        design="Original mechanical dry-access fixture; no OEM internal geometry claim",
        bearing_radial_clearance_m=0.0006,
        closed_lid_clearance_m=0.002,
        opening_diameter_m=0.452,
        limitations=[
            "No vacuum, gas flow, pressure, deposition or seal performance model",
            "Front cover is opaque; observation uses the open top passage",
            "Ideal constrained bearings have authored running clearance",
            "Service connectors follow the estimated process-bay layout",
        ],
    )
