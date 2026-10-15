"""Source-informed compact Boston SEM mechanism, explicitly versioned as revision 3.

Published family chamber/stage envelopes inform this original geometry. Hidden
slides, trunnions, drive settings, cabinet and detector dimensions are authored.
"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from real_labs.mounting import annular_mount

SCENE = "boston_photonics_pml_sem"
ID = "sem_station"


def numbers(v):
    return " ".join(f"{x:.12g}" for x in v)


class Parts:
    def __init__(self, b):
        self.b = b

    def name(self, key):
        return self.b.name + "__" + key

    def box(self, body, key, pos, half, material="metal", **kw):
        return self.b.box(
            body,
            pos,
            half,
            material,
            collision=True,
            mass="0",
            name=self.name(key),
            **kw,
        )

    def cylinder(self, body, key, pos, radius, half, material="bright", **kw):
        return self.b.cylinder(
            body,
            pos,
            radius,
            half,
            material,
            collision=True,
            mass="0",
            name=self.name(key),
            **kw,
        )

    def rod(self, body, key, a, z, radius, material="bright"):
        return self.b.rod(
            body, a, z, radius, material, collision=True, mass="0", name=self.name(key)
        )

    def ring(self, body, key, pos, inner, outer, half, axis="z"):
        frame = ET.SubElement(
            body, "body", name=self.name(key + "_frame"), pos=numbers(pos), gravcomp="1"
        )
        if axis == "y":
            frame.set("quat", numbers((math.sqrt(0.5), 0, 0, math.sqrt(0.5))))
            axis = "x"
        ET.SubElement(
            frame, "inertial", pos="0 0 0", mass=".002", diaginertia="1e-7 1e-7 1e-7"
        )
        before = set(frame.iter("geom"))
        annular_mount(
            self.b,
            frame,
            key,
            (0, 0, 0),
            inner,
            outer,
            half,
            axis=axis,
            proxies=True,
            namespace="bearing",
        )
        for g in frame.iter("geom"):
            if g not in before:
                g.set("mass", "0")
        return frame


def camera(world, name, pos, target, fovy):
    forward = np.array(target) - np.array(pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ET.SubElement(
        world,
        "camera",
        name=name,
        pos=numbers(pos),
        xyaxes=numbers(np.r_[right, up]),
        fovy=str(fovy),
    )


def cabinet(p, base):
    for x in (-0.255, 0.255):
        for y in (-0.255, 0.255):
            p.cylinder(base, f"foot_{x}_{y}", (x, y, 0.025), 0.027, 0.025, "rubber")
            p.cylinder(base, f"level_{x}_{y}", (x, y, 0.060), 0.010, 0.012)
    p.box(base, "cabinet", (0, 0, 0.399), (0.310, 0.310, 0.339), "cream")
    p.box(base, "desktop", (0, 0, 0.759), (0.365, 0.365, 0.021), "dark")
    # Nominal chamber330mm envelope, with a genuinely open front.
    p.box(base, "chamber_floor", (0, 0.025, 0.970), (0.185, 0.190, 0.010))
    p.box(base, "chamber_roof", (0, 0.025, 1.260), (0.185, 0.190, 0.010), "cream")
    for x in (-0.1725, 0.1725):
        p.box(
            base,
            f"straight_wall_{x}",
            (x, -0.070, 1.115),
            (0.0075, 0.095, 0.135),
            "cream",
        )
    for i in range(32):
        a = (i + 0.5) * math.pi / 32
        radius = 0.1725
        p.box(
            base,
            f"rear_curve_{i}",
            (radius * math.cos(a), 0.025 + radius * math.sin(a), 1.115),
            (0.0075, 0.1725 * math.tan(math.pi / 64), 0.135),
            "cream",
            euler=(0, 0, a),
        )
    for x in (-0.155, 0.155):
        for y in (-0.070, 0.125):
            p.box(
                base, f"chamber_support_{x}_{y}", (x, y, 0.870), (0.018, 0.024, 0.090)
            )
    # Support the tall, narrower cream column from the real roof; no floating case.
    p.cylinder(base, "column_neck", (0, 0.025, 1.2925), 0.10, 0.0225)
    p.cylinder(base, "column_case", (0, 0.025, 1.600), 0.120, 0.285, "cream")
    p.box(base, "column_side", (0.080, 0.025, 1.595), (0.120, 0.123, 0.290), "cream")
    p.box(base, "vent_panel", (0.080, -0.100, 1.590), (0.116, 0.003, 0.250), "dark")
    for row in range(22):
        for col in range(12):
            p.box(
                base,
                f"vent_slot_{row}_{col}",
                (-0.026 + col * 0.019, -0.104, 1.354 + row * 0.021),
                (0.0016, 0.001, 0.007),
                "metal",
            )
    # Connected source-visible top loop, with explicit two glands.
    for side in (-1, 1):
        p.cylinder(
            base,
            f"loop_gland_{side}",
            (side * 0.085, 0.025, 1.897),
            0.016,
            0.012,
            "dark",
        )
    last = None
    for i in range(25):
        a = math.pi * i / 24
        point = (0.085 * math.cos(a), 0.025, 1.909 + 0.085 * math.sin(a))
        if last is not None:
            p.rod(base, f"top_loop_{i}", last, point, 0.012, "dark")
        last = point
    # Static objective exterior, not an electron-optical model.
    p.cylinder(base, "pole_upper", (0, 0.025, 1.242), 0.055, 0.008)
    p.cylinder(base, "pole_lower", (0, 0.025, 1.2145), 0.023, 0.0115)
    p.cylinder(base, "pole_neck", (0, 0.025, 1.229), 0.035, 0.008)
    p.b.site(base, "pole_face", (0, 0.025, 1.203), 0.0008)
    # Closed side detector housings and connected flange stalks; no detector physics.
    for side in (-1, 1):
        p.cylinder(
            base,
            f"port_mount_{side}",
            (side * 0.186, 0.055, 1.150),
            0.045,
            0.014,
            "bright",
            quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
        )
        p.cylinder(
            base,
            f"port_stalk_{side}",
            (side * 0.232, 0.055, 1.150),
            0.020,
            0.032,
            "metal",
            quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
        )
        p.box(
            base,
            f"port_detector_{side}",
            (side * 0.285, 0.055, 1.150),
            (0.023, 0.039, 0.047),
            "cream",
        )
        for i in range(8):
            a = 2 * math.pi * i / 8
            p.cylinder(
                base,
                f"port_bolt_{side}_{i}",
                (
                    side * 0.203,
                    0.055 + 0.036 * math.cos(a),
                    1.150 + 0.036 * math.sin(a),
                ),
                0.003,
                0.003,
                "dark",
                quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
            )


def channel(p, body, key, x, width, height, pads=False):
    # True open C-section; all walls and bearing pads are enabled solids.
    z = 0.925
    t = 0.003
    p.box(
        body,
        key + "_base",
        (x, 0.010, z - height / 2 + t / 2),
        (width / 2, 0.150, t / 2),
        "bright",
    )
    for side in (-1, 1):
        p.box(
            body,
            key + f"_wall_{side}",
            (x + side * (width / 2 - t / 2), 0.010, z + t / 2),
            (t / 2, 0.150, (height - t) / 2),
            "bright",
        )
    if pads:
        child_width = width - 0.012
        child_height = height - 0.010
        for y in (-0.120, -0.075):
            # Two forward bearings remain inside the90mm minimal overlap.
            for side in (-1, 1):
                lo = child_width / 2 + 0.0006
                hi = width / 2 - t
                p.box(
                    body,
                    key + f"_shoe_{side}_{y}",
                    (x + side * (lo + hi) / 2, y, z + 0.001),
                    ((hi - lo) / 2, 0.010, 0.004),
                    "dark",
                )
            top = -child_height / 2 - 0.0006
            bottom = -height / 2 + t
            p.box(
                body,
                key + f"_floor_shoe_{y}",
                (x, y, z + (top + bottom) / 2),
                (child_width / 2, 0.010, (top - bottom) / 2),
                "dark",
            )


def telescopes(p, base):
    b = p.b
    for side in (-1, 1):
        x = side * 0.205
        for y in (-0.115, 0.135):
            p.box(
                base,
                f"telescope_post_{side}_{y}",
                (x, y, 0.84125),
                (0.030, 0.020, 0.06125),
            )
        channel(p, base, f"fixed_rail_{side}", x, 0.050, 0.045, True)
    middle = b.moving(
        base,
        "door_slide_outer",
        (0, 0, 0),
        (0, -1, 0),
        (0, 0.210),
        mass=0.32,
        kp=500,
        force=150,
    )
    for side in (-1, 1):
        channel(
            p, middle, f"intermediate_rail_{side}", side * 0.205, 0.038, 0.035, True
        )
    # Couple both lateral members through a lower forward cross tie, away from the fixed supports.
    p.box(middle, "intermediate_bridge", (0, -0.120, 0.955), (0.1885, 0.006, 0.0025))
    for side in (-1, 1):
        p.box(
            middle,
            f"intermediate_bridge_post_{side}",
            (side * 0.1885, -0.120, 0.9475),
            (0.002, 0.006, 0.005),
        )
    drawer = b.moving(
        middle,
        "door_slide_inner",
        (0, 0, 0),
        (0, -1, 0),
        (0, 0.210),
        mass=0.35,
        kp=500,
        force=150,
    )
    for side in (-1, 1):
        x = side * 0.205
        p.box(drawer, f"inner_rail_{side}", (x, 0.010, 0.925), (0.013, 0.150, 0.0125))
        p.box(drawer, f"inner_stay_{side}", (x, -0.122, 0.945), (0.010, 0.014, 0.013))
        p.box(drawer, f"door_link_{side}", (x, -0.150, 0.960), (0.015, 0.030, 0.005))
    p.box(drawer, "door_panel", (0, -0.176, 1.110), (0.230, 0.010, 0.150), "cream")
    for x in (-0.195, 0.195):
        p.box(drawer, f"door_inner_stay_{x}", (x, -0.169, 1.012), (0.014, 0.003, 0.047))
    # Stage load path wraps around the front floor edge, entering above it.
    p.box(drawer, "cantilever_crossbar", (0, -0.179, 0.996), (0.207, 0.008, 0.009))
    for x in (-0.125, 0.125):
        p.box(drawer, f"cantilever_{x}", (x, -0.063, 0.996), (0.012, 0.119, 0.009))
    p.box(drawer, "stage_foundation", (0, 0.025, 0.996), (0.140, 0.085, 0.009), "dark")
    for side in (-1, 1):
        p.cylinder(
            drawer,
            f"handle_standoff_{side}",
            (side * 0.060, -0.202, 1.110),
            0.007,
            0.016,
            "dark",
            quat=(math.sqrt(0.5), math.sqrt(0.5), 0, 0),
        )
    p.cylinder(
        drawer,
        "door_handle",
        (0, -0.218, 1.110),
        0.008,
        0.067,
        "dark",
        quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
    )
    return drawer


def horizontal_slide(p, parent, key, axis, origin, span=0.130):
    b = p.b
    axis_index = 0 if axis == "x" else 1
    rod_half = 0.130 if axis == "x" else 0.109
    post_offset = 0.124 if axis == "x" else 0.103
    post_half = 0.010 if axis == "x" else 0.006
    bearing_offset = 0.033 if axis == "x" else 0.023
    bearing_half = 0.011 if axis == "x" else 0.008
    # Body origin is20mm above rail centers. Shoes join plate without filling the bores.
    rod_z = origin[2] - 0.020
    for side in (-1, 1):
        cross = side * 0.047
        a = [origin[0], origin[1], rod_z]
        c = a.copy()
        a[axis_index] -= rod_half
        c[axis_index] += rod_half
        a[1 - axis_index] += cross
        c[1 - axis_index] += cross
        p.rod(parent, key + f"_rail_{side}", a, c, 0.004)
        for end in (-1, 1):
            q = np.array(origin)
            q[axis_index] += end * post_offset
            q[1 - axis_index] += cross
            q[2] = rod_z - 0.005
            p.box(parent, key + f"_post_{side}_{end}", q, (post_half, post_half, 0.005))
    direction = (1, 0, 0) if axis == "x" else (0, 1, 0)
    body = b.moving(
        parent,
        key,
        origin,
        direction,
        (-span / 2, span / 2),
        mass=0.12,
        kp=500,
        force=100,
    )
    p.box(body, key + "_plate", (0, 0, 0), (0.078, 0.078, 0.006), "metal")
    for side in (-1, 1):
        for end in (-1, 1):
            center = [0, 0, -0.020]
            center[axis_index] = end * bearing_offset
            center[1 - axis_index] = side * 0.047
            p.ring(
                body,
                key + f"_bearing_{side}_{end}",
                center,
                0.0046,
                0.009,
                bearing_half,
                axis,
            )
            crown = center.copy()
            crown[2] = -0.008
            p.box(
                body, key + f"_crown_{side}_{end}", crown, (0.011, 0.011, 0.004), "dark"
            )
    return body


def five_axes(p, drawer):
    b = p.b
    # Foundation top1.005 meets X anchor bottoms; rod centers are1.015.
    x = horizontal_slide(p, drawer, "stage_x", "x", (0, 0.025, 1.035))
    for xx in (-0.047, 0.047):
        for yy in (-0.092, 0.092):
            p.box(
                x, f"y_guide_foundation_{xx}_{yy}", (xx, yy, 0), (0.006, 0.017, 0.006)
            )
    y = horizontal_slide(p, x, "stage_y", "y", (0, 0, 0.036))
    # Z columns at loading-local Y0.05, clear of the tilt cradle.
    for side in (-1, 1):
        xx = side * 0.075
        p.rod(y, f"z_guide_{side}", (xx, 0.050, 0.008), (xx, 0.050, 0.120), 0.004)
        for zz in (0.007, 0.121):
            p.box(y, f"z_anchor_{side}_{zz}", (xx, 0.050, zz), (0.011, 0.010, 0.003))
    z = b.moving(
        y, "stage_z", (0, 0, 0.019), (0, 0, 1), (0, 0.050), mass=0.12, kp=450, force=80
    )
    for side in (-1, 1):
        xx = side * 0.075
        p.ring(z, f"z_bearing_{side}", (xx, 0.050, 0.015), 0.0046, 0.009, 0.012, "z")
        p.box(z, f"z_arm_{side}", (xx, -0.004, 0), (0.012, 0.046, 0.006), "dark")
    p.box(z, "z_crossbar", (0, -0.060, 0), (0.093, 0.025, 0.006), "dark")
    # Trunnion axis passes through the loading specimen top at1.150m.
    pivot = 0.060
    for side in (-1, 1):
        xx = side * 0.080
        p.box(
            z, f"trunnion_bottom_tie_{side}", (xx, -0.0215, 0), (0.013, 0.0335, 0.006)
        )
        p.box(
            z, f"trunnion_post_{side}", (xx, 0, 0.024), (0.013, 0.012, 0.025), "metal"
        )
        p.ring(z, f"trunnion_bearing_{side}", (xx, 0, pivot), 0.0066, 0.013, 0.009, "x")
    tilt = b.moving(
        z,
        "stage_tilt",
        (0, 0, pivot),
        (1, 0, 0),
        (math.radians(-3), math.radians(70)),
        kind="hinge",
        mass=0.10,
        kp=80,
        force=12,
    )
    for side in (-1, 1):
        p.cylinder(
            tilt,
            f"tilt_axle_{side}",
            (side * 0.077, 0, 0),
            0.006,
            0.016,
            "bright",
            quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0),
        )
        p.box(tilt, f"cradle_top_{side}", (side * 0.050, 0, 0), (0.017, 0.007, 0.007))
        p.box(
            tilt, f"cradle_arm_{side}", (side * 0.048, 0, -0.020), (0.006, 0.007, 0.027)
        )
    p.box(tilt, "cradle_base", (0, 0, -0.046), (0.054, 0.024, 0.005), "dark")
    # Real circular aperture through the bottom cross member: split its middle.
    base_geom = tilt.find(f"geom[@name='{b.name}__cradle_base']")
    tilt.remove(base_geom)
    for side in (-1, 1):
        p.box(
            tilt,
            f"cradle_split_base_{side}",
            (side * 0.034, 0, -0.046),
            (0.020, 0.024, 0.005),
            "dark",
        )
    for yy in (-0.019, 0.019):
        p.box(
            tilt,
            f"cradle_base_web_{yy}",
            (0, yy, -0.046),
            (0.014, 0.005, 0.005),
            "dark",
        )
    p.ring(tilt, "rotary_bearing", (0, 0, -0.040), 0.0066, 0.015, 0.007, "z")
    rotation = b.moving(
        tilt,
        "stage_rotation",
        (0, 0, -0.024),
        (0, 0, 1),
        (-math.pi, math.pi),
        kind="hinge",
        mass=0.05,
        kp=80,
        force=12,
    )
    p.cylinder(rotation, "rotary_shaft", (0, 0, -0.019), 0.006, 0.016)
    p.cylinder(rotation, "rotary_platen", (0, 0, 0), 0.040, 0.006, "copper")
    p.cylinder(rotation, "stub_seat", (0, 0, 0.009), 0.020, 0.003, "bright")
    p.cylinder(rotation, "retained_stub", (0, 0, 0.018), 0.015, 0.006, "orange")
    p.box(rotation, "stub_marker", (0.004, 0, 0.0242), (0.001, 0.010, 0.0002), "dark")
    b.site(rotation, "stub_surface", (0, 0, 0.024), 0.0008)
    return rotation


def build_compact(b, base):
    """Preserve the qualified candidate mechanics under the existing room pose."""
    option = b.root.find("option")
    option.set("timestep", ".001")
    option.set("iterations", "100")
    before = set(base.iter("geom"))
    p = Parts(b)
    cabinet(p, base)
    drawer = telescopes(p, base)
    stage = five_axes(p, drawer)
    # Preserve the candidate response despite the room's different default.
    for geom in base.iter("geom"):
        if geom not in before:
            geom.set("solref", ".02 1")
    b.metadata["mechanism_revision"] = dict(
        version=3,
        retired_controls=["specimen_exchange", "chamber_access", "specimen_azimuth"],
        automatic_legacy_mapping=False,
        control_order=[
            "door_slide_outer",
            "door_slide_inner",
            "stage_x",
            "stage_y",
            "stage_z",
            "stage_tilt",
            "stage_rotation",
        ],
        migration="Revision 3 rebases the stage and replaces the 600 mm door axis with two 210 mm telescopes. The azimuth frame/range changes to a bounded rotation within a tilt cradle. Replan old actions; no key/value aliases.",
        access_m=0.420,
        family_chamber_inner_width_height_m=[0.330, 0.270],
        physics=dict(
            timestep_s=0.001, solver_iterations=100, instrument_contact_solref=[0.02, 1]
        ),
        sample_datum="Actual top of the clamped 30 mm diameter by 12 mm stub; old above-surface site retired.",
        support_model="Authored ideal-joint guides with explicit open bearings and ideal gravity compensation; no calibrated stiffness/preload or vacuum seal",
    )
    b.metadata["capabilities"] = [
        "Two supported telescopes expose and return the retained specimen stage",
        "Five real stage axes with measured collision-aware dry inspection",
        "Actual pole-to-stub surface registration at an authored 8 mm reference gap",
    ]
    b.metadata["limitations"] += [
        "Family-informed chamber/stage envelopes; hidden mechanisms, inertia and drives are authored, not BU or OEM CAD.",
        "The full coupled joint box is obstructed. Use checked trajectories; arbitrary raw commands may collide.",
        "The 30 mm inert stub is rigidly retained; no gripping, removable specimen transfer, vacuum, electron optics, image formation or scientific response model.",
        "Rotation has finite plus/minus pi limits; it is not continuous multi-turn rotation. Closed internal camera visibility is unqualified.",
    ]
    return stage
