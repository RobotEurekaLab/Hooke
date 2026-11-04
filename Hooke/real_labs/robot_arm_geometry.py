"""Supported generic six-axis arm with contact-bearing exterior geometry.

These are authored robot dimensions and ideal constrained bearing interfaces,
not a manufacturer replica. Open collars retain explicit running gaps; lumped
body inertias and drives keep the established mechanical surrogate contract.
"""

import math
import xml.etree.ElementTree as ET

from real_labs.mounting import annular_mount


class ArmParts:
    def __init__(self, builder, prefix, scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Arm scale must be finite and positive")
        self.b, self.prefix, self.scale = builder, prefix, scale
        self.interfaces = []

    def key(self, key):
        return f"{self.b.name}__arm_contact_{self.prefix}{key}"

    def point(self, *xyz):
        return tuple(v * self.scale for v in xyz)

    def box(self, parent, key, pos, half, material="metal"):
        return self.b.box(
            parent,
            self.point(*pos),
            self.point(*half),
            material,
            collision=True,
            mass=0,
            name=self.key(key),
        )

    def cylinder(self, parent, key, pos, radius, half, material="metal", axis="z"):
        quaternion = {
            "z": (1, 0, 0, 0),
            "x": (0.70710678, 0, 0.70710678, 0),
            "y": (0.70710678, 0.70710678, 0, 0),
        }[axis]
        return self.b.cylinder(
            parent,
            self.point(*pos),
            radius * self.scale,
            half * self.scale,
            material,
            quat=quaternion,
            collision=True,
            mass=0,
            name=self.key(key),
        )

    def rod(self, parent, key, start, end, radius, material="metal"):
        return self.b.rod(
            parent,
            self.point(*start),
            self.point(*end),
            radius * self.scale,
            material,
            collision=True,
            mass=0,
            name=self.key(key),
        )

    def collar(self, parent, key, pos, inner, outer, half, axis):
        # One hollow visual surface plus 16 convex contact sectors. Neither
        # representation fills the bearing bore with a convex mesh hull.
        node = ET.SubElement(
            parent,
            "body",
            name=self.key(key + "_mount"),
            pos=" ".join(map(str, self.point(*pos))),
            gravcomp="1",
        )
        if axis == "y":
            node.set("quat", ".7071067811865476 .7071067811865476 0 0")
        elif axis == "x":
            node.set("quat", ".7071067811865476 0 .7071067811865476 0")
        elif axis != "z":
            raise ValueError("Unknown collar axis")
        annular_mount(
            self.b,
            node,
            self.prefix + key,
            (0, 0, 0),
            inner * self.scale,
            outer * self.scale,
            half * self.scale,
            axis="z",
            proxies=False,
            namespace="arm_shell",
        )
        node.find("geom").set("mass", str(0.005 * self.scale))
        for i in range(16):
            angle = (i + 0.5) * math.tau / 16
            r = (inner + outer) / 2
            g = self.box(
                node,
                f"{key}_sector_{i}",
                (r * math.cos(angle), r * math.sin(angle), 0),
                ((outer - inner) / 2, outer * math.tan(math.pi / 16), half),
            )
            g.set("euler", f"0 0 {angle}")
            g.set("rgba", ".5 .5 .5 0")
        self.interfaces.append(
            dict(
                collar=self.key(key),
                radial_clearance_m=0.0006 * self.scale,
                axis=axis,
                scope="Ideal-joint bearing, not rolling-contact load simulation",
            )
        )

    def move(
        self, parent, key, pos, axis, limits, mass, *, kp=180, force=120, slide=False
    ):
        return self.b.moving(
            parent,
            self.prefix + key,
            self.point(*pos),
            axis,
            limits,
            kind="slide" if slide else "hinge",
            mass=mass * self.scale,
            kp=kp,
            force=force,
        )

    def cross_pin(self, parent, key, center, radius, length, axis="y"):
        self.cylinder(
            parent, key + "_shaft", center, radius, length / 2, "bright", axis
        )


def _yaw_and_shoulder(p, base):
    p.cylinder(base, "foundation", (0, 0, 0.040), 0.095, 0.040, "dark").set(
        "mass", str(2 * p.scale)
    )
    p.collar(base, "yaw_bearing", (0, 0, 0.092), 0.0366, 0.064, 0.012, "z")
    # Mounting plate reaches the supporting work surface at local z=0.
    for angle in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):
        p.cylinder(
            base,
            f"foundation_bolt_{angle}",
            (0.079 * math.cos(angle), 0.079 * math.sin(angle), 0.080),
            0.005,
            0.004,
            "bright",
        )
    yaw = p.move(
        base, "base_yaw", (0, 0, 0.115), (0, 0, 1), (-math.pi, math.pi), 3, kp=250
    )
    p.cylinder(yaw, "yaw_spindle", (0, 0, -0.013), 0.036, 0.021, "bright")
    p.cylinder(yaw, "yaw_housing", (0, 0, 0), 0.073, 0.009, "blue")
    for side in (-1, 1):
        p.box(
            yaw,
            f"shoulder_support_{side}",
            (0, side * 0.060, 0.030),
            (0.025, 0.011, 0.0225),
            "blue",
        )
        p.collar(
            yaw,
            f"shoulder_bearing_{side}",
            (0, side * 0.060, 0.075),
            0.0186,
            0.028,
            0.010,
            "y",
        )
    shoulder = p.move(
        yaw, "shoulder", (0, 0, 0.075), (0, 1, 0), (-1.65, 1.65), 2, kp=250
    )
    p.cylinder(shoulder, "shoulder_housing", (0, 0, 0), 0.065, 0.040, "blue", "y")
    p.cross_pin(shoulder, "shoulder", (0, 0, 0), 0.018, 0.150)
    for side in (-1, 1):
        p.cylinder(
            shoulder,
            f"shoulder_pin_end_{side}",
            (0, side * 0.075, 0),
            0.023,
            0.003,
            "bright",
            "y",
        )
    return shoulder


def _elbow(p, shoulder):
    length = math.hypot(0.21, 0.25)
    u = (0.21 / length, 0.25 / length)
    endpoint = (0.21 - 0.095 * u[0], 0, 0.25 - 0.095 * u[1])
    p.rod(
        shoulder,
        "upper_link_neck",
        (0.04 * u[0], 0, 0.04 * u[1]),
        (0.13 * u[0], 0, 0.13 * u[1]),
        0.024,
        "metal",
    )
    p.rod(
        shoulder,
        "upper_link",
        (0.115 * u[0], 0, 0.115 * u[1]),
        endpoint,
        0.040,
        "shell",
    )
    p.rod(
        shoulder,
        "elbow_fork_tie",
        (endpoint[0], -0.060, endpoint[2]),
        (endpoint[0], 0.060, endpoint[2]),
        0.014,
        "blue",
    )
    for side in (-1, 1):
        p.rod(
            shoulder,
            f"elbow_fork_{side}",
            (endpoint[0], side * 0.055, endpoint[2]),
            (0.21 - 0.029 * u[0], side * 0.055, 0.25 - 0.029 * u[1]),
            0.007,
            "shell",
        )
        p.collar(
            shoulder,
            f"elbow_bearing_{side}",
            (0.21, side * 0.055, 0.25),
            0.0166,
            0.026,
            0.009,
            "y",
        )
    elbow = p.move(
        shoulder, "elbow", (0.21, 0, 0.25), (0, 1, 0), (-2.15, 2.15), 1.4, kp=220
    )
    p.cylinder(elbow, "elbow_housing", (0, 0, 0), 0.052, 0.035, "blue", "y")
    p.cross_pin(elbow, "elbow", (0, 0, 0), 0.016, 0.146)
    for side in (-1, 1):
        p.cylinder(
            elbow,
            f"elbow_pin_end_{side}",
            (0, side * 0.073, 0),
            0.021,
            0.003,
            "bright",
            "y",
        )
    return elbow


def _wrists(p, elbow):
    p.rod(elbow, "forearm_link", (0.035, 0, 0.011), (0.160, 0, 0.05), 0.030, "shell")
    for side in (-1, 1):
        p.rod(
            elbow,
            f"roll_support_{side}",
            (0.16, side * 0.015, 0.05),
            (0.215, side * 0.015, 0.039),
            0.010,
            "metal",
        )
    p.collar(elbow, "roll_bearing", (0.215, 0, 0.075), 0.0196, 0.042, 0.012, "x")
    roll = p.move(
        elbow, "wrist_roll", (0.24, 0, 0.075), (1, 0, 0), (-math.pi, math.pi), 0.45
    )
    p.cylinder(roll, "roll_spindle", (-0.0125, 0, 0), 0.019, 0.0275, "bright", "x")
    p.cylinder(roll, "roll_housing", (0.005, 0, 0), 0.038, 0.015, "blue", "x")
    p.box(roll, "pitch_fork_tie", (0.014, 0, -0.032), (0.009, 0.049, 0.006), "blue")
    for side in (-1, 1):
        p.rod(
            roll,
            f"pitch_fork_{side}",
            (0.014, side * 0.042, -0.024),
            (0.065, side * 0.042, -0.024),
            0.007,
            "metal",
        )
        p.collar(
            roll,
            f"pitch_bearing_{side}",
            (0.065, side * 0.042, 0),
            0.0106,
            0.020,
            0.007,
            "y",
        )
    pitch = p.move(roll, "wrist_pitch", (0.065, 0, 0), (0, 1, 0), (-1.9, 1.9), 0.25)
    p.cylinder(pitch, "pitch_housing", (0, 0, 0), 0.032, 0.025, "shell", "y")
    p.cross_pin(pitch, "pitch", (0, 0, 0), 0.010, 0.110)
    for side in (-1, 1):
        p.cylinder(
            pitch,
            f"pitch_pin_end_{side}",
            (0, side * 0.055, 0),
            0.014,
            0.003,
            "bright",
            "y",
        )
    for side in (-1, 1):
        p.rod(
            pitch,
            f"tool_bearing_support_{side}",
            (0.023, 0, side * 0.014),
            (0.039, 0, side * 0.014),
            0.004,
            "metal",
        )
    p.collar(pitch, "tool_bearing", (0.043, 0, 0), 0.0086, 0.019, 0.006, "x")
    wrist = p.move(
        pitch, "tool_yaw", (0.045, 0, 0), (1, 0, 0), (-math.pi, math.pi), 0.15
    )
    p.cylinder(wrist, "tool_spindle", (0.0045, 0, 0), 0.008, 0.0145, "bright", "x")
    p.cylinder(wrist, "tool_flange", (0.01755, 0, 0), 0.028, 0.01295, "metal", "x")
    return wrist


def _hand(p, wrist):
    p.box(wrist, "palm", (0.03385, 0, 0), (0.01285, 0.072, 0.022), "dark")
    for z in (-0.012, 0.012):
        p.cylinder(
            wrist, f"finger_rail_{z}", (0.054, 0, z), 0.0025, 0.073, "bright", "y"
        )
        for side in (-1, 1):
            p.box(
                wrist,
                f"finger_rail_mount_{z}_{side}",
                (0.0495, side * 0.073, z),
                (0.0075, 0.008, 0.006),
                "metal",
            )
    for suffix, sign in (("left", -1), ("right", 1)):
        finger = p.move(
            wrist,
            f"gripper_{suffix}",
            (0.063, sign * 0.027, 0),
            (0, sign, 0),
            (0, 0.025 * p.scale),
            0.035,
            force=20,
            slide=True,
        )
        for i, z in enumerate((-0.012, 0.012)):
            p.collar(
                finger,
                f"{suffix}_guide_{i}",
                (-0.009, 0, z),
                0.0031,
                0.0065,
                0.007,
                "y",
            )
        p.box(
            finger, f"finger_{suffix}", (0.025, 0, 0), (0.0285, 0.005, 0.019), "metal"
        )
        p.box(
            finger,
            f"finger_pad_{suffix}",
            (0.047, -sign * 0.006, 0),
            (0.008, 0.003, 0.018),
            "rubber",
        )
        p.b.site(
            finger,
            p.prefix + f"finger_{suffix}",
            p.point(0.05, -sign * 0.009, 0),
            0.002,
        )
    p.b.site(wrist, p.prefix + "tcp", p.point(0.112, 0, 0))


def _pair_classes(p):
    """Exact designed interface surfaces, never an adjacent-body blanket mask."""

    def sectors(*collars):
        return [f"{c}_sector_{i}" for c in collars for i in range(16)]

    definitions = [
        (
            "yaw_bearing",
            ["foundation"] + sectors("yaw_bearing"),
            ["yaw_spindle", "yaw_housing"],
        ),
        (
            "shoulder_pivot",
            ["yaw_spindle", "yaw_housing", "shoulder_support_-1", "shoulder_support_1"]
            + sectors("shoulder_bearing_-1", "shoulder_bearing_1"),
            [
                "shoulder_housing",
                "shoulder_shaft",
                "shoulder_pin_end_-1",
                "shoulder_pin_end_1",
            ],
        ),
        (
            "elbow_pivot",
            ["upper_link", "elbow_fork_tie", "elbow_fork_-1", "elbow_fork_1"]
            + sectors("elbow_bearing_-1", "elbow_bearing_1"),
            ["elbow_housing", "elbow_shaft", "elbow_pin_end_-1", "elbow_pin_end_1"],
        ),
        (
            "roll_bearing",
            ["forearm_link", "roll_support_-1", "roll_support_1"]
            + sectors("roll_bearing"),
            ["roll_spindle", "roll_housing"],
        ),
        (
            "pitch_pivot",
            ["roll_housing", "pitch_fork_tie", "pitch_fork_-1", "pitch_fork_1"]
            + sectors("pitch_bearing_-1", "pitch_bearing_1"),
            ["pitch_housing", "pitch_shaft", "pitch_pin_end_-1", "pitch_pin_end_1"],
        ),
        (
            "tool_bearing",
            ["pitch_housing", "tool_bearing_support_-1", "tool_bearing_support_1"]
            + sectors("tool_bearing"),
            ["tool_spindle", "tool_flange"],
        ),
    ]
    fixed_hand = (
        ["palm"]
        + [f"finger_rail_{z}" for z in (-0.012, 0.012)]
        + [f"finger_rail_mount_{z}_{side}" for z in (-0.012, 0.012) for side in (-1, 1)]
    )
    for side in ("left", "right"):
        definitions.append(
            (
                f"{side}_finger_guide",
                fixed_hand,
                [f"finger_{side}"] + sectors(f"{side}_guide_0", f"{side}_guide_1"),
            )
        )
    return [
        dict(
            name=label,
            fixed_or_proximal=[p.key(k) for k in first],
            moving_or_distal=[p.key(k) for k in second],
            minimum_clearance_m=0.00025 * p.scale,
            basis="Exact named bearing, shaft, end-cap and mounting-interface surfaces; no whole-link or whole-body exemption",
        )
        for label, first, second in definitions
    ]


def build_arm(builder, base, *, prefix="", scale=1.0):
    p = ArmParts(builder, prefix, float(scale))
    shoulder = _yaw_and_shoulder(p, base)
    elbow = _elbow(p, shoulder)
    wrist = _wrists(p, elbow)
    _hand(p, wrist)
    builder.metadata.setdefault("arm_interfaces", {})[prefix or "main"] = dict(
        scale=scale,
        bearing_interfaces=p.interfaces,
        engineered_pair_classes=_pair_classes(p),
        primary_outer_surfaces="Enabled housing, link, flange, palm and finger contact geometry; hollow visual collars paired with explicit convex sectors",
        inertia_basis="Established joint-body lumped inertias and authored actuator gains; each fixed collar adds an explicitly assumed 5 g times scale, foundation 2 kg times scale; all added collar masses share the original ideal gravity compensation",
        support="Grounded foundation; ideal-joint constrained shafts and open collars with running clearance, not a calibrated bearing-force model",
        fidelity="Original representative six-axis robot; not Waterloo or NASA manufacturer CAD",
    )
