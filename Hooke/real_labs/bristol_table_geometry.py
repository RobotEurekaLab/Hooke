"""An original eight-leg EQUALS mechanical surrogate, in metres.

Bristol publishes the 3 m square platform, eight 300 mm stroke actuators,
70 kN dynamic capacity per actuator, and four vertical static support sections.
Anchor coordinates, joints, masses, preload law and controller are authored
estimates. This model is not a hydraulic, earthquake or structural solver.
"""

import math
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from real_labs.architecture import numbers
from real_labs.mounting import annular_mount

ID = "shake_platform"
PREFIX = ID + "__equals_v2_"
SCENE = "bristol_equals_shaking_table"
PLATFORM_ORIGIN = np.array([0.0, 0.0, -0.065])
LEGS = (
    ("vertical_0", (-0.95, -0.95, -1.44), (-0.95, -0.95, -0.24)),
    ("vertical_1", (0.95, -0.95, -1.44), (0.95, -0.95, -0.24)),
    ("vertical_2", (0.95, 0.95, -1.44), (0.95, 0.95, -0.24)),
    ("vertical_3", (-0.95, 0.95, -1.44), (-0.95, 0.95, -0.24)),
    ("horizontal_x_0", (-1.95, -1.68, -0.35), (-0.35, -1.68, -0.35)),
    ("horizontal_x_1", (1.95, 1.68, -0.35), (0.35, 1.68, -0.35)),
    ("horizontal_y_0", (-1.68, 1.95, -0.55), (-1.68, 0.35, -0.55)),
    ("horizontal_y_1", (1.68, -1.95, -0.55), (1.68, -0.35, -0.55)),
)
CONTROLS = tuple("v2_" + name for name, _, _ in LEGS)


def leg_lengths(translation, rotation=None):
    """Eight physical length changes for a rigid platform pose; no state writes."""
    translation = np.asarray(translation, dtype=float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    values = np.array(
        [
            np.linalg.norm(
                PLATFORM_ORIGIN
                + translation
                + rotation @ (np.array(p) - PLATFORM_ORIGIN)
                - a
            )
            - np.linalg.norm(np.array(p) - a)
            for _, a, p in LEGS
        ]
    )
    # Subtracting nominal length can make an exact 150 mm geometric endpoint
    # evaluate as 0.15000000000000013. Normalize only this private arithmetic
    # residue; external target validation remains strict and no live state is
    # changed. This is not extra physical travel.
    boundary = np.abs(np.abs(values) - 0.15) <= 1e-12
    values[boundary] = np.copysign(0.15, values[boundary])
    return values


def orientation(vector):
    result = np.zeros(4)
    mujoco.mju_quatZ2Vec(result, np.asarray(vector) / np.linalg.norm(vector))
    return result


class Parts:
    def __init__(self, builder):
        self.b = builder

    def box(self, body, key, pos, half, material="metal", **extra):
        return self.b.box(
            body, pos, half, material, name=PREFIX + key, collision=True, **extra
        )

    def sphere(self, body, key, pos, radius, material="bright"):
        return self.b.geom(
            body, "sphere", (radius,), pos, material, name=PREFIX + key, collision=True
        )

    def rod(self, body, key, first, last, radius, material="bright", **extra):
        return self.b.rod(
            body,
            first,
            last,
            radius,
            material,
            name=PREFIX + key,
            collision=True,
            **extra,
        )

    def annulus(self, body, key, center, inner, outer, half, material="metal"):
        """Closed visual wall with an open bore and conservative convex contacts.

        Sixteen overlapping sector boxes are welded to their parent. Their
        inner faces preserve the declared running clearance. The visual mesh
        has a genuine opening and is never treated as a solid convex cylinder.
        """
        annular_mount(
            self.b,
            body,
            key,
            center,
            inner,
            outer,
            half,
            axis="z",
            namespace="equals_v2",
            material=material,
        )
        for i in range(16):
            angle = (i + 0.5) * math.tau / 16
            r = (inner + outer) / 2
            self.box(
                body,
                f"{key}_contact_{i}",
                (
                    center[0] + r * math.cos(angle),
                    center[1] + r * math.sin(angle),
                    center[2],
                ),
                ((outer - inner) / 2, outer * math.tan(math.pi / 16), half),
                material,
                euler=(0, 0, angle),
                group="3",
                rgba=(0.5, 0.5, 0.5, 0),
            )


def _platform_frame(p, body):
    p.box(body, "platform", (0, 0, 0), (1.5, 1.5, 0.065), "bright")
    for x in (-1.3, -0.78, -0.26, 0.26, 0.78, 1.3):
        for y in (-1.3, -0.78, -0.26, 0.26, 0.78, 1.3):
            p.b.cylinder(
                body, (x, y, 0.067), 0.014, 0.002, "dark", name=PREFIX + f"bolt_{x}_{y}"
            )
    for x in (-1.05, 1.05):
        for y in (-1.05, 1.05):
            p.box(
                body, f"foot_{x}_{y}", (x, y, 0.0925), (0.17, 0.17, 0.0275), "warning"
            )
            p.box(
                body, f"post_{x}_{y}", (x, y, 1.475), (0.065, 0.065, 1.355), "warning"
            )
            for dx in (-0.115, 0.115):
                p.b.cylinder(
                    body,
                    (x + dx, y, 0.126),
                    0.018,
                    0.008,
                    "metal",
                    name=PREFIX + f"footbolt_{x}_{y}_{dx}",
                )
    for z in (1.43, 2.78):
        for y in (-1.05, 1.05):
            p.box(body, f"cross_x_{y}_{z}", (0, y, z), (1.13, 0.07, 0.075), "warning")
        for x in (-1.05, 1.05):
            p.box(body, f"cross_y_{x}_{z}", (x, 0, z), (0.07, 1.13, 0.075), "warning")
    # Black diagonal braces and yellow bolted members are visible in the photo.
    for y in (-1.05, 1.05):
        p.rod(body, f"diagonal_{y}", (-1.04, y, 0.17), (1.04, y, 1.39), 0.027, "dark")
    p.box(body, "accelerometer", (0.94, -1.135, 1.48), (0.06, 0.025, 0.035), "dark")
    p.rod(
        body,
        "frame_sensor_lead",
        (0.94, -1.16, 1.48),
        (0.94, -1.16, 0.15),
        0.006,
        "dark",
    )
    p.b.site(body, "structural_frame", (0, 0, 1.475), size=0.025)
    p.b.site(body, "physical_platform_center", (0, 0, 0.065), size=0.012)
    p.box(
        body,
        "inert_registration_patch",
        (0.35, -0.8, 0.066),
        (0.04, 0.04, 0.001),
        "copper",
    )
    p.b.site(body, "physical_witness", (0.35, -0.8, 0.067), size=0.005)


def _leg(p, base, platform, equality, name, anchor, tip):
    b = p.b
    a, end = np.array(anchor), np.array(tip)
    vector = end - a
    length = float(np.linalg.norm(vector))
    quat = orientation(vector)
    key = "v2_" + name
    fixed = ET.SubElement(
        base, "body", name=PREFIX + name + "_socket", pos=numbers(a), quat=numbers(quat)
    )
    p.annulus(fixed, name + "_base_bearing", (0, 0, 0), 0.111, 0.16, 0.045)
    p.box(fixed, name + "_anchor_plate", (0, 0, -0.18), (0.17, 0.17, 0.03))
    for x in (-0.145, 0.145):
        p.box(fixed, name + f"_anchor_cheek_{x}", (x, 0, -0.09), (0.025, 0.13, 0.095))
    # Concrete-filled box-section anchor stands reach the bearing bracket.
    # Positions/depth are engineering estimates, not a survey of the pit.
    pedestal = a - 0.18 * vector / length
    top = pedestal[2] - (0.15 if abs(vector[2]) < 1e-8 else 0.015)
    bottom = -1.72
    p.box(
        base,
        name + "_anchor_box",
        (pedestal[0], pedestal[1], (top + bottom) / 2),
        (0.18, 0.18, (top - bottom) / 2),
        "dark",
    )
    p.box(
        base,
        name + "_anchor_foot",
        (pedestal[0], pedestal[1], -1.70),
        (0.25, 0.25, 0.025),
    )
    barrel = ET.SubElement(
        base, "body", name=PREFIX + name + "_barrel", pos=numbers(a), quat=numbers(quat)
    )
    ET.SubElement(barrel, "inertial", pos="0 0 .4", mass="45", diaginertia="3 3 .4")
    for axis, direction in (("pitch", (1, 0, 0)), ("yaw", (0, 1, 0))):
        ET.SubElement(
            barrel,
            "joint",
            name=PREFIX + name + "_" + axis,
            type="hinge",
            axis=numbers(direction),
            limited="true",
            range="-.25 .25",
            damping="10",
            armature=".02",
        )
    p.sphere(barrel, name + "_base_ball", (0, 0, 0), 0.11)
    barrel_end = 0.86 if name.startswith("vertical") else 1.10
    p.annulus(
        barrel,
        name + "_barrel",
        (0, 0, (barrel_end + 0.06) / 2),
        0.046,
        0.095,
        (barrel_end - 0.06) / 2,
    )
    p.annulus(
        barrel, name + "_gland", (0, 0, barrel_end - 0.022), 0.046, 0.111, 0.022, "dark"
    )
    # Static-section exterior on the four vertical cylinders. Its support law
    # is explicit below; no hydraulic response or pressure circuit is claimed.
    if name.startswith("vertical"):
        p.annulus(
            barrel, name + "_static_section", (0, 0, 0.28), 0.096, 0.13, 0.13, "blue"
        )
    p.box(
        barrel,
        name + "_lvdt_mount",
        (0.106, 0, barrel_end / 2),
        (0.014, 0.035, 0.045),
        "dark",
    )
    p.rod(
        barrel, name + "_lvdt", (0.125, 0, 0.17), (0.125, 0, barrel_end - 0.11), 0.011
    )
    # Captive connector and a capped local lead: there is deliberately no
    # straight rigid hose pretending to flex between ground and the leg.
    p.rod(
        barrel,
        name + "_local_lead",
        (0.125, 0, 0.17),
        (0.12, 0.08, 0.17),
        0.008,
        "dark",
    )
    p.box(barrel, name + "_lead_cap", (0.12, 0.08, 0.17), (0.015, 0.015, 0.015), "dark")
    rod = ET.SubElement(
        barrel, "body", name=PREFIX + name + "_rod", pos=numbers((0, 0, length))
    )
    ET.SubElement(rod, "inertial", pos="0 0 -.3", mass="18", diaginertia="1.2 1.2 .04")
    joint = b.name + "__joint_" + key
    attrs = dict(
        name=joint,
        type="slide",
        axis="0 0 1",
        limited="true",
        range="-.15 .15",
        damping="200",
        armature="2",
    )
    if name.startswith("vertical"):
        attrs.update(stiffness="20000", springref=".249")
    ET.SubElement(rod, "joint", **attrs)
    shaft = 0.8 if name.startswith("vertical") else 1.0
    # Cylinder ends are flat: a capsule would extend beyond the specified bore.
    b.geom(
        rod,
        "cylinder",
        (0.045, (shaft - 0.04) / 2),
        (0, 0, -(shaft + 0.04) / 2),
        "bright",
        name=PREFIX + name + "_shaft",
        collision=True,
    )
    p.sphere(rod, name + "_tip_ball", (0, 0, 0), 0.075)
    socket = ET.SubElement(
        platform,
        "body",
        name=PREFIX + name + "_platform_socket",
        pos=numbers(end - PLATFORM_ORIGIN),
        quat=numbers(quat),
    )
    p.annulus(socket, name + "_tip_bearing", (0, 0, 0), 0.077, 0.12, 0.035)
    # Fixed-to-platform fork bars attach around the open bearing, keeping the
    # ball and entering rod free. Two extended webs reach the plate underside.
    lateral = (
        np.array([1.0, 0.0, 0.0])
        if name.startswith("vertical")
        else np.cross(vector / length, [0.0, 0.0, 1.0])
    )
    for sign in (-1, 1):
        start = end + sign * 0.108 * lateral - PLATFORM_ORIGIN
        finish = start.copy()
        finish[2] = -0.035
        p.box(
            platform,
            name + f"_upper_web_{sign}",
            (start + finish) / 2,
            (
                (0.018, 0.065, abs(finish[2] - start[2]) / 2 + 0.008)
                if name.startswith("vertical")
                else (
                    (0.035, 0.022, abs(finish[2] - start[2]) / 2 + 0.008)
                    if name.startswith("horizontal_x")
                    else (0.022, 0.035, abs(finish[2] - start[2]) / 2 + 0.008)
                )
            ),
            "metal",
        )
        foot = finish.copy()
        # A short plate runs inward to the 3 m deck at each exterior bracket.
        inward = 1 if name.startswith("horizontal_x") else 0
        if name.startswith("horizontal"):
            edge = foot.copy()
            edge[inward] = math.copysign(1.40, foot[inward])
            half = [0.045, 0.045, 0.022]
            half[inward] = abs(foot[inward] - edge[inward]) / 2 + 0.045
            p.box(platform, name + f"_bracket_flange_{sign}", (foot + edge) / 2, half)
    ET.SubElement(
        equality,
        "connect",
        name=PREFIX + name + "_spherical_connection",
        body1=rod.get("name"),
        body2=platform.get("name"),
        anchor="0 0 0",
        solref=".004 1",
        solimp=".99 .999 .001",
    )
    actuator = b.name + "__actuator_" + key
    ET.SubElement(
        b.actuator,
        "position",
        name=actuator,
        joint=joint,
        kp="2000000",
        kv="45000",
        forcelimited="true",
        forcerange="-70000 70000",
        ctrllimited="true",
        ctrlrange="-.15 .15",
    )
    b.metadata["joints"][key] = joint
    b.metadata["actuators"][key] = actuator
    b.metadata["joint_ranges"][key] = [-0.15, 0.15]
    b.metadata["actuator_modes"][key] = "position"


def build_table(b, base, params):
    if params.get("mechanical_revision", 2) != 2:
        raise ValueError(
            "Bristol version 2 requires its explicit eight leg controls; no legacy XY mapping"
        )
    p = Parts(b)
    p.box(base, "strong_block", (0, 0, -1.80), (2.43, 2.43, 0.08), "metal")
    for sign in (-1, 1):
        p.box(
            base,
            f"pit_wall_x_{sign}",
            (sign * 2.37, 0, -0.89),
            (0.06, 2.43, 0.83),
            "cream",
        )
        p.box(
            base,
            f"pit_wall_y_{sign}",
            (0, sign * 2.37, -0.89),
            (2.31, 0.06, 0.83),
            "cream",
        )
    # A fixed, supported service manifold is visible context. Its capped leads
    # deliberately stop on the strong block; no rigid line impersonates a hose
    # flexing to a moving cylinder.
    for x in (-0.5, 0.5):
        p.box(base, f"manifold_mount_{x}", (x, 2.21, -1.0), (0.05, 0.10, 0.72))
    p.box(base, "service_manifold", (0, 2.21, -0.38), (0.62, 0.11, 0.10), "dark")
    for i in range(8):
        x = -0.49 + i * 0.14
        p.rod(base, f"manifold_port_{i}", (x, 2.10, -0.38), (x, 2.04, -0.38), 0.019)
        previous = np.array([x, 2.04, -0.38])
        for j in range(1, 9):
            point = np.array(
                [
                    x + 0.04 * math.sin(j * math.pi / 8),
                    2.04 - 0.03 * math.sin(j * math.pi / 8),
                    -0.38 - 0.12 * j,
                ]
            )
            p.rod(base, f"parked_service_{i}_{j}", previous, point, 0.009, "dark")
            previous = point
        p.box(base, f"service_clip_{i}", (x, 2.10, -1.34), (0.025, 0.07, 0.025))
    world = b.root.find("worldbody")
    platform = ET.SubElement(
        world, "body", name=PREFIX + "platform_body", pos=numbers(PLATFORM_ORIGIN)
    )
    ET.SubElement(platform, "freejoint", name=PREFIX + "platform_free")
    ET.SubElement(
        platform, "inertial", pos="0 0 .45", mass="2000", diaginertia="1800 1800 2600"
    )
    _platform_frame(p, platform)
    equality = b.root.find("equality")
    if equality is None:
        equality = ET.SubElement(b.root, "equality")
    for name, anchor, tip in LEGS:
        _leg(p, base, platform, equality, name, anchor, tip)
    b.metadata["mechanical_interface"] = dict(
        version=2,
        controls=list(CONTROLS),
        retired_controls=["table_x", "table_y"],
        migration="The old two ideal slides are replaced by eight coupled leg servos and a free rigid platform. No automatic legacy action mapping.",
        coordination="Use the source-bound platform-pose task. Eight actuator commands are redundant constraints; arbitrary independent settings are not a feasible workspace.",
        source_dimensions=dict(
            platform_m=[3, 3],
            actuator_stroke_m=0.300,
            actuator_dynamic_capacity_N=70000,
        ),
        authored=dict(
            platform_and_frame_mass_kg=2000,
            leg_barrel_mass_kg=45,
            leg_rod_mass_kg=18,
            servo_stiffness_N_m=2000000,
            servo_damping_N_s_m=45000,
            vertical_static_spring_N_m=20000,
            vertical_static_spring_reference_m=0.249,
        ),
        static_support="Four passive preload laws approximate the documented vertical static support sections. They are not simulated hydraulic pressure or calibrated load balancing.",
    )
    b.metadata["capabilities"] = [
        "Eight contact-capable telescopic legs move a mechanically connected rigid platform under gravity",
        "Measured coordinated inert platform displacement and return",
    ]
    b.metadata["coordinated_inspection_targets"] = dict(
        zip(CONTROLS, leg_lengths((0.05, -0.04, 0.06)).tolist())
    )
    b.metadata["limitations"] += [
        "Hidden anchors, masses, bearing clearances, preload and controller are authored estimates. No OEM CAD or measured actuator layout.",
        "Published 15 tonne capacity and about 5g performance are source context, not validated simulation capability. No hydraulics, earthquake waveform, material failure or structural dynamics validation.",
        "Ideal angular joints and spherical equality constraints transfer loads across explicit running clearances. The retained yellow frame is rigid, not a calibrated structural specimen.",
    ]
