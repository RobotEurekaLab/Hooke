"""Manchester reference-informed bay using the local BSD-licensed Kinova Gen3."""

from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from real_labs.architecture import box


SOURCE = "https://sites.manchester.ac.uk/robotics/research/"
GEN3 = Path(__file__).resolve().parents[1] / "model/robot_menagerie/kinova_gen3"
HOME = (0.0, 0.26179939, math.pi, -2.26892803, 0.0, 0.95993109, math.pi / 2)


def _numbers(values):
    return " ".join(f"{value:.10g}" for value in values)


def _rotate_local_z(quaternion, angle):
    w, x, y, z = quaternion
    length = math.sqrt(w * w + x * x + y * y + z * z)
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return (
        (w * c - z * s) / length,
        (x * c + y * s) / length,
        (y * c - x * s) / length,
        (z * c + w * s) / length,
    )


def _import_gen3(b, mount):
    """Keep OEM-derived meshes, link transforms and inertias; expose bounded home offsets."""
    source = ET.parse(GEN3 / "gen3.xml").getroot()
    for source_mesh in source.findall("asset/mesh"):
        mesh = deepcopy(source_mesh)
        mesh.set("name", f"{b.name}__gen3_{source_mesh.get('name')}")
        mesh.set("file", str(GEN3 / "assets" / source_mesh.get("file")))
        b.root.find("asset").append(mesh)
    robot = deepcopy(source.find("worldbody/body"))
    robot.set("pos", "0 0 0.028")
    robot.set("quat", "0 0 0 1")
    mount.append(robot)
    # The base is welded to the scene's static body, so MuJoCo's usual
    # parent/child filter no longer covers this adjacent joint bearing.
    # Its overlapping OEM convex hulls are not a self-collision obstacle.
    contact = b.root.find("contact")
    if contact is None:
        contact = ET.SubElement(b.root, "contact")
    ET.SubElement(
        contact,
        "exclude",
        body1=f"{b.name}__gen3_base_link",
        body2=f"{b.name}__gen3_shoulder_link",
    )
    joint_index = 0
    for body in robot.iter("body"):
        original_name = body.get("name")
        body.set("name", f"{b.name}__gen3_{original_name}")
        body.set("gravcomp", "1")
        joint = body.find("joint")
        if joint is not None:
            # Fold the supplied home pose into the body frame. Controls report
            # offsets from that pose, never incorrectly report OEM joint angles.
            q = tuple(float(v) for v in body.get("quat", "1 0 0 0").split())
            body.set("quat", _numbers(_rotate_local_z(q, HOME[joint_index])))
            key = f"joint_{joint_index + 1}_offset"
            name = f"{b.name}__{key}"
            joint.attrib.clear()
            joint.attrib.update(
                name=name,
                type="hinge",
                axis="0 0 1",
                limited="true",
                range="-0.14 0.14",
                damping="1",
                armature="0.01",
            )
            actuator = name + "_servo"
            kp, kv, force = (2500, 100, 105) if joint_index < 4 else (800, 35, 52)
            ET.SubElement(
                b.actuator,
                "position",
                name=actuator,
                joint=name,
                kp=str(kp),
                kv=str(kv),
                ctrllimited="true",
                ctrlrange="-0.14 0.14",
                forcelimited="true",
                forcerange=f"-{force} {force}",
            )
            b.metadata["joints"][key] = name
            b.metadata["actuators"][key] = actuator
            b.metadata["joint_ranges"][key] = [-0.14, 0.14]
            b.metadata["actuator_modes"][key] = "position"
            joint_index += 1
        for geom in body.findall("geom"):
            visual = geom.attrib.pop("class") == "visual"
            geom.set("name", b.unique("gen3_visual" if visual else "gen3_collision"))
            geom.set("type", "mesh")
            geom.set("mesh", f"{b.name}__gen3_{geom.get('mesh')}")
            geom.set("group", "0" if visual else "3")
            geom.set("contype", "0" if visual else "1")
            geom.set("conaffinity", "0" if visual else "1")
            geom.set("material", f"{b.name}__mat_dark")
            if not visual:
                geom.set("rgba", "0 0 0 0")
        for child in list(body):
            if child.tag in ("camera", "site"):
                body.remove(child)
    # Narrow cream bands echo the reference appearance, without manufacturer labels.
    for name, z, radius in (
        ("base_link", 0.115, 0.048),
        ("half_arm_2_link", -0.060, 0.039),
        ("bracelet_link", -0.030, 0.040),
    ):
        body = next(
            part
            for part in robot.iter("body")
            if part.get("name").endswith("__gen3_" + name)
        )
        b.cylinder(body, (0, 0, z), radius, 0.007, "cream")
    return next(
        part
        for part in robot.iter("body")
        if part.get("name").endswith("__gen3_bracelet_link")
    )


def _manchester(b, base, params):
    b.box(base, (0, 0, 0.014), (0.115, 0.105, 0.014), "dark", collision=True)
    for x in (-0.09, 0.09):
        b.box(base, (x, -0.15, -0.025), (0.015, 0.025, 0.075), "blue")
        b.box(base, (x, -0.09, 0.03), (0.015, 0.085, 0.015), "blue")
        b.box(base, (x, -0.09, -0.085), (0.015, 0.085, 0.015), "blue")
        b.cylinder(base, (x, -0.065, -0.066), 0.008, 0.024, "metal")
        b.cylinder(base, (x, -0.065, -0.10), 0.022, 0.008, "dark")
    wrist = _import_gen3(b, base)
    hand = ET.SubElement(
        wrist,
        "body",
        name=f"{b.name}__original_fixed_three_finger_tool",
        pos="0 0 -0.061525",
        quat="0 1 0 0",
    )
    b.cylinder(hand, (0, 0, 0.024), 0.043, 0.024, "dark")
    for angle in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6):
        c, s = math.cos(angle), math.sin(angle)
        b.rod(
            hand,
            (0.027 * c, 0.027 * s, 0.040),
            (0.057 * c, 0.057 * s, 0.098),
            0.012,
            "cream",
        )
        b.geom(hand, "sphere", (0.012,), (0.057 * c, 0.057 * s, 0.098), "metal")
        b.rod(
            hand,
            (0.057 * c, 0.057 * s, 0.098),
            (0.065 * c, 0.065 * s, 0.145),
            0.008,
            "cream",
        )
        b.rod(
            hand,
            (0.065 * c, 0.065 * s, 0.145),
            (0.049 * c, 0.049 * s, 0.157),
            0.006,
            "dark",
        )
    b.site(hand, "tool_center", (0, 0, 0.125), 0.002)
    # The inert target is fixed in a genuine table-supported inspection jig;
    # no rigidly attached target is presented as a completed robotic grasp.
    b.box(base, (-0.55, 0.24, 0.012), (0.11, 0.09, 0.012), "metal", collision=True)
    b.box(base, (-0.55, 0.24, 0.047), (0.025, 0.025, 0.023), "sample", collision=True)
    for x in (-0.590, -0.510):
        b.box(base, (x, 0.24, 0.035), (0.013, 0.045, 0.011), "blue", collision=True)
    b.site(base, "inspection_block", (-0.55, 0.24, 0.047), 0.001)
    b.rod(base, (-0.045, -0.01, 0.12), (-0.16, -0.15, 0.05), 0.006, "warning")
    b.rod(base, (-0.16, -0.15, 0.05), (-0.22, -0.21, -0.28), 0.006, "warning")
    b.rod(base, (-0.22, -0.21, -0.28), (-0.12, -0.17, -0.39), 0.006, "warning")
    b.metadata["capabilities"] = [
        "seven_independent_gen3_joint_offset_servos",
        "visible_clamped_inspection_block",
        "fixed_three_finger_visual_tool",
    ]
    b.metadata["asset_provenance"] = {
        "robot": "Local MuJoCo Menagerie Kinova Gen3, OEM-derived meshes/inertias; substitute for the photographed robot",
        "source": "Hooke/model/robot_menagerie/kinova_gen3/gen3.xml",
        "license": "BSD-3-Clause",
        "license_file": "Hooke/model/robot_menagerie/kinova_gen3/LICENSE",
        "license_text": (GEN3 / "LICENSE").read_text(encoding="utf-8"),
        "original_home_rad": list(HOME),
        "control_semantics": "Offsets in radians around imported home pose, restricted to +/- 0.14 rad",
    }
    b.metadata["limitations"].append(
        "Gen3 is a locally available licensed substitute, not the exact photographed Kinova/Quanser model. Original three-finger tool is fixed open; no grasp success, OEM calibration, torque fidelity, full workspace or human interaction is claimed."
    )


def _features(world, definition):
    # Background equipment case sits on the window-side counter.
    box(
        world,
        "manchester_equipment_case",
        (0.34, 0.10, 0.32),
        (1.20, 1.59, 1.17),
        (0.055, 0.065, 0.075, 1),
    )
    box(
        world,
        "manchester_case_handle",
        (0.09, 0.014, 0.02),
        (1.20, 1.59, 1.51),
        (0.13, 0.14, 0.15, 1),
    )
    for index in range(3):
        box(
            world,
            f"manchester_case_rib_{index}",
            (0.025, 0.01, 0.29),
            (0.95 + index * 0.25, 1.48, 1.17),
            (0.10, 0.11, 0.12, 1),
            collision=False,
        )


BUILDERS = {"manchester_gen3_inspection_workspace": _manchester}
SOURCES = {
    "manchester_gen3_inspection_workspace": dict(
        reference="Manchester installed collaborative manipulation bay; local Gen3 substitute",
        url=SOURCE,
        dimensions_m=[1.05, 0.90, 1.10],
        dimension_basis="Robot meshes and link transforms from local licensed Gen3 MJCF. Room, clamp, cable and fixed three-finger appearance estimated from official photograph; photograph robot is not claimed to be Gen3.",
    )
}
SAMPLE_INTERFACES = {
    "manchester_gen3_inspection_workspace": (
        "inspection_block",
        (0.050, 0.050, 0.046),
        "clamped",
        "Inert block clamped to a table jig, independent of the moving arm",
    )
}
FEATURES = {"manchester_collaborative_robotics": _features}
