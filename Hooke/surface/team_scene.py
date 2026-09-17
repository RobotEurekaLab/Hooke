"""Attach the original free-floating G1 and a physical confirmation button."""

import hashlib
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from surface.artifacts import SCENES
from surface.humanoid import DEFAULT_LEGS, HOME, KD, KP, LEG_NAMES, PREFIX
from surface.scene import SOURCE_ROOT, camera


def attach_team(spec, mission):
    source = SOURCE_ROOT / "model/robot_menagerie/unitree_g1/g1_with_hands.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    root = ET.fromstring(source.read_text())
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(source.parent / "assets" / mesh.get("file")))
    default = root.find("default/default/position")
    default.attrib.pop("dampratio")
    rate = math.sqrt(mission.environment.gravity_m_s2 / 9.81)
    for element in root.find("actuator"):
        name = element.get("name")
        actuator = original.actuator(name).id
        if name in LEG_NAMES:
            index = LEG_NAMES.index(name)
            kp, kv = KP[index] * rate**2, KD[index] * rate
        else:
            kp, kv = (
                original.actuator_gainprm[actuator, 0],
                -original.actuator_biasprm[actuator, 2],
            )
        element.set("kp", str(kp))
        element.set("kv", str(kv))
    root.find("default/default/joint").set("frictionloss", str(0.3 * rate**2))
    index = root.find(".//body[@name='left_hand_index_1_link']")
    ET.SubElement(
        index, "site", name="press_tip", pos=".04 0 0", size=".006", group="5"
    )
    ET.SubElement(
        index,
        "geom",
        name="index_touchpad",
        type="sphere",
        size=".006",
        pos=".04 0 0",
        rgba=".45 .5 .55 1",
        mass="0",
        friction="1 .01 .001",
    )
    encoded = ET.tostring(root, encoding="unicode")
    directory = SCENES / (
        "team-g1-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "g1.xml"
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as temporary:
        temporary.write(encoded)
    Path(temporary.name).replace(path)
    child = mujoco.MjSpec.from_file(str(path))
    frame = spec.worldbody.add_frame(name="g1_origin", pos=[HOME[0], HOME[1], 0])
    frame.attach_body(child.body("pelvis"), prefix=PREFIX)
    rover = spec.body("rover")
    button_height, stand_half_height = 0.60, 0.29
    spring, damping = 20.0, 1.0
    rover.add_geom(
        name="confirmation_stand",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.02, stand_half_height, 0],
        pos=[-0.5, -0.43, 0.29],
        rgba=[0.6, 0.63, 0.65, 1],
        mass=0.1,
    )
    button = rover.add_body(name="team_button", pos=[-0.5, -0.45, button_height])
    button.add_joint(
        name="team_button_joint",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[0, 0.016],
        limited=True,
        damping=0.05,
    )
    button.add_geom(
        name="team_button_cap",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.075, 0.024, 0.075],
        rgba=[0.15, 0.8, 0.65, 1],
        mass=0.03,
    )
    spec.add_actuator(
        name="team_button_return",
        target="team_button_joint",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[spring, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -spring, -damping, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=True,
        ctrlrange=[0, 0.016],
        forcelimited=True,
        forcerange=[-3, 3],
    )
    for body, name, position, target in [
        (
            spec.body(PREFIX + "pelvis"),
            "team_follow",
            [2.8, -3.5, 1.05],
            [0.5, 0.4, 0.05],
        ),
        (rover, "team_closeup", [1.7, -2.8, 1.5], [-0.1, -0.3, 0.45]),
    ]:
        node = ET.Element("body")
        camera(node, name, position, target, 54)
        attrs = node[0].attrib
        body.add_camera(
            name=name,
            pos=position,
            xyaxes=np.fromstring(attrs["xyaxes"], sep=" "),
            fovy=54,
        )
    return spec


def reset_humanoid(model, data):
    source = SOURCE_ROOT / "model/robot_menagerie/unitree_g1/g1_with_hands.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    for index in range(1, original.njnt):
        name = original.joint(index).name
        data.qpos[int(model.joint(PREFIX + name).qposadr[0])] = original.key_qpos[
            0, int(original.jnt_qposadr[index])
        ]
        data.ctrl[model.actuator(PREFIX + name).id] = original.key_ctrl[
            0, original.actuator(name).id
        ]
    joint = model.joint(PREFIX + "floating_base_joint")
    qa = int(joint.qposadr[0])
    data.qpos[qa : qa + 7] = [*HOME, 0.785, 1, 0, 0, 0]
    for name, value in zip(LEG_NAMES, DEFAULT_LEGS):
        data.qpos[int(model.joint(PREFIX + name).qposadr[0])] = value
        data.ctrl[model.actuator(PREFIX + name).id] = value
    mujoco.mj_forward(model, data)
