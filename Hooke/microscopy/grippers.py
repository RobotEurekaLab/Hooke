"""Authentic parallel-gripper geometry with original fine contact extensions.

The OEM reference CAD has one translating jaw. Servo gains, extension fingers
and contact forces are demonstration parameters, not a calibrated SGP device.
"""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import body, cylinder, geom

SGP_SHA256 = "404f0c0375210c37044f18ea82f0ed013f1fdf074e7191daead317fa837ef046"
MOVING_PARTS = frozenset((31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 45, 50, 51, 52, 53))


def parallel_gripper(cad, asset, parent, actuators):
    report = cad.read("smaract-sgp17f", SGP_SHA256)
    if len(report["parts"]) != 56 or report["parts"][31]["source_name"] != "Slide":
        raise ValueError("Parallel gripper differs from the inspected assembly")
    # Factory reference opening is about 10 mm. Centre the two factory tip
    # points at a 1.44 mm spacing using the actual tessellated tip coordinates.
    nominal_shift = .00874512
    centre = np.array([-.01062744, -.038934686, .0135])
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    translation = np.array([.015, 0., .025])-rotation@centre
    jaw = body(parent, "jaw_a", gravcomp="1")
    ET.SubElement(jaw, "inertial", pos="0 0 0", mass=".0004", diaginertia="1e-9 1e-9 1e-9")
    ET.SubElement(jaw, "joint", name="jaw_a", type="slide", axis="0 1 0",
                  range="-.00025 .0016", damping=".001")
    ET.SubElement(actuators, "position", name="jaw_a_drive", joint="jaw_a", kp="50", kv=".02",
                  ctrlrange="-.00025 .0016", forcelimited="true", forcerange="-.005 .005")
    for index in range(56):
        moving = index in MOVING_PARTS
        position = translation+rotation@np.array([nominal_shift if moving else 0., 0., 0.])
        cad.attach(asset, jaw if moving else parent, "smaract-sgp17f", index,
                   position, rotation, "metal", prefix="gripper")
    for side, sign, node in (("a", 1., jaw), ("b", -1., parent)):
        geom(node, f"jaw_{side}_pad", "box", (.00065, .00012, .00035),
             (0., sign*.00072, 0.), "metal", contype="1", conaffinity="1",
             friction="2 .00001 .000001")
        cylinder(node, f"jaw_{side}_shank", (.00065, sign*.00072, 0.),
                 (.015, sign*.00072, .025), .00018)
    # The original mounting extension ends at the factory base underside.
    holder = np.fromstring(parent.find("geom[@name='gripper_boom']").get("fromto"), sep=" ")[:3]
    parent.remove(parent.find("geom[@name='gripper_boom']"))
    parent.remove(parent.find("geom[@name='gripper_collet']"))
    base_point = translation+rotation@np.array([-.015, 0., 0.])
    cylinder(parent, "gripper_boom", holder, base_point-[0, 0, .006], .004)
    geom(parent, "gripper_mount_plate", "box", (.017, .012, .002), base_point-[0, 0, .002], "metal")
    cad.evidence["gripper"] = dict(name="SmarAct SGP-17F reference CAD with original fine extensions",
        source_sha256=report["sha256"], parts=56, moving_parts=sorted(MOVING_PARTS),
        factory_reference_opening_m=.010, nominal_carriage_shift_m=nominal_shift,
        cad_rotation=rotation.tolist(), cad_translation_m=translation.tolist(),
        opening_dof="One translating jaw; the opposite jaw is fixed",
        demo_force_limit_n=.005, rights=report["rights"],
        original_parts=["15 mm forward / 25 mm downward contact extensions", "Mount plate and tool extension"],
        limitations=["Uncalibrated force and servo dynamics; no load-cell electronics",
                     "Limited demo travel; bearing roll and cage translation omitted"])
