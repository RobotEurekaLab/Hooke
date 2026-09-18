"""Micrometre-scale blunt probe and electric forceps around a suspended cell."""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.cell_scene import build_xml as cell_xml
from microscopy.geometry import cylinder, frustum, numbers

CELL_RADII = np.full(3, 18e-6)
CELL_CENTER = np.array([0., 0., 18e-6])
PROBE_RADIUS_M = 2e-6
PAD_HALFSIZE_M = np.array([8e-6, 2e-6, 8e-6])
JAW_OPEN_M = 40e-6
JAW_CLOSED_M = 16e-6
PUSH_TARGET = np.array([26e-6, -8e-6])
PICK_TARGET = np.array([20e-6, -25e-6])
LIFT_M = 24e-6


def build_xml(operation):
    if operation not in ("push", "pick_place"):
        raise ValueError("Unknown cell handling experiment")
    tool = "probe" if operation == "push" else "gripper"
    root = ET.fromstring(cell_xml(tools=("injector", tool), focal_reference_m=CELL_CENTER[2]))
    root.set("model", "cell_"+operation)
    world, asset = root.find("worldbody"), root.find("asset")
    for name, radii in (("cell_0_shell", CELL_RADII), ("cell_0_nucleus", np.full(3, 6e-6))):
        node = world.find(f".//geom[@name='{name}']")
        node.set("pos", numbers(CELL_CENTER))
        node.set("size", numbers(radii))
        node.set("rgba", "0 0 0 0")
        node.attrib.pop("material", None)
    world.find(".//site[@name='cell_center']").set("pos", numbers(CELL_CENTER))
    if tool == "probe":
        parent = next(node for node in world.iter("body")
                      if node.find("site[@name='probe_tcp']") is not None)
        taper = parent.find("geom[@name='probe_taper']")
        frustum(asset, "cell_blunt_probe_mesh", .009, .00015, PROBE_RADIUS_M)
        taper.set("mesh", "cell_blunt_probe_mesh")
        parent.find("geom[@name='probe_tip']").set("size", str(PROBE_RADIUS_M))
    else:
        for side, sign in (("a", 1), ("b", -1)):
            jaw = world.find(f".//body[@name='jaw_{side}']")
            fixed = jaw is None
            if fixed:
                jaw = next(node for node in world.iter("body")
                           if node.find(f"geom[@name='jaw_{side}_pad']") is not None)
            else:
                jaw.set("pos", numbers((0, sign*PAD_HALFSIZE_M[1], 0)))
                joint = jaw.find("joint")
                joint.set("range", f"0 {JAW_OPEN_M}")
                drive = root.find(f"actuator/position[@name='jaw_{side}_drive']")
                drive.set("ctrlrange", f"0 {JAW_OPEN_M}")
                drive.set("kp", "1")
                drive.set("kv", str(2*np.sqrt(.0004)))
                drive.set("forcerange", "-.0001 .0001")
            pad = jaw.find(f"geom[@name='jaw_{side}_pad']")
            pad.set("size", numbers(PAD_HALFSIZE_M))
            offset = np.array([0., -20e-6, 0.]) if fixed else np.zeros(3)
            pad.set("pos", numbers(offset))
            shank = jaw.find(f"geom[@name='jaw_{side}_shank']")
            jaw.remove(shank)
            # Original fine extensions taper into the existing motor housing.
            cylinder(jaw, f"jaw_{side}_fine_tip", offset+[8e-6, 0, 0],
                     offset+[.0002, sign*30e-6, 80e-6], 1.5e-6)
            cylinder(jaw, f"jaw_{side}_shank", offset+[.0002, sign*30e-6, 80e-6],
                     (.009, sign*.001, .002), 70e-6)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
