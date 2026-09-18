"""Two electric pipette stages around a suspended, density-matched phantom."""

import json
import xml.etree.ElementTree as ET

import numpy as np

from microscopy.cell_scene import NEEDLE_LENGTH_M, build_xml as cell_xml, hollow_pipette
from microscopy.geometry import numbers
from microscopy.pipettes import HOLDER_INNER_M, HOLDER_OUTER_M, HOLDING_PROFILE, pipette_back

CELL_RADII = np.full(3, 18e-6)
CELL_CENTER = np.array([0., 0., 26e-6])
HOLDER_BACK = pipette_back("holder")


def build_xml():
    root = ET.fromstring(cell_xml(tools=("holder", "injector"), focal_reference_m=CELL_CENTER[2]))
    root.set("model", "suspended_cell_suction_injection_phantom")
    asset, world = root.find("asset"), root.find("worldbody")
    for name in ("cell_0_shell", "cell_0_nucleus"):
        node = world.find(f".//geom[@name='{name}']")
        node.set("pos", numbers(CELL_CENTER))
        node.set("size", numbers(CELL_RADII if name == "cell_0_shell" else (6e-6,)*3))
        node.set("rgba", "0 0 0 0")
        node.attrib.pop("material", None)
    world.find(".//site[@name='cell_center']").set("pos", numbers(CELL_CENTER))
    holder = next(node for node in world.iter("body")
                  if node.find("site[@name='holder_tcp']") is not None)
    holder.remove(holder.find("geom[@name='holder_taper']"))
    hollow_pipette(asset, "holding_pipette_mesh", profile=HOLDING_PROFILE)
    quaternion = np.r_[1-HOLDER_BACK[2], np.cross([0., 0., -1.], HOLDER_BACK)]
    quaternion /= np.linalg.norm(quaternion)
    ET.SubElement(holder, "geom", name="holding_hollow_pipette", type="mesh",
                  mesh="holding_pipette_mesh", quat=numbers(quaternion), material="glass")
    holder.find("geom[@name='holder_tip']").set("size", str(HOLDER_OUTER_M))
    holder.find("geom[@name='holder_tip']").set("group", "3")
    holder.find("site[@name='holder_tcp']").set("size", ".0000005")
    # The original support ends at the actual capillary base, not its mouth.
    metadata = root.find("custom/text[@name='microscopy_assets']")
    compact = metadata is not None and any(
        item.get("tool") == "holder" and item.get("needle_holder") is not None
        for item in json.loads(metadata.get("data")).get("assemblies", []))
    if compact:
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
    if metadata is not None and holder.find("geom[@name='holder_boom2']") is not None:
        boom = holder.find("geom[@name='holder_boom2']")
        start = np.fromstring(boom.get("fromto"), sep=" ")[:3]
    else:
        boom = holder.find("geom[@name='holder_boom']")
        start = np.fromstring(boom.get("fromto"), sep=" ")[:3]
    boom.set("fromto", numbers([*start, *(HOLDER_BACK*.018)]))
    holder.find("geom[@name='holder_collet']").set(
        "fromto", numbers([*(HOLDER_BACK*.018), *(HOLDER_BACK*NEEDLE_LENGTH_M)]))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
