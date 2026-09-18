"""Original hollow pipette and nominal geometry for an adherent-cell phantom.

Cells are non-colliding visual references. Their interaction is computed by
the explicitly declared reduced model, rather than a microscopic rigid body.
"""

import json
import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import camera, geom, mesh, numbers
from microscopy.scene import ORIGIN, build_xml as workstation_xml
from microscopy.pipettes import (BASE_INNER_M, BASE_OUTER_M, INJECTION_PROFILE,
                                 NEEDLE_LENGTH_M, TIP_INNER_M, TIP_OUTER_M, pipette_back)
from microscopy.cell_appearance import background_geometry

CELL_RADII = np.array([18e-6, 14e-6, 4e-6])
CELL_CENTER = np.array([0., 0., 4e-6])
NUCLEAR_RADII = np.array([6e-6, 4e-6, 2e-6])
NEEDLE_BACK = pipette_back("injector")
BACKGROUND_CELLS = ((-52e-6, 19e-6), (43e-6, 43e-6), (-34e-6, -48e-6),
                    (57e-6, -23e-6), (-17e-6, 65e-6))


def hollow_pipette(asset, name="cell_hollow_pipette_mesh", *, profile=INJECTION_PROFILE):
    sides = 48
    vertices, faces = [], []
    # Close the glass wall around the mouth and rear opening; preserve the bore.
    rings = [(distance, outer) for distance, outer, _ in reversed(profile.sections)]
    rings += [(distance, inner) for distance, _, inner in profile.sections]
    for distance, radius in rings:
        for angle in np.linspace(0, 2*np.pi, sides, endpoint=False):
            vertices.append((radius*np.cos(angle), radius*np.sin(angle), -distance))
    for ring in range(len(rings)):
        a, b = ring*sides, ((ring+1) % len(rings))*sides
        for i in range(sides):
            j = (i+1) % sides
            triangles = ((a+i, a+j, b+j), (a+i, b+j, b+i))
            faces.extend(triangles)
    mesh(asset, name, vertices, faces)


def build_xml(tools=("injector",), *, focal_reference_m=CELL_CENTER[2]):
    root = ET.fromstring(workstation_xml(tools=tools, focal_reference_m=focal_reference_m,
                                        cell_pipettes=True))
    root.set("model", "adherent_cell_microinjection_phantom")
    root.find("visual/map").set("znear", ".0000001")
    world, asset, actuators = (root.find(name) for name in ("worldbody", "asset", "actuator"))
    for name in ("bead_push", "bead_pick"):
        world.remove(world.find(f"body[@name='{name}']"))
    metadata = root.find("custom/text[@name='microscopy_assets']")
    if metadata is not None:
        evidence = json.loads(metadata.get("data"))
        evidence["assemblies"] = [part for part in evidence["assemblies"] if part["tool"] in tools]
        evidence.pop("gripper", None)
        metadata.set("data", json.dumps(evidence))
    for drive in list(actuators):
        name = drive.get("name", "")
        if ((name.startswith("probe_") and "probe" not in tools)
                or (name.startswith(("gripper_", "jaw_")) and "gripper" not in tools)):
            actuators.remove(drive)
    # The sample belongs to the final carriage, independent of slide order.
    stage = next(node for node in world.iter("body")
                 if node.find("site[@name='sample_plane']") is not None)
    for node in list(stage):
        if node.get("name", "").startswith(("well_wall", "well_center")):
            stage.remove(node)
    for name, color in (("cell_membrane", ".76 .84 .79 .45"),
                        ("cell_nucleus", ".32 .46 .60 .9")):
        ET.SubElement(asset, "material", name=name, rgba=color)
    for i, xy in enumerate(((0., 0.), *BACKGROUND_CELLS)):
        centre = CELL_CENTER + [*xy, 0.]
        radii, angle = (CELL_RADII, 0.) if i == 0 else background_geometry(i)
        nucleus = NUCLEAR_RADII if i == 0 else radii*np.array([.33, .3, .5])
        geom(stage, f"cell_{i}_shell", "ellipsoid", radii, centre, "cell_membrane", euler=f"0 0 {angle}")
        geom(stage, f"cell_{i}_nucleus", "ellipsoid", nucleus, centre, "cell_nucleus", euler=f"0 0 {angle}")
    ET.SubElement(stage, "site", name="cell_center", pos=numbers(CELL_CENTER),
                  size=".0000005", group="5")
    injector = next(node for node in world.iter("body")
                    if node.find("site[@name='injector_tcp']") is not None)
    injector.remove(injector.find("geom[@name='injector_taper']"))
    hollow_pipette(asset)
    # Rotate local -Z onto the upward/rightward pipette back direction.
    axis = np.cross([0., 0., -1.], NEEDLE_BACK)
    quaternion = np.r_[1-NEEDLE_BACK[2], axis]
    quaternion /= np.linalg.norm(quaternion)
    ET.SubElement(injector, "geom", name="cell_hollow_pipette", type="mesh",
                  mesh="cell_hollow_pipette_mesh", quat=numbers(quaternion), material="glass")
    tip = injector.find("geom[@name='injector_tip']")
    tip.set("size", str(TIP_OUTER_M))
    # Keep the spherical contact proxy out of the image so it cannot cap
    # the hollow glass mouth. Group 3 remains a collider in both backends.
    tip.set("group", "3")
    injector.find("site[@name='injector_tcp']").set("size", ".0000005")
    horizontal = np.array([2**-.5, 2**-.5, 0.])
    boom = injector.find("geom[@name='injector_boom']")
    collet = injector.find("geom[@name='injector_collet']")
    if boom is not None and collet is not None:
        metadata = root.find("custom/text[@name='microscopy_assets']")
        if metadata is not None and json.loads(metadata.get("data"))["profile"] == "cad":
            holder = np.fromstring(boom.get("fromto"), sep=" ")[:3]
            boom.set("fromto", numbers([*holder, *(NEEDLE_BACK*.018)]))
            collet.set("fromto", numbers([*(NEEDLE_BACK*.018), *(NEEDLE_BACK*NEEDLE_LENGTH_M)]))
        else:
            boom.set("fromto", numbers([*(horizontal*.31), *(horizontal*.035+[0, 0, .009])]))
            collet.set("fromto", numbers([*(horizontal*.035+[0, 0, .009]), *(NEEDLE_BACK*NEEDLE_LENGTH_M)]))
    camera(world, "cell_detail", ORIGIN+[.00014, -.00021, .00019], ORIGIN+[0, 0, .000012], 45)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
