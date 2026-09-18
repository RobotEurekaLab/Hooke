"""Independent workstation mounting of the licensed parallel mechanism."""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, geom, numbers
from microscopy.needle_holder import CAPILLARY_EXPOSED_M, HOLDER_LENGTH_M, mount_holder
from microscopy.parallel_scene import build_xml

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "temp/microscopy_research/stepper_snapshot_meshes_v1"
TOOL_EXTENSION_M = .180


def requested_parallel():
    profile = os.environ.get("HOOKE_MICROSCOPY_MANIPULATOR", "auto")
    if profile not in ("auto", "parallel-v4"):
        raise ValueError("HOOKE_MICROSCOPY_MANIPULATOR must be auto or parallel-v4")
    return Path(os.environ.get("HOOKE_MICROSCOPY_PARALLEL_ROOT", DEFAULT_ROOT)) if profile == "parallel-v4" else None


def mount_parallel(world, asset, actuators, equality, tool, home, origin, bench_top, back, root):
    """Preserve CAD scale and articulation; define the original needle interface separately."""
    mechanism = ET.fromstring(build_xml(root))
    mapping = {node.get("name"): (tool + "_" + node.get("name").removeprefix("parallel_"))
               for node in mechanism.iter() if node.get("name")}
    evidence = json.loads(mechanism.find("custom/text[@name='parallel_cad']").get("data"))
    for node in mechanism.iter():
        for key in ("name", "joint", "mesh", "body1", "body2", "site1", "site2"):
            if node.get(key) in mapping:
                node.set(key, mapping[node.get(key)])
    tangent = np.cross([0., 0., 1.], back)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack((tangent, np.cross(back, tangent), back))
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    platform_position = origin + home + back * TOOL_EXTENSION_M
    for node in mechanism.find("worldbody"):
        node.set("pos", numbers(platform_position))
        node.set("quat", numbers(quaternion))
        world.append(node)
    asset.extend(mechanism.find("asset"))
    actuators.extend(mechanism.find("actuator"))
    equality.extend(mechanism.find("equality"))
    platform = world.find(f"body[@name='{tool}_platform']")
    tcp = body(platform, tool + "_tcp_frame", (0, 0, -TOOL_EXTENSION_M),
               quat=numbers(np.r_[quaternion[0], -quaternion[1:]]), gravcomp="1")
    ET.SubElement(tcp, "site", name=tool + "_tcp", size=".00005", group="5")
    # The estimated plate contacts the CAD base's local Z=0 face. Its post is
    # independent hardware, not an upstream or commercial mounting claim.
    plate = platform_position + rotation @ np.array([-.03025, -.03025, -.0495])
    support = body(world, tool + "_support")
    outward = back[:2] / np.linalg.norm(back[:2])
    foot = np.r_[plate[:2] + outward * .160, bench_top + .012]
    geom(support, tool + "_foot", "box", (.043, .037, .012), foot, "graphite")
    arm_end = plate - back * .004
    post_top = np.r_[foot[:2], arm_end[2]]
    cylinder(support, tool + "_post", foot + [0, 0, .012], post_top, .014, "metal")
    cylinder(support, tool + "_support_arm", post_top, arm_end, .008, "metal")
    geom(support, tool + "_mount_plate", "box", (.01725, .01725, .002), plate, "metal", quat=numbers(quaternion))
    needle_holder = None
    if tool in ("injector", "holder"):
        cylinder(tcp, tool + "_holder_adapter", back * (TOOL_EXTENSION_M - .0145),
                 back * (CAPILLARY_EXPOSED_M + HOLDER_LENGTH_M), .002, "metal")
        needle_holder = mount_holder(tcp, tool, back)
    else:
        cylinder(tcp, tool + "_boom", back * (TOOL_EXTENSION_M - .0145), back * .025, .004, "metal")
        cylinder(tcp, tool + "_collet", back * .025, back * .009, .003, "graphite")
    for item in evidence["bindings"]:
        item["body"] = mapping[item["body"]]
        item["geoms"] = [mapping[name] for name in item["geoms"]]
    evidence.update(tool=tool, profile="parallel-v4", parts=173, needle_holder=needle_holder,
                    motor_joints=[f"{tool}_motor_{i}" for i in range(3)],
                    tool_back_direction=back.tolist(), tool_elevation_deg=float(np.rad2deg(np.arcsin(back[2]))),
                    tool_extension_m=TOOL_EXTENSION_M,
                    mounting="Original inclined plate, outboard post, cantilever and holder adapter; no verified hardware fit")
    return tcp, evidence
