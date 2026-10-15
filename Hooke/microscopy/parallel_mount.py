"""Independent workstation mounting of the licensed parallel mechanism."""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import body, cylinder, numbers
from microscopy.manipulator_support import add_support
from microscopy.needle_holder import CAPILLARY_EXPOSED_M, HOLDER_LENGTH_M, mount_holder
from microscopy.parallel_scene import build_xml

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "temp/microscopy_research/stepper_snapshot_meshes_v1"
TOOL_EXTENSION_M = .180


def requested_parallel():
    from microscopy.manipulator_profile import requested_profile
    profile = requested_profile()
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
    # Keep the motor assembly upright; only its tool holder is inclined.
    # The CAD occupies mostly local -X/-Y, so yaw its bulk away from the sample.
    yaw = np.arctan2(back[1], back[0]) + 3 * np.pi / 4
    rotation = Rotation.from_euler("z", yaw).as_matrix()
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
    tcp = body(platform, tool + "_tcp_frame", -rotation.T @ back * TOOL_EXTENSION_M,
               quat=numbers(np.r_[quaternion[0], -quaternion[1:]]), gravcomp="1")
    ET.SubElement(tcp, "site", name=tool + "_tcp", size=".00005", group="5")
    # The upright CAD base and bench fixture use separate frames from the tool.
    # This is the measured lower face centre of BaseBlock.FCStd, not a promise
    # of hardware fit; the surrounding motors require a side cradle below it.
    plate = platform_position + rotation @ np.array([-.03025, -.03025, -.0475])
    support_evidence = add_support(asset, world, tool, plate, rotation, bench_top)
    needle_holder = None
    if tool in ("injector", "holder"):
        # The fork loads the rotation clamp from the side. The axial holder
        # tail stays open for the pneumatic inlet instead of carrying a rod.
        rear = back * (TOOL_EXTENSION_M - .0145)
        clamp = back * (CAPILLARY_EXPOSED_M + .118)
        side = -tangent * .010
        # Retain the previous numerical mass distribution without showing the
        # old axial rod through the pressure inlet. The new fork is visual.
        cylinder(tcp, tool + "_holder_adapter", rear,
                 back * (CAPILLARY_EXPOSED_M + HOLDER_LENGTH_M), .002, "metal", rgba="0 0 0 0")
        cylinder(tcp, tool + "_holder_fork_rear", rear, rear + side,
                 .002, "graphite", mass="0")
        cylinder(tcp, tool + "_holder_fork", rear + side, clamp + side,
                 .002, "graphite", mass="0")
        cylinder(tcp, tool + "_holder_clamp_bridge", clamp + side,
                 clamp - tangent * .0035, .002, "graphite", mass="0")
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
                    mounting="Original upright bench column and side cradle; independently inclined holder; no verified hardware fit",
                    mechanism_yaw_deg=float(np.rad2deg(yaw)),
                    mechanism_vertical_axis="world +Z",
                    base_face_centre_m=plate.tolist(), bench_support=support_evidence)
    if needle_holder is not None:
        evidence["holder_adapter_inertia"] = (
            "Original axial adapter retained as an invisible numerical mass proxy; "
            "side fork and clamp details are massless appearance geometry")
    return tcp, evidence
