"""Actuated closed-loop prototype of the CAD-verified v4 parallel mechanism.

Three clockwise motor hinges drive six rigid rods and a free rigid platform.
Three colocated hinges represent each passive spherical rod attachment; six
connect constraints close the platform. These proxies are not OEM appearance
meshes. Mass, inertia, drive gains and gravity compensation are uncalibrated.
"""

import xml.etree.ElementTree as ET

import numpy as np

from microscopy.geometry import body, cylinder, geom, numbers
from microscopy.parallel_kinematics import ACTUATOR_ORIGINS_M, ACTUATOR_ROTATIONS, CAD_GEOMETRY, rotor_attachments


def build_xml(cad_root=None):
    root = ET.Element("mujoco", model="licensed_parallel_v4_closure_prototype")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=".0005", gravity="0 0 -9.81", iterations="100", integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", mass="0", rgba=".65 .68 .72 1")
    world = ET.SubElement(root, "worldbody")
    mount = body(world, "parallel_base", (0, 0, .20))
    ET.SubElement(mount, "inertial", pos="0 0 0", mass=".001", diaginertia=".0000001 .0000001 .0000001")
    actuators = ET.SubElement(root, "actuator")
    closures = ET.SubElement(root, "equality")
    platform = body(world, "parallel_platform", (0, 0, .20), gravcomp="1")
    ET.SubElement(platform, "freejoint", name="platform_free")
    ET.SubElement(platform, "inertial", pos="0 0 0", mass=".02", diaginertia=".0000004 .0000004 .0000004")
    geom(platform, "platform_proxy", "box", (.0145, .0145, .0145), material="")
    neutral = np.deg2rad([42.]*3)
    rotor_points = rotor_attachments(neutral, CAD_GEOMETRY)
    attachments = np.asarray(CAD_GEOMETRY.platform_attachments_m)
    for index, (origin, rotation, point, attachment) in enumerate(zip(ACTUATOR_ORIGINS_M, ACTUATOR_ROTATIONS, rotor_points, attachments)):
        axis = rotation[:, 2]
        cylinder(mount, f"motor_{index}_proxy", origin-axis*.055, origin-axis*.010, .021, material="")
        motor = body(mount, f"parallel_motor_{index}", origin, gravcomp="1")
        ET.SubElement(motor, "inertial", pos="0 0 0", mass=".01", diaginertia=".000002 .000002 .000002")
        ET.SubElement(motor, "joint", name=f"motor_{index}", type="hinge", axis=numbers(-axis))
        ET.SubElement(actuators, "position", name=f"motor_{index}_drive", joint=f"motor_{index}", kp="30", kv=".3")
        cylinder(motor, f"horn_{index}_proxy", (0, 0, 0), point-origin, .002, material="")
        for side in (-1, 1):
            start = point+axis*side*.0085
            end = attachment+axis*side*.0085
            vector = end-start
            rod_name = f"parallel_rod_{index}_{'a' if side < 0 else 'b'}"
            rod = body(motor, rod_name, start-origin, gravcomp="1")
            inertia = np.full(3, .001*np.dot(vector, vector)/12)
            inertia[np.argmax(abs(vector))] = .001*.0008**2/2
            ET.SubElement(rod, "inertial", pos=numbers(vector/2), mass=".001", diaginertia=numbers(inertia))
            for label, hinge_axis in zip("xyz", np.eye(3)):
                ET.SubElement(rod, "joint", name=rod_name+"_"+label, type="hinge", axis=numbers(hinge_axis), damping=".00001")
            cylinder(rod, rod_name+"_proxy", (0, 0, 0), vector, .0008, material="")
            ET.SubElement(rod, "site", name=rod_name+"_end", pos=numbers(vector), size=".0005")
            ET.SubElement(closures, "connect", name=rod_name+"_closure", body1=rod_name,
                          body2="parallel_platform", anchor=numbers(vector), solref=".002 1")
    for node in root.iter("geom"):
        node.attrib.pop("material", None)
    if cad_root is not None:
        from microscopy.parallel_cad import ParallelCad

        asset = ET.SubElement(root, "asset")
        ParallelCad(cad_root).attach(root, asset)
    return ET.tostring(root, encoding="unicode")
