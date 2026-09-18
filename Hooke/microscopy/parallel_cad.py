"""Bind the licensed v4 physical CAD instances to the actuated rigid mechanism."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from microscopy.geometry import numbers
from microscopy.parallel_kinematics import (
    ACTUATOR_ORIGINS_M, ACTUATOR_ROTATIONS, CAD_GEOMETRY, SOURCE_REVISION, rotor_attachments,
)

LICENSE_SHA256 = "98f8b354b9dff41ea4b2e5523b6e5ad1644899ee8d10d6ef92d80be350815765"
CAD_ORIGIN_M = np.full(3, .0475)
_LEGS = ("Actuator", "Actuator001", "Actuator002")
_ROTOR_FASTENERS = frozenset(("Pin", "Pin001", "Pin002", "Screw004", "Screw005", "Screw007", "Screw008"))
_COLOURS = {
    "MotorHorn.FCStd": (.62, .06, .045, 1.),
    "LinkageRod.FCStd": (.74, .61, .28, 1.),
    "MT6835.FCStd": (.08, .22, .11, 1.),
    "JointBall.FCStd": (.68, .70, .72, 1.),
    "RubberBand.FCStd": (.025, .028, .030, 1.),
    "CrimpBead.FCStd": (.55, .57, .59, 1.),
}


class ParallelCad:
    """Runtime needs inspected metre meshes, JSON and the exact license only."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.report = json.loads((self.root / "assembly.json").read_text())
        if (self.report.get("source_revision") != SOURCE_REVISION
                or self.report.get("mesh_length_unit") != "m"
                or len(self.report.get("instances", [])) != 173
                or len(self.report.get("shapes", {})) != 53):
            raise ValueError("Parallel CAD does not match the inspected v4 physical assembly")
        if (self.report.get("license_sha256") != LICENSE_SHA256
                or hashlib.sha256((self.root / "LICENSE").read_bytes()).hexdigest() != LICENSE_SHA256):
            raise ValueError("Parallel CAD license differs from the pinned upstream notice")
        self.rotor_points = rotor_attachments(np.deg2rad([42.] * 3), CAD_GEOMETRY)

    def ownership(self, part):
        """Return body and its neutral position in the mechanism frame."""
        path = part["instance"].split("/")
        document, name = part["source"]["document"], part["source"]["object"]
        if document == "EndEffector.FCStd":
            return "parallel_platform", np.zeros(3)
        if len(path) < 3 or path[1] not in _LEGS:
            return "parallel_base", np.zeros(3)
        index = _LEGS.index(path[1])
        point = np.asarray(part["transform"])[:3, 3] * 1e-3 - CAD_ORIGIN_M
        axis = ACTUATOR_ROTATIONS[index, :, 2]
        rotor, attachment = self.rotor_points[index], np.asarray(CAD_GEOMETRY.platform_attachments_m[index])
        if document in ("LinkageRod.FCStd", "RubberBandCollet.FCStd", "RubberBand.FCStd"):
            side = -1 if np.dot(point - rotor, axis) < 0 else 1
            suffix = "a" if side < 0 else "b"
            return f"parallel_rod_{index}_{suffix}", rotor + axis * side * .0085
        if document in ("JointBall.FCStd", "BallJointPlate.FCStd", "CrimpBead.FCStd"):
            if np.linalg.norm(point - attachment) < np.linalg.norm(point - rotor):
                return "parallel_platform", np.zeros(3)
            return f"parallel_motor_{index}", ACTUATOR_ORIGINS_M[index]
        moving = (document in ("MotorHorn.FCStd", "EncoderMagnetArray.FCStd")
                  or document == "StepperMotorNema17.FCStd" and name == "Body003"
                  or document == "Assembly_Actuator.FCStd" and name in _ROTOR_FASTENERS)
        return ((f"parallel_motor_{index}", ACTUATOR_ORIGINS_M[index]) if moving
                else ("parallel_base", np.zeros(3)))

    def attach(self, root, asset):
        bodies = {node.get("name"): node for node in root.iter("body")}
        for node in root.iter():
            for child in list(node):
                if child.tag == "geom" and child.get("name", "").endswith("_proxy"):
                    node.remove(child)
        registered, bindings = set(), []
        for index, instance in enumerate(self.report["instances"]):
            owner, reference = self.ownership(instance)
            transform = np.asarray(instance["transform"], dtype=float)
            position = transform[:3, 3] * 1e-3 - CAD_ORIGIN_M - reference
            quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            digest = instance["source"]["shape_sha256"]
            shape = self.report["shapes"][digest]
            if shape.get("mesh_length_unit") != "m":
                raise ValueError("Parallel CAD shape is not in metres")
            geoms = []
            for mesh_index, part in enumerate(shape["parts"]):
                mesh_name = f"parallel_cad_{digest}_{mesh_index}"
                file = self.root / digest / part["file"]
                if file.resolve().parent != (self.root / digest).resolve():
                    raise ValueError("Parallel CAD mesh is outside its inspected shape directory")
                if mesh_name not in registered:
                    if hashlib.sha256(file.read_bytes()).hexdigest() != part["mesh_sha256"]:
                        raise ValueError("Parallel CAD mesh differs from its inspected artifact")
                    ET.SubElement(asset, "mesh", name=mesh_name, file=str(file))
                    registered.add(mesh_name)
                geom_name = f"parallel_cad_part_{index:03d}_{mesh_index}"
                colour = _COLOURS.get(instance["source"]["document"], (.095, .10, .11, 1.))
                ET.SubElement(bodies[owner], "geom", name=geom_name, type="mesh", mesh=mesh_name,
                              pos=numbers(position), quat=numbers(quaternion), rgba=numbers(colour),
                              contype="0", conaffinity="0", mass="0")
                geoms.append(geom_name)
            bindings.append(dict(instance=instance["instance"], body=owner, geoms=geoms,
                                 source_shape_sha256=digest))
        evidence = dict(source="https://github.com/0x23/MicroManipulatorStepper", source_revision=SOURCE_REVISION,
                        license_sha256=LICENSE_SHA256, physical_instances=173, meshes=len(registered),
                        cad_origin_m=CAD_ORIGIN_M.tolist(), bindings=bindings,
                        appearance="Independent material colours guided by the upstream hardware photo",
                        dynamics="Three motor hinges, six rigid rods and a free platform closed by six connects",
                        limitations=["Mass, inertia, drive gains and gravity compensation are uncalibrated",
                                     "Rubber-band meshes follow their rods rigidly; no elastic strain or preload model",
                                     "No claimed hardware accuracy, motor electronics or whole-workstation installation proof"])
        custom = root.find("custom")
        if custom is None:
            custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", name="parallel_cad", data=json.dumps(evidence))
        return evidence
