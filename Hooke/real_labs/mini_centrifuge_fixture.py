"""Persistable inert-gauge variants, separate from the ordinary laboratory scene.

The free gauge begins on a supported parking cradle. Loading uses an explicitly
authored handling wrench in the inspection task; no robot gripper or specimen
processing protocol is implied by this small mechanical fixture.
"""

import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.mini_centrifuge import SCENES, _numbers


GAUGE_MASS = 0.002
GAUGE_RADIUS = 0.0044
GAUGE_HALF_LENGTH = 0.006
FLANGE_RADIUS = 0.0062
FLANGE_HALF_HEIGHT = 0.0007


def build_inspection_scene(scene, *, port=0):
    """Build the unchanged lab plus a supported free inert loading fixture."""
    from real_labs.builder import build_scene

    if scene not in SCENES or type(port) is not int or not 0 <= port < 8:
        raise ValueError("Choose a miniature centrifuge scene and an integer port 0–7")
    root, manifest = build_scene(scene)
    equipment = next(
        item for item in manifest["equipment"] if item["id"] == "centrifuge"
    )
    interface = equipment["mechanical_interface"]
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mount_id = model.body("centrifuge__mount").id
    mount = root.find(".//body[@name='centrifuge__mount']")
    fixture_name = "centrifuge__gauge_fixture"
    fixture = ET.SubElement(mount, "body", name=fixture_name, pos=".11 0 0")
    # The base rests on the same support plane as the instrument. Four short
    # guides leave a real 13.2 mm opening around the 12.4 mm gauge flange.
    parts = [("base", (0, 0, 0.002), (0.016, 0.016, 0.002))]
    for side in (-1, 1):
        parts += [
            (f"x_{side}", (side * 0.0076, 0, 0.008), (0.001, 0.0086, 0.004)),
            (f"y_{side}", (0, side * 0.0076, 0.008), (0.0066, 0.001, 0.004)),
        ]
    for name, pos, size in parts:
        ET.SubElement(
            fixture,
            "geom",
            name=fixture_name + "_" + name,
            type="box",
            pos=_numbers(pos),
            size=_numbers(size),
            contype="1",
            conaffinity="1",
            rgba=".32 .35 .39 1",
        )
    park_local = np.array([0.11, 0, 0.0100001])
    park_site = fixture_name + "_park"
    ET.SubElement(fixture, "site", name=park_site, pos="0 0 .0100001", size=".0003")
    rotation = data.xmat[mount_id].reshape(3, 3)
    position = data.xpos[mount_id] + rotation @ park_local
    body_name = "centrifuge__inert_gauge"
    gauge = ET.SubElement(
        root.find("worldbody"),
        "body",
        name=body_name,
        pos=_numbers(position),
        quat=_numbers(data.xquat[mount_id]),
    )
    ET.SubElement(gauge, "freejoint", name=body_name + "_free")
    ET.SubElement(
        gauge,
        "inertial",
        pos="0 0 .001",
        mass=str(GAUGE_MASS),
        diaginertia="4e-8 4e-8 2.5e-8",
    )
    gauge_geoms = []
    for part, z, radius, height in (
        ("shaft", 0, GAUGE_RADIUS, GAUGE_HALF_LENGTH),
        (
            "flange",
            GAUGE_HALF_LENGTH + FLANGE_HALF_HEIGHT,
            FLANGE_RADIUS,
            FLANGE_HALF_HEIGHT,
        ),
    ):
        name = body_name + "_" + part
        gauge_geoms.append(name)
        ET.SubElement(
            gauge,
            "geom",
            name=name,
            type="cylinder",
            pos=_numbers((0, 0, z)),
            size=_numbers((radius, height)),
            rgba=".35 .48 .65 1",
            contype="1",
            conaffinity="1",
            friction=".25 .003 .0001",
            solref=".004 1",
            solimp=".95 .99 .001",
            density="0",
        )
    manifest["inspection_variant"] = dict(
        kind="mini_inert_gauge_loading",
        revision=1,
        equipment="centrifuge",
        port=port,
        port_site=interface["port_sites"][port],
        gauge_body=body_name,
        gauge_geoms=gauge_geoms,
        fixture_body=fixture_name,
        park_site=park_site,
        mass_kg=GAUGE_MASS,
        initial_state="Free gauge resting on a supported cradle; no weld or hidden pose constraint",
        handling="Explicit bounded external wrench; no modeled robot gripper",
        scope="Dry inert mechanical loading, release, low-speed rotation and unloading only",
    )
    return root, manifest


def create_inspection_session(scene, *, port=0):
    from real_labs.runtime import InstrumentSession

    root, manifest = build_inspection_scene(scene, port=port)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def export_inspection_scene(scene, destination, *, port=0):
    """Write a portable MJCF and manifest accepted by InstrumentSession.from_files."""
    from real_labs.export import _portable_resources, write_json

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root, manifest = build_inspection_scene(scene, port=port)
    resources = _portable_resources(root)
    for item in resources:
        target = destination / item["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["source"], target)
    path = destination / "scene.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    manifest.update(
        resources=resources, scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    write_json(destination / "manifest.json", manifest)
    return manifest
