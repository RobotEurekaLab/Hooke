"""Namespaced reuse of existing Hooke MJCF assets, without modifying originals."""

import copy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
REFERENCES = {
    "name",
    "class",
    "childclass",
    "mesh",
    "material",
    "texture",
    "body",
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "site1",
    "site2",
    "geom",
    "geom1",
    "geom2",
    "tendon",
    "actuator",
    "target",
}


def _repair_mechanics(source, filename):
    """Correct specific legacy proxy clearances in the copy, retaining contacts."""
    kind = Path(filename).stem
    changes = []
    if kind == "analytical_balance":
        door = source.find(".//body[@name='draft_door']")
        position = list(map(float, door.get("pos").split()))
        position[2] += 0.001
        position[1] -= 0.007
        door.set("pos", " ".join(map(str, position)))
        changes.append(
            "Raised sliding shield 1 mm above bench and shifted it 7 mm forward to clear rear-glass seam"
        )
    if kind == "tip_box.gen":
        # MJCF plane dimensions only bound its drawing, not its collisions.
        # Preserve this rack's local support without leaking a room-wide plane.
        for surface in source.findall(".//geom[@type='plane']"):
            surface.set("type", "box")
            surface.set("size", ".026 .018 .0005")
            surface.set("pos", "0 0 .0199")
        changes.append("Replaced infinite rack support plane with a bounded plate")
    if kind in {"oscilloscope", "function_generator", "muffle_furnace"}:
        knob = source.find(".//body[@name='knob']")
        position = list(map(float, knob.get("pos").split()))
        position[2] += 0.004
        if kind == "oscilloscope":
            position[0] += 0.008
        knob.set("pos", " ".join(map(str, position)))
        changes.append(
            "Raised knob 4 mm to clear housing; scope knob shifted 8 mm to clear adjacent dial"
        )
    if kind in {"water_bath", "spin_coater", "quartz_crystal_microbalance"}:
        name = "sensor_lid" if kind == "quartz_crystal_microbalance" else "lid"
        lid = source.find(f".//body[@name='{name}']")
        lid.find("joint").set("axis", "-1 0 0")
        position = list(map(float, lid.get("pos").split()))
        position[2] += 0.003
        lid.set("pos", " ".join(map(str, position)))
        if kind != "water_bath":
            lid.find("geom").set("pos", f"0 {-position[1]} 0")
        changes.append(
            "Recentered cover above chamber; hinge opens upwards with 3 mm clearance"
        )
    if kind in {"xrf_rock_analyzer", "water_bath"}:
        base = source.find("./worldbody/body")
        solid = base.find("geom")
        x, y, z = map(float, solid.get("size").split())
        base.remove(solid)
        thickness = 0.006
        # Replace the filled proxy with the actual shell required for tray access.
        shell = [
            ((x, y, thickness), (0, 0, thickness)),
            ((thickness, y, z), (x - thickness, 0, z)),
            ((thickness, y, z), (-x + thickness, 0, z)),
            ((x, thickness, z), (0, y - thickness, z)),
        ]
        if kind == "xrf_rock_analyzer":
            shell += [((x, y, thickness), (0, 0, 2 * z - thickness))]
        else:
            shell += [((x, thickness, z), (0, -y + thickness, z))]
        for size, pos in shell:
            panel = copy.deepcopy(solid)
            panel.set("size", " ".join(map(str, size)))
            panel.set("pos", " ".join(map(str, pos)))
            base.append(panel)
        changes.append(
            "Replaced filled-box proxy with physical thin-wall cavity; collision remains enabled"
        )
        if kind == "xrf_rock_analyzer":
            display = base.find("geom")
            position = list(map(float, display.get("pos").split()))
            position[2] = 0.04
            display.set("pos", " ".join(map(str, position)))
            changes.append(
                "Moved front display below the tray entrance to keep the slot clear"
            )
    return changes


def attach_asset(root, world, filename, name, pos, yaw=0):
    path = (MODEL_ROOT / filename).resolve()
    if not path.is_relative_to(MODEL_ROOT.resolve()):
        raise ValueError("Asset must be inside Hooke/model")
    # A few original files contain '--' inside comments, which XML disallows.
    source = ET.fromstring(re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S))
    adaptations = _repair_mechanics(source, filename)
    if (
        source.find("include") is not None
        or source.find("extension/plugin") is not None
    ):
        raise ValueError(f"Asset requires a dedicated composition adapter: {filename}")
    compiler = source.find("compiler")
    options = {} if compiler is None else compiler.attrib
    if options.get("angle", "degree") != "radian":
        raise ValueError(f"Asset needs explicit angular-unit conversion: {filename}")
    original_joints = {j.get("name"): j for j in source.findall(".//worldbody//joint")}
    for element in source.iter():
        for key, value in list(element.attrib.items()):
            if key in REFERENCES:
                element.set(key, name + "__" + value)
        if element.tag in ("mesh", "texture") and element.get("file"):
            folder = options.get(
                "meshdir" if element.tag == "mesh" else "texturedir",
                options.get("assetdir", "."),
            )
            resource = (path.parent / folder / element.get("file")).resolve()
            if not resource.is_file():
                raise FileNotFoundError(resource)
            element.set("file", str(resource))
    defaults = source.find("default")
    if defaults is not None:
        wrapper = ET.SubElement(
            root.find("default"), "default", {"class": name + "__defaults"}
        )
        wrapper.extend(copy.deepcopy(list(defaults)))
    mount = ET.SubElement(
        world,
        "body",
        name=name + "__mount",
        pos=" ".join(map(str, pos)),
        quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
    )
    if defaults is not None:
        mount.set("childclass", name + "__defaults")
    mount.extend(copy.deepcopy(list(source.find("worldbody"))))
    for tag in ("asset", "actuator", "equality", "contact", "sensor", "tendon"):
        section = source.find(tag)
        if section is not None and len(section):
            target = root.find(tag)
            if target is None:
                target = ET.SubElement(root, tag)
            target.extend(copy.deepcopy(list(section)))
    joints, actuators = {}, {}
    for semantic, original in original_joints.items():
        if not semantic or original.get("type", "hinge") not in ("hinge", "slide"):
            continue
        actual = name + "__" + semantic
        joint = next(j for j in mount.iter("joint") if j.get("name") == actual)
        joints[semantic] = actual
        if "range" not in joint.attrib:
            continue
        low, high = map(float, joint.get("range").split())
        if not low <= 0 <= high:
            continue
        existing = next(
            (a for a in root.find("actuator") if a.get("joint") == actual), None
        )
        if existing is not None:
            actuators[semantic] = existing.get("name")
            continue
        is_slide = joint.get("type", "hinge") == "slide"
        # Adapt manual handles to an explicit motor for this scene's direct-control mode.
        drive = actual + "__drive"
        joint.set("damping", "5" if is_slide else "1")
        ET.SubElement(
            root.find("actuator"),
            "position",
            name=drive,
            joint=actual,
            kp="500" if is_slide else "80",
            kv="20" if is_slide else "4",
            ctrlrange=f"{low} {high}",
            forcerange="-100 100",
        )
        actuators[semantic] = drive
    return dict(
        id=name,
        kind="reuse",
        asset=filename,
        joints=joints,
        actuators=actuators,
        sites={},
        capabilities=["existing_geometry", "direct_joint_control"],
        mechanical_adaptations=adaptations,
        fidelity="Existing Hooke geometry; added position drives for manual handles. "
        "Original instrument-specific Python process logic is not attached.",
        provenance="Hooke repository; retain upstream notices when exporting",
    )
