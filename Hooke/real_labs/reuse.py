"""Namespaced reuse of existing Hooke MJCF assets, without modifying originals."""

import copy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from real_labs.reuse_drives import MINI_CENTRIFUGE, add_inspection_drive

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
    if str(filename) == MINI_CENTRIFUGE:
        from real_labs.mini_centrifuge import adapt_geometry

        changes.append(adapt_geometry(source))
    if kind == "analytical_balance":
        base = source.find("./worldbody/body")
        # The legacy glass extended through the solid base. Keep its top at
        # 180 mm and lift the bottom to 42 mm, above the 40 mm base surface.
        for glass in base.findall("geom"):
            size = list(map(float, glass.get("size").split()))
            if size[2] != 0.09:
                continue
            position = list(map(float, glass.get("pos").split()))
            size[2], position[2] = 0.069, 0.111
            glass.set("size", " ".join(map(str, size)))
            glass.set("pos", " ".join(map(str, position)))
        display = base.findall("geom")[1]
        display.set("pos", "-0.06 -0.12 0.06")
        display.set("name", "front_display")
        door = source.find(".//body[@name='draft_door']")
        position = list(map(float, door.get("pos").split()))
        position[2] = 0.111
        position[1] -= 0.007
        door.set("pos", " ".join(map(str, position)))
        door.find("geom").set("size", ".001 .09 .069")
        door.find("geom").set("name", "sliding_shield")
        changes.append(
            "Set glass bottoms 2 mm above the 40 mm base while retaining 180 mm top height; shifted sliding shield 7 mm forward to clear the rear seam. Moved display to the supported front edge, clear of the full 100 mm shield travel; contacts remain enabled."
        )
    if kind == "ph_meter":
        base = source.find("./worldbody/body")
        probe = source.find(".//body[@name='probe']")
        probe.set("pos", ".03 .09 .10")
        probe.find("geom").set("name", "probe_tip")
        ET.SubElement(
            base,
            "geom",
            name="probe_stand_foot",
            type="box",
            size=".03 .06 .005",
            pos="-.02 .04 .005",
            rgba=".55 .58 .60 1",
            contype="1",
            conaffinity="1",
        )
        ET.SubElement(
            base,
            "geom",
            name="probe_stand_post",
            type="cylinder",
            size=".006 .05",
            pos="-.03 .09 .055",
            rgba=".55 .58 .60 1",
            contype="1",
            conaffinity="1",
        )
        # The bearing pin is visual geometry at the modeled joint, rather than
        # a second colliding solid inside the rotating probe's hinge bearing.
        ET.SubElement(
            base,
            "geom",
            name="probe_hinge_pin",
            type="cylinder",
            size=".004 .03",
            pos="0 .09 .10",
            euler=f"0 {math.pi / 2} 0",
            rgba=".45 .48 .50 1",
            contype="0",
            conaffinity="0",
        )
        changes.append(
            "Moved probe hinge 30 mm outward to clear the housing through the unchanged +/-0.6 rad travel. Added a supported stand and visual hinge pin; probe, foot and post collisions remain enabled. Geometry and joint bearing are authored mechanical proxies."
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
    joints, actuators, actuator_modes, control_contracts = {}, {}, {}, {}
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
        contract = add_inspection_drive(
            root.find("actuator"), filename, semantic, joint, drive
        )
        if contract is not None:
            actuators[semantic] = drive
            actuator_modes[semantic] = contract["mode"]
            control_contracts[semantic] = contract
            continue
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
        actuator_modes[semantic] = "position"
    record = dict(
        id=name,
        kind="reuse",
        asset=filename,
        joints=joints,
        actuators=actuators,
        actuator_modes=actuator_modes,
        control_contracts=control_contracts,
        sites={},
        capabilities=["existing_geometry", "direct_joint_control"],
        mechanical_adaptations=adaptations,
        fidelity="Existing Hooke geometry; declared inspection drives adapt manual handles or rotors. "
        "Original instrument-specific Python process logic is not attached.",
        provenance="Hooke repository; retain upstream notices when exporting",
    )
    if str(filename) == MINI_CENTRIFUGE:
        from real_labs.mini_centrifuge import configure

        interface = configure(root, name)
        record["mechanical_interface"] = interface
        record["sites"] = {
            f"port_{index}": site for index, site in enumerate(interface["port_sites"])
        }
        record["capabilities"].extend(
            ["open_lid_contacts", "rotor_bore_contacts", "inert_gauge_fixture_variants"]
        )
    return record
