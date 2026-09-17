"""Dry sample handling and a spring instrument in the attributed worlds.

Dimensions and masses describe illustrative Hooke hardware, not NASA hardware.
The cartridge is a closed rigid shell; gas sealing is not simulated.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

from worlds.asset_scenes import ASSET_ROOT, build_scene_with_assets
from worlds.build import SCENES, SOURCE_ROOT, camera, geom
from worlds.profiles import WORLDS

SAMPLES = ("reference", "candidate")
STORAGE = {"reference": (-0.18, -0.08, 1.0), "candidate": (0.0, -0.08, 1.0)}
HALF_SIZE = (0.023, 0.020, 0.035)
REFERENCE_MASS_KG = 0.1
TRAY_OFFSET = (0.0, 0.0, 0.043)


def numbers(values):
    return " ".join(map(str, values))


def build_scene(profile, rows):
    root = build_scene_with_assets(profile, rows)
    root.set("model", f"space_{profile.name}_experiment")
    world = root.find("worldbody")
    world.remove(world.find("body[@name='/external_free']"))
    world.remove(world.find("camera[@name='experiment_closeup']"))
    camera(world, "experiment_closeup", (0.40, -0.94, 1.55), (-0.05, -0.16, 1.04), 45)
    bench = world.find("body[@name='anchored_workstation']")
    for item in list(bench):
        name = item.get("name", "")
        if name.startswith(
            (
                "/containment",
                "/balance",
                "/retained",
                "sample_retention",
                "retention_clip",
                "inertial_",
                "glovebox_mount",
            )
        ):
            bench.remove(item)
    equality = ET.SubElement(root, "equality")
    for name, position in STORAGE.items():
        x, y, z = position
        holder = ET.SubElement(
            bench, "body", name=f"storage_{name}", pos=numbers((x, y, z - 0.043))
        )
        geom(
            holder,
            f"storage_{name}_seat",
            "box",
            ".032 .025 .008",
            "0 0 0",
            ".18 .29 .39 1",
        )
        geom(
            holder,
            f"storage_{name}_post",
            "box",
            ".018 .018 .0625",
            "0 0 -.0705",
            ".25 .32 .38 1",
        )
        sample = ET.SubElement(
            world, "body", name=f"sample_{name}", pos=numbers(position)
        )
        ET.SubElement(sample, "freejoint", name=f"sample_{name}_free")
        a, b, c = HALF_SIZE
        mass = REFERENCE_MASS_KG if name == "reference" else 0.15
        inertia = (
            mass * (b * b + c * c) / 3,
            mass * (a * a + c * c) / 3,
            mass * (a * a + b * b) / 3,
        )
        ET.SubElement(
            sample,
            "inertial",
            pos="0 0 0",
            mass=str(mass),
            diaginertia=numbers(inertia),
        )
        geom(
            sample,
            f"sample_{name}_shell",
            "box",
            numbers(HALF_SIZE),
            "0 0 0",
            ".70 .76 .82 1",
            friction="1 .01 .001",
            solref=".005 1",
            mass="0",
        )
        geom(
            sample,
            f"sample_{name}_cap",
            "box",
            ".023 .020 .003",
            "0 0 .038",
            ".12 .28 .39 1",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        ET.SubElement(
            sample,
            "site",
            name=f"sample_{name}_grasp",
            pos="0 0 .008",
            size=".003",
            group="5",
        )
        ET.SubElement(
            equality,
            "weld",
            name=f"storage_{name}_lock",
            body1=f"storage_{name}",
            body2=f"sample_{name}",
            active="true",
            solref=".004 1",
            solimp=".9999 .9999 .001",
        )
        ET.SubElement(
            equality,
            "weld",
            name=f"instrument_{name}_lock",
            body1="measurement_tray",
            body2=f"sample_{name}",
            active="false",
            solref=".004 1",
            solimp=".9999 .9999 .001",
        )
    tray = ET.SubElement(bench, "body", name="measurement_tray", pos="-.01 -.32 .957")
    orbital = profile.name == "orbital"
    ET.SubElement(
        tray,
        "joint",
        name="measurement_slide",
        type="slide",
        axis="1 0 0" if orbital else "0 0 1",
        range="-.03 .03",
        stiffness="20" if orbital else "100",
        damping=".02" if orbital else "3",
    )
    ET.SubElement(
        tray, "inertial", pos="0 0 0", mass=".2", diaginertia=".0002 .0002 .0002"
    )
    geom(
        tray,
        "instrument_seat",
        "box",
        ".035 .027 .008",
        "0 0 0",
        ".24 .49 .65 1",
        mass="0",
    )
    for side in (-1, 1):
        geom(
            tray,
            f"instrument_guide_{side}",
            "box",
            ".003 .024 .016",
            f"{side*.030} 0 .024",
            ".77 .54 .17 1",
            mass="0",
        )
    ET.SubElement(
        tray,
        "site",
        name="instrument_sample_pose",
        pos=numbers(TRAY_OFFSET),
        size=".003",
        group="5",
    )
    # Visible lock indication moves through a real, independently driven joint.
    latch = ET.SubElement(bench, "body", name="instrument_latch", pos="-.01 -.366 .99")
    ET.SubElement(
        latch,
        "joint",
        name="instrument_latch_slide",
        type="slide",
        axis="1 0 0",
        range="0 .025",
        damping="1",
    )
    geom(
        latch,
        "instrument_latch_handle",
        "box",
        ".012 .008 .008",
        "0 0 0",
        ".87 .58 .12 1",
        contype="0",
        conaffinity="0",
        mass=".02",
    )
    actuators = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuators,
        "position",
        name="instrument_latch_drive",
        joint="instrument_latch_slide",
        kp="50",
        kv="2",
        ctrlrange="0 .025",
    )
    shutter = ET.SubElement(
        bench, "body", name="spectrometer_shutter", pos="-.01 -.32 1.055"
    )
    ET.SubElement(
        shutter,
        "joint",
        name="spectrometer_shutter_slide",
        type="slide",
        axis="1 0 0",
        range="0 .12",
        ref=".12",
        damping="1",
    )
    geom(
        shutter,
        "spectrometer_shutter_panel",
        "box",
        ".043 .030 .004",
        ".12 0 0",
        ".15 .24 .32 1",
        mass=".03",
    )
    ET.SubElement(
        actuators,
        "position",
        name="spectrometer_shutter_drive",
        joint="spectrometer_shutter_slide",
        kp="200",
        kv="5",
        ctrlrange="0 .12",
    )
    ET.indent(root, space="  ")
    return root


def build():
    rows = {
        row["id"]: row
        for row in json.loads((ASSET_ROOT / "manifest.json").read_text())["assets"]
    }
    records = []
    for profile in WORLDS.values():
        path = SCENES / f"space_{profile.name}_experiment.gen.xml"
        ET.ElementTree(build_scene(profile, rows)).write(path, encoding="unicode")
        records.append(
            {
                "world": profile.name,
                "scene": path.relative_to(SOURCE_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "gravity_m_s2": profile.gravity_m_s2,
                "mass_sensor": (
                    "spring_encoder_inertial"
                    if profile.name == "orbital"
                    else "spring_encoder_load_cell"
                ),
                "hardware": "illustrative_hooke_dry_cartridge_and_fixture",
                "seal": "closed_geometry_not_gas_qualified",
            }
        )
    destination = SOURCE_ROOT / "experiments/scenes.json"
    destination.write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({"generated_scenes": len(build())}))
