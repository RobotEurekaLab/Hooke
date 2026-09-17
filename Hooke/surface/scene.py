"""Portable outdoor scenes; generated files are cached in ignored local output."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from surface.profiles import MISSIONS
from surface.artifacts import SCENES as CACHE
from surface.appearance import APPEARANCES, add_appearance
from surface.terrain import landscape, write_heightfield
from worlds.build import SOURCE_ROOT, attach, camera, geom

ARM_PREFIX = "/sampler:"
WHEELS = tuple(
    f"wheel_{side}_{index}" for side in ("left", "right") for index in range(3)
)
BIN_CENTER = np.array([-0.30, -0.28, 0.17])


def element_geom(parent, name, kind, size, pos, color, **attributes):
    return geom(
        parent,
        name,
        kind,
        " ".join(map(str, size)),
        " ".join(map(str, pos)),
        color,
        **attributes,
    )


def add_rover(world, assets, actuators, mission):
    ET.SubElement(
        assets,
        "model",
        name="sampler",
        file=str(SOURCE_ROOT / "model/robot/ur5e_gripper.xml"),
        content_type="text/xml",
    )
    rover = ET.SubElement(
        world, "body", name="rover", pos=f"0 0 {2 * mission.wheel_radius_m + .005}"
    )
    ET.SubElement(rover, "freejoint", name="rover_free")
    ET.SubElement(rover, "inertial", pos="0 0 0", mass="100", diaginertia="6 10 12")
    element_geom(
        rover,
        "chassis",
        "box",
        (0.57, 0.32, 0.12),
        (0, 0, 0),
        ".78 .81 .78 1",
        mass="0",
    )
    element_geom(
        rover,
        "thermal_skirt",
        "box",
        (0.54, 0.33, 0.03),
        (0, 0, -0.08),
        ".64 .47 .16 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    element_geom(
        rover,
        "front_camera",
        "box",
        (0.055, 0.12, 0.045),
        (0.59, 0, 0.06),
        ".08 .12 .16 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    for side, y in (("left", mission.half_track_m), ("right", -mission.half_track_m)):
        for index, x in enumerate((-0.44, 0, 0.44)):
            name = f"wheel_{side}_{index}"
            wheel = ET.SubElement(
                rover, "body", name=name, pos=f"{x} {y} {-mission.wheel_radius_m}"
            )
            ET.SubElement(
                wheel,
                "joint",
                name=name,
                type="hinge",
                axis="0 1 0",
                damping=".05",
                armature=".05",
            )
            element_geom(
                wheel,
                f"{name}_tire",
                "cylinder",
                (mission.wheel_radius_m, 0.055),
                (0, 0, 0),
                ".10 .12 .13 1",
                quat=".70710678 .70710678 0 0",
                mass="2",
                friction="1.3 .02 .001",
                condim="4",
            )
            element_geom(
                wheel,
                f"{name}_hub",
                "cylinder",
                (0.08, 0.057),
                (0, 0, 0),
                ".67 .69 .64 1",
                quat=".70710678 .70710678 0 0",
                mass="0",
                contype="0",
                conaffinity="0",
            )
            for tick in range(12):
                angle = tick * math.tau / 12
                element_geom(
                    wheel,
                    f"{name}_tread_{tick}",
                    "box",
                    (0.015, 0.052, 0.008),
                    (
                        mission.wheel_radius_m * math.sin(angle),
                        0,
                        mission.wheel_radius_m * math.cos(angle),
                    ),
                    ".35 .36 .32 1",
                    quat=f"{math.cos(angle/2)} 0 {math.sin(angle/2)} 0",
                    mass="0",
                    contype="0",
                    conaffinity="0",
                )
            ET.SubElement(
                actuators,
                "velocity",
                name=name,
                joint=name,
                kv="24",
                ctrllimited="true",
                ctrlrange="-6 6",
                forcelimited="true",
                forcerange="-36 36",
            )
    attach(rover, "sampler", ARM_PREFIX, ".40 0 .145")
    bin_body = ET.SubElement(
        rover, "body", name="sample_bin", pos=" ".join(map(str, BIN_CENTER))
    )
    element_geom(
        bin_body,
        "bin_floor",
        "box",
        (0.13, 0.11, 0.012),
        (0, 0, 0),
        ".14 .24 .32 1",
        mass=".25",
    )
    for axis in range(2):
        for side in (-1, 1):
            size = [0.13, 0.11, 0.07]
            size[axis] = 0.01
            pos = [0, 0, 0.07]
            pos[axis] = side * (0.13 if axis == 0 else 0.11)
            element_geom(
                bin_body,
                f"bin_wall_{axis}_{side}",
                "box",
                size,
                pos,
                ".56 .60 .60 1",
                mass=".05",
            )
    mast = ET.SubElement(rover, "body", name="navigation_mast", pos="-.53 .25 .13")
    element_geom(
        mast,
        "mast",
        "capsule",
        (0.02, 0.32),
        (0, 0, 0.32),
        ".69 .72 .72 1",
        mass=".1",
        contype="0",
        conaffinity="0",
    )
    element_geom(
        mast,
        "stereo_head",
        "box",
        (0.065, 0.11, 0.045),
        (0, 0, 0.64),
        ".22 .27 .30 1",
        mass=".05",
        contype="0",
        conaffinity="0",
    )
    camera(rover, "rover_follow", (3.2, -4.5, 1.65), (0.1, 0, 0.35), 54)
    camera(rover, "sampling_closeup", (1.2, -1.6, 0.9), (0.35, -0.2, -0.15), 56)


def add_lander(world):
    lander = ET.SubElement(world, "body", name="lander", pos="-2.8 3.5 .935")
    element_geom(
        lander,
        "lander_module",
        "cylinder",
        (0.9, 0.65),
        (0, 0, 0),
        ".78 .80 .78 1",
        mass="0",
    )
    element_geom(
        lander,
        "lander_blanket",
        "cylinder",
        (0.92, 0.18),
        (0, 0, -0.35),
        ".67 .48 .13 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    for index, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        ET.SubElement(
            lander,
            "geom",
            name=f"lander_leg_{index}",
            type="capsule",
            size=".045",
            fromto=f"{x*.55} {y*.55} -.3 {x*1.2} {y*1.2} -.9",
            rgba=".65 .69 .70 1",
        )
        element_geom(
            lander,
            f"lander_foot_{index}",
            "cylinder",
            (0.23, 0.035),
            (x * 1.2, y * 1.2, -0.9),
            ".25 .28 .30 1",
            mass="0",
        )
    for index, x in enumerate((-1.6, 1.6)):
        element_geom(
            lander,
            f"solar_array_{index}",
            "box",
            (0.62, 0.83, 0.025),
            (x, 0, 0.55),
            ".08 .17 .33 1",
            mass="0",
            contype="0",
            conaffinity="0",
        )
        for cell in range(6):
            element_geom(
                lander,
                f"solar_grid_{index}_{cell}",
                "box",
                (0.62, 0.006, 0.003),
                (x, -0.7 + cell * 0.28, 0.58),
                ".43 .54 .64 1",
                mass="0",
                contype="0",
                conaffinity="0",
            )
    element_geom(
        lander,
        "antenna_mast",
        "capsule",
        (0.025, 0.5),
        (0, 0, 1.13),
        ".73 .74 .71 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    element_geom(
        lander,
        "antenna_dish",
        "ellipsoid",
        (0.3, 0.3, 0.07),
        (0, 0, 1.66),
        ".81 .81 .75 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )


def add_rocks(world, assets, mission, height):
    obj = SOURCE_ROOT / "assets/space/apollo_sample/part_0.obj"
    ET.SubElement(assets, "mesh", name="apollo_sample", file=str(obj))
    ET.SubElement(
        assets,
        "texture",
        name="apollo_diffuse",
        type="2d",
        file=str(SOURCE_ROOT / "assets/space/apollo_sample/diffuse_0.png"),
    )
    ET.SubElement(
        assets,
        "material",
        name="apollo_material",
        texture="apollo_diffuse",
        rgba="1 1 1 1",
        shininess="0",
    )
    sample = ET.SubElement(
        world,
        "body",
        name="field_sample",
        pos=" ".join(map(str, mission.sample_position)),
    )
    ET.SubElement(sample, "freejoint", name="field_sample_free")
    ET.SubElement(
        sample,
        "inertial",
        pos="0 0 .0257",
        mass=".1",
        diaginertia=".00004 .00004 .000036",
    )
    ET.SubElement(
        sample,
        "geom",
        name="field_sample_geom",
        type="mesh",
        mesh="apollo_sample",
        material="apollo_material",
        mass="0",
        friction="1.4 .01 .001",
    )
    ET.SubElement(
        sample,
        "site",
        name="field_sample_grasp",
        pos="0 0 .044",
        size=".003",
        group="5",
    )
    rng = np.random.default_rng(741)
    axis = np.linspace(
        -mission.terrain_half_size_m, mission.terrain_half_size_m, height.shape[0]
    )
    ground = RegularGridInterpolator((axis, axis), height)
    # All background rocks reuse Apollo geometry; their placements/scales are generated.
    for index in range(30):
        x, y = rng.uniform(-25, 25, 2)
        if -4 < x < 11 and abs(y) < 3:
            y = math.copysign(4 + abs(y), y)
        scale = rng.uniform(3, 11)
        name = f"background_rock_mesh_{index}"
        ET.SubElement(
            assets,
            "mesh",
            name=name,
            file=str(obj),
            scale=f"{scale} {scale*.8} {scale}",
        )
        z = float(ground([[y, x]])[0]) - 0.01
        ET.SubElement(
            world,
            "geom",
            name=f"background_rock_{index}",
            type="mesh",
            mesh=name,
            material="apollo_material",
            pos=f"{x} {y} {z}",
            quat=f"{math.cos(index)} 0 0 {math.sin(index)}",
            contype="0",
            conaffinity="0",
            mass="0",
        )


def build(mission, destination=CACHE):
    destination = Path(destination)
    fingerprint = hashlib.sha256(repr(mission).encode())
    dependencies = [
        Path(__file__).with_name(name)
        for name in ("scene.py", "terrain.py", "profiles.py", "appearance.py")
    ]
    dependencies += [
        SOURCE_ROOT / "worlds" / name
        for name in ("build.py", "pds_terrain.py", "profiles.py")
    ]
    dependencies += [
        SOURCE_ROOT / "assets/space/terrain" / mission.world / name
        for name in ("height_m.npy", "manifest.json")
    ]
    dependencies += [
        SOURCE_ROOT / "assets/space/apollo_sample" / name
        for name in ("part_0.obj", "diffuse_0.png")
    ]
    for dependency in dependencies:
        fingerprint.update(dependency.read_bytes())
    destination /= f"{mission.world}-{fingerprint.hexdigest()[:12]}"
    destination.mkdir(parents=True, exist_ok=True)
    scene_path = destination / f"{mission.world}-sampling.xml"
    if scene_path.is_file():
        return scene_path
    height, provenance = landscape(mission)
    height_path = destination / f"{mission.world}-terrain.bin"
    minimum, span = write_heightfield(height_path, height)
    root = ET.Element("mujoco", model=mission.task_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "statistic", extent="40", center="0 0 0")
    ET.SubElement(
        root,
        "option",
        timestep=".002",
        gravity=f"0 0 {-mission.environment.gravity_m_s2}",
        integrator="implicitfast",
        cone="elliptic",
        noslip_iterations="2",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "map", znear=".0025", zfar="75", shadowclip=".6")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(
        visual, "headlight", diffuse="0 0 0", ambient="0 0 0", specular="0 0 0"
    )
    assets, world, actuators = (
        ET.SubElement(root, tag) for tag in ("asset", "worldbody", "actuator")
    )
    appearance = APPEARANCES[mission.world]
    ET.SubElement(
        assets,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=appearance.sky_top,
        rgb2=appearance.sky_bottom,
        width="512",
        height="3072",
    )
    ET.SubElement(
        assets,
        "hfield",
        name="surface",
        file=str(height_path.resolve()),
        size=f"{mission.terrain_half_size_m} {mission.terrain_half_size_m} {span} .2",
    )
    provenance["visual_environment"] = add_appearance(
        world, assets, mission, height, destination
    )
    ET.SubElement(
        world,
        "geom",
        name="surface",
        type="hfield",
        hfield="surface",
        pos=f"0 0 {minimum}",
        material="soil",
        group="5",
        friction="1.2 .02 .001",
    )
    ET.SubElement(
        world,
        "light",
        name="sun",
        directional="true",
        pos="-15 -10 20",
        dir=appearance.sun_direction,
        diffuse=appearance.sunlight,
        ambient=appearance.ambient,
    )
    add_rover(world, assets, actuators, mission)
    add_lander(world)
    add_rocks(world, assets, mission, height)
    # Marker posts identify the survey corridor, not a painted road on the planet.
    for index, x in enumerate((0, 2.5, 5.5)):
        element_geom(
            world,
            f"survey_post_{index}",
            "capsule",
            (0.012, 0.2),
            (x, -0.95, 0.2),
            ".93 .61 .18 1",
            mass="0",
            contype="0",
            conaffinity="0",
        )
        element_geom(
            world,
            f"survey_flag_{index}",
            "box",
            (0.09, 0.006, 0.05),
            (x + 0.08, -0.95, 0.36),
            ".93 .61 .18 1",
            mass="0",
            contype="0",
            conaffinity="0",
        )
    camera(world, "expedition_overview", (11, -16, 5), (1, 3, 1.0), 54)
    camera(world, "landing_site", (7, -9, 2.7), (0, 2, 0.8), 54)
    ET.indent(root, space="  ")
    serialized = ET.tostring(root, encoding="unicode")
    provenance.update(
        world=mission.world,
        scene_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        sample_geometry="NASA JSC Apollo sample via SRB; illustrative specimen mass, not measured composition",
        martian_sample_scope="Apollo geometry reused as an illustrative analog, not a measured Martian rock",
    )
    (destination / f"{mission.world}-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    # Publish the scene last, so readers never see a partially written XML file.
    with tempfile.NamedTemporaryFile("w", dir=destination, delete=False) as temporary:
        temporary.write(serialized)
    Path(temporary.name).replace(scene_path)
    return scene_path


if __name__ == "__main__":
    for mission in MISSIONS.values():
        print(build(mission))
