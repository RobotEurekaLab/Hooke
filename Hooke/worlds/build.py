"""Generate meter-scale worlds from shared, local Hooke assets."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from worlds.profiles import WORLDS

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCENES = SOURCE_ROOT / "model/scene"


def geom(parent, name, kind, size, pos, rgba, **attributes):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type=kind,
        size=size,
        pos=pos,
        rgba=rgba,
        **attributes,
    )


def attach(parent, model, prefix, pos, **attributes):
    body = ET.SubElement(parent, "body", name=prefix.rstrip(":"), pos=pos, **attributes)
    ET.SubElement(body, "attach", model=model, body="world", prefix=prefix)
    return body


def camera(parent, name, position, target, fovy):
    import numpy as np

    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(
        parent,
        "camera",
        name=name,
        pos=" ".join(map(str, position)),
        xyaxes=" ".join(map(str, [*x, *y])),
        fovy=str(fovy),
    )


def build_scene(profile):
    root = ET.Element("mujoco", model=f"space_{profile.name}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        gravity=f"0 0 {-profile.gravity_m_s2}",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        noslip_iterations="2",
        density="0",
        viscosity="0",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    assets = ET.SubElement(root, "asset")
    colors = {
        "orbital": ("0.02 0.025 0.05", "0.04 0.05 0.08"),
        "lunar": ("0.005 0.006 0.009", "0.01 0.01 0.015"),
        "martian": ("0.38 0.23 0.14", "0.13 0.09 0.07"),
    }
    top, bottom = colors[profile.name]
    ET.SubElement(
        assets,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=top,
        rgb2=bottom,
        width="512",
        height="3072",
    )
    dependencies = {
        "table": "../misc/simple_table.xml",
        "ur5e": "../robot/ur5e_gripper.xml",
        "cartridge": "../instrument/space_cartridge.xml",
        "balance": "../instrument/analytical_balance.xml",
        "glovebox": "../instrument/glovebox.xml",
    }
    if profile.name == "orbital":
        del dependencies["table"]
        del dependencies["balance"]
    for name, path in dependencies.items():
        ET.SubElement(assets, "model", name=name, file=path, content_type="text/xml")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="work_light",
        directional="true",
        pos="0 -2 4",
        dir="0 0.3 -1",
        diffuse="0.8 0.8 0.8",
        ambient="0.15 0.15 0.15",
    )
    floor_color = {
        "orbital": "0.29 0.34 0.40 1",
        "lunar": "0.30 0.30 0.31 1",
        "martian": "0.42 0.22 0.12 1",
    }[profile.name]
    geom(
        world,
        "terrain",
        "plane",
        "10 10 0.05",
        "0 0 0",
        floor_color,
        friction="0.8 0.01 0.001",
    )
    if profile.name == "orbital":
        cabin(world)
    else:
        terrain(world, assets, profile.name)
        # Reference sealed station module. No gas pressure or active life support is simulated.
        geom(
            world,
            "habitat_shell",
            "box",
            "0.8 0.5 0.7",
            "1.6 1.6 0.7",
            "0.7 0.73 0.74 1",
        )
        geom(
            world,
            "habitat_hatch",
            "box",
            "0.28 0.015 0.45",
            "1.6 1.085 0.65",
            "0.16 0.21 0.24 1",
        )
        for x in (1.25, 1.95):
            geom(
                world,
                f"hatch_rail_{x}",
                "capsule",
                "0.015 0.33",
                f"{x} 1.05 0.75",
                "0.9 0.59 0.15 1",
            )
        for x in (-0.5, 0.5):
            geom(
                world,
                f"ground_anchor_{x}",
                "box",
                "0.1 0.1 0.025",
                f"{x} 0 0.025",
                "0.16 0.17 0.18 1",
            )
    bench = ET.SubElement(world, "body", name="anchored_workstation")
    if profile.name == "orbital":
        orbital_work_surface(bench)
    else:
        attach(bench, "table", "bench:", "0 0 0")
    attach(bench, "ur5e", "/ur:", "0.48 0.12 0.824", quat="0 0 0 1")
    attach(bench, "glovebox", "/containment:", "-0.30 0.08 0.824")
    if profile.name != "orbital":
        attach(bench, "balance", "/balance:", "0.02 -0.20 0.824")
    geom(
        bench,
        "sample_retention_base",
        "box",
        "0.15 0.06 0.008",
        "-0.32 -0.24 0.832",
        "0.13 0.22 0.30 1",
    )
    for index, x in enumerate((-0.42, -0.32, -0.22)):
        attach(bench, "cartridge", f"/retained_{index}:", f"{x} -0.24 0.883")
        for side in (-1, 1):
            geom(
                bench,
                f"retention_clip_{index}_{side}",
                "box",
                "0.004 0.035 0.024",
                f"{x + side * 0.031} -0.24 0.87",
                "0.8 0.47 0.10 1",
            )
    # A prototype single-axis slide for future inertial measurements, not a calibrated instrument.
    geom(
        bench,
        "inertial_rail",
        "box",
        "0.16 0.045 0.012",
        "0.29 -0.24 0.838",
        "0.15 0.20 0.26 1",
    )
    slide = ET.SubElement(
        bench, "body", name="inertial_fixture", pos="0.29 -0.24 0.873"
    )
    ET.SubElement(
        slide,
        "joint",
        name="inertial_slide",
        type="slide",
        axis="1 0 0",
        range="-0.09 0.09",
        stiffness="5",
        damping="0.1",
    )
    geom(
        slide,
        "slide_tray",
        "box",
        "0.04 0.035 0.02",
        "0 0 0",
        "0.27 0.50 0.66 1",
        mass="0.2",
    )
    # Unretained dry cartridge deliberately clear of the bench and floor during the 0.5 s gravity test.
    loose = attach(world, "cartridge", "/free:", "-0.9 -0.70 1.5")
    ET.SubElement(loose, "freejoint", name="free_cartridge_joint")
    camera(world, "world_overview", (0, -3.9, 2.8), (0, 0.15, 0.95), 52)
    camera(world, "experiment_closeup", (0.1, -1.6, 1.8), (0, 0, 1.0), 48)
    ET.indent(root, space="  ")
    return root, dependencies


def orbital_work_surface(parent):
    """Station-fixed panel and side brackets, with no floor-supported legs.

    This welded rigid assembly does not simulate bolt compliance or station
    recoil. Positive retention is independent of contact weight/friction.
    """
    geom(
        parent,
        "rack_work_panel",
        "box",
        "0.66 0.40 0.016",
        "0 0 0.808",
        "0.68 0.72 0.76 1",
    )
    for index, y in enumerate((-0.30, 0.30)):
        geom(
            parent,
            f"rack_wall_bracket_{index}",
            "box",
            "0.45 0.025 0.025",
            f"1.11 {y} 0.79",
            "0.20 0.30 0.40 1",
        )
        geom(
            parent,
            f"rack_wall_anchor_{index}",
            "box",
            "0.025 0.07 0.08",
            f"1.55 {y} 0.79",
            "0.82 0.55 0.18 1",
        )
    for name, position, size in (
        ("robot_mount", "0.48 0.12 0.812", "0.13 0.13 0.012"),
        ("glovebox_mount", "-0.30 0.08 0.812", "0.30 0.22 0.012"),
    ):
        geom(parent, name, "box", size, position, "0.20 0.30 0.40 1")
    for index, (x, y) in enumerate(
        ((0.38, 0.02), (0.58, 0.02), (0.38, 0.22), (0.58, 0.22))
    ):
        geom(
            parent,
            f"robot_mount_bolt_{index}",
            "cylinder",
            "0.009 0.005",
            f"{x} {y} 0.829",
            "0.82 0.55 0.18 1",
            contype="0",
            conaffinity="0",
        )


def cabin(world):
    for index, x in enumerate((-1.6, 1.6)):
        geom(
            world,
            f"cabin_side_{index}",
            "box",
            "0.06 1.9 1.25",
            f"{x} 0 1.25",
            "0.70 0.73 0.77 1",
        )
        for height in (0.45, 1.1, 1.9):
            geom(
                world,
                f"handrail_{index}_{height}",
                "capsule",
                "0.02 0.32",
                f"{x * 0.95} 0.6 {height}",
                "0.94 0.64 0.16 1",
                quat="0.7071068 0.7071068 0 0",
            )
    geom(world, "cabin_back", "box", "1.55 0.06 1.25", "0 1.9 1.25", "0.77 0.79 0.82 1")
    for index, x in enumerate((-0.9, 0, 0.9)):
        geom(
            world,
            f"rack_panel_{index}",
            "box",
            "0.30 0.04 0.45",
            f"{x} 1.8 1.05",
            "0.24 0.29 0.35 1",
        )
        geom(
            world,
            f"rack_display_{index}",
            "box",
            "0.19 0.012 0.12",
            f"{x} 1.75 1.30",
            "0.12 0.53 0.64 1",
            contype="0",
            conaffinity="0",
        )
    # Cutaway roof/front keep both cameras usable; this is not a pressure-tight habitat simulation.
    for x in (-1.3, 1.3):
        geom(
            world,
            f"cabin_rib_{x}",
            "box",
            "0.035 1.85 0.035",
            f"{x} 0 2.4",
            "0.8 0.82 0.85 1",
        )


def terrain(world, assets, name):
    # Bake anisotropy into vertices. PhysX's scaled sphere primitive cannot
    # preserve the collision shape of a nonuniform MuJoCo ellipsoid.
    phi = (1 + math.sqrt(5)) / 2
    vertices = [
        (-1, phi, 0),
        (1, phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1, phi),
        (0, 1, phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    normalization = math.sqrt(1 + phi**2)
    rng = random.Random(17)
    color = "0.34 0.33 0.32 1" if name == "lunar" else "0.36 0.19 0.11 1"
    for index in range(28):
        angle = rng.uniform(0, 6.283185)
        radius = rng.uniform(2.4, 7.5)
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        size = rng.uniform(0.12, 0.42)
        mesh_name = f"rock_mesh_{index}"
        baked = [
            (
                vx * size / normalization,
                vy * size * 0.8 / normalization,
                vz * size * 0.6 / normalization,
            )
            for vx, vy, vz in vertices
        ]
        ET.SubElement(
            assets,
            "mesh",
            name=mesh_name,
            vertex=" ".join(str(v) for point in baked for v in point),
            face=" ".join(str(v) for face in faces for v in face),
        )
        ET.SubElement(
            world,
            "geom",
            name=f"procedural_rock_{index}",
            type="mesh",
            mesh=mesh_name,
            pos=f"{x} {y} {size * 0.3}",
            rgba=color,
        )


def build():
    scenes = []
    assets = {}
    for profile in WORLDS.values():
        root, dependencies = build_scene(profile)
        path = SCENES / f"space_{profile.name}.gen.xml"
        ET.ElementTree(root).write(path, encoding="unicode")
        scenes.append(
            {
                "world": profile.name,
                "scene": path.relative_to(SOURCE_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "environment": profile.report(),
            }
        )
        for name, relative in dependencies.items():
            source = (SCENES / relative).resolve()
            assets[name] = {
                "path": source.relative_to(SOURCE_ROOT).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "origin": (
                    "local_procedural"
                    if name == "cartridge"
                    else "existing_hooke_asset"
                ),
                "license_scope": "existing_repository_provenance_no_new_external_license_audit",
            }
    manifest = {
        "schema_version": 1,
        "scenes": scenes,
        "assets": assets,
        "geometry": "meter_scale_procedural_not_real_spacecraft_or_measured_planetary_terrain",
        "mounts": "explicit_fixed_workstation_robot_instruments_and_retained_cartridges",
        "qualification": "scene_and_rigid_body_tests_only_no_grasp_latch_pressure_seal_or_E1_E2_certification",
    }
    destination = SOURCE_ROOT / "worlds/assets_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({"generated_scenes": len(build()["scenes"])}))
