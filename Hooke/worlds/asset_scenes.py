"""Opt-in scene variants with attributed, portable external visual assets.

Collision proxies and example masses are explicit Hooke settings. Imported
appearance does not establish real mass, gas seals or calibrated materials.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET

from worlds.build import SCENES, SOURCE_ROOT, attach, build_scene, camera, geom
from worlds.profiles import WORLDS

ASSET_ROOT = SOURCE_ROOT / "assets/space"


def attach_asset(parent, model, prefix, position, **attributes):
    body = attach(parent, model, prefix, position, **attributes)
    body.find("attach").set("body", "asset_root")
    return body


def add_sample(parent, name, prefix, position, assets, rows, *, free=False):
    body = attach_asset(parent, name, prefix, position)
    if free:
        ET.SubElement(body, "freejoint", name="free_cartridge_joint")
    row = rows[name]
    width, depth, height = row["dimensions_m"]
    if name == "apollo_sample":
        mesh_name = f"{prefix}collision_mesh"
        ET.SubElement(
            assets,
            "mesh",
            name=mesh_name,
            file=f"../../assets/space/{name}/part_0.obj",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{prefix}collision",
            type="mesh",
            mesh=mesh_name,
            group="3",
            mass="0",
            friction="0.8 0.01 0.001",
        )
    else:
        radius = max(width, depth) / 2
        geom(
            body,
            f"{prefix}collision",
            "capsule",
            f"{radius} {height / 2 - radius}",
            f"0 0 {height / 2}",
            "0.5 0.5 0.5 1",
            group="3",
            mass="0",
            friction="0.8 0.01 0.001",
        )
    if not free:
        mount = ET.SubElement(
            parent, "body", name=f"{prefix.rstrip(':')}_mount", pos=position
        )
        geom(
            mount,
            f"{prefix}mount_plate",
            "box",
            f"{width / 2 + 0.012} {depth / 2 + 0.01} 0.004",
            "0 0 -0.004",
            "0.18 0.28 0.38 1",
            contype="0",
            conaffinity="0",
            mass="0.02",
        )
        heights = (
            (height * 0.3,)
            if name == "apollo_sample"
            else (height * 0.25, height * 0.75)
        )
        for level, z in enumerate(heights):
            for side in (-1, 1):
                geom(
                    mount,
                    f"{prefix}mount_clip_{level}_{side}",
                    "box",
                    f"0.004 {depth / 2 + 0.004} 0.006",
                    f"{side * (width / 2 + 0.003)} 0 {z}",
                    "0.82 0.55 0.18 1",
                    contype="0",
                    conaffinity="0",
                    mass="0.001",
                )
    return body


def build_scene_with_assets(profile, rows):
    root, _ = build_scene(profile)
    root.set("model", f"space_{profile.name}_assets")
    assets, world = root.find("asset"), root.find("worldbody")
    for name in rows:
        ET.SubElement(
            assets,
            "model",
            name=name,
            file=f"../../assets/space/{name}/asset.xml",
            content_type="text/xml",
        )
    if profile.name == "orbital":
        for item in list(world):
            if item.tag == "geom" and item.get("name", "").startswith(
                ("cabin_", "rack_", "handrail_")
            ):
                world.remove(item)
        attach_asset(
            world,
            "iss_us_lab",
            "/iss:",
            "0 0 -0.94",
            quat=f"{math.sqrt(0.5)} 0 0 {math.sqrt(0.5)}",
        )
        length, width, height = rows["iss_us_lab"]["dimensions_m"]
        # Keep the imported concave cabin visual-only. Explicit boundary
        # panels retain the interior; no opaque whole-cabin convex collider.
        for side in (-1, 1):
            geom(
                world,
                f"iss_wall_{side}",
                "box",
                f"0.04 {length / 2} {height / 2}",
                f"{side * width / 2} 0 {height / 2 - 0.94}",
                "0.5 0.5 0.5 1",
                group="3",
            )
            geom(
                world,
                f"iss_end_{side}",
                "box",
                f"{width / 2} 0.04 {height / 2}",
                f"0 {side * length / 2} {height / 2 - 0.94}",
                "0.5 0.5 0.5 1",
                group="3",
            )
        geom(
            world,
            "iss_ceiling",
            "box",
            f"{width / 2} {length / 2} 0.04",
            f"0 0 {height - 0.94}",
            "0.5 0.5 0.5 1",
            group="3",
        )
        rail = attach_asset(world, "iss_handrail", "/rail:", "-0.55 -0.60 0.80")
        geom(
            rail,
            "rail_collision",
            "capsule",
            "0.012 0.365",
            "-0.019 0 0.3917",
            "0.5 0.5 0.5 1",
            group="3",
        )
        overview = world.find("camera[@name='world_overview']")
        world.remove(overview)
        camera(world, "world_overview", (0, -2.20, 1.70), (0, 0.3, 1.0), 62)
        camera(world, "asset_closeup", (0, -1.2, 1.50), (-0.55, -0.60, 1.2), 45)
        for index, y in enumerate((-2, 0, 2)):
            ET.SubElement(
                world,
                "light",
                name=f"cabin_fixture_{index}",
                pos=f"0 {y} 1.92",
                dir="0 0 -1",
                directional="false",
                cutoff="90",
                diffuse="0.6 0.6 0.6",
                ambient="0.05 0.05 0.05",
            )
    else:
        position = (
            (-0.53, -0.65, 1.06) if profile.name == "lunar" else (-0.53, -0.78, 1.16)
        )
        target = (
            (-0.53, -0.29, 0.86) if profile.name == "lunar" else (-0.53, -0.29, 0.917)
        )
        camera(
            world,
            "asset_closeup",
            position,
            target,
            25 if profile.name == "lunar" else 32,
        )
    bench = world.find("body[@name='anchored_workstation']")
    rock_position = (
        "-0.53 -0.05 0.832" if profile.name == "martian" else "-0.53 -0.29 0.832"
    )
    tube_position = (
        "-0.53 -0.29 0.832" if profile.name == "martian" else "-0.53 -0.05 0.832"
    )
    add_sample(bench, "apollo_sample", "/apollo_display:", rock_position, assets, rows)
    add_sample(bench, "mars_sample_tube", "/tube_display:", tube_position, assets, rows)
    # Reuse the source triangle geometry in the moving gravity witness too.
    loose = world.find("body[@name='/free']")
    world.remove(loose)
    name = "apollo_sample" if profile.name == "lunar" else "mars_sample_tube"
    free_position = "-0.40 -0.85 1.5" if profile.name == "orbital" else "-0.9 -0.70 1.5"
    add_sample(world, name, "/external_free:", free_position, assets, rows, free=True)
    ET.indent(root, space="  ")
    return root


def build():
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    rows = {row["id"]: row for row in manifest["assets"]}
    result = []
    for profile in WORLDS.values():
        path = SCENES / f"space_{profile.name}_assets.gen.xml"
        root = build_scene_with_assets(profile, rows)
        ET.ElementTree(root).write(path, encoding="unicode")
        result.append(
            {
                "world": profile.name,
                "task": f"space_{profile.name}_assets_workstation",
                "scene": path.relative_to(SOURCE_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "collision_scope": "Convex Apollo hull, capsule tube/rail and approximate cabin boundary panels; imported cabin triangles have no collisions.",
                "mass_scope": "0.1 kg illustrative sample mass, not measured NASA specimen/tube mass.",
            }
        )
    (ASSET_ROOT / "scenes.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({"generated_scenes": len(build())}))
