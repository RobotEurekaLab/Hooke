"""Offline Blender CPU rendering of exported real-lab geometry; no GPU context.

Run with Blender's Python: blender -b --python real_labs/render.py -- --input DIR.
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microscopy.blender_render import _area, build_scene
from real_labs.finishes import equipment_finishes, manufactured_edges, primitive_normals
from real_labs.presentation import fixture_emitter_bounds, inspection_softbox_visibility
from real_labs.render_version import FINISH_REVISION, render_source_digest
from real_labs.volume_presentation import apply_presentation_volumes


def architectural_finishes():
    """Original procedural surface finishes; no downloaded photo textures."""
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.data.materials:
            continue
        if obj.name == "arch_floor" or obj.name.startswith("arch_wall_"):
            material = obj.active_material.copy()
            obj.active_material = material
            nodes, links = material.node_tree.nodes, material.node_tree.links
            shader = nodes.get("Principled BSDF")
            noise = nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = (
                320 if obj.name == "arch_floor" else 650
            )
            noise.inputs["Detail"].default_value = 2
            coordinates = nodes.new("ShaderNodeNewGeometry")
            links.new(coordinates.outputs["Position"], noise.inputs["Vector"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.12
            bump.inputs["Distance"].default_value = 0.00008
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], shader.inputs["Normal"])
            shader.inputs["Roughness"].default_value = (
                0.48 if obj.name == "arch_floor" else 0.76
            )
            obj["presentation_finish"] = (
                "estimated sealed floor"
                if obj.name == "arch_floor"
                else "estimated matte wall"
            )
        elif "_glass" in obj.name:
            material = obj.active_material.copy()
            obj.active_material = material
            shader = material.node_tree.nodes.get("Principled BSDF")
            shader.inputs["Transmission Weight"].default_value = 0.98
            shader.inputs["Base Color"].default_value = (0.91, 0.95, 0.96, 1)
            shader.inputs["Roughness"].default_value = 0.035
        elif obj.name.endswith("_soil"):
            material = obj.active_material.copy()
            obj.active_material = material
            nodes, links = material.node_tree.nodes, material.node_tree.links
            shader = nodes.get("Principled BSDF")
            coordinates = nodes.new("ShaderNodeNewGeometry")
            grain = nodes.new("ShaderNodeTexNoise")
            grain.inputs["Scale"].default_value = 350
            grain.inputs["Detail"].default_value = 3
            links.new(coordinates.outputs["Position"], grain.inputs["Vector"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.65
            bump.inputs["Distance"].default_value = 0.002
            links.new(grain.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], shader.inputs["Normal"])
            variation = nodes.new("ShaderNodeTexNoise")
            variation.inputs["Scale"].default_value = 3
            links.new(coordinates.outputs["Position"], variation.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = (0.60, 0.60, 0.60, 1)
            ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
            links.new(variation.outputs["Fac"], ramp.inputs["Fac"])
            multiply = nodes.new("ShaderNodeMixRGB")
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1
            multiply.inputs[1].default_value = shader.inputs["Base Color"].default_value
            links.new(ramp.outputs["Color"], multiply.inputs[2])
            links.new(multiply.outputs["Color"], shader.inputs["Base Color"])
            shader.inputs["Roughness"].default_value = 0.95


def render_package(path, args):
    manifest = json.loads((path / "manifest.json").read_text())
    softboxes_visible = inspection_softbox_visibility(manifest)
    snapshot = build_scene(path / "snapshot")
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        shader = material.node_tree.nodes.get("Principled BSDF")
        if shader and "__mat_glass" in material.name:
            shader.inputs["Base Color"].default_value = (0.91, 0.96, 0.98, 1)
            shader.inputs["Transmission Weight"].default_value = 1
            shader.inputs["Roughness"].default_value = 0.018
    manufactured_edges()
    curved_surface_count = primitive_normals(snapshot)
    scene = bpy.context.scene
    architectural_finishes()
    finish_counts = equipment_finishes(manifest)
    volume_records = apply_presentation_volumes(manifest)
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 10
    scene.render.threads_mode, scene.render.threads = "FIXED", args.threads
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.1
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.75,
        0.80,
        0.86,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.30
    w, d, h = manifest["room"]["size"]
    area = w * d
    _area(
        "Ceiling softbox",
        (0, 0, h + 0.2),
        (0, 0, 0),
        area * 15,
        w * 0.7,
        (1, 0.97, 0.93),
    )
    _area(
        "Front softbox",
        (0, -d * 0.6, h * 0.9),
        (0, 0.5, 0.8),
        area * 9,
        w * 0.65,
        (0.90, 0.95, 1),
    )
    _area(
        "Right fill", (w * 0.55, 0.3, h * 0.8), (0, 0, 1), area * 7, d * 0.65, (1, 1, 1)
    )
    for name in ("Ceiling softbox", "Front softbox", "Right fill"):
        light = bpy.data.objects[name]
        light.data.specular_factor = 0.55
        light.visible_glossy = softboxes_visible
        light.visible_transmission = softboxes_visible
    bpy.context.view_layer.update()
    fixtures = [
        obj
        for obj in scene.objects
        if obj.name.startswith(("arch_luminaire_", "wetlab_luminaire_"))
    ]
    ceiling_count = sum("clean_work" not in obj.name for obj in fixtures)
    for index, obj in enumerate(fixtures):
        diffuser = obj.active_material.copy()
        obj.active_material = diffuser
        shader = diffuser.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (1, 0.985, 0.965, 1)
        shader.inputs["Emission Strength"].default_value = 2
        (x, y, z), (size_x, size_y) = fixture_emitter_bounds(
            [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        )
        _area(
            f"Ceiling fixture {index}",
            (x, y, z),
            (x, y, 0),
            7 if "clean_work" in obj.name else area * 19 / max(1, ceiling_count),
            size_x,
            (1, 0.985, 0.965),
        )
        lamp = bpy.data.objects[f"Ceiling fixture {index}"].data
        lamp.shape, lamp.size_y = "RECTANGLE", size_y
    for index, obj in enumerate(list(scene.objects)):
        if not obj.name.startswith("facility_growth_lamp_face_"):
            continue
        diffuser = obj.active_material.copy()
        obj.active_material = diffuser
        shader = diffuser.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (1, 0.87, 0.68, 1)
        shader.inputs["Emission Strength"].default_value = 3
        (x, y, z), (size_x, size_y) = fixture_emitter_bounds(
            [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        )
        _area(
            f"Growth fixture {index}",
            (x, y, z),
            (x, y, 0),
            18,
            size_x,
            (1, 0.87, 0.68),
        )
        lamp = bpy.data.objects[f"Growth fixture {index}"].data
        lamp.shape, lamp.size_y = "RECTANGLE", size_y
    # Institutional names belong in the gallery; reference photos do not
    # establish the giant wall signs used in earlier presentation renders.
    scene.camera = bpy.data.objects["overview"]
    roofs = [
        obj
        for obj in scene.objects
        if obj.name.startswith(("arch_roof_", "arch_cutaway_wall_", "arch_luminaire_"))
    ]
    for roof in roofs:
        roof.hide_render = True
    if not args.preview_only:
        bpy.ops.wm.save_as_mainfile(filepath=str(path / "laboratory.blend"))
        bpy.ops.export_scene.gltf(
            filepath=str(path / "laboratory.glb"),
            export_format="GLB",
            export_cameras=True,
            export_lights=True,
        )
        bpy.ops.wm.usd_export(filepath=str(path / "laboratory.usdc"))
    views = args.views or manifest["camera_names"]
    for name in views:
        for roof in roofs:
            roof.hide_render = name not in ("interior", "reference")
        scene.camera = bpy.data.objects[name]
        scene.render.filepath = str(path / (name + ".png"))
        bpy.ops.render.render(write_still=True)
    (path / "render.json").write_text(
        json.dumps(
            dict(
                renderer="Blender Cycles",
                device="CPU",
                dimensions=[args.width, args.height],
                samples=args.samples,
                views=views,
                scene_sha256=manifest["scene_sha256"],
                renderer_source_sha256=render_source_digest(),
                cutaway="Roof, roof-mounted fixtures and front/right envelope hidden for overview/detail; full enclosure shown for interior/reference",
                blender=bpy.app.version_string,
                visual_export_formats=(
                    [] if args.preview_only else ["blend", "glb", "usdc"]
                ),
                physics_format="MJCF",
                fidelity="Snapshot of actual generated scene geometry; estimated lighting, edge radii and procedural finishes added",
                finish_revision=FINISH_REVISION,
                presentation_material_counts=finish_counts,
                presentation_volumes=volume_records,
                volume_interchange_scope=(
                    (
                        "Blender shader is authoritative. Tested Blender 4.5 exports "
                        "omit this absorption in GLB and USD; USD Preview Surface "
                        "also does not preserve the clear-water transmission."
                    )
                    if volume_records
                    else None
                ),
                analytic_curved_surfaces_smoothed=curved_surface_count,
                texture_coordinates="World metres; procedural estimated finishes",
                fixture_light_placement="Below the actual world-space fixture bounds",
                fixture_light_clearance_m=0.015,
                calibrated_lighting=False,
                inspection_softbox_visible_in_glass=softboxes_visible,
            ),
            indent=2,
        )
        + "\n"
    )
    print("LAB_RENDER_COMPLETE " + manifest["id"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("overview", "workstation", "interior", "reference", "instrument"),
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Render images without replacing editable exports",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    paths = (
        [args.input / args.scene]
        if args.scene
        else sorted(p.parent for p in args.input.glob("*/manifest.json"))
    )
    for path in paths:
        render_package(path.resolve(), args)


if __name__ == "__main__":
    main()
