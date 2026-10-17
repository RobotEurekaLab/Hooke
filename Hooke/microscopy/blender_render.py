"""Render an exported experiment snapshot with Blender's bundled Python.

Example: blender -b --python blender_render.py -- --snapshot DIR --output DIR
This is an offline presentation renderer; physics stays in the source backend.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def _image(record, arrays):
    pixels = arrays[record["array"]]
    rgba = np.ones((*pixels.shape[:2], 4), dtype=np.float32)
    channels = pixels.shape[2]
    if channels == 1:
        rgba[..., :3] = pixels/255.
    elif channels in (3, 4):
        rgba[..., :channels] = pixels/255.
    else:
        raise ValueError(f"Unsupported texture channel count: {channels}")
    image = bpy.data.images.new(record["name"], width=record["width"], height=record["height"])
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def _material(name, rgba, record, textures, surface):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    colour = np.asarray(rgba[:3])
    linear = np.where(colour <= .04045, colour/12.92, ((colour+.055)/1.055)**2.4)
    shader.inputs["Base Color"].default_value = (*linear, 1)
    # Translate the simple simulator materials into named, documented finishes.
    # Geometry, colour assignments, camera poses and all encoder positions remain
    # those of the export. Lighting and the BRDF are presentation choices.
    # MuJoCo defaults metallic to 1 even for dielectric ivory/rubber. Its
    # legacy OpenGL renderer ignores this field; copying it makes all paint
    # look like chrome. Use the named physical finish instead.
    metallic = 0.
    roughness = max(.18, 1-math.sqrt(record.get("shininess", .3)))
    if name in ("metal", "breadboard"):
        metallic, roughness = .8, .3 if name == "metal" else .42
        if colour.max()-colour.min() > .15:
            metallic, roughness = 0., .4  # Painted CAD part colour override.
    elif name == "bronze":
        metallic, roughness = .8, .32
    elif name in ("ivory", "lab_wall", "lab_floor"):
        roughness = .34 if name == "ivory" else .75
    elif name in ("graphite", "rubber"):
        roughness = .48 if name == "graphite" else .7
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = surface.get("roughness", roughness)
    shader.inputs["IOR"].default_value = 1.46
    shader.inputs["Specular IOR Level"].default_value = .25
    if rgba[3] < .99:
        shader.inputs["Transmission Weight"].default_value = 1-float(rgba[3])
        shader.inputs["Roughness"].default_value = .13
        if name == "lens":
            shader.inputs["Transmission Weight"].default_value = .04
    texture_ids = record.get("texture_ids", [])
    texture_id = next((texture_ids[role] for role in (8, 1)
                       if len(texture_ids) > role and texture_ids[role] in textures), None)
    if texture_id is not None:
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = textures[texture_id]
        material.node_tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
        if record.get("emission", 0):
            material.node_tree.links.new(node.outputs["Color"], shader.inputs["Emission Color"])
            shader.inputs["Emission Strength"].default_value = record["emission"]
    return material


def _mesh(record, arrays):
    mesh = bpy.data.meshes.new(record["name"])
    mesh.from_pydata(arrays[record["vertices"]], [], arrays[record["faces"]])
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    normals = arrays[record["normals"]]
    indices = arrays[record["face_normals"]]
    if len(normals) and (indices >= 0).all():
        mesh.normals_split_custom_set(normals[indices].reshape(-1, 3))
    if record["texcoords"] is not None:
        uv = mesh.uv_layers.new(name="UVMap")
        coords = arrays[record["texcoords"]]
        indices = arrays[record["face_texcoords"]]
        if (indices < 0).any():
            raise ValueError(f"Incomplete UV mapping: {record['name']}")
        uv.data.foreach_set("uv", coords[indices].ravel())
    return mesh


def _primitive(geom, cache):
    kind, size = geom["type"], geom["size"]
    key = (kind, tuple(size) if kind == 3 else ())
    if key in cache:
        obj = bpy.data.objects.new(geom["name"], cache[key])
        bpy.context.collection.objects.link(obj)
        if kind in (0, 2, 4, 5, 6):
            obj.scale = ((size[0] or 3, size[1] or 3, 1) if kind == 0 else
                         [size[0]]*3 if kind == 2 else
                         (size[0], size[0], size[1]) if kind == 5 else size)
        return obj
    if kind == 0:
        bpy.ops.mesh.primitive_plane_add(size=2)
        bpy.context.object.scale = (size[0] or 3, size[1] or 3, 1)
    elif kind == 6:
        bpy.ops.mesh.primitive_cube_add(size=2)
        bpy.context.object.scale = size
    elif kind in (2, 4):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)
        bpy.context.object.scale = [size[0]]*3 if kind == 2 else size
    elif kind == 5:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1, depth=2)
        bpy.context.object.scale = (size[0], size[0], size[1])
    elif kind == 3:
        # Exact capsule profile: two hemispheres separated by the cylinder span.
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=size[0])
        for vertex in bpy.context.object.data.vertices:
            vertex.co.z += size[1] if vertex.co.z > 0 else -size[1]
    else:
        raise ValueError(f"Unsupported geometry type {kind}")
    obj = bpy.context.object
    if kind in (2, 3, 4, 5):
        for polygon in obj.data.polygons:
            polygon.use_smooth = kind != 5 or len(polygon.vertices) == 4
    cache[key] = obj.data
    return obj


def build_scene(snapshot):
    record = json.loads((snapshot/"scene.json").read_text())
    if record.get("schema_version") != 1 or record.get("units") != "m":
        raise ValueError("Expected a version 1 snapshot in metres")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    with np.load(snapshot/record["arrays"], allow_pickle=False) as arrays:
        meshes = {item["id"]: _mesh(item, arrays) for item in record["meshes"]}
        textures = {item["id"]: _image(item, arrays) for item in record["textures"] if item["type"] == 0}
    definitions = {item["id"]: item for item in record["materials"]}
    materials, primitives = {}, {}
    for geom in record["geoms"]:
        definition = definitions.get(geom["material"], {})
        name = definition.get("name", "runtime_surface")
        key = (geom["material"], tuple(geom["rgba"]), json.dumps(geom.get("surface", {}), sort_keys=True))
        if key not in materials:
            materials[key] = _material(name, geom["rgba"], definition, textures, geom.get("surface", {}))
        if geom["type"] == 7:
            # Copy only when materials differ between instances of one mesh.
            obj = bpy.data.objects.new(geom["name"], meshes[geom["mesh"]])
            bpy.context.collection.objects.link(obj)
        else:
            obj = _primitive(geom, primitives)
            obj.name = geom["name"]
        obj.location = geom["pos"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Matrix(np.array(geom["mat"]).reshape(3, 3)).to_quaternion()
        if not obj.data.materials:
            obj.data.materials.append(materials[key])
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials[key]
        obj["snapshot_source"] = geom["source"]
    for camera in record["cameras"]:
        data = bpy.data.cameras.new(camera["name"])
        data.sensor_fit = "VERTICAL"
        data.sensor_height = 32
        data.lens = 16/math.tan(math.radians(camera["fovy"])/2)
        data.clip_start, data.clip_end = .0001, 100
        obj = bpy.data.objects.new(camera["name"], data)
        bpy.context.collection.objects.link(obj)
        obj.location = camera["pos"]
        obj.rotation_euler = Matrix(np.array(camera["mat"]).reshape(3, 3)).to_euler()
    return record


def _area(name, position, target, energy, size, colour):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size, data.color = energy, "DISK", size, colour
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


def render(args):
    record = build_scene(args.snapshot)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transparent_max_bounces = 12
    if args.device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = args.device
        preferences.get_devices()
        enabled = []
        for device in preferences.devices:
            device.use = device.type == args.device
            if device.use:
                enabled.append(device.name)
        if not enabled:
            raise RuntimeError(f"No {args.device} device available")
        scene.cycles.device = "GPU"
    else:
        enabled = ["CPU"]
    scene.render.threads_mode, scene.render.threads = "FIXED", args.threads
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.6, .65, .7, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .35
    _area("Softbox key", (-.6, -.75, 2.25), (0, 0, 1), 120, 1.2, (1, .955, .90))
    _area("Softbox fill", (1.0, -.1, 1.8), (0, 0, 1), 60, .9, (.90, .95, 1))
    _area("Top rim", (-.3, .65, 2.0), (0, -.1, 1), 80, .7, (1, 1, 1))
    if args.camera == "hero":
        data = bpy.data.cameras.new("hero")
        data.lens, data.clip_start = 48, .001
        camera = bpy.data.objects.new("hero", data)
        bpy.context.collection.objects.link(camera)
        camera.location = (.82, -1.32, 1.48)
        camera.rotation_euler = (Vector((0, -.04, 1.01))-camera.location).to_track_quat("-Z", "Y").to_euler()
    else:
        camera = bpy.data.objects.get(args.camera)
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"Unknown exported camera: {args.camera}")
    scene.camera = camera
    args.output.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output/(args.camera+".png"))
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/"workstation.blend"))
    bpy.ops.render.render(write_still=True)
    report = dict(renderer="Blender Cycles", blender=bpy.app.version_string, devices=enabled,
                  source=str(args.snapshot), geometry_count=len(record["geoms"]), units="m",
                  camera=args.camera, resolution=[args.width, args.height], samples=args.samples,
                  presentation_changes=["PBR material translation", "area lighting", "AgX colour transform"],
                  simulation_advanced=False)
    (args.output/"render.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="hero")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--threads", type=int, default=8)
    render(parser.parse_args(sys.argv[sys.argv.index("--")+1:]))


if __name__ == "__main__":
    main()
