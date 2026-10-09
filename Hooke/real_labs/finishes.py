"""Original, metre-scale presentation finishes for offline Blender exports.

These shaders approximate material classes, not calibrated optical properties.
No reference photographs are embedded or used as textures.
"""

import bpy
from mathutils import Matrix


def manufactured_edges():
    """Round authored boxes in metres without anisotropic bevel distortion."""
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or len(obj.data.vertices) != 8:
            continue
        if obj.name.startswith(("arch_floor", "arch_wall", "arch_roof")):
            continue
        thickness = min(obj.dimensions)
        if thickness < 0.003 or "_glass" in obj.name:
            continue
        # The snapshot shares primitive meshes. Copy before baking object scale
        # so a wide countertop cannot change another instance's geometry.
        obj.data = obj.data.copy()
        obj.data.transform(Matrix.Diagonal((*obj.scale, 1)))
        obj.scale = (1, 1, 1)
        bevel = obj.modifiers.new("Manufactured edge radius (metres)", "BEVEL")
        bevel.width = min(0.0015, thickness * 0.12)
        bevel.segments = 3
        bevel.harden_normals = True


def _noise(material, scale, stretch=None):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    coord = nodes.new("ShaderNodeTexCoord")
    vector = coord.outputs["Object"]
    if stretch:
        mapping = nodes.new("ShaderNodeVectorMath")
        mapping.operation = "MULTIPLY"
        mapping.inputs[1].default_value = stretch
        links.new(vector, mapping.inputs[0])
        vector = mapping.outputs["Vector"]
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 2
    links.new(vector, noise.inputs["Vector"])
    return noise


def _variation(material, noise, strength, bump_distance):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    ramp = nodes.new("ShaderNodeValToRGB")
    color = tuple(shader.inputs["Base Color"].default_value)
    ramp.color_ramp.elements[0].color = tuple(c * (1 - strength) for c in color[:3]) + (
        1,
    )
    ramp.color_ramp.elements[1].color = tuple(
        min(1, c * (1 + strength)) for c in color[:3]
    ) + (1,)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.2
    bump.inputs["Distance"].default_value = bump_distance
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])


def equipment_finishes(manifest):
    wood_benches = tuple(
        b["id"] + "_"
        for b in manifest["benches"]
        if b.get("color", manifest["room"]["accent"])[0]
        > b.get("color", manifest["room"]["accent"])[2] * 1.5
    )
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.data.materials:
            continue
        original = obj.active_material
        material_name = original.name.split(".")[0]
        name = obj.name
        shader = original.node_tree.nodes.get("Principled BSDF")
        if not shader:
            continue
        color = shader.inputs["Base Color"].default_value
        wood = (
            (
                (wood_benches and name.startswith(wood_benches))
                or name.startswith("wetlab_")
            )
            and color[0] > color[2] * 2
            and color[1] > color[2] * 1.3
        )
        leaf = "_leaf_" in name and name.startswith("facility_")
        metal = material_name.endswith("__mat_metal") or any(
            token in name
            for token in ("_steel_", "_aluminium_", "_reflector", "_rail_")
        )
        if not (wood or leaf or metal):
            continue
        material = original.copy()
        obj.active_material = material
        shader = material.node_tree.nodes.get("Principled BSDF")
        if wood:
            shader.inputs["Roughness"].default_value = 0.39
            _variation(material, _noise(material, 1, (95, 95, 2.5)), 0.12, 0.00007)
            obj["presentation_finish"] = "estimated sealed wood veneer"
        elif leaf:
            shader.inputs["Roughness"].default_value = 0.56
            shader.inputs["Subsurface Weight"].default_value = 0.035
            shader.inputs["Subsurface Radius"].default_value = (0.001, 0.002, 0.0005)
            shader.inputs["Specular IOR Level"].default_value = 0.25
            nodes, links = material.node_tree.nodes, material.node_tree.links
            translucent = nodes.new("ShaderNodeBsdfTranslucent")
            translucent.inputs["Color"].default_value = tuple(color)
            mix = nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 0.12
            links.new(shader.outputs["BSDF"], mix.inputs[1])
            links.new(translucent.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
            obj["presentation_finish"] = (
                "estimated leaf surface; no optical calibration"
            )
        else:
            shader.inputs["Metallic"].default_value = 0.78
            shader.inputs["Roughness"].default_value = 0.30
            _variation(material, _noise(material, 170), 0.035, 0.000015)
            obj["presentation_finish"] = "estimated satin metal"
