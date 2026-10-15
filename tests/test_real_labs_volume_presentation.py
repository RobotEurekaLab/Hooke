"""Pure boundary checks and actual Blender contract tests (no rendering)."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.volume_presentation import (
    absorption_entries,
    absorption_node_values,
    apply_presentation_volumes,
    closed_convex_volume,
)

try:
    import bpy
except ImportError:
    bpy = None

VERTICES = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
]
FACES = [
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]


def entry(name="water", sigma=(0.34, 0.065, 0.01)):
    return dict(
        object=name,
        kind="estimated_rgb_absorption",
        units="m^-1",
        absorption_rgb=list(sigma),
        dimension_basis="Estimated test coefficients",
        scientific_optics_calibrated=False,
    )


def manifest(*rows):
    return dict(units="m", presentation_volumes=list(rows))


class VolumeContractTests(unittest.TestCase):
    def test_default_is_noop_without_importing_blender(self):
        self.assertEqual(apply_presentation_volumes({}), [])
        self.assertEqual(apply_presentation_volumes(manifest()), [])

    def test_metadata_rejects_ambiguous_or_nonfinite_contracts(self):
        for bad in (None, {}, "water"):
            with self.assertRaises(ValueError):
                absorption_entries(dict(presentation_volumes=bad))
        for key, value in (
            ("units", "cm"),
            ("kind", "glass"),
            ("object", ""),
            ("dimension_basis", ""),
            ("scientific_optics_calibrated", True),
            ("absorption_rgb", [1, 2]),
            ("absorption_rgb", [True, 0, 0]),
            ("absorption_rgb", [float("nan"), 0, 0]),
            ("absorption_rgb", [float("inf"), 0, 0]),
            ("absorption_rgb", [-1, 0, 0]),
        ):
            row = entry()
            row[key] = value
            with self.assertRaises(ValueError):
                absorption_entries(manifest(row))
        with self.assertRaises(ValueError):
            absorption_entries(manifest(entry(), entry()))
        row = entry()
        row["unvalidated"] = True
        with self.assertRaises(ValueError):
            absorption_entries(manifest(row))

    def test_node_coefficients_cover_zero_and_above_one(self):
        for sigma in ((0, 0, 0), (0.34, 0.065, 0.01), (2.7, 0.3, 0)):
            color, density = absorption_node_values(sigma)
            for requested, channel in zip(sigma, color):
                self.assertAlmostEqual((1 - channel) * density, requested)
                self.assertTrue(0 <= channel <= 1)

    def test_unrepresentable_positive_coefficients_reject(self):
        for sigma in ((1e100, 1, 0), (1e-100, 1e-100, 0), (1, 1e-12, 0)):
            with self.assertRaises(ValueError):
                absorption_entries(manifest(entry(sigma=sigma)))

    def test_convex_closed_volume_in_world_metres(self):
        self.assertAlmostEqual(
            closed_convex_volume(VERTICES, FACES)["world_volume_m3"], 8
        )
        translated = [(100 + x * 2, -50 + y * 3, 10 + z * 4) for x, y, z in VERTICES]
        self.assertAlmostEqual(
            closed_convex_volume(translated, FACES)["world_volume_m3"], 192
        )

    def test_open_inward_disconnected_nonconvex_boundaries_reject(self):
        for faces in (
            FACES[:-1],
            [tuple(reversed(f)) for f in FACES],
            FACES + [FACES[0]],
        ):
            with self.assertRaises(ValueError):
                closed_convex_volume(VERTICES, faces)
        with self.assertRaises(ValueError):
            closed_convex_volume(
                VERTICES * 2, FACES + [tuple(i + 8 for i in f) for f in FACES]
            )
        nonconvex = list(VERTICES)
        nonconvex[6] = (0, 0, 0)
        with self.assertRaises(ValueError):
            closed_convex_volume(nonconvex, FACES)
        with self.assertRaises(ValueError):
            closed_convex_volume(VERTICES + [(9, 9, 9)], FACES)


def graph(material):
    return {
        "nodes": [
            (
                n.name,
                n.type,
                [
                    (s.identifier, str(s.default_value))
                    for s in n.inputs
                    if hasattr(s, "default_value")
                ],
            )
            for n in material.node_tree.nodes
        ],
        "links": [
            (
                link.from_node.name,
                link.from_socket.identifier,
                link.to_node.name,
                link.to_socket.identifier,
            )
            for link in material.node_tree.links
        ],
    }


def material_state():
    return {
        "materials": {m.name: graph(m) for m in bpy.data.materials if m.use_nodes},
        "slots": {
            o.name: [
                (slot.link, slot.material.name if slot.material else None)
                for slot in o.material_slots
            ]
            for o in bpy.context.scene.objects
            if o.type == "MESH"
        },
    }


@unittest.skipIf(bpy is None, "Run with Blender Python for actual material tests")
class BlenderVolumeTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def cube(self, name="water", material=None):
        mesh = bpy.data.meshes.new(name + " mesh")
        mesh.from_pydata(VERTICES, [], FACES)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        if material is None:
            material = bpy.data.materials.new("test__mat_clear_water")
            material.use_nodes = True
        mesh.materials.append(material)
        return obj

    def test_absent_empty_and_zero_are_actual_graph_noops(self):
        water = self.cube()
        for name in (
            "tank__mat_glass",
            "tank__mat_shell",
            "tank__mat_metal",
            "tank__mat_unknown",
        ):
            material = bpy.data.materials.new(name)
            material.use_nodes = True
        before = material_state()
        for config in ({}, manifest(), manifest(entry(sigma=(0, 0, 0)))):
            apply_presentation_volumes(config)
            self.assertEqual(material_state(), before)
        self.assertEqual(water.material_slots[0].link, "DATA")

    def test_shared_data_material_is_isolated_and_surface_unchanged(self):
        water = self.cube()
        neighbor = bpy.data.objects.new("neighbor", water.data)
        bpy.context.scene.collection.objects.link(neighbor)
        original = water.active_material
        before = graph(original)
        surface = [
            (s.identifier, str(s.default_value))
            for s in original.node_tree.nodes.get("Principled BSDF").inputs
        ]
        result = apply_presentation_volumes(manifest(entry()))
        self.assertEqual(graph(original), before)
        self.assertIs(neighbor.active_material, original)
        self.assertIsNot(water.active_material, original)
        self.assertEqual(water.material_slots[0].link, "OBJECT")
        self.assertEqual(
            surface,
            [
                (s.identifier, str(s.default_value))
                for s in water.active_material.node_tree.nodes.get(
                    "Principled BSDF"
                ).inputs
            ],
        )
        node = next(
            n
            for n in water.active_material.node_tree.nodes
            if n.type == "VOLUME_ABSORPTION"
        )
        for expected, color in zip(
            (0.34, 0.065, 0.01), node.inputs["Color"].default_value
        ):
            self.assertAlmostEqual(
                (1 - color) * node.inputs["Density"].default_value, expected, places=7
            )
        self.assertEqual(result[0]["world_volume_m3"], 8)

    def test_all_targets_validate_before_any_material_mutation(self):
        self.cube()
        before = material_state()
        with self.assertRaises(ValueError):
            apply_presentation_volumes(manifest(entry(), entry("missing")))
        self.assertEqual(material_state(), before)

    def test_huge_tiny_and_cancelled_float_sockets_do_not_mutate(self):
        self.cube()
        self.cube("second")
        for sigma in ((1e100, 1, 0), (1e-100, 1e-100, 0), (1, 1e-12, 0)):
            before = material_state()
            with self.assertRaises(ValueError):
                apply_presentation_volumes(
                    manifest(entry(), entry("second", sigma=sigma))
                )
            self.assertEqual(material_state(), before)

    def test_bad_geometry_roles_units_and_slots_reject_before_mutation(self):
        for problem in (
            "open",
            "inward",
            "mirror",
            "zero_scale",
            "modifier",
            "slots",
            "role",
            "units",
            "unit_scale",
            "already_volume",
        ):
            with self.subTest(problem=problem):
                self.setUp()
                obj = self.cube()
                config = manifest(entry())
                if problem in ("open", "inward"):
                    obj.data.clear_geometry()
                    obj.data.from_pydata(
                        VERTICES,
                        [],
                        (
                            FACES[:-1]
                            if problem == "open"
                            else [tuple(reversed(f)) for f in FACES]
                        ),
                    )
                elif problem == "mirror":
                    obj.scale.x = -1
                elif problem == "zero_scale":
                    obj.scale.x = 0
                elif problem == "modifier":
                    obj.modifiers.new("unsupported shell", "SOLIDIFY")
                elif problem == "slots":
                    obj.data.materials.append(obj.active_material)
                elif problem == "role":
                    obj.active_material.name = "looks_blue_but_no_role"
                elif problem == "units":
                    config["units"] = "cm"
                elif problem == "unit_scale":
                    bpy.context.scene.unit_settings.scale_length = 0.01
                elif problem == "already_volume":
                    material = obj.active_material
                    n = material.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
                    material.node_tree.links.new(
                        n.outputs["Volume"],
                        material.node_tree.nodes.get("Material Output").inputs[
                            "Volume"
                        ],
                    )
                bpy.context.view_layer.update()
                before = material_state()
                with self.assertRaises(ValueError):
                    apply_presentation_volumes(config)
                self.assertEqual(material_state(), before)

    def test_shape_keys_reject_all_targets_before_any_material_mutation(self):
        for strength in (0.0, 1.0):
            with self.subTest(shape_key_value=strength):
                self.setUp()
                first = self.cube("first")
                target = self.cube("water", material=first.active_material)
                target.shape_key_add(name="Basis")
                key = target.shape_key_add(name="Nonconvex rendered boundary")
                key.data[6].co = (0, 0, 0)
                key.value = strength
                bpy.context.view_layer.update()
                if strength:
                    evaluated = target.evaluated_get(
                        bpy.context.evaluated_depsgraph_get()
                    )
                    mesh = evaluated.to_mesh()
                    try:
                        with self.assertRaises(ValueError):
                            closed_convex_volume(
                                [tuple(target.matrix_world @ v.co) for v in mesh.vertices],
                                [tuple(p.vertices) for p in mesh.polygons],
                            )
                    finally:
                        evaluated.to_mesh_clear()
                before = material_state()
                with self.assertRaisesRegex(ValueError, "shape keys"):
                    apply_presentation_volumes(
                        manifest(entry("first"), entry("water"))
                    )
                self.assertEqual(material_state(), before)

    def test_positive_nonuniform_transform_and_large_coefficients(self):
        obj = self.cube()
        obj.scale = (2, 3, 4)
        obj.location = (100, -50, 10)
        bpy.context.view_layer.update()
        result = apply_presentation_volumes(manifest(entry(sigma=(2.7, 0.3, 0))))
        self.assertAlmostEqual(result[0]["world_volume_m3"], 192)
        node = next(
            n
            for n in obj.active_material.node_tree.nodes
            if n.type == "VOLUME_ABSORPTION"
        )
        self.assertAlmostEqual(node.inputs["Density"].default_value, 2.7, places=6)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
