"""Source dimensions and exact filtering survive the native USD conversion."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.collision_filters import collision_filter_plan
from backends.collision_rules import collision_participants
from backends.ellipsoid import ellipsoid_surface

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from backends.usd_scene import SceneBridge
except ImportError:
    Usd = None


class EllipsoidGeometry(unittest.TestCase):
    def test_metric_surface_is_closed_outward_and_within_analytic_support(self):
        directions = np.random.default_rng(91).normal(size=(1000, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for axes in ([.63, .13, .011], [.28, .055, .014], [18e-6, 12e-6, 9e-6]):
            with self.subTest(axes=axes):
                axes = np.asarray(axes)
                points, faces, normals = ellipsoid_surface(axes)
                np.testing.assert_allclose(points.min(axis=0), -axes, rtol=1e-7)
                np.testing.assert_allclose(points.max(axis=0), axes, rtol=1e-7)
                unit = points / axes
                np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1., atol=1e-7)
                triangles = points[faces]
                outward = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                self.assertTrue(np.all(np.sum(outward * triangles.mean(axis=1), axis=1) > 0))
                edges = Counter(tuple(sorted((int(a), int(b)))) for triangle in faces
                                for a, b in zip(triangle, np.roll(triangle, -1)))
                self.assertEqual(set(edges.values()), {2})
                self.assertEqual(len(points) - len(edges) + len(faces), 2)
                analytic_normals = unit / axes
                analytic_normals /= np.linalg.norm(analytic_normals, axis=1)[:, None]
                np.testing.assert_allclose(normals, analytic_normals, atol=1e-7)
                # Validate the display tessellation in normalized coordinates;
                # cooked PhysX hull fidelity needs a separate native check.
                support = np.max(unit @ directions.T, axis=0)
                self.assertTrue(np.all((support <= 1. + 1e-7) & (support >= .99)))

    def test_invalid_dimensions_are_rejected(self):
        for axes in ([1, 0, 1], [1, -1, 1], [1, np.nan, 1], [1, 1], [1, np.inf, 1]):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                ellipsoid_surface(axes)


def model_fixture(seed=0):
    random = np.random.default_rng(seed)
    return dict(
        geom_bodyid=np.repeat(np.arange(7), 3),
        geom_contype=random.integers(0, 4, 21),
        geom_conaffinity=random.integers(0, 4, 21),
        body_weldid=np.array([0, 0, 2, 2, 4, 5, 5]),
        body_parentid=np.array([0, 0, 1, 2, 2, 4, 5]),
        exclude_signature=np.array([(2 << 16) + 5]),
        pair_geom1=np.array([0, 6, 7, 12]),
        pair_geom2=np.array([3, 9, 16, 17]),
    )


def source_forbids(model, first, second, pairs):
    """Scalar source rule, independent of the compact planner's matrices."""
    a, b = int(model['geom_bodyid'][first]), int(model['geom_bodyid'][second])
    if a == b:
        return False  # Same-body filtering is handled by the native actor.
    if tuple(sorted((first, second))) in pairs:
        return False
    wa, wb = model['body_weldid'][a], model['body_weldid'][b]
    pa = model['body_weldid'][model['body_parentid'][wa]]
    pb = model['body_weldid'][model['body_parentid'][wb]]
    if wa == wb or (wa and wb and (wa == pb or wb == pa)):
        return True
    if (min(a, b) << 16) + max(a, b) in model['exclude_signature']:
        return True
    return not ((model['geom_contype'][first] & model['geom_conaffinity'][second])
                or (model['geom_contype'][second] & model['geom_conaffinity'][first]))


class CollisionFiltering(unittest.TestCase):
    def test_mask_weld_exclude_and_explicit_pair_decisions_are_preserved(self):
        for seed in range(30):
            model = model_fixture(seed)
            colliders, pairs = collision_participants(model)
            plan = collision_filter_plan(model, colliders, pairs)
            body_pairs = set(plan.body_pairs)
            geom_pairs = {tuple(sorted(pair)) for pair in plan.geom_pairs}
            for a in colliders:
                for b in colliders:
                    if a >= b:
                        continue
                    bodies = tuple(sorted(map(int, model['geom_bodyid'][[a, b]])))
                    actual = bodies in body_pairs or (a, b) in geom_pairs
                    self.assertEqual(actual, source_forbids(model, a, b, pairs), (seed, a, b))

    def test_large_static_collections_do_not_expand_into_geom_products(self):
        counts = [10345, 399] + [27] * 168
        model = dict(geom_bodyid=np.repeat(np.arange(len(counts)), counts),
                     geom_contype=np.ones(sum(counts), dtype=int),
                     geom_conaffinity=np.ones(sum(counts), dtype=int),
                     body_weldid=np.zeros(len(counts), dtype=int),
                     body_parentid=np.zeros(len(counts), dtype=int), exclude_signature=[])
        plan = collision_filter_plan(model, range(sum(counts)), set())
        self.assertEqual(len(plan.geom_pairs), 0)
        self.assertEqual(len(plan.body_pairs), len(counts) * (len(counts) - 1) // 2)
        self.assertEqual(sum(map(len, plan.body_geoms.values())), sum(counts))


@unittest.skipIf(Usd is None, 'Run with installed Isaac USD bindings')
class USDConversion(unittest.TestCase):
    def test_explicit_adjacent_articulation_pairs_are_rejected_before_physics(self):
        bridge = object.__new__(SceneBridge)
        bridge.rigid = {1, 3}
        bridge.explicit_pairs = {(0, 1)}
        bridge.names = {'geom': ['parent_shape', 'child_shape']}
        bridge.m = dict(geom_bodyid=np.array([1, 3]), body_weldid=np.array([0, 1, 1, 3]),
                        body_parentid=np.array([0, 0, 1, 1]), body_jntnum=np.array([0, 1, 0, 1]))
        with self.assertRaisesRegex(NotImplementedError, 'parent_shape.*child_shape'):
            bridge.validate_explicit_contacts()
        bridge.m['geom_bodyid'][0] = 2  # The source geom is on a welded descendant.
        with self.assertRaisesRegex(NotImplementedError, 'rigid bodies 1/3'):
            bridge.validate_explicit_contacts()
        bridge.m['body_parentid'][3] = 0  # Independent free bodies remain supported.
        bridge.validate_explicit_contacts()
        bridge.m['body_parentid'][3] = 1
        bridge.m['body_jntnum'][3] = 2  # Auxiliary link separates terminal collider.
        bridge.validate_explicit_contacts()
        bridge.explicit_pairs = set()
        bridge.validate_explicit_contacts()

    def test_position_iterations_are_bounded_and_authored_with_default_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / 'model.npz', geom_type=np.array([6]))
            (folder / 'scene.json').write_text(json.dumps({'names': {'body': ['world']}}))
            for invalid in (0, 256, -1, True, False, 32.5, 32., '32', None):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    SceneBridge(Usd.Stage.CreateInMemory(), folder, folder,
                                {'solver_position_iterations': invalid})
            with patch.dict(os.environ, {'HOOKE_ISAAC_POSITION_ITERATIONS': '32'}):
                bridge = SceneBridge(Usd.Stage.CreateInMemory(), folder, folder)
                self.assertEqual(bridge.physics_options['solver_position_iterations'], 32)
            with patch.dict(os.environ, {'HOOKE_ISAAC_POSITION_ITERATIONS': '64'}):
                bridge = SceneBridge(Usd.Stage.CreateInMemory(), folder, folder)
                UsdGeom.Xform.Define(bridge.stage, '/World/b0')
                bridge.articulation(0, '/World/b0')
                api = PhysxSchema.PhysxArticulationAPI(bridge.stage.GetPrimAtPath('/World/b0'))
                self.assertEqual(api.GetSolverPositionIterationCountAttr().Get(), 64)

    def test_free_bodies_author_requested_solver_counts_in_the_exported_stage(self):
        fixture = os.environ.get('HOOKE_TEST_USD_FREE_BODY_SOURCE')
        if not fixture:
            self.skipTest('Set HOOKE_TEST_USD_FREE_BODY_SOURCE to a compiled free-body snapshot')
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ, {
                'HOOKE_ISAAC_POSITION_ITERATIONS': '32', 'HOOKE_ISAAC_VELOCITY_ITERATIONS': '8'}):
            for options, expected in ((None, (32, 8)), ({
                    'solver_position_iterations': 11, 'solver_velocity_iterations': 3}, (11, 3))):
                with self.subTest(options=options):
                    bridge = SceneBridge(Usd.Stage.CreateInMemory(), fixture, output, options)
                    bridge.build()
                    self.assertTrue(bridge.free_bodies, 'The fixture must contain an independent free body')
                    for body in bridge.rigid:
                        api = PhysxSchema.PhysxRigidBodyAPI(
                            bridge.stage.GetPrimAtPath(bridge.body_paths[body]))
                        attributes = (api.GetSolverPositionIterationCountAttr(),
                                      api.GetSolverVelocityIterationCountAttr())
                        self.assertTrue(all(attribute.HasAuthoredValueOpinion() for attribute in attributes))
                        self.assertEqual(tuple(attribute.Get() for attribute in attributes), expected)

    def test_stabilization_defaults_off_and_requires_an_explicit_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / 'model.npz', geom_type=np.array([6]))
            (folder / 'scene.json').write_text(json.dumps({'names': {'body': ['world']}}))
            bridge = SceneBridge(Usd.Stage.CreateInMemory(), folder, folder)
            self.assertFalse(bridge.physics_options['enable_stabilization'])
            enabled = SceneBridge(Usd.Stage.CreateInMemory(), folder, folder,
                                  {'enable_stabilization': True})
            self.assertTrue(enabled.physics_options['enable_stabilization'])
            self.assertTrue(any('rescale effective gravity' in item for item in enabled.limits))
            for invalid in (0, 1, 'false', None, np.bool_(False)):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'boolean'):
                    SceneBridge(Usd.Stage.CreateInMemory(), folder, folder,
                                {'enable_stabilization': invalid})

    def material_bridge(self, textured=False):
        from backends.render_settings import RenderSettings
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        bridge = object.__new__(SceneBridge)
        bridge.stage = Usd.Stage.CreateInMemory()
        bridge.output = Path(temporary.name)
        bridge.material_templates, bridge.written_textures = {}, set()
        bridge.render_settings = RenderSettings.from_environment()
        bridge.physics_options = {'compliant_contact_scale': 1., 'compliant_impedance_fraction': 0.}
        bridge.meta = {'timestep_s': .001}
        bridge.colliders = {0, 1, 2}
        bridge.m = dict(geom_matid=np.array([0, 0, 0]), geom_rgba=np.array([[.5, .5, .5, 1.]] * 3),
                        mat_rgba=np.array([[.3, .5, .7, 1.]]), mat_shininess=np.array([.2]),
                        mat_texid=np.array([[-1, 0 if textured else -1]]), tex_type=np.array([0]),
                        tex_width=np.array([2]), tex_height=np.array([2]), tex_nchannel=np.array([3]),
                        tex_adr=np.array([0]), tex_data=np.arange(12, dtype=np.uint8),
                        geom_friction=np.array([[.5, 0, 0], [.5, 0, 0], [.7, 0, 0]]),
                        geom_contype=np.ones(3, dtype=int), geom_conaffinity=np.ones(3, dtype=int),
                        geom_solref=np.array([[.02, 1.]] * 3),
                        geom_solimp=np.array([[.95, .95, .001, .5, 2.]] * 3))
        for i in range(3):
            bridge.material(i, UsdGeom.Cube.Define(bridge.stage, f'/World/g{i}').GetPrim())
        return bridge

    def test_shared_constant_materials_preserve_distinct_physical_parameters(self):
        bridge = self.material_bridge()
        materials = [UsdShade.MaterialBindingAPI(bridge.stage.GetPrimAtPath(f'/World/g{i}'))
                     .ComputeBoundMaterial('physics')[0] for i in range(3)]
        self.assertEqual(materials[0].GetPath(), materials[1].GetPath())
        self.assertNotEqual(materials[0].GetPath(), materials[2].GetPath())
        for material, expected in zip(materials, (.5, .5, .7)):
            self.assertAlmostEqual(UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get(), expected)
        bridge.m['geom_solref'][1] = [.04, 1.]
        bridge.material(1, bridge.stage.GetPrimAtPath('/World/g1'))
        changed = UsdShade.MaterialBindingAPI(bridge.stage.GetPrimAtPath('/World/g1')).ComputeBoundMaterial('physics')[0]
        self.assertNotEqual(changed.GetPath(), materials[0].GetPath())
        old_stiffness = PhysxSchema.PhysxMaterialAPI(materials[0].GetPrim()).GetCompliantContactStiffnessAttr().Get()
        new_stiffness = PhysxSchema.PhysxMaterialAPI(changed.GetPrim()).GetCompliantContactStiffnessAttr().Get()
        self.assertAlmostEqual(old_stiffness / new_stiffness, 4.)

    def test_textured_alias_paths_keep_independent_runtime_overrides(self):
        bridge = self.material_bridge(textured=True)
        shaders = [UsdShade.Shader.Get(bridge.stage, f'/World/Materials/g{i}/Texture') for i in range(2)]
        self.assertTrue(all(shaders))
        previous = shaders[1].GetInput('file').Get()
        shaders[0].GetInput('file').Set('runtime-override.png')
        self.assertEqual(shaders[1].GetInput('file').Get(), previous)
        self.assertNotEqual(shaders[0].GetInput('file').Get(), previous)

    def test_ellipsoid_has_baked_dimensions_and_convex_collision_without_scale(self):
        bridge = object.__new__(SceneBridge)
        bridge.stage = Usd.Stage.CreateInMemory()
        bridge.m = dict(geom_type=np.array([4]), geom_size=np.array([[.63, .13, .011]]),
                        geom_bodyid=np.array([0]), geom_pos=np.array([[.1, .2, .3]]),
                        geom_quat=np.array([[1., 0., 0., 0.]]), geom_group=np.array([0]),
                        geom_margin=np.array([0.]))
        bridge.body_paths, bridge.geom_paths = {0: '/World/b0'}, {}
        bridge.collider_paths = {}
        bridge.colliders, bridge.sdf_contacts = {0}, set()
        bridge.names = {'geom': ['wing']}
        bridge.physics_options = {'contact_offset_m': .001}
        bridge.material = lambda index, prim: 1.
        bridge.geometry(0)
        prim = bridge.stage.GetPrimAtPath(bridge.geom_paths[0])
        self.assertTrue(prim.IsA(UsdGeom.Mesh))
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), 'convexHull')
        hull = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        self.assertEqual(hull.GetHullVertexLimitAttr().Get(), 128)
        self.assertEqual(hull.GetMinThicknessAttr().Get(), 0.)
        self.assertFalse(prim.GetAttribute('xformOp:scale').HasAuthoredValueOpinion())
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        np.testing.assert_allclose(np.ptp(points, axis=0), 2 * bridge.m['geom_size'][0], rtol=1e-7)

    def test_groups_include_only_exact_colliders_not_nested_bodies(self):
        bridge = object.__new__(SceneBridge)
        bridge.stage = Usd.Stage.CreateInMemory()
        bridge.m = model_fixture()
        bridge.colliders, bridge.explicit_pairs = collision_participants(bridge.m)
        bridge.geom_paths = {i: f'/World/b{bridge.m["geom_bodyid"][i]}/g{i}' for i in bridge.colliders}
        bridge.collider_paths = {geom: [path] for geom, path in bridge.geom_paths.items()}
        for path in bridge.geom_paths.values():
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(bridge.stage, path).GetPrim())
        bridge.filters()
        for prim in bridge.stage.Traverse():
            if prim.IsA(UsdPhysics.CollisionGroup):
                group = UsdPhysics.CollisionGroup(prim)
                collection = group.GetCollidersCollectionAPI()
                self.assertEqual(collection.GetExpansionRuleAttr().Get(), 'explicitOnly')
                targets = collection.GetIncludesRel().GetTargets()
                self.assertTrue(targets)
                self.assertTrue(all(bridge.stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in targets))
        table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(bridge.stage)
        forbidden = set(collision_filter_plan(bridge.m, bridge.colliders, bridge.explicit_pairs).body_pairs)
        groups = table.GetGroups()
        for first in groups:
            for second in groups:
                a, b = int(first.name[1:]), int(second.name[1:])
                self.assertEqual(table.IsCollisionEnabled(first, second),
                                 tuple(sorted((a, b))) not in forbidden)


if __name__ == '__main__':
    unittest.main()
