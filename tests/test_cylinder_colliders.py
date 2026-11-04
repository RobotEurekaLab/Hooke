"""CPU geometry, cooked-bound witnesses, and actual USD child-shape contracts."""
from collections import Counter
import math
import json
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.sector_cylinder import cylinder_sectors, radial_deficit, sector_count, uses_box_fallback, RADIAL_BUDGET_M
from backends.box_cylinder import MIN_BOXES, MAX_BOXES, box_count, cylinder_boxes, radial_excess

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from backends.usd_scene import SceneBridge
except ImportError:
    Usd = None


class SectorGeometry(unittest.TestCase):
    def test_count_is_minimal_across_rounding_boundaries_and_cap(self):
        for count in (3, 4, 25, 75, 132, 180, 511):
            boundary = RADIAL_BUDGET_M / (2 * math.sin(math.pi / (2*count))**2)
            for factor in (1-1e-10, 1., 1+1e-10):
                radius = boundary * factor
                measured = sector_count(radius)
                self.assertIn(measured, (count, count+1))
                self.assertLessEqual(radial_deficit(radius), RADIAL_BUDGET_M)
                if measured > 3:
                    self.assertGreater(radius*2*math.sin(math.pi/(2*(measured-1)))**2, RADIAL_BUDGET_M)
        self.assertEqual(sector_count(.14), 132)
        self.assertEqual(sector_count(.26), 180)
        with self.assertRaisesRegex(ValueError, 'more than 512'):
            sector_count(15.)

    def test_closed_outward_six_corner_prisms_tile_disjoint_interiors(self):
        for radius, half in ((.14, .054), (.26, .42), (5e-6, 2e-6), (.003, 1.3)):
            sectors = cylinder_sectors(radius, half)
            total_volume = 0.
            for index, (points, faces) in enumerate(sectors):
                self.assertEqual(points.shape, (6, 3))
                self.assertEqual(len(np.unique(points, axis=0)), 6)
                edges = Counter(tuple(sorted(edge)) for f in faces for edge in zip(f, np.roll(f, -1)))
                self.assertTrue(all(count == 2 for count in edges.values()))
                triangles = points[faces]
                normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                self.assertTrue(np.all(np.einsum('ij,ij->i', normals, triangles.mean(axis=1)-points.mean(axis=0)) > 0))
                total_volume += np.einsum('ij,ij->', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6
                np.testing.assert_array_equal(points[[2, 5]], sectors[(index+1) % len(sectors)][0][[1, 4]])
                self.assertEqual(float(points[:, 2].min()), -half)
                self.assertEqual(float(points[:, 2].max()), half)
                self.assertTrue(np.all(np.linalg.norm(points[:, :2], axis=1) <= radius*(1+1e-14)))
            ideal_volume = len(sectors)*radius**2*math.sin(2*math.pi/len(sectors))*half
            self.assertAlmostEqual(total_volume/ideal_volume, 1., places=12)
            # Equal signed angular wedges with shared boundaries tile one ring.
            centroids = np.array([points[:3, :2].mean(axis=0) for points, _ in sectors])
            angles = np.mod(np.arctan2(centroids[:, 1], centroids[:, 0]), 2*math.pi)
            np.testing.assert_allclose(angles, (np.arange(len(sectors))+.5)*2*math.pi/len(sectors), atol=1e-13)

    def test_invalid_dimensions(self):
        for radius, half in ((0, 1), (1, 0), (-1, 1), (np.nan, 1), (1, np.inf)):
            with self.assertRaises(ValueError):
                cylinder_sectors(radius, half)



@unittest.skipIf(Usd is None, 'Run with installed Isaac USD bindings')
class SectorUSD(unittest.TestCase):
    def bridge(self, kinds, colliders, body_ids=None, sdf=()):
        count = len(kinds)
        bridge = object.__new__(SceneBridge)
        bridge.stage = Usd.Stage.CreateInMemory()
        bridge.m = dict(geom_type=np.array(kinds), geom_size=np.tile([.14, .054, 0.], (count, 1)),
                        geom_bodyid=np.array(body_ids or [0]*count), geom_pos=np.zeros((count, 3)),
                        geom_quat=np.tile([1., 0., 0., 0.], (count, 1)), geom_group=np.zeros(count),
                        geom_margin=np.zeros(count))
        bridge.body_paths = {i: f'/World/b{i}' for i in range(count+1)}
        bridge.geom_paths, bridge.collider_paths = {}, {}
        bridge.colliders, bridge.sdf_contacts = set(colliders), set(sdf)
        bridge.names = {'geom': [f'source{i}' for i in range(count)]}
        bridge.physics_options = {'contact_offset_m': .001, 'sdf_contact_offset_m': .00005,
                                  'compound_cylinder_colliders': True}
        material = UsdShade.Material.Define(bridge.stage, '/World/Materials/shared')
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(.61)
        def bind(index, prim):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
            return 1.
        bridge.material = bind
        for geom in range(count):
            bridge.geometry(geom)
        return bridge

    def test_hidden_convex_children_keep_identity_material_source_visual_and_offsets(self):
        bridge = self.bridge([5, 5], [0], sdf=(0,))
        parent = bridge.stage.GetPrimAtPath(bridge.geom_paths[0])
        self.assertTrue(parent.IsA(UsdGeom.Cylinder))
        self.assertFalse(parent.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(parent.GetAttribute('hooke:cylinderSectorCount').Get(), 132)
        self.assertNotIn(1, bridge.collider_paths)
        for path in bridge.collider_paths[0]:
            prim = bridge.stage.GetPrimAtPath(path)
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(prim.GetAttribute('hooke:sourceId').Get(), 0)
            self.assertEqual(len(UsdGeom.Mesh(prim).GetPointsAttr().Get()), 6)
            self.assertEqual(UsdGeom.Imageable(prim).ComputeVisibility(), 'invisible')
            self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), 'convexHull')
            self.assertEqual(PhysxSchema.PhysxConvexHullCollisionAPI(prim).GetMinThicknessAttr().Get(), 0.)
            self.assertAlmostEqual(PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(), .00005, places=9)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            self.assertEqual(str(material.GetPath()), '/World/Materials/shared')

    def test_partial_filters_expand_exactly_to_all_child_paths(self):
        bridge = self.bridge([5, 5, 5, 5], [0, 1, 2, 3], [1, 1, 2, 3])
        bridge.m.update(body_weldid=np.array([0, 1, 2, 3]), body_parentid=np.zeros(4, dtype=int),
                        geom_contype=np.array([1, 2, 1, 4]), geom_conaffinity=np.array([1, 2, 1, 4]),
                        exclude_signature=np.array([], dtype=int))
        bridge.explicit_pairs = {(1, 3)}
        bridge.filters()
        for source, target in ((1, 2), (0, 3)):
            for path in bridge.collider_paths[source]:
                prim = bridge.stage.GetPrimAtPath(path)
                actual = set(map(str, UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()))
                self.assertEqual(actual, set(bridge.collider_paths[target]))


    def test_thin_rail_uses_native_boxes_and_metadata(self):
        from pxr import UsdGeom, UsdPhysics
        bridge = self.bridge([5], [0])
        # Rebuild the source geom under a new path after selecting the thin rail.
        bridge.stage.RemovePrim(bridge.geom_paths[0])
        bridge.m['geom_size'][0] = [.00155, .04526, 0.]
        bridge.geometry(0)
        self.assertEqual(len(bridge.collider_paths[0]), box_count(.00155))
        self.assertEqual(bridge.cylinder_collision_profile[0]['strategy'], 'native_boxes')
        parent = bridge.stage.GetPrimAtPath(bridge.geom_paths[0])
        self.assertTrue(parent.IsA(UsdGeom.Cylinder))
        self.assertFalse(parent.HasAPI(UsdPhysics.CollisionAPI))
        for path in bridge.collider_paths[0]:
            prim = bridge.stage.GetPrimAtPath(path)
            self.assertTrue(prim.IsA(UsdGeom.Cube))
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
            self.assertEqual(prim.GetAttribute('hooke:sourceId').Get(), 0)


class BoxCylinderGeometry(unittest.TestCase):
    def test_minimum_count_budget_and_rounding_boundaries(self):
        import math
        for target in (3, 4, 7, 66, 90, 128, 255):
            angle = math.pi / (2 * target)
            radius = RADIAL_BUDGET_M / (2 * math.sin(angle / 2)**2 / math.cos(angle))
            for factor in (1-1e-10, 1., 1+1e-10):
                current = radius * factor
                count = box_count(current)
                self.assertIn(count, (target, target+1))
                self.assertLessEqual(radial_excess(current), RADIAL_BUDGET_M)
                if count > MIN_BOXES:
                    previous_angle = math.pi / (2 * (count-1))
                    previous_error = current * (2 * math.sin(previous_angle/2)**2 / math.cos(previous_angle))
                    self.assertGreater(previous_error, RADIAL_BUDGET_M)
        self.assertEqual(box_count(1e-12), MIN_BOXES)
        self.assertEqual(box_count(.14), 66)
        self.assertEqual(box_count(.26), 90)
        self.assertLessEqual(box_count(2.), MAX_BOXES)
        with self.assertRaisesRegex(ValueError, 'more than 256'):
            box_count(15.)

    def test_disk_coverage_and_outward_bound_at_axes_corners_and_random_rays(self):
        for radius, half_length in ((.14, .054), (.26, .42), (5e-6, 2e-6)):
            extents, angles = cylinder_boxes(radius, half_length)
            self.assertEqual(len(angles), box_count(radius))
            self.assertEqual(extents[2], half_length)
            directions = np.r_[angles, angles + np.pi/(2*len(angles)),
                               np.random.default_rng(23).uniform(-np.pi, np.pi, 4000)]
            cosines = np.abs(np.cos(directions[:, None]-angles))
            sines = np.abs(np.sin(directions[:, None]-angles))
            # A radius-r circle point belongs to at least one native box.
            inside = (radius*cosines <= extents[0]+radius*1e-14) & (radius*sines <= extents[1]+radius*1e-14)
            self.assertTrue(np.all(inside.any(axis=1)))
            # Each box's furthest corner bounds the entire union in every direction.
            corner_radius = np.linalg.norm(extents[:2])
            self.assertAlmostEqual(corner_radius-radius, radial_excess(radius), places=14)
            support = (extents[0]*cosines + extents[1]*sines).max(axis=1)
            self.assertTrue(np.all(support >= radius-radius*1e-14))
            self.assertTrue(np.all(support <= corner_radius+radius*1e-14))
            self.assertLessEqual(radial_excess(radius), RADIAL_BUDGET_M)

    def test_invalid_dimensions_are_rejected(self):
        for radius, half in ((0, 1), (1, 0), (-1, 1), (np.nan, 1), (1, np.inf)):
            with self.assertRaises(ValueError):
                cylinder_boxes(radius, half)


class HybridGeometry(unittest.TestCase):
    def test_geometric_fallback_boundaries(self):
        self.assertTrue(uses_box_fallback(.00155, .04526))
        self.assertTrue(uses_box_fallback(.002, .032))
        self.assertFalse(uses_box_fallback(.002000001, .0321))
        self.assertFalse(uses_box_fallback(.002, .031999999))
        self.assertFalse(uses_box_fallback(.003, 1.3))
        self.assertFalse(uses_box_fallback(.14, .054))


@unittest.skipIf(Usd is None, 'Run with installed Isaac USD bindings')
class CylinderOptions(unittest.TestCase):
    def test_profile_is_opt_in_and_strictly_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'model.npz', geom_type=np.array([5]))
            (root / 'scene.json').write_text(json.dumps({'names': {'body': ['world']}}))
            self.assertFalse(SceneBridge(Usd.Stage.CreateInMemory(), root, root).physics_options['compound_cylinder_colliders'])
            self.assertTrue(SceneBridge(Usd.Stage.CreateInMemory(), root, root,
                {'compound_cylinder_colliders': True}).physics_options['compound_cylinder_colliders'])
            for value in (0, 1, None, 'true', np.bool_(True)):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                    SceneBridge(Usd.Stage.CreateInMemory(), root, root, {'compound_cylinder_colliders': value})
            with self.assertRaisesRegex(ValueError, 'Unknown physics options'):
                SceneBridge(Usd.Stage.CreateInMemory(), root, root, {'convex_cylinder_colliders': True})

if __name__ == '__main__':
    unittest.main()
