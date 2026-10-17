"""Conservative world-support partitioning never removes moving source links."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

import numpy as np

from backends.articulation_layout import articulation_layout


def fixed_branches(count):
    return dict(body_parentid=np.array([0, 0] + [1] * count),
                body_weldid=np.array([0, 0] + list(range(2, count + 2))),
                body_jntnum=np.array([0, 0] + [1] * count),
                body_jntadr=np.array([-1, -1] + list(range(count))),
                jnt_type=np.full(count, 3), eq_type=np.array([], dtype=int),
                tendon_adr=np.array([], dtype=int))


class ArticulationPartition(unittest.TestCase):
    def test_large_independent_fixed_branches_split_without_mutating_source(self):
        model = fixed_branches(168)
        before = {key: value.copy() for key, value in model.items()}
        rigid, report = articulation_layout(model, range(1, 170))
        self.assertEqual(rigid, set(range(2, 170)))
        self.assertEqual(report['removed_fixed_rigid_bodies'], [1])
        self.assertEqual(report['original_articulations'][0]['link_count'], 169)
        self.assertEqual(len(report['articulations']), 3)
        self.assertEqual(sorted(item['link_count'] for item in report['articulations']), [43, 64, 64])
        self.assertEqual(sorted(root for group in report['shared_world_anchor_roots'] for root in group), sorted(rigid))
        self.assertTrue(all(item['link_count'] == 2 for item in report['source_branches']))
        for key in model:
            np.testing.assert_array_equal(model[key], before[key])

    def test_small_coupled_tree_is_unchanged(self):
        model = fixed_branches(12)
        model['eq_type'] = np.array([2])
        model['tendon_adr'] = np.array([0])
        rigid, report = articulation_layout(model, range(1, 14))
        self.assertEqual(rigid, set(range(1, 14)))
        self.assertEqual(report['removed_fixed_rigid_bodies'], [])
        self.assertEqual(len(report['articulations']), 1)

    def test_large_coupled_tree_requires_independence_analysis(self):
        for field in ('eq_type', 'tendon_adr'):
            model = fixed_branches(64)
            model[field] = np.array([0])
            with self.subTest(field=field), self.assertRaisesRegex(NotImplementedError, 'equality or tendon'):
                articulation_layout(model, range(1, 66))

    def test_large_connected_moving_tree_is_rejected_even_with_fixed_support(self):
        model = fixed_branches(64)
        model['body_parentid'][2:] = np.arange(1, 65)
        with self.assertRaisesRegex(NotImplementedError, 'Moving branches still exceed'):
            articulation_layout(model, range(1, 66))
        with self.assertRaisesRegex(NotImplementedError, 'moving links cannot be split safely'):
            articulation_layout(model, range(2, 66))

    def test_compound_joints_and_fixed_anchor_count_toward_limit(self):
        model = fixed_branches(31)
        model['body_jntnum'][2:] = 2
        _, report = articulation_layout(model, range(1, 33))
        self.assertEqual(report['articulations'][0]['link_count'], 63)
        model['body_jntnum'][2] = 4
        _, report = articulation_layout(model, range(1, 33))
        self.assertEqual(report['removed_fixed_rigid_bodies'], [1])

    def test_large_pure_static_tree_needs_no_articulation(self):
        model = fixed_branches(200)
        model['body_jntnum'][:] = 0
        rigid, report = articulation_layout(model, range(1, 202))
        self.assertEqual(len(rigid), 201)
        self.assertEqual(report['articulations'], [])
        self.assertEqual(report['removed_fixed_rigid_bodies'], [])

    def test_fixed_intermediate_rigid_links_are_not_assumed_merged(self):
        model = fixed_branches(168)
        model['body_parentid'] = np.array([0, 0] + [parent for i in range(168) for parent in (1, 2 + 2 * i)])
        model['body_weldid'] = np.array([0, 0] + [owner for i in range(168) for owner in (0, 3 + 2 * i)])
        model['body_jntnum'] = np.array([0, 0] + [0, 1] * 168)
        model['body_jntadr'] = np.array([-1, -1] + [joint for i in range(168) for joint in (-1, i)])
        rigid, report = articulation_layout(model, range(1, 338))
        self.assertEqual(report['original_articulations'][0]['link_count'], 337)
        self.assertEqual(len(report['removed_fixed_rigid_bodies']), 169)
        self.assertEqual(len(rigid), 168)
        self.assertEqual(len(report['articulations']), 3)
        self.assertEqual(sorted(item['link_count'] for item in report['articulations']), [43, 64, 64])
        self.assertEqual(sorted(root for group in report['shared_world_anchor_roots'] for root in group), sorted(rigid))
        self.assertTrue(all(item['link_count'] == 2 for item in report['source_branches']))
