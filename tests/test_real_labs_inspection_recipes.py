"""Measured dry recipes keep original contacts, limits and failure evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from real_labs.inspection import _sequence, recipe_for, run_inspection
from real_labs.inspection_contacts import InspectionAccess
from real_labs.motion import JointMotion, _model_signature
from test_real_labs_motion import fixture


class DryInspectionTests(unittest.TestCase):
    def setUp(self):
        self.session = fixture()
        self.session.manifest['id'] = 'fixture'
        self.recipe = dict(scene='fixture', version=1, limits=['Dry inspection only'], devices=[
            dict(equipment='cart', controls=['x', 'y'], clearance_m=.015, speed_scale=.5,
                 actions=[dict(name='approach', targets={'x': .006}),
                          dict(name='align', targets={'y': .004})])
        ])

    def test_explicit_recipe_reverses_the_access_sequence(self):
        sequence = _sequence(self.recipe['devices'][0], np.array([.001, -.002]))
        self.assertEqual([n for n, _ in sequence], ['approach', 'align', 'return_approach', 'return_measured_home'])
        np.testing.assert_array_equal(sequence[0][1], [.006, -.002])
        np.testing.assert_array_equal(sequence[1][1], [.006, .004])
        np.testing.assert_array_equal(sequence[-1][1], [.001, -.002])

    def test_registry_copy_cannot_change_the_authored_targets(self):
        first = recipe_for('auckland_aerodynamics_tunnel')
        saved = deepcopy(first)
        first['devices'][0]['actions'][0]['targets'].clear()
        self.assertEqual(recipe_for('auckland_aerodynamics_tunnel'), saved)
        self.assertIsNone(recipe_for('unknown'))

    def test_actual_motion_returns_with_every_step_checked(self):
        before = _model_signature(self.session.model)
        with patch('real_labs.inspection.recipe_for', return_value=self.recipe), \
             patch('real_labs.inspection.require_binding'), \
             patch('real_labs.rigid_access.require_binding', return_value={}), \
             patch('real_labs.inspection_contacts.registered_interfaces', return_value={'pairs': []}):
            result = run_inspection(self.session)
        self.assertTrue(result['success'], result['failure'])
        self.assertEqual(len(result['phases']), 4)
        self.assertGreater(result['executed_steps'], 0)
        self.assertEqual(result['executed_steps'], result['checked_steps'])
        self.assertTrue(result['returns'][0]['registered'])
        self.assertEqual(_model_signature(self.session.model), before)
        for phase in result['phases']:
            self.assertFalse(phase['report']['reset_performed'])
            self.assertIn('tcp', phase['observation']['sites'])

    def test_observer_queries_up_to_the_requested_clearance(self):
        with patch('real_labs.rigid_access.require_binding', return_value={}), \
             patch('real_labs.inspection_contacts.registered_interfaces', return_value={'pairs': []}):
            access = InspectionAccess(self.session, 'fixture', 'cart', ['x', 'y'], minimum_m=.02)
        self.assertEqual(access.cap_m, .02)
        self.session.data.qpos[access.motion.qadr[0]] = .435
        import mujoco
        mujoco.mj_forward(self.session.model, self.session.data)
        with self.assertRaisesRegex(RuntimeError, 'inspection_clearance_lost'):
            access.check(self.session.model, self.session.data)

    def test_observer_failure_cannot_count_as_completed_recipe(self):
        with patch('real_labs.inspection.recipe_for', return_value=self.recipe), \
             patch('real_labs.inspection.require_binding'), \
             patch('real_labs.rigid_access.require_binding', return_value={}), \
             patch('real_labs.inspection_contacts.registered_interfaces', return_value={'pairs': []}), \
             patch.object(InspectionAccess, 'check', side_effect=RuntimeError('injected_clearance_failure')):
            result = run_inspection(self.session)
        self.assertFalse(result['success'])
        self.assertFalse(result['complete_declared_recipe'])
        self.assertFalse(result['coverage_complete'])
        self.assertIn('injected_clearance_failure', result['failure']['reason'])

    def test_slowing_motion_preserves_original_tracking_and_contact_limits(self):
        reports = []
        for scale in (1., .5):
            session = fixture()
            motion = JointMotion(session, 'cart', ['x', 'y'])
            report = motion.execute(motion.plan_joint_target([.006, 0]), speed_scale=scale)
            self.assertEqual(report['status'], 'succeeded', report['failure'])
            reports.append(report)
        self.assertGreater(reports[1]['physics_steps'], reports[0]['physics_steps'])
        for field in ('tracking_limits', 'contact_penetration_threshold_m', 'clearance_planning_m'):
            self.assertEqual(reports[0][field], reports[1][field])

    def test_planner_projects_only_measured_machine_roundoff(self):
        from real_labs.motion import PlanningError
        motion = JointMotion(self.session, 'cart', ['x', 'y'])
        address = motion.qadr[0]
        self.session.data.qpos[address] = -5.639875177500858e-24
        plan = motion.plan_joint_target([.006, 0])
        self.assertEqual(plan.waypoints[0][0], 0)
        self.assertLess(self.session.data.qpos[address], 0)
        with self.assertRaisesRegex(ValueError, 'Configuration exceeds'):
            motion.plan_joint_target([-1e-24, 0])
        self.session.data.qpos[address] = -1e-8
        with self.assertRaisesRegex(PlanningError, 'Measured start exceeds'):
            motion.plan_joint_target([.006, 0])
        self.assertEqual(self.session.data.qpos[address], -1e-8)
        self.assertEqual(self.session.data.time, 0)

    def test_invalid_speed_is_rejected_before_live_state_mutation(self):
        motion = JointMotion(self.session, 'cart', ['x', 'y'])
        plan = motion.plan_joint_target([.006, 0])
        before = [getattr(self.session.data, name).copy() for name in ('qpos', 'qvel', 'ctrl')]
        for scale in (0, -1, 1.01, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'Speed scale'):
                motion.execute(plan, speed_scale=scale)
            for name, expected in zip(('qpos', 'qvel', 'ctrl'), before):
                np.testing.assert_array_equal(getattr(self.session.data, name), expected)
            self.assertEqual(self.session.data.time, 0)


if __name__ == '__main__':
    unittest.main()
