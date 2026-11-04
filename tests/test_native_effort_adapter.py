"""Native input availability must replace, and never reuse, source forces."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from backends.closed_loop import PhysXTaskAdapter
from backends.effort_observation import summarize_applied_efforts


class NativeEffortAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><joint name="j" type="slide"/><geom size=".1" mass="1"/></body>
          </worldbody><actuator><motor joint="j"/></actuator></mujoco>''')
        self.data = mujoco.MjData(self.model)
        self.data.ctrl[:] = 7.
        mujoco.mj_forward(self.model, self.data)
        self.session = SimpleNamespace(model=self.model, data=self.data,
                                       physics_observations={'actuator_force': True})
        self.adapter = PhysXTaskAdapter.__new__(PhysXTaskAdapter)
        self.adapter.task = SimpleNamespace(model=self.model, data=self.data)
        self.adapter.observation_target = self.session
        self.adapter.max_fk_position_error = self.adapter.max_fk_rotation_error = 0.
        self.state = dict(qpos=[0.], qvel=[0.], time=.002, contacts=[])
        self.effort = dict(available=True, quantity='applied_explicit_actuator_input',
                           native_input_verified=True, last=[-2.], mean=[1.],
                           peak_abs=[5.], signed_peak=[5.], sample_count=4, interval_s=.002)

    def test_unavailable_native_effort_invalidates_old_source_force(self):
        self.assertEqual(self.data.actuator_force[0], 7.)
        self.adapter.apply_state(self.state)
        self.assertTrue(np.isnan(self.data.actuator_force).all())
        self.assertIs(self.session.physics_observations['actuator_force'], False)

    def test_preserves_last_and_substep_peak_with_distinct_meaning(self):
        self.adapter.apply_state(dict(self.state, actuator_effort=self.effort))
        np.testing.assert_array_equal(self.data.actuator_force, [-2.])
        self.assertEqual(self.session.physics_observations['actuator_force_peak_abs'], [5.])
        with self.adapter:
            self.data.actuator_force[:] = 999.
            mujoco.mj_forward(self.model, self.data)
        np.testing.assert_array_equal(self.data.actuator_force, [-2.])

    def test_missing_or_malformed_effort_fails_before_state_update(self):
        self.adapter.apply_state(dict(self.state, actuator_effort=self.effort))
        candidates = [None, dict(available=False)]
        for key, value in [('last', [float('nan')]), ('peak_abs', [1.]),
                           ('signed_peak', [-3.]), ('mean', []), ('native_input_verified', False),
                           ('mean', [100.]), ('sample_count', 0), ('sample_count', True),
                           ('interval_s', 0.), ('interval_s', .1)]:
            candidate = deepcopy(self.effort)
            candidate[key] = value
            candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                state = dict(self.state, time=.004)
                if candidate is not None:
                    state['actuator_effort'] = candidate
                with self.assertRaises(RuntimeError):
                    self.adapter.apply_state(state)
                self.assertEqual(self.data.time, .002)
                np.testing.assert_array_equal(self.data.actuator_force, [-2.])

    def test_zero_effort_interval_only_at_explicit_reset(self):
        effort = dict(self.effort, last=[0.], mean=[0.], peak_abs=[0.], signed_peak=[0.],
                      sample_count=0, interval_s=0.)
        self.adapter.apply_state(dict(self.state, time=0., actuator_effort=effort))
        self.assertTrue(self.session.physics_observations['actuator_force'])
        with self.assertRaises(RuntimeError):
            self.adapter.apply_state(dict(self.state, actuator_effort=effort))
        self.assertEqual(self.data.time, 0.)

    def test_sample_count_matches_declared_native_substeps(self):
        self.adapter.loaded = {'conversion': {'integration': {'native_substeps': 2}}}
        with self.assertRaises(RuntimeError):
            self.adapter.apply_state(dict(self.state, actuator_effort=self.effort))
        self.assertEqual(self.data.time, 0.)

    def test_weighted_mean_roundoff_is_not_a_force_violation(self):
        for count in range(1, 33):
            for value in (.1, 1e-20, 12.):
                with self.subTest(count=count, value=value):
                    effort = dict(self.effort, **summarize_applied_efforts(
                        [[value]] * count, [.002 / count] * count))
                    self.adapter.apply_state(dict(self.state, actuator_effort=effort))
                    self.assertTrue(self.session.physics_observations['actuator_force'])


if __name__ == '__main__':
    unittest.main()
