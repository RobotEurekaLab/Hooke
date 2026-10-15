"""Coupled native feedback uses observed state, without source integration."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.mechanisms import NativeMechanismFeedback, mechanism_kinematics
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class NativeMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene, cls.manifest = build_scene('epfl_lspn_chemistry')
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))

    def state(self, data):
        return {'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'time': data.time,
                'eq_active': data.eq_active.copy()}

    def test_mass_bias_and_equality_residuals_match_forward_at_observed_state(self):
        expected = InstrumentSession(self.model, self.manifest).data
        expected.qpos[:] += np.linspace(0, 1e-6, self.model.nq)
        expected.qvel[:] = np.linspace(-.001, .001, self.model.nv)
        mujoco.mj_forward(self.model, expected)
        actual = mujoco.MjData(self.model)
        actual.qpos[:] = expected.qpos
        actual.qvel[:] = expected.qvel
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('No source integration')):
            mechanism_kinematics(self.model, actual)
        self.assertEqual(actual.ncon, 0)
        np.testing.assert_allclose(actual.qM, expected.qM, atol=1e-12)
        np.testing.assert_allclose(actual.qfrc_bias, expected.qfrc_bias, atol=1e-12)
        kind = mujoco.mjtConstraint.mjCNSTR_EQUALITY
        np.testing.assert_allclose(actual.efc_pos[actual.efc_type == kind],
                                   expected.efc_pos[expected.efc_type == kind], atol=1e-12)

    def test_initial_holding_torque_matches_authored_controller_without_source_step(self):
        session = InstrumentSession(self.model, self.manifest)
        state = self.state(session.data)
        expected = session.data.ctrl.copy()
        feedback = NativeMechanismFeedback(self.model, self.manifest)
        with patch.object(mujoco, 'mj_step', side_effect=AssertionError('No source integration')):
            feedback.reset(state, expected)
            controls = feedback.update(state, expected)
        np.testing.assert_allclose(controls, expected, atol=1e-12)
        np.testing.assert_array_equal(state['qpos'], session.data.qpos)
        self.assertEqual(feedback.data.time, state['time'])

    def test_guards_reject_nonfinite_state_and_reversed_native_time(self):
        session = InstrumentSession(self.model, self.manifest)
        state = self.state(session.data)
        feedback = NativeMechanismFeedback(self.model, self.manifest)
        feedback.reset(state, session.data.ctrl)
        feedback.update(state, session.data.ctrl)
        with self.assertRaisesRegex(RuntimeError, 'time_reversed'):
            feedback.update(dict(state, time=-.01), session.data.ctrl)
        invalid = dict(state, qpos=np.full(self.model.nq, np.nan))
        with self.assertRaisesRegex(ValueError, 'finite'):
            feedback.update(invalid, session.data.ctrl)

    def test_disabled_native_equality_cannot_use_source_default_activation(self):
        session = InstrumentSession(self.model, self.manifest)
        state = self.state(session.data)
        feedback = NativeMechanismFeedback(self.model, self.manifest)
        feedback.reset(state, session.data.ctrl)
        state['eq_active'][:] = False
        with self.assertRaisesRegex(RuntimeError, 'equality_disabled'):
            feedback.update(state, session.data.ctrl)


    def test_native_equality_accepts_booleans_and_exact_numeric_bits(self):
        session = InstrumentSession(self.model, self.manifest)
        state = self.state(session.data)
        feedback = NativeMechanismFeedback(self.model, self.manifest)
        feedback.reset(state, session.data.ctrl)
        for dtype in (bool, np.uint8, np.int32, np.float64):
            with self.subTest(dtype=dtype):
                active = np.asarray(state['eq_active'], dtype=dtype)
                controls = feedback.update(dict(state, eq_active=active), session.data.ctrl)
                np.testing.assert_array_equal(feedback.data.eq_active, state['eq_active'])
                np.testing.assert_allclose(controls, session.data.ctrl, atol=1e-12)
        feedback.update(dict(state, eq_active=state['eq_active'].tolist()), session.data.ctrl)

    def test_invalid_equality_values_do_not_replace_observed_state(self):
        session = InstrumentSession(self.model, self.manifest)
        state = self.state(session.data)
        feedback = NativeMechanismFeedback(self.model, self.manifest)
        feedback.reset(state, session.data.ctrl)
        for value in (np.nan, np.inf, -np.inf, -.01, .5, 2, 'false', '0', None, 1+0j):
            with self.subTest(value=value):
                invalid = dict(state, qpos=state['qpos']+.01, time=3.,
                               eq_active=np.full(self.model.neq, value))
                with self.assertRaisesRegex(ValueError, 'boolean or exact 0/1'):
                    feedback.update(invalid, session.data.ctrl)
                np.testing.assert_array_equal(feedback.data.qpos, state['qpos'])
                np.testing.assert_array_equal(feedback.data.qvel, state['qvel'])
                np.testing.assert_array_equal(feedback.data.eq_active, state['eq_active'])
                self.assertEqual(feedback.data.time, state['time'])
        with self.assertRaisesRegex(ValueError, 'include equality activation'):
            feedback.update(dict(state, eq_active=[]), session.data.ctrl)

if __name__ == '__main__':
    unittest.main()
