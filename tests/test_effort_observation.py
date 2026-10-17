"""Explicit effort provenance must never inherit joint-reaction semantics."""
import unittest
import numpy as np

from backends.effort_observation import (
    explicit_affine_joint_map, summarize_applied_efforts, verify_native_effort_inputs,
)


class EffortObservationTests(unittest.TestCase):
    def model(self):
        return dict(actuator_trntype=np.array([0, 0]), actuator_dyntype=np.array([0, 0]),
                    actuator_gaintype=np.array([0, 0]), actuator_biastype=np.array([0, 1]),
                    actuator_gear=np.tile([1., 0., 0., 0., 0., 0.], (2, 1)),
                    actuator_trnid=np.array([[0, -1], [1, -1]]), jnt_type=np.array([2, 3]))

    def test_scalar_motor_and_position_supported(self):
        np.testing.assert_array_equal(explicit_affine_joint_map(self.model()), [0, 1])

    def test_unsupported_transmissions_fail_closed(self):
        for key, value in [('actuator_trntype', 3), ('actuator_dyntype', 1),
                           ('actuator_gaintype', 1), ('actuator_biastype', 2)]:
            model = self.model(); model[key][0] = value
            with self.subTest(key=key), self.assertRaises(NotImplementedError):
                explicit_affine_joint_map(model)

    def test_shared_joint_and_nonunit_gearing_rejected(self):
        model = self.model(); model['actuator_trnid'][1, 0] = 0
        with self.assertRaises(NotImplementedError): explicit_affine_joint_map(model)
        model = self.model(); model['actuator_gear'][0, 0] = 1.0000001
        with self.assertRaises(NotImplementedError): explicit_affine_joint_map(model)

    def test_ball_joint_and_invalid_index_rejected(self):
        model = self.model(); model['jnt_type'][0] = 1
        with self.assertRaises(NotImplementedError): explicit_affine_joint_map(model)
        model = self.model(); model['actuator_trnid'][0, 0] = -1
        with self.assertRaises(ValueError): explicit_affine_joint_map(model)

    def test_readback_accepts_float32_quantization(self):
        applied = np.array([.1, -123.123456, 0., 1e-10])
        report = verify_native_effort_inputs(applied, applied.astype(np.float32))
        self.assertTrue(report['verified'])
        self.assertEqual(report['max_absolute_error'], 0.)
        self.assertIn('explicit', report['observed_quantity'])

    def test_wrong_or_missing_native_input_rejected(self):
        with self.assertRaises(RuntimeError): verify_native_effort_inputs([10.], [0.])
        with self.assertRaises(RuntimeError): verify_native_effort_inputs([0.], [1e-5])
        with self.assertRaises(ValueError): verify_native_effort_inputs([1.], [])
        with self.assertRaises(ValueError): verify_native_effort_inputs([np.nan], [0.])
        with self.assertRaises(ValueError): verify_native_effort_inputs([0.], [np.inf])
        with self.assertRaises(ValueError): verify_native_effort_inputs([1e100], [1e100])

    def test_peak_preserves_transient_guard_and_sign(self):
        report = summarize_applied_efforts([[12., -3.], [-13., 4.], [1., -8.]], [.1, .2, .1])
        self.assertEqual(report['signed_peak'], [-13., -8.])
        self.assertEqual(report['peak_abs'], [13., 8.])
        self.assertEqual(report['last'], [1., -8.])
        np.testing.assert_allclose(report['mean'], [-3.25, -.75])
        self.assertEqual(report['sample_count'], 3)

    def test_unavailable_sample_is_not_zero(self):
        for samples, intervals in [([], []), ([[1.]], []), ([[1.]], [0.]),
                                    ([[np.nan]], [1.]), ([[1.]], [np.inf])]:
            with self.subTest(samples=samples, intervals=intervals), self.assertRaises(ValueError):
                summarize_applied_efforts(samples, intervals)

    def test_zero_actuator_model_remains_well_formed(self):
        report = summarize_applied_efforts(np.empty((1, 0)), [.001])
        self.assertEqual(report['signed_peak'], [])
        self.assertEqual(report['mean'], [])


if __name__ == '__main__': unittest.main()
