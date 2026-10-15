"""Observe explicitly applied actuator inputs at the native stepping boundary."""

import numpy as np

from backends.effort_observation import (
    explicit_affine_joint_map,
    summarize_applied_efforts,
    verify_native_effort_inputs,
)


class AppliedEffortRecorder:
    """Keep actuator inputs separate from passive forces and constraint loads.

    Enabled scenes use one stateless affine actuator per scalar joint. Native
    DOF input readback verifies the sum actually submitted to PhysX; the source
    actuator contribution is recorded separately after all source force caps.
    """

    def __init__(self, model, enabled):
        self.enabled = enabled
        self.count = len(model['reset_ctrl'])
        self.dofs = (np.asarray(model['jnt_dofadr'])[explicit_affine_joint_map(model)]
                     if enabled else np.array([], dtype=int))
        self.state = {'available': False, 'reason': 'Implicit drive effort is not independently observable'}
        self.begin_interval()

    def invalidate(self, reason):
        self.state = {'available': False, 'reason': str(reason)}

    def begin_interval(self):
        if self.enabled:
            self.invalidate('Native effort interval has not been fully verified')
        self.samples = []
        self.durations = []
        self.maximum_readback_error = 0.0

    def _verify(self, submitted, views):
        for item in views:
            proof = verify_native_effort_inputs(
                submitted[item['va']], item['view'].get_dof_actuation_forces()[0])
            self.maximum_readback_error = max(self.maximum_readback_error, proof['max_absolute_error'])

    def reset(self, views, dof_count):
        self.begin_interval()
        if not self.enabled:
            self.invalidate('Implicit drive effort is not independently observable')
            return
        self._verify(np.zeros(dof_count), views)
        zeros = [0.0] * self.count
        self.state = dict(self._provenance(), last=zeros, mean=zeros.copy(),
                          peak_abs=zeros.copy(), signed_peak=zeros.copy(),
                          sample_count=0, interval_s=0.0,
                          semantics='Explicit input buffer cleared and verified at reset; no physics interval elapsed.')

    def record(self, actuator_forces, submitted, views, duration):
        if not self.enabled:
            return
        self._verify(submitted, views)
        self.samples.append(np.asarray(actuator_forces)[self.dofs].copy())
        self.durations.append(duration)

    def _provenance(self):
        return {'available': True, 'quantity': 'applied_explicit_actuator_input',
                'native_input_verified': True,
                'max_native_input_readback_error': self.maximum_readback_error,
                'readback_api': 'omni.physics.tensors.ArticulationView.get_dof_actuation_forces',
                'excluded_contributions': ['caller generalized forces', 'passive spring and damping',
                                           'dry friction', 'gravity', 'constraints and contact loads'],
                'measured_motor_reaction': False}

    def finish_interval(self):
        if self.enabled:
            self.state = {**summarize_applied_efforts(self.samples, self.durations),
                          **self._provenance()}
