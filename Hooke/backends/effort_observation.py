"""Contracts for explicit native actuator inputs, distinct from joint loads.

PhysX projected incoming joint loads include passive forces and constraints.
They cannot identify an individual implicit drive. An explicit input can instead
be checked against the native DOF input buffer and reported as an applied input.
"""
from __future__ import annotations

import numpy as np


def explicit_affine_joint_map(model):
    """Return source joint IDs for supported, independently observable inputs.

    This deliberately excludes tendons, gearing, dynamic actuators and several
    actuators sharing a joint. Joint force limits are unambiguous for the
    one-actuator-per-joint case; callers must report the input after those caps.
    """
    transmissions = np.asarray(model['actuator_trntype'])
    count = len(transmissions)
    if (np.any(transmissions != 0)
            or np.any(np.asarray(model['actuator_dyntype']) != 0)
            or np.any(np.asarray(model['actuator_gaintype']) != 0)
            or np.any(~np.isin(model['actuator_biastype'], (0, 1)))):
        raise NotImplementedError('Explicit effort requires stateless affine joint actuators')
    gear = np.asarray(model['actuator_gear'], dtype=float)
    if gear.shape != (count, 6) or not np.array_equal(gear, np.tile([1., 0., 0., 0., 0., 0.], (count, 1))):
        raise NotImplementedError('Explicit effort requires unit scalar transmission gearing')
    joints = np.asarray(model['actuator_trnid'])[:, 0].astype(np.int64)
    if np.any(joints < 0) or np.any(joints >= len(model['jnt_type'])):
        raise ValueError('Actuator joint IDs are invalid')
    if np.any(~np.isin(np.asarray(model['jnt_type'])[joints], (2, 3))):
        raise NotImplementedError('Explicit effort requires slide or hinge joints')
    if len(np.unique(joints)) != count:
        raise NotImplementedError('Separate actuator effort is ambiguous for shared joint inputs')
    return joints


def verify_native_effort_inputs(submitted, readback):
    """Check the native explicit DOF input buffer against the submitted sum.

    The sum includes actuator and named non-actuator inputs. Matching it proves
    input submission, not a measured motor reaction or its dynamic accuracy.
    Float32 conversion is expected because the installed PhysX tensor API uses
    float32. Non-finite inputs, shape mismatches and excess error fail closed.
    """
    expected = np.asarray(submitted, dtype=np.float64)
    actual = np.asarray(readback, dtype=np.float64)
    if expected.ndim != 1 or actual.shape != expected.shape:
        raise ValueError('Native effort input readback shape differs from submission')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Native effort input readback must be finite')
    if np.any(np.abs(expected) > np.finfo(np.float32).max):
        raise ValueError('Native effort input exceeds the float32 range')
    rounded = expected.astype(np.float32).astype(np.float64)
    error = np.abs(actual-rounded)
    tolerance = 1e-7 + 4*np.finfo(np.float32).eps*np.abs(rounded)
    if np.any(error > tolerance):
        raise RuntimeError('Native explicit effort input readback does not match submission')
    return {'verified': True, 'max_absolute_error': float(np.max(error, initial=0.)),
            'observed_quantity': 'native_explicit_dof_input',
            'readback_api': 'omni.physics.tensors.ArticulationView.get_dof_actuation_forces',
            'includes_non_actuator_inputs': True}


def summarize_applied_efforts(samples, durations):
    """Summarize verified per-actuator inputs over one public interval.

    Supply one row per actual native substep, after source actuator and joint
    caps. A signed maximum-magnitude sample preserves existing absolute-value
    guards; the last input and duration-weighted mean retain separate meanings.
    An absent sample is unavailable, never a manufactured zero observation.
    """
    values = np.asarray(samples, dtype=np.float64)
    intervals = np.asarray(durations, dtype=np.float64)
    if values.ndim != 2 or not len(values) or intervals.shape != (len(values),):
        raise ValueError('Applied effort requires one nonempty sample per native interval')
    if (not np.isfinite(values).all() or not np.isfinite(intervals).all()
            or np.any(intervals <= 0)):
        raise ValueError('Applied effort samples and positive intervals must be finite')
    total = float(intervals.sum())
    if not np.isfinite(total):
        raise ValueError('Applied effort interval sum must be finite')
    peak_indices = np.argmax(np.abs(values), axis=0)
    signed_peak = values[peak_indices, np.arange(values.shape[1])]
    mean = (values*(intervals/total)[:, None]).sum(axis=0)
    if not np.isfinite(mean).all():
        raise ValueError('Applied effort mean must be finite')
    return {'quantity': 'applied_explicit_actuator_input',
            'last': values[-1].tolist(), 'mean': mean.tolist(),
            'signed_peak': signed_peak.tolist(), 'peak_abs': np.abs(signed_peak).tolist(),
            'sample_count': len(values), 'interval_s': total,
            'semantics': 'Source affine actuator input applied to native DOFs, after actuator and joint caps; excludes passive, caller and constraint loads. Not a measured motor reaction.'}
