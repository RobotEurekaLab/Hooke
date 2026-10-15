"""Client-side coupled controllers driven only by observed native state.

MuJoCo evaluates the source mass matrix and geometric equality residuals.
It never integrates a second trajectory or solves a second contact problem.
"""
from __future__ import annotations

import mujoco
import numpy as np

from backends.source_forces import update_kinematics
from real_labs.runtime import InstrumentSession


def mechanism_kinematics(model, data):
    """Refresh mass/bias and equality geometry without contact or acceleration solves."""
    update_kinematics(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeConstraint(model, data)


class NativeMechanismFeedback:
    """Reuse authored mechanism controllers, retaining their guards and limits.

    Scalar targets remain caller-owned. Registered mechanisms own their motor
    channels and must be commanded through ``start`` in their semantic units.
    Every returned control vector must be integrated by the native worker.
    """

    def __init__(self, model, manifest):
        self.session = InstrumentSession(model, manifest)
        self.model, self.data = model, self.session.data
        self.drives = self.session.mechanisms
        self.ready = False

    def _observe(self, state):
        qpos, qvel = np.asarray(state['qpos']), np.asarray(state['qvel'])
        if qpos.shape != self.data.qpos.shape or qvel.shape != self.data.qvel.shape:
            raise ValueError('Native mechanism observation size differs from the source model')
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(state['time']):
            raise ValueError('Native mechanism observation must be finite')
        if self.drives:
            active = np.asarray(state.get('eq_active', []))
            if active.shape != self.data.eq_active.shape:
                raise ValueError('Native mechanism observation must include equality activation')
            if (active.dtype.kind not in 'biuf' or not np.isfinite(active).all()
                    or not np.isin(active, (0, 1)).all()):
                raise ValueError('Native equality activation must be boolean or exact 0/1')
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = state['time']
        if self.drives:
            self.data.eq_active[:] = active
        mechanism_kinematics(self.model, self.data)

    def reset(self, state, controls):
        self.ready = False
        self.data.ctrl[:] = controls
        self._observe(state)
        for drive in self.drives.values():
            drive.reset(self.data)
        self.ready = True

    def start(self, key, state, target, duration):
        if not self.ready:
            raise RuntimeError('Reset native mechanism feedback before commanding it')
        self._observe(state)
        return self.drives[key].start(self.data, target, duration)

    def update(self, state, scalar_controls):
        if not self.ready:
            raise RuntimeError('Reset native mechanism feedback before stepping it')
        requested = np.asarray(scalar_controls, dtype=float)
        if requested.shape != self.data.ctrl.shape or not np.isfinite(requested).all():
            raise ValueError('Native mechanism controls must have the source size and finite values')
        previous = self.data.ctrl.copy()
        self.data.ctrl[:] = requested
        for drive in self.drives.values():
            self.data.ctrl[drive.actuator] = previous[drive.actuator]
        self._observe(state)
        for drive in self.drives.values():
            drive.update(self.data)
        return self.data.ctrl.copy()

    def measured(self, key, state):
        self._observe(state)
        return self.drives[key].measured(self.data)
