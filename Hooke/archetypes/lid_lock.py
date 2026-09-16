"""Observe a closed lid from its actual joint and constant joint lock."""
import math

import mujoco


def lid_lock_passes(data, instrument):
    model, lock = instrument.model, instrument.lid_lock
    if model.eq_type[lock] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[lock] != -1:
        raise ValueError('Lid completion requires a constant joint lock')
    position = float(data.qpos[instrument.lid_qposadr])
    target = float(model.eq_data[lock, 0])
    return bool(data.eq_active[lock] and math.isfinite(position) and abs(position-target) < .01)
