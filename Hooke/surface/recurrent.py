"""Pure NumPy implementation of the pinned Unitree G1 recurrent actor."""

from pathlib import Path
import hashlib

import numpy as np
from scipy.special import expit


class RecurrentPolicy:
    """The pinned Unitree one-layer LSTM and ELU actor, without runtime Torch."""

    def __init__(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(
                "Prepare the G1 gait with python -m surface.gait_assets"
            )
        with np.load(path, allow_pickle=False) as weights:
            self.weights = {
                key: weights[key].astype(np.float32) for key in weights.files
            }
        expected = {
            "memory.weight_ih_l0": (256, 47),
            "memory.weight_hh_l0": (256, 64),
            "memory.bias_ih_l0": (256,),
            "memory.bias_hh_l0": (256,),
            "actor.0.weight": (32, 64),
            "actor.0.bias": (32,),
            "actor.2.weight": (12, 32),
            "actor.2.bias": (12,),
        }
        if set(expected) != set(self.weights) or any(
            self.weights[k].shape != shape or not np.isfinite(self.weights[k]).all()
            for k, shape in expected.items()
        ):
            raise ValueError("G1 policy arrays do not match the declared architecture")
        self.h = np.zeros(64, np.float32)
        self.c = self.h.copy()
        digest = hashlib.sha256()
        for key in sorted(self.weights):
            digest.update(key.encode())
            digest.update(self.weights[key].astype("<f4").tobytes())
        self.sha256 = digest.hexdigest()

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (47,) or not np.isfinite(observation).all():
            raise ValueError("G1 actor requires 47 finite observation values")
        w = self.weights
        gates = (
            w["memory.weight_ih_l0"] @ observation
            + w["memory.weight_hh_l0"] @ self.h
            + w["memory.bias_ih_l0"]
            + w["memory.bias_hh_l0"]
        )
        i, f, g, o = np.split(gates, 4)
        self.c = expit(f) * self.c + expit(i) * np.tanh(g)
        self.h = expit(o) * np.tanh(self.c)
        hidden = w["actor.0.weight"] @ self.h + w["actor.0.bias"]
        hidden = np.where(hidden > 0, hidden, np.expm1(np.minimum(hidden, 0)))
        return w["actor.2.weight"] @ hidden + w["actor.2.bias"]
