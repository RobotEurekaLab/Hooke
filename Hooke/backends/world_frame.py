"""Translate native solver coordinates while keeping the source world unchanged."""

import numpy as np


class WorldFrame:
    def __init__(self, origin=(0., 0., 0.)):
        self.origin = np.asarray(origin, dtype=float)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("world_origin_m must contain three finite coordinates")

    def to_native(self, position):
        return np.asarray(position, dtype=float) - self.origin

    def to_source(self, position):
        return np.asarray(position, dtype=float) + self.origin
