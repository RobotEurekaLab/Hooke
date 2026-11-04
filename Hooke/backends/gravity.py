"""Exact gravity compensation wrenches for welded native rigid bodies."""
from __future__ import annotations

import numpy as np

from backends.mass_properties import rotation


def needs_gravity_wrench(properties):
    """Fully compensated groups use native disabled gravity; mixed groups do not."""
    return not properties['disable_gravity'] and (
        properties['compensated_mass'] != 0.
        or np.any(properties['compensated_first_moment'])
    )


def gravity_wrench(properties, orientation, gravity):
    """Return world force and torque about the source body's origin.

    A welded child's compensation acts at its own center of mass. Averaging
    coefficients alone loses the moment from an asymmetric compensated load.
    The returned wrench must be applied at the body origin, not at its COM.
    """
    if not needs_gravity_wrench(properties):
        return np.zeros(3), np.zeros(3)
    acceleration = -np.asarray(gravity, dtype=float)
    force = properties['compensated_mass'] * acceleration
    first_moment = rotation(orientation) @ properties['compensated_first_moment']
    return force, np.cross(first_moment, acceleration)
