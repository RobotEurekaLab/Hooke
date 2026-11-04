"""Conservative signed box separation for dry inspection geometry queries.

MuJoCo's convex query can report a large negative distance for nearly touching
parallel boxes. The fifteen normalized OBB separating axes provide an independent
bound while the raw query is retained as diagnostic evidence. This does not
replace the simulator's actual contact solver or enabled-contact checks.
"""

import mujoco
import numpy as np


def box_separation_lower_bound(first_center, first_axes, first_half,
                               second_center, second_axes, second_half):
    """Return a padded SAT separation/penetration bound for two oriented boxes."""
    first_axes, second_axes = np.asarray(first_axes), np.asarray(second_axes)
    delta = np.asarray(second_center) - np.asarray(first_center)
    first_half, second_half = np.asarray(first_half), np.asarray(second_half)
    cross = np.cross(first_axes.T[:, None, :], second_axes.T[None, :, :]).reshape(-1, 3)
    candidates = np.concatenate((first_axes.T, second_axes.T, cross))
    norms = np.linalg.norm(candidates, axis=1)
    # Near-parallel cross axes are unstable to normalize. Omitting such an axis
    # only makes the resulting separation bound more conservative.
    axes = candidates[norms > 1e-12] / norms[norms > 1e-12, None]
    radii = np.abs(axes @ first_axes) @ first_half + np.abs(axes @ second_axes) @ second_half
    separation = np.abs(axes @ delta) - radii
    scale = max(1., np.linalg.norm(first_center) + np.linalg.norm(second_center)
                + np.sum(first_half) + np.sum(second_half))
    padding = 64 * np.finfo(float).eps * scale
    return float(np.max(separation) - padding)


def geometry_separation(model, data, first, second, cap):
    """Return conservative distance, raw engine query, and the method used."""
    raw = float(mujoco.mj_geomDistance(model, data, first, second, cap, None))
    if not np.isfinite(raw):
        raise RuntimeError("inspection_nonfinite_geometry_distance")
    if (model.geom_type[first] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_type[second] == mujoco.mjtGeom.mjGEOM_BOX):
        distance = box_separation_lower_bound(
            data.geom_xpos[first], data.geom_xmat[first].reshape(3, 3), model.geom_size[first],
            data.geom_xpos[second], data.geom_xmat[second].reshape(3, 3), model.geom_size[second],
        )
        return min(distance, cap), raw, "padded_obb_sat_lower_bound"
    return min(raw, cap), raw, "mujoco_geometry_distance"
