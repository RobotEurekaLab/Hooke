"""Conservative whole-range bounds for the supported scalar-joint SEM inspection scene.

Boxes are propagated through every ancestor. Hinge extrema use the complete
angular interval, including derivative roots, rather than a pose sample.
"""

import itertools
import math
import mujoco
import numpy as np


def corners(low, high):
    return np.array(list(itertools.product(*zip(low, high))))


def rotated_interval(low, high, axis, pivot, lower, upper):
    vertices = corners(low, high) - pivot
    projection = np.outer(vertices @ axis, axis)
    cosine = vertices - projection
    sine = np.cross(axis, vertices)
    constant = projection + pivot
    values = []
    for a, b, c in zip(cosine, sine, constant):
        out_low, out_high = [], []
        for aa, bb, cc in zip(a, b, c):
            angles = [lower, upper]
            stationary = math.atan2(bb, aa)
            first = math.ceil((lower - stationary) / math.pi)
            last = math.floor((upper - stationary) / math.pi)
            angles.extend(stationary + n * math.pi for n in range(first, last + 1))
            outcomes = [aa * math.cos(q) + bb * math.sin(q) + cc for q in angles]
            out_low.append(min(outcomes))
            out_high.append(max(outcomes))
        values.append((out_low, out_high))
    return np.min([v[0] for v in values], axis=0), np.max(
        [v[1] for v in values], axis=0
    )


def transform_box(low, high, quat, position):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    matrix = matrix.reshape(3, 3)
    center = matrix @ ((low + high) / 2) + position
    half = abs(matrix) @ ((high - low) / 2)
    return center - half, center + half


def whole_range_boxes(model, domain_padding=0.0):
    lows, highs = [], []
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            lows.append(np.full(3, -np.inf))
            highs.append(np.full(3, np.inf))
            continue
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        low, high = transform_box(
            center - half, center + half, model.geom_quat[geom], model.geom_pos[geom]
        )
        body = int(model.geom_bodyid[geom])
        while body:
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError("Only one scalar joint per body is supported")
            if count:
                joint = int(model.body_jntadr[body])
                kind = model.jnt_type[joint]
                if kind not in (
                    mujoco.mjtJoint.mjJNT_SLIDE,
                    mujoco.mjtJoint.mjJNT_HINGE,
                ):
                    raise ValueError(
                        "Unbounded/free joints are outside this candidate contract"
                    )
                if not model.jnt_limited[joint]:
                    raise ValueError("Explicit finite candidate ranges are required")
                lower, upper = (
                    model.jnt_range[joint] - model.qpos0[model.jnt_qposadr[joint]]
                )
                lower -= domain_padding
                upper += domain_padding
                axis = model.jnt_axis[joint]
                if kind == mujoco.mjtJoint.mjJNT_SLIDE:
                    offsets = np.array([axis * lower, axis * upper])
                    low, high = low + offsets.min(0), high + offsets.max(0)
                else:
                    low, high = rotated_interval(
                        low, high, axis, model.jnt_pos[joint], lower, upper
                    )
            low, high = transform_box(
                low, high, model.body_quat[body], model.body_pos[body]
            )
            body = int(model.body_parentid[body])
        lows.append(low - 1e-9)
        highs.append(high + 1e-9)
    return np.array(lows), np.array(highs)


def candidate_pairs(model, cap=0.002, domain_padding=0.0):
    ids = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    weld = model.body_weldid[model.geom_bodyid]
    pairs = np.array(
        [(a, b) for a, b in itertools.combinations(ids, 2) if weld[a] != weld[b]],
        dtype=int,
    )
    low, high = whole_range_boxes(model, domain_padding)
    a, b = pairs.T
    separation = np.maximum(np.maximum(low[a] - high[b], low[b] - high[a]), 0)
    keep = np.linalg.norm(separation, axis=1) < cap
    return pairs[keep], pairs[~keep], (low, high)
