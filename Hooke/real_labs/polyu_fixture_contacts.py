"""PolyU finite contact faces and conservative complete-range pair bounds.

Free specimen pairs are never pruned. A distance capped at 2 mm is a lower
bound; it is not a reported exact far-field separation.
"""

import itertools
import mujoco
import numpy as np


def finite_face_gap(model, data, coupon, pad, outward_sign, *, axis=1):
    """Exact axial gap over the finite, axis-aligned retaining face.

    Clip the six oriented coupon faces against the pad's X/Z rectangle. The
    nearest face-normal coordinate of the remaining surface is the physical material
    under that finite face; corners outside the jaw do not inflate the depth.
    This retains native contacts/forces and avoids their half-depth convention.
    """
    if not np.allclose(
        data.geom_xmat[pad].reshape(3, 3), np.eye(3), atol=1e-12, rtol=0
    ):
        raise RuntimeError("retaining_face_frame_changed")
    corners = np.array(list(itertools.product((-1, 1), repeat=3)))
    vertices = (
        data.geom_xpos[coupon]
        + (corners * model.geom_size[coupon]) @ data.geom_xmat[coupon].reshape(3, 3).T
    )
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    center, half = data.geom_xpos[pad], model.geom_size[pad]
    tangent = [i for i in range(3) if i != axis]
    planes = tuple(
        (a, center[a] + side * half[a], -side) for a in tangent for side in (-1, 1)
    )
    retained = []
    for face in faces:
        polygon = [vertices[i] for i in face]
        for clip_axis, bound, sign in planes:
            clipped = []
            for index, first in enumerate(polygon):
                second = polygon[(index + 1) % len(polygon)]
                a = sign * (first[clip_axis] - bound)
                b = sign * (second[clip_axis] - bound)
                inside_a, inside_b = a >= -1e-14, b >= -1e-14
                if inside_a:
                    clipped.append(first)
                if inside_a != inside_b:
                    clipped.append(first + a / (a - b) * (second - first))
            polygon = clipped
            if not polygon:
                break
        retained.extend(polygon)
    if not retained:
        return None
    ys = np.asarray(retained)[:, axis]
    extreme = float(ys.min() if outward_sign < 0 else ys.max())
    return float(outward_sign * (center[axis] - extreme) - half[axis])


def ranges(model, data):
    m, d = model, data
    rotation = d.geom_xmat.reshape(-1, 3, 3)
    center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
    low, high = center - half, center + half
    unbounded = np.zeros(m.ngeom, dtype=bool)
    corners = np.array(list(itertools.product((-1, 1), repeat=3)))
    for g in range(m.ngeom):
        body = int(m.geom_bodyid[g])
        joints = []
        while body:
            joints.extend(
                range(
                    int(m.body_jntadr[body]),
                    int(m.body_jntadr[body] + m.body_jntnum[body]),
                )
            )
            body = int(m.body_parentid[body])
        if any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in joints):
            unbounded[g] = True
            continue
        hinges = [j for j in joints if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if hinges:
            if len(joints) != 1 or not np.allclose(
                abs(d.xaxis[hinges[0]]), [0, 0, 1], atol=1e-12
            ):
                raise ValueError("Unsupported swept hierarchy")
            pivot = d.xanchor[hinges[0]]
            vertices = center[g] + corners * half[g]
            radius = np.linalg.norm(vertices[:, :2] - pivot[:2], axis=1).max()
            low[g, :2] = pivot[:2] - radius
            high[g, :2] = pivot[:2] + radius
        else:
            for j in joints:
                if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE or not m.jnt_limited[j]:
                    raise ValueError("Unsupported joint")
                # Matches actual task's strict one-micrometre numerical limit guard.
                current = d.qpos[m.jnt_qposadr[j]]
                a = d.xaxis[j] * (m.jnt_range[j, 0] - current - 1e-6)
                b = d.xaxis[j] * (m.jnt_range[j, 1] - current + 1e-6)
                low[g] += np.minimum(a, b)
                high[g] += np.maximum(a, b)
    return low, high, unbounded


def retain_pairs(model, data, pairs, cap=0.002):
    low, high, free = ranges(model, data)
    first, second = pairs.T
    separation = np.linalg.norm(
        np.maximum(np.maximum(low[first] - high[second], low[second] - high[first]), 0),
        axis=1,
    )
    keep = (separation <= cap) | free[first] | free[second]
    return pairs[keep].copy(), dict(
        full_pairs=len(pairs),
        retained_pairs=int(keep.sum()),
        excluded_pairs=int((~keep).sum()),
        every_free_coupon_pair_retained=True,
        query_cap_m=cap,
        whole_range_bounds="Complete original/additional scalar ranges plus1um numerical slide guard; full-circle door envelope; free body unbounded",
    )
