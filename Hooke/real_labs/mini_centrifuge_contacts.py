"""Direct contact checks for the miniature instrument and its free inert gauge.

Fixed-axis envelopes remove only pairs whose complete authored motion cannot
approach. Free/foreign moving bodies keep unbounded envelopes. Parent-body
collision filtering is never used to omit a potentially interacting pair.
"""

import itertools
import math

import mujoco
import numpy as np


_CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)))


def _swept_bounds(points, anchor, axis, low, high):
    relative = points - anchor
    along = (relative @ axis)[:, None] * axis
    cosine = relative - along
    sine = np.cross(axis, relative)
    center = anchor + along
    lower = np.minimum(
        center + cosine * math.cos(low) + sine * math.sin(low),
        center + cosine * math.cos(high) + sine * math.sin(high),
    )
    upper = np.maximum(
        center + cosine * math.cos(low) + sine * math.sin(low),
        center + cosine * math.cos(high) + sine * math.sin(high),
    )
    critical = np.arctan2(sine, cosine)
    for turn in range(-4, 5):
        angle = critical + turn * math.pi
        value = center + cosine * np.cos(angle) + sine * np.sin(angle)
        inside = (angle >= low) & (angle <= high)
        lower = np.minimum(lower, np.where(inside, value, np.inf))
        upper = np.maximum(upper, np.where(inside, value, -np.inf))
    return lower.min(axis=0), upper.max(axis=0)


class ContactCheck:
    """Conservative screening followed by enabled, direct geometry queries."""

    cap = 0.0003
    # A strictly separated enclosing box proves no overlap. This is only a
    # numerical broad-phase pad, independent of the physical contact limits.
    screen_padding = 1e-7
    instrument_overlap = 1e-7
    gauge_overlap = 0.0002
    limit_padding = 0.005

    def __init__(self, model, data, interface, fixture):
        self.model = model
        self.gauge = model.body(fixture["gauge_body"]).id
        self.fixture = model.body(fixture["fixture_body"]).id
        self.rotor = model.body(interface["rotor_body"]).id
        self.lid = model.body(interface["lid_body"]).id
        groups = model.body_weldid[[self.lid, self.rotor, self.gauge]]
        self.moving = np.isin(model.body_weldid[model.geom_bodyid], groups)
        explicit = np.r_[model.pair_geom1, model.pair_geom2]
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        enabled[explicit] = True
        ids = np.flatnonzero(enabled)
        a, b = np.triu_indices(len(ids), 1)
        a, b = ids[a], ids[b]
        selected = (self.moving[a] | self.moving[b]) & (
            model.body_weldid[model.geom_bodyid[a]]
            != model.body_weldid[model.geom_bodyid[b]]
        )
        a, b = a[selected], b[selected]
        self.unculled_pairs = np.column_stack((a, b))
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum(
            "nij,nj->ni", rotation, model.geom_aabb[:, :3]
        )
        half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
        lower = np.full_like(center, -np.inf)
        upper = np.full_like(center, np.inf)
        fixed = model.body_weldid[model.geom_bodyid] == 0
        lower[fixed], upper[fixed] = (
            center[fixed] - half[fixed],
            center[fixed] + half[fixed],
        )
        for body in (self.lid, self.rotor):
            if model.body_jntnum[body] != 1:
                raise ValueError(
                    "Inspection envelope requires a single fixed-axis hinge"
                )
            joint = int(model.body_jntadr[body])
            parent = int(model.body_parentid[body])
            while parent:
                if model.body_jntnum[parent]:
                    raise ValueError("Inspection hinge anchor must be fixed")
                parent = int(model.body_parentid[parent])
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Inspection envelope requires hinge geometry")
            if body == self.rotor:
                low, high = -math.pi, math.pi
            else:
                angle = float(data.qpos[model.jnt_qposadr[joint]])
                low, high = (
                    model.jnt_range[joint]
                    - angle
                    + np.array([-1, 1]) * self.limit_padding
                )
            for geom in np.flatnonzero(
                model.body_weldid[model.geom_bodyid] == model.body_weldid[body]
            ):
                local = model.geom_aabb[geom, :3] + _CORNERS * model.geom_aabb[geom, 3:]
                points = local @ rotation[geom].T + data.geom_xpos[geom]
                lower[geom], upper[geom] = _swept_bounds(
                    points, data.xanchor[joint], data.xaxis[joint], low, high
                )
        self.envelope_lower, self.envelope_upper = lower, upper
        distance = np.linalg.norm(
            np.maximum(np.maximum(lower[a] - upper[b], lower[b] - upper[a]), 0), axis=1
        )
        keep = distance <= self.screen_padding
        self.pairs = self.unculled_pairs[keep]
        self.excluded_pairs = self.unculled_pairs[~keep]
        self.queries = self.fallbacks = 0
        self.maximum_penetration = 0.0
        self._mesh_normals = {}

    def _support_bound(self, data, cylinder, mesh):
        """An exact support projection bounds separating translation, not distance."""
        m = self.model
        if (
            m.geom_type[cylinder] != mujoco.mjtGeom.mjGEOM_CYLINDER
            or m.geom_type[mesh] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            return None
        asset = int(m.geom_dataid[mesh])
        local = m.mesh_vert[
            m.mesh_vertadr[asset] : m.mesh_vertadr[asset] + m.mesh_vertnum[asset]
        ]
        if asset not in self._mesh_normals:
            faces = m.mesh_face[
                m.mesh_faceadr[asset] : m.mesh_faceadr[asset] + m.mesh_facenum[asset]
            ]
            triangle = local[faces]
            normals = np.cross(
                triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
            ).astype(float)
            length = np.linalg.norm(normals, axis=1)
            normals = normals[length > 1e-15] / length[length > 1e-15, None]
            self._mesh_normals[asset] = normals
        rotation = data.geom_xmat[mesh].reshape(3, 3)
        vertices = local @ rotation.T + data.geom_xpos[mesh]
        axis = data.geom_xmat[cylinder].reshape(3, 3)[:, 2]
        normals = np.vstack((self._mesh_normals[asset] @ rotation.T, axis))
        projection = vertices @ normals.T
        center = data.geom_xpos[cylinder] @ normals.T
        axial = normals @ axis
        radius, half = m.geom_size[cylinder, :2]
        support = radius * np.sqrt(np.maximum(0, 1 - axial**2)) + half * abs(axial)
        return float(
            np.max(
                np.maximum(
                    projection.min(0) - center - support,
                    center - support - projection.max(0),
                )
            )
        )

    def _limit(self, a, b):
        first, second = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
        if self.gauge != first and self.gauge != second:
            return self.instrument_overlap
        other = second if first == self.gauge else first
        return (
            self.gauge_overlap
            if other in (self.rotor, self.fixture)
            else self.instrument_overlap
        )

    def check(self, data):
        m = self.model
        if data.model is not m:
            raise RuntimeError("contact_model_state_binding_changed")
        normal_load = 0.0
        gauge_contacts = 0
        force = np.empty(6)
        for index, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if not (self.moving[a] or self.moving[b]):
                continue
            depth = max(0.0, -float(contact.dist))
            self.maximum_penetration = max(self.maximum_penetration, depth)
            if depth > self._limit(a, b):
                raise RuntimeError(
                    f"native_contact_limit:{m.geom(a).name}:{m.geom(b).name}"
                )
            if self.gauge in m.geom_bodyid[[a, b]]:
                mujoco.mj_contactForce(m, data, index, force)
                normal_load += float(force[0])
                gauge_contacts += 1
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        candidates = self.pairs
        # Filter each axis successively to avoid building a large pair-by-axis
        # temporary array. This is the same enclosing-box overlap proof.
        for coordinate in range(3):
            a, b = candidates.T
            candidates = candidates[
                abs(center[a, coordinate] - center[b, coordinate])
                <= half[a, coordinate] + half[b, coordinate] + self.screen_padding
            ]
        for a, b in candidates:
            a, b = int(a), int(b)
            gap = float(mujoco.mj_geomDistance(m, data, a, b, self.cap, None))
            self.queries += 1
            limit = self._limit(a, b)
            if gap < -limit and self.gauge in m.geom_bodyid[[a, b]]:
                cylinder, mesh = (a, b) if m.geom_bodyid[a] == self.gauge else (b, a)
                bound = self._support_bound(data, cylinder, mesh)
                if bound is not None and bound >= -limit:
                    gap = bound
                    self.fallbacks += 1
            self.maximum_penetration = max(self.maximum_penetration, max(0.0, -gap))
            if gap < -limit:
                raise RuntimeError(
                    f"direct_contact_limit:{m.geom(a).name}:{m.geom(b).name}"
                )
        return dict(normal_load_N=normal_load, gauge_contact_count=gauge_contacts)
