"""Bounded full-array mechanical inspection, without a wave-generation model.

All 168 targets are radians. Real joint integration is observed at every step;
no live pose assignment, reset, fluid response or force-feedback calibration is
provided. A conservative full-angle enclosure only removes distant collision
pairs; all nearby enabled cross-weld pairs, including parent-filtered pairs,
are queried directly. Excluded pairs have full-domain separation bounds. Retained near pairs are
checked at every executed state and at explicit diagnostic samples; those
samples are not a proof of all 168-dimensional configurations.
"""

import hashlib
import itertools
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.flowave_geometry import COUNT, LIMIT, PREFIX, key
from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.rigid_access import require_binding

SCENE = "edinburgh_flowave_basin"
EQUIPMENT = "wave_basin"
STATE_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "act",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
)


def full_angle_bounds(model):
    """Exact coordinate extrema of each local AABB corner under one hinge.

    Unknown or nested moving assemblies get infinite bounds, never exclusion.
    The 1 microradian domain pad contains numerical endpoint residue; leaving
    that padded domain is rejected before any retained-pair observation.
    """
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    matrices = d.geom_xmat.reshape(-1, 3, 3)
    center = d.geom_xpos + np.einsum("nij,nj->ni", matrices, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(matrices), model.geom_aabb[:, 3:])
    lower, upper = center - half, center + half
    corners = np.array(list(itertools.product((-1, 1), repeat=3)))
    for g in range(model.ngeom):
        if model.body_weldid[model.geom_bodyid[g]] == 0:
            continue
        body, joints = int(model.geom_bodyid[g]), []
        while body:
            joints.extend(
                range(
                    model.body_jntadr[body],
                    model.body_jntadr[body] + model.body_jntnum[body],
                )
            )
            body = int(model.body_parentid[body])
        if len(joints) != 1 or model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_HINGE:
            lower[g], upper[g] = -np.inf, np.inf
            continue
        j = joints[0]
        low, high = model.jnt_range[j] + np.array([-1e-6, 1e-6])
        local = model.geom_aabb[g, :3] + corners * model.geom_aabb[g, 3:]
        points = d.geom_xpos[g] + local @ matrices[g].T
        axis, anchor = d.xaxis[j], d.xanchor[j]
        v = points - anchor
        constant = anchor + np.outer(v @ axis, axis)
        cosine = v - np.outer(v @ axis, axis)
        sine = np.cross(axis, v)
        lo, hi = [], []
        for n in range(8):
            for coordinate in range(3):
                a, b, c = (
                    cosine[n, coordinate],
                    sine[n, coordinate],
                    constant[n, coordinate],
                )
                candidates = [low, high]
                stationary = math.atan2(b, a)
                candidates.extend(
                    stationary + k * math.pi
                    for k in range(-2, 3)
                    if low <= stationary + k * math.pi <= high
                )
                values = [c + a * math.cos(t) + b * math.sin(t) for t in candidates]
                lo.append(min(values))
                hi.append(max(values))
        lower[g] = np.asarray(lo).reshape(8, 3).min(axis=0)
        upper[g] = np.asarray(hi).reshape(8, 3).max(axis=0)
    return lower - 1e-7, upper + 1e-7


class FlapInspection:
    """Exclusive-session array positioning with no public arbitrary callback."""

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session, self.m, self.d = session, session.model, session.data
        self.signature = _model_signature(self.m)
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.controls = [session.controls[EQUIPMENT, key(i)] for i in range(COUNT)]
        self.actuators = np.array([c.actuator for c in self.controls])
        self.addresses = np.array([self.m.jnt_qposadr[c.joint] for c in self.controls])
        self.velocities = np.array([self.m.jnt_dofadr[c.joint] for c in self.controls])
        self.masks = self.m.geom_contype.copy(), self.m.geom_conaffinity.copy()
        self.flags = int(self.m.opt.disableflags), int(self.m.opt.enableflags)
        self.cap = 0.002
        self.required = 0.00025
        self._build_pairs()
        self.reset_counts()
        self.preflight()

    def _build_pairs(self):
        m = self.m
        self.weld = m.body_weldid[m.geom_bodyid]
        enabled = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        moving = enabled[self.weld[enabled] != 0]
        self.lower, self.upper = full_angle_bounds(m)
        pairs, count = [], 0
        # Stream rows instead of materializing tens of millions of distant pairs.
        for a in moving:
            others = enabled[
                (self.weld[enabled] != self.weld[a])
                & ((self.weld[enabled] == 0) | (enabled > a))
            ]
            count += len(others)
            gap = np.linalg.norm(
                np.maximum(
                    np.maximum(
                        self.lower[a] - self.upper[others],
                        self.lower[others] - self.upper[a],
                    ),
                    0,
                ),
                axis=1,
            )
            pairs.extend((int(a), int(b)) for b in others[gap <= self.cap])
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.all_pair_count = count

    def reset_counts(self):
        self.executed = self.entries = self.checked = self.queries = 0
        self.minimum = self.cap
        self.maximum_torque = 0.0

    def state(self):
        return dict(
            time_s=float(self.d.time),
            model_signature=self.signature,
            **{field: getattr(self.d, field).tolist() for field in STATE_FIELDS},
        )

    def preflight(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Flap model/state ownership changed")
        require_binding(self.session, SCENE)
        if _model_signature(self.m) != self.signature:
            raise ValueError("Flap geometry, drives or physics changed")
        if not math.isfinite(float(self.d.time)) or any(
            not np.isfinite(getattr(self.d, field)).all() for field in STATE_FIELDS
        ):
            raise ValueError("Finite state and time are required")
        if np.any(_warning_state(self.d)[:, 0]):
            raise ValueError("Existing native warnings must be inspected first")
        if (
            np.max(abs(self.d.qpos[self.addresses])) > LIMIT + 1e-6
            or np.max(abs(self.d.ctrl[self.actuators])) > LIMIT
        ):
            raise ValueError("Measured/commanded start is outside the authored range")

    def direct_clearance(self, data):
        if data.model is not self.m:
            raise ValueError("Foreign diagnostic data")
        if np.max(abs(data.qpos[self.addresses])) > LIMIT + 1e-6:
            raise RuntimeError("Angle exceeds the conservative pruning domain")
        m = self.m
        matrices = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum("nij,nj->ni", matrices, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(matrices), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap
        )
        minimum, queries = self.cap, 0
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, data, int(a), int(b), self.cap, None)),
                self.cap,
            )
            minimum = min(minimum, gap)
            queries += 1
            if gap < self.required:
                raise RuntimeError(
                    f"Flap clearance lost: {m.geom(a).name} / {m.geom(b).name}: {gap:.9g} m"
                )
        return minimum, queries

    def check(self):
        self.entries += 1
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("Flap model/state ownership changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("Flap contact/physics configuration changed")
        if (
            not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or not math.isfinite(float(d.time))
        ):
            raise RuntimeError("Nonfinite flap state")
        if np.max(abs(d.qvel[self.velocities])) > 0.5:
            raise RuntimeError(
                "Flap angular speed exceeds the authored inspection limit"
            )
        for contact in d.contact:
            if contact.dist < -1e-7:
                raise RuntimeError(
                    "Unexpected native penetration during dry inspection"
                )
        minimum, queries = self.direct_clearance(d)
        self.minimum = min(self.minimum, minimum)
        self.queries += queries
        self.maximum_torque = max(
            self.maximum_torque, float(np.max(abs(d.actuator_force[self.actuators])))
        )
        self.checked += 1

    def observe(self):
        self.preflight()
        return dict(
            angles_rad=self.d.qpos[self.addresses].tolist(),
            top_edges_m=[
                self.d.site_xpos[
                    self.m.site(f"{EQUIPMENT}__site_flap_{i:03d}_top").id
                ].tolist()
                for i in range(COUNT)
            ],
            scope="Actual hinge positions and top-edge geometry; no wave-height or flow observation",
            **self.state(),
        )

    def move(self, targets, *, ramp_s=2.0, settle_s=1.0):
        self.preflight()
        keys = {key(i) for i in range(COUNT)}
        if not isinstance(targets, dict) or set(targets) != keys:
            raise ValueError(
                "A complete version-2 vector of all 168 radian keys is required; legacy keys are rejected"
            )
        target = np.array([targets[key(i)] for i in range(COUNT)], dtype=float)
        if not np.isfinite(target).all() or np.max(abs(target)) > LIMIT:
            raise ValueError("Every requested angle must lie in ±0.12 rad")
        if not (
            math.isfinite(ramp_s)
            and math.isfinite(settle_s)
            and 0.5 <= ramp_s <= 30
            and 0 <= settle_s <= 30
        ):
            raise ValueError("Finite ramp 0.5–30 s and settling 0–30 s required")
        initial = self.state()
        beginning = self.d.ctrl[self.actuators].copy()
        self.reset_counts()
        reason, events = None, []
        warning = _warning_state(self.d)
        steps = math.ceil((ramp_s + settle_s) / self.m.opt.timestep)
        try:
            self.check()
            self.reset_counts()
            for step in range(steps):
                f = min(1.0, (step + 1) * self.m.opt.timestep / ramp_s)
                blend = f * f * (3 - 2 * f)
                self.d.ctrl[self.actuators] = (
                    target if f == 1 else beginning + (target - beginning) * blend
                )
                before_time = float(self.d.time)
                reset_inputs = {
                    name: getattr(self.d, name).copy()
                    for name in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(self.m, self.d)
                self.executed += 1
                mujoco.mj_forward(self.m, self.d)
                event, warning = _physics_event(
                    self.m, self.d, before_time, warning, reset_inputs
                )
                if event is not None:
                    events.append(dict(executed_step=self.executed, **event))
                    raise RuntimeError(
                        "Native warning, reset evidence or time discontinuity"
                    )
                self.check()
            if np.max(abs(self.d.qpos[self.addresses] - target)) > 0.0002:
                raise RuntimeError(
                    "Array did not settle within 0.2 mrad of every target"
                )
            if np.max(abs(self.d.qvel[self.velocities])) > 0.001:
                raise RuntimeError(
                    "Array residual speed exceeds 1 mrad/s settling criterion"
                )
        except RuntimeError as error:
            reason = str(error)
            self.d.ctrl[self.actuators] = np.clip(
                self.d.qpos[self.addresses], -LIMIT, LIMIT
            )
        complete = self.executed > 0 and self.executed == self.checked
        native_reset = any(e["automatic_reset_detected"] for e in events)
        unknown_reset = any(e["reset_performed"] is None for e in events)
        reset_performed = True if native_reset else (None if unknown_reset else False)
        return dict(
            status="succeeded" if reason is None else "failed",
            reason=reason,
            physics_steps=self.executed,
            checked_steps=self.checked,
            observer_entries=self.entries,
            unchecked_steps=self.executed - self.checked,
            coverage_complete=complete,
            candidate_pair_count=self.all_pair_count,
            retained_pair_count=len(self.pairs),
            direct_queries=self.queries,
            clearance_cap_m=self.cap,
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            maximum_actuator_torque_Nm=self.maximum_torque,
            maximum_target_error_rad=float(
                np.max(abs(self.d.qpos[self.addresses] - target))
            ),
            maximum_terminal_speed_rad_s=float(
                np.max(abs(self.d.qvel[self.velocities]))
            ),
            settling_speed_limit_rad_s=0.001,
            targets_rad=target.tolist(),
            hold_requested=reason is not None,
            reset_requested=False,
            reset_used=reset_performed,
            reset_performed=reset_performed,
            explicit_reset_performed=False,
            automatic_reset_detected=native_reset,
            time_discontinuity_detected=any(not e["time_continuous"] for e in events),
            physics_events=events,
            initial_state=initial,
            final_state=self.state(),
            scene_sha256=self.binding["scene_sha256"],
            task_sha256=self.source_sha256,
            scientific_model_qualified=False,
        )
