"""Measured wheel-contact traversal and physical sample staging for NASA bay v1.

This authored Earth-gravity, rigid-flat-floor task is not a NASA vehicle or soil
model. No live qpos/qvel writes, resets, welds, mocap attachments or external body
forces are used. Direct geometry uses exact box SAT and analytic cylinder/floor
support where MuJoCo 3.3's generic distance can be inaccurate near contact.
"""

from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from real_labs.nasa_rover_geometry import START, VERSION
from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports

ARM = ("base_yaw", "shoulder", "elbow", "wrist_roll", "wrist_pitch", "tool_yaw")


def _execution_provenance(session):
    """Capture descriptive observation identity without claiming qualification."""
    observations = getattr(session, "physics_observations", {})
    keys = ("backend", "physics_engine", "actuator_force_provenance")
    return tuple(
        (key, value if isinstance(value := observations.get(key), str) else "unknown")
        for key in keys
    )


def _observed_actuator_peaks(session):
    """Use applied actuator-input peaks when supplied, retaining source limits."""
    observations = getattr(session, "physics_observations", {})
    if observations.get("actuator_force") is not True:
        raise ValueError(
            "The rover task requires observed actuator forces for its drive "
            "limits; this backend has not supplied that observation channel"
        )
    values = observations.get("actuator_force_peak_abs")
    peaks = np.abs(session.data.actuator_force) if values is None else np.asarray(values, dtype=float)
    if peaks.shape != (session.model.nu,) or not np.isfinite(peaks).all() or np.any(peaks < 0):
        raise ValueError("Observed actuator-force peaks must be finite nonnegative values")
    return peaks


@lru_cache(maxsize=1)
def _binding():
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources

    root, manifest = build_scene("nasa_planetary")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    _portable_resources(root)
    return dict(
        signature=_model_signature(m),
        equipment=manifest["equipment"],
        scene_sha256=hashlib.sha256(
            ET.tostring(root, encoding="unicode").encode()
        ).hexdigest(),
    )


def _require_binding(session):
    if session.manifest.get("id") != "nasa_planetary":
        raise ValueError("NASA task requires its authored scene")
    b = _binding()
    if (
        session.data.model is not session.model
        or _model_signature(session.model) != b["signature"]
    ):
        raise ValueError("NASA task geometry, dynamics or transmissions changed")
    if session.manifest["equipment"] != b["equipment"]:
        raise ValueError("NASA task metadata changed")
    if session.manifest.get("scene_sha256", b["scene_sha256"]) != b["scene_sha256"]:
        raise ValueError("NASA task source digest changed")
    if session.equipment["rover"].get("nasa_rover_task", {}).get("version") != VERSION:
        raise ValueError("NASA task version missing")
    return b


def box_separation(model, data, a, b):
    """Exact 15-axis OBB signed separating bound; negative means overlap.

    For separated boxes the bound may be smaller than Euclidean distance. It is
    therefore conservative for clearance. No box pair or parent pair is masked.
    """
    A = data.geom_xmat[a].reshape(3, 3)
    B = data.geom_xmat[b].reshape(3, 3)
    axes = np.r_[A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(-1, 3)]
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    gaps = (
        abs(axes @ (data.geom_xpos[b] - data.geom_xpos[a]))
        - abs(axes @ A) @ model.geom_size[a]
        - abs(axes @ B) @ model.geom_size[b]
    )
    return float(np.max(gaps))


def direct_gap(model, data, a, b, cap=0.001):
    box = mujoco.mjtGeom.mjGEOM_BOX
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    if model.geom_type[a] == box and model.geom_type[b] == box:
        return min(cap, box_separation(model, data, a, b))
    # The named floor is a finite axis-aligned box. Only use the plane formula
    # when the full cylinder projection lies strictly inside its XY footprint.
    if model.geom(a).name == "arch_floor" and model.geom_type[b] == cylinder:
        R = data.geom_xmat[b].reshape(3, 3)
        axis = R[:, 2]
        r, h = model.geom_size[b, :2]
        support = r * np.sqrt(np.maximum(0, 1 - axis * axis)) + h * abs(axis)
        floor_R = data.geom_xmat[a].reshape(3, 3)
        if np.allclose(floor_R, np.eye(3), atol=1e-12, rtol=0) and np.all(
            abs(data.geom_xpos[b, :2] - data.geom_xpos[a, :2]) + support[:2]
            < model.geom_size[a, :2]
        ):
            return min(
                cap,
                float(
                    data.geom_xpos[b, 2]
                    - support[2]
                    - data.geom_xpos[a, 2]
                    - model.geom_size[a, 2]
                ),
            )
    return min(
        cap, float(mujoco.mj_geomDistance(model, data, int(a), int(b), cap, None))
    )


class NasaRoverTask:
    """Exclusively own a bound session through the declared physical task.

    Only actuator controls are changed in live state. IK uses separate scratch
    MjData. The route is a finite, authored flat-bay operation, not a general
    planner. Every native step and sampled direct geometry are checked.
    """

    def __init__(self, session, *, record=True, require_start=True):
        _observed_actuator_peaks(session)
        self.execution_provenance = _execution_provenance(session)
        if not math.isfinite(float(session.data.time)):
            raise ValueError("NASA task requires finite simulation time")
        self.binding = _require_binding(session)
        self.session = session
        self.m = session.model
        self.d = session.data
        self.record = bool(record)
        self.initial_time = float(self.d.time)
        self.initial_warning = self.d.warning.number.copy()
        self.initial_state = {
            name: getattr(self.d, name).copy()
            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act")
        }
        self.geometry_checks = 0
        self.minimum_direct = 0.0
        self.direct_queries = 0
        self.direct_worst = None
        self.operation = "traverse_and_stage"
        self.mast_captures = []
        self.max_parked_drift = 0.0
        self.ik_records = []
        self.max_arm_tracking_error = 0.0
        self.completed = False
        self.max_retention_error = 0.0
        self.minimum_carried_height = 10.0
        m, d = self.m, self.d
        self.armj = np.array([m.joint("rover__joint_arm_" + s).id for s in ARM])
        self.armq = m.jnt_qposadr[self.armj]
        self.armv = m.jnt_dofadr[self.armj]
        self.mastj = m.joint("rover__joint_mast_pan").id
        self.mastq = int(m.jnt_qposadr[self.mastj])
        self.masta = m.actuator("rover__actuator_mast_pan").id
        self.mast_target = float(d.qpos[self.mastq])
        self.arma = np.array([m.actuator("rover__actuator_arm_" + s).id for s in ARM])
        self.target = np.zeros(6)
        self.finga = np.array(
            [
                m.actuator("rover__actuator_arm_gripper_" + s).id
                for s in ("left", "right")
            ]
        )
        self.fingj = np.array(
            [m.joint("rover__joint_arm_gripper_" + s).id for s in ("left", "right")]
        )
        self.fingq = m.jnt_qposadr[self.fingj]
        self.wheela = np.array(
            [
                m.actuator("rover__actuator_wheel_" + s + "_" + str(i)).id
                for s in ("left", "right")
                for i in (1, 2, 3)
            ]
        )
        self.steera = np.array(
            [
                m.actuator("nasa_task__steer_wheel_" + s + "_" + str(i)).id
                for s in ("left", "right")
                for i in (1, 2, 3)
            ]
        )
        self.steerj = np.array(
            [
                m.joint("nasa_task__steer_wheel_" + s + "_" + str(i)).id
                for s in ("left", "right")
                for i in (1, 2, 3)
            ]
        )
        self.steerq = m.jnt_qposadr[self.steerj]
        self.steer_target = np.zeros(6)
        self.xy = np.array([(x, y) for y in (-0.416, 0.416) for x in (-0.4, 0, 0.4)])
        self.tcp = m.site("rover__site_arm_tcp").id
        self.base = m.body("rover").id
        self.sample = m.body("nasa_task__sample").id
        self.grip = 0.015  # XML joint references define the open initial state.
        self.park = np.array(START[:2])
        self.park_yaw = 0.0
        self.parking_anchor = None
        self.rows = []
        self.events = []
        self.steps = 0
        self.completed_checks = 0
        self.physics_events = []
        self.phase = "initial"
        self.drive = None
        self.v = 0.0
        self.w = 0.0
        self.minimum_contact = 0.0
        self.worst_contact = None
        self.max_wheel_torque = 0.0
        self.minimum_height = 10.0
        self.raw = []
        self.padids = {
            m.geom("rover__arm_contact_arm_finger_pad_" + s).id
            for s in ("left", "right")
        }
        self.sg = m.geom("nasa_task__sample").id
        mujoco.mj_forward(m, d)
        if not np.isfinite(np.r_[d.qpos, d.qvel, d.ctrl]).all():
            raise ValueError("Nonfinite initial state")
        if require_start and (
            np.linalg.norm(d.xpos[self.base][:2] - np.array(START[:2])) > 0.03
            or abs(self.pose()[1]) > 0.03
        ):
            raise ValueError("Task requires its declared pickup start pose")
        if (
            require_start
            and np.linalg.norm(
                d.xpos[self.sample] - np.array([START[0] + 0.6554, START[1], 0.7263])
            )
            > 0.002
        ):
            raise ValueError("Sample is not on the pickup fixture")
        if np.max(abs(d.qvel)) > 0.05 or (
            require_start and np.max(abs(d.qpos[self.armq])) > 0.08
        ):
            raise ValueError("Task requires settled initial joints")
        if not require_start:
            self.target = d.qpos[self.armq].copy()
            self.grip = float(np.mean(d.qpos[self.fingq]))
            self.steer_target = d.qpos[self.steerq].copy()
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] != 0
                    or m.body_weldid[m.geom_bodyid[b]] != 0
                )
            ],
            dtype=int,
        )
        self._direct_check()

    def pose(self):
        d = self.d
        R = d.xmat[self.base].reshape(3, 3)
        return d.xpos[self.base].copy(), math.atan2(R[1, 0], R[0, 0])

    def step(self, v=0, w=0):
        m, d = self.m, self.d
        _observed_actuator_peaks(self.session)
        if _execution_provenance(self.session) != self.execution_provenance:
            raise RuntimeError("Task physics observation provenance changed")
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("Task model/state replaced")
        if not math.isfinite(float(d.time)):
            raise ValueError("NASA task requires finite simulation time")
        if not math.isfinite(v + w) or abs(v) > 0.13 + 1e-10 or abs(w) > 0.35 + 1e-10:
            raise ValueError("Drive twist outside authored limits")
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
            raise RuntimeError("External applied forces are outside this task")
        if v == 0 and w == 0:
            from real_labs.rover_parking import parking_twist
            position, heading = self.pose()
            if self.parking_anchor is None:
                self.parking_anchor = (position.copy(), heading)
            base_joint = int(m.body_jntadr[self.base])
            velocity_address = int(m.jnt_dofadr[base_joint])
            world_yaw_rate = float(d.xmat[self.base].reshape(3, 3)[2] @ d.qvel[velocity_address+3:velocity_address+6])
            v, w = parking_twist(position, heading, d.qvel[velocity_address:velocity_address+3],
                                 world_yaw_rate, *self.parking_anchor)
        else:
            self.parking_anchor = None
        # Joint gravity compensation through the finite original actuators.
        d.ctrl[self.arma] = np.clip(
            self.target + d.qfrc_bias[self.armv] / m.actuator_gainprm[self.arma, 0],
            m.actuator_ctrlrange[self.arma, 0],
            m.actuator_ctrlrange[self.arma, 1],
        )
        d.ctrl[self.finga] = self.grip
        d.ctrl[self.masta] = self.mast_target
        # Labels follow existing model: 'left' occupies negative local Y.
        self.v += np.clip(v - self.v, -0.2 * m.opt.timestep, 0.2 * m.opt.timestep)
        self.w += np.clip(w - self.w, -0.6 * m.opt.timestep, 0.6 * m.opt.timestep)
        v, w = self.v, self.w
        velocities = np.c_[v - w * self.xy[:, 1], w * self.xy[:, 0]]
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        speeds = np.linalg.norm(velocities, axis=1) / 0.14
        for i in range(6):
            if angles[i] > math.pi / 2:
                angles[i] -= math.pi
                speeds[i] *= -1
            if angles[i] < -math.pi / 2:
                angles[i] += math.pi
                speeds[i] *= -1
        if abs(v) + abs(w) < 1e-9:
            angles = d.qpos[self.steerq].copy()
        self.steer_target += np.clip(
            angles - self.steer_target, -0.6 * m.opt.timestep, 0.6 * m.opt.timestep
        )
        d.ctrl[self.steera] = np.clip(self.steer_target, -1.57, 1.57)
        d.ctrl[self.wheela] = (
            speeds * np.maximum(0, np.cos(angles - d.qpos[self.steerq])) ** 8
        )
        before_time = float(d.time)
        before_warnings = _warning_state(d)
        reset_inputs = {
            key: getattr(d, key).copy()
            for key in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        mujoco.mj_step(m, d)
        self.steps += 1
        mujoco.mj_forward(m, d)
        event, _ = _physics_event(m, d, before_time, before_warnings, reset_inputs)
        if event is not None:
            self.physics_events.append(dict(executed_step=self.steps, **event))
            raise RuntimeError("Native physics warning or time discontinuity")
        if (
            not np.all(np.isfinite(d.qpos))
            or abs(d.time - self.initial_time - self.steps * m.opt.timestep) > 1e-7
        ):
            raise RuntimeError("nonfinite/time")
        p, y = self.pose()
        pads = set()
        pen = 0
        for c in d.contact:
            pen = min(pen, c.dist)
            if c.dist < self.minimum_contact:
                self.minimum_contact = float(c.dist)
                self.worst_contact = dict(
                    t=float(d.time),
                    geoms=[m.geom(c.geom1).name, m.geom(c.geom2).name],
                    distance=float(c.dist),
                )
            if self.sg in (c.geom1, c.geom2):
                pads.update(set((c.geom1, c.geom2)) & self.padids)
        if self.record:
            self.raw.append(np.r_[d.time, d.qpos, d.qvel, d.ctrl, d.actuator_force])
        if not np.isfinite(
            np.r_[d.qpos, d.qvel, d.actuator_force]
        ).all() or not np.array_equal(d.warning.number, self.initial_warning):
            raise RuntimeError("Nonfinite state or native physics warning")
        if self.minimum_contact < -5e-5:
            raise RuntimeError(f"Native contact exceeds 50 um: {self.worst_contact}")
        effort_peaks = _observed_actuator_peaks(self.session)
        if (
            np.max(effort_peaks[self.wheela]) > 12.000001
            or np.max(effort_peaks[self.steera]) > 20.000001
        ):
            raise RuntimeError("Drive force limit exceeded")
        if self.phase.startswith("mast_"):
            drift = float(np.linalg.norm(d.xpos[self.base, :2] - self.park[:2]))
            self.max_parked_drift = max(self.max_parked_drift, drift)
            if drift > 0.003:
                raise RuntimeError("Parked mast base drift exceeds 3 mm")
        if self.phase in (
            "reverse_clear_pickup",
            "traverse_aisle_1",
            "traverse_aisle_2",
            "approach_inspection_station",
        ):
            retention = float(
                np.linalg.norm(d.xpos[self.sample] - d.site_xpos[self.tcp])
            )
            self.max_retention_error = max(self.max_retention_error, retention)
            self.minimum_carried_height = min(
                self.minimum_carried_height, float(d.xpos[self.sample, 2])
            )
            if retention > 0.025 or d.xpos[self.sample, 2] < 0.7763:
                raise RuntimeError("Sample retention failed during traversal")
        if self.steps % round(0.05 / m.opt.timestep) == 0:
            self._direct_check()
        self.max_wheel_torque = max(
            self.max_wheel_torque, float(np.max(effort_peaks[self.wheela]))
        )
        if self.steps % round(0.02 / m.opt.timestep) == 0:
            self.rows.append(
                dict(
                    t=float(d.time),
                    phase=self.phase,
                    base=p.tolist(),
                    yaw=y,
                    tcp=d.site_xpos[self.tcp].tolist(),
                    sample=d.xpos[self.sample].tolist(),
                    gripper=d.qpos[self.fingq].tolist(),
                    pads=len(pads),
                    min_contact=pen,
                    qpos=d.qpos.tolist(),
                    qvel=d.qvel.tolist(),
                    ctrl=d.ctrl.tolist(),
                    wheel_torque=d.actuator_force[self.wheela].tolist(),
                    wheel_peak_abs_torque=effort_peaks[self.wheela].tolist(),
                )
            )

        self.completed_checks += 1

    def wait(self, t):
        for _ in range(round(t / self.m.opt.timestep)):
            self.step()

    def checkpoint(self, s):
        self.phase = s
        p, y = self.pose()
        e = dict(
            name=s,
            t=float(self.d.time),
            base=p.tolist(),
            yaw=y,
            tcp=self.d.site_xpos[self.tcp].tolist(),
            sample=self.d.xpos[self.sample].tolist(),
            arm=self.d.qpos[self.armq].tolist(),
        )
        self.events.append(e)
        wheel_bodies = set(self.m.jnt_bodyid[self.m.actuator_trnid[self.wheela, 0]])
        wheel_geoms = set(
            np.flatnonzero(np.isin(self.m.geom_bodyid, list(wheel_bodies)))
        )
        wheel_load = 0.0
        loaded = set()
        contacts = []
        for i, c in enumerate(self.d.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, force)
            if (
                self.m.geom("arch_floor").id in (c.geom1, c.geom2)
                and set((c.geom1, c.geom2)) & wheel_geoms
            ):
                wheel_load += force[0]
                loaded.update(set((c.geom1, c.geom2)) & wheel_geoms)
            if self.sg in (c.geom1, c.geom2):
                contacts.append(
                    dict(
                        geoms=[self.m.geom(c.geom1).name, self.m.geom(c.geom2).name],
                        normal_force_n=float(force[0]),
                        gap_m=float(c.dist),
                    )
                )
        e.update(
            wheel_normal_load_n=wheel_load,
            loaded_wheels=len(loaded),
            sample_contacts=contacts,
        )
        self._direct_check()

    def hand(self, target, t=2):
        if target not in (0, 0.015) or not 0 < t <= 5:
            raise ValueError("Invalid gripper command")
        start = self.grip
        n = round(t / self.m.opt.timestep)
        for k in range(n):
            u = (k + 1) / n
            self.grip = start + (target - start) * (u * u * (3 - 2 * u))
            self.step()

    def arm(self, q, t=2):
        a = self.target.copy()
        q = np.asarray(q, dtype=float)
        if (
            q.shape != (6,)
            or not np.isfinite(q).all()
            or np.any(q < self.m.jnt_range[self.armj, 0])
            or np.any(q > self.m.jnt_range[self.armj, 1])
            or not 0 < t <= 10
        ):
            raise ValueError("Invalid arm target")
        n = round(t / self.m.opt.timestep)
        for k in range(n):
            u = (k + 1) / n
            self.target = a + (q - a) * (u * u * (3 - 2 * u))
            self.step()
        self.wait(0.5)
        error = float(np.max(abs(self.d.qpos[self.armq] - q)))
        self.max_arm_tracking_error = max(self.max_arm_tracking_error, error)
        if error > 0.012:
            raise RuntimeError(f"Arm target tracking failed: {error} rad")

    def ik(self, xyz, orientation=None):
        m, d = self.m, self.d
        s = mujoco.MjData(m)
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        q = self.target.copy()
        for _ in range(150):
            s.qpos[self.armq] = q
            mujoco.mj_forward(m, s)
            e = np.asarray(xyz) - s.site_xpos[self.tcp]
            J = np.zeros((3, m.nv))
            Jr = np.zeros_like(J)
            mujoco.mj_jacSite(m, s, J, Jr, self.tcp)
            if orientation is not None:
                R = s.site_xmat[self.tcp].reshape(3, 3)
                er = sum(np.cross(R[:, i], orientation[:, i]) for i in range(3)) / 2
                err = np.r_[e, er * 0.15]
                jac = np.r_[J[:, self.armv], Jr[:, self.armv] * 0.15]
            else:
                err = e
                jac = J[:, self.armv]
            if np.linalg.norm(err) < 1e-5:
                break
            q += np.clip(
                jac.T @ np.linalg.solve(jac @ jac.T + np.eye(len(err)) * 1e-5, err),
                -0.08,
                0.08,
            )
            q = np.clip(
                q, m.jnt_range[self.armj, 0] + 0.02, m.jnt_range[self.armj, 1] - 0.02
            )
        position_error = float(np.linalg.norm(e))
        weighted_error = float(np.linalg.norm(err))
        self.ik_records.append(
            dict(
                target=np.asarray(xyz).tolist(),
                position_error_m=position_error,
                weighted_error=weighted_error,
                joints=q.tolist(),
            )
        )
        if position_error > 2e-5 or weighted_error > 2e-5:
            raise RuntimeError(f"IK target unreachable: {position_error} m")
        return q, position_error

    def go(self, xy, yaw=None, timeout=50, reverse=False):
        xy = np.array(xy)
        n = round(timeout / self.m.opt.timestep)
        reached = False
        for k in range(n):
            p, a = self.pose()
            delta = xy - p[:2]
            dist = np.linalg.norm(delta)
            err = (math.atan2(delta[1], delta[0]) - a + math.pi) % (
                2 * math.pi
            ) - math.pi
            direction = 1
            if reverse and abs(err) > math.pi / 2:
                err = ((err + math.pi) + math.pi) % (2 * math.pi) - math.pi
                direction = -1
            if dist < 0.005:
                reached = True
            if reached:
                err = (
                    0 if yaw is None else (yaw - a + math.pi) % (2 * math.pi) - math.pi
                )
                v = 0
                w = np.clip(2 * err, -0.35, 0.35)
                if abs(err) < 0.01:
                    break
            else:
                v = (
                    direction * min(0.13, 0.65 * dist) * max(0, math.cos(err))
                    if abs(err) < 0.6
                    else 0
                )
                w = np.clip(2 * err, -0.35, 0.35)
            self.step(v, w)
        else:
            raise RuntimeError("navigation timeout")
        self.wait(1)
        self.checkpoint(self.phase + "_arrived")

    def _direct_check(self):
        m, d = self.m, self.d
        R = d.geom_xmat.reshape(-1, 3, 3)
        centers = d.geom_xpos + np.einsum("nij,nj->ni", R, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(R), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(abs(centers[a] - centers[b]) - half[a] - half[b], 0)
        near = np.flatnonzero(np.linalg.norm(separation, axis=1) < 0.001)
        self.geometry_checks += 1
        for i in near:
            a, b = self.pairs[i]
            gap = direct_gap(m, d, int(a), int(b))
            self.direct_queries += 1
            if gap < self.minimum_direct:
                self.minimum_direct = gap
                self.direct_worst = dict(
                    t=float(d.time), geoms=[m.geom(a).name, m.geom(b).name], gap_m=gap
                )
            if gap < -0.00005:
                raise RuntimeError(
                    f"Direct geometry exceeds 50 um: {self.direct_worst}"
                )

    def _sample_contact_force(self, geom):
        force = 0.0
        for i, c in enumerate(self.d.contact):
            if set((c.geom1, c.geom2)) == set((geom, self.sg)):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, i, f)
                force += f[0]
        return float(force)

    def inspect_mast(self):
        """Pan the real mounted cameras to their supported target and return."""
        from real_labs.nasa_mast_motion import MastInspection

        self.park = self.d.xpos[self.base].copy()
        self.park_yaw = self.pose()[1]
        home = float(self.d.qpos[self.mastq])
        target_site = self.session.equipment["rover"]["mast_inspection"]["target"][
            "site"
        ]
        vector = (
            self.d.site_xpos[self.m.site(target_site).id] - self.d.xanchor[self.mastj]
        )
        target = math.atan2(float(vector[1]), float(vector[0])) - self.pose()[1]
        target = (target + math.pi) % (2 * math.pi) - math.pi
        for label, value in (("mast_align", target), ("mast_return", home)):
            self.phase = label
            start = self.mast_target
            n = round(3 / self.m.opt.timestep)
            for k in range(n):
                u = (k + 1) / n
                self.mast_target = start + (value - start) * (u * u * (3 - 2 * u))
                self.step()
            self.wait(1)
            if abs(self.d.qpos[self.mastq] - value) > 0.0001:
                raise RuntimeError("Parked mast did not reach commanded pan")
            if label == "mast_align":
                capture = MastInspection(self.session)
                capture.capture()
                self.mast_captures.extend(capture.report["captures"])
            self.checkpoint(label + "_complete")
        return self.mast_captures

    def perform(self):
        """Grip, back clear, traverse the bay aisle, physically stage and retreat."""
        self.phase = "settle"
        self.wait(2)
        self.checkpoint("settled")
        self.inspect_mast()
        R = self.d.site_xmat[self.tcp].reshape(3, 3).copy()
        self.phase = "align_pickup"
        q, _ = self.ik(self.d.xpos[self.sample] + np.array([0.00124, 0, 0]), R)
        self.arm(q)
        self.phase = "close_cradle"
        self.hand(0)
        self.wait(0.5)
        self.checkpoint("gripped")
        self.phase = "lift_sample"
        q, _ = self.ik(self.d.site_xpos[self.tcp] + np.array([-0.08, 0, 0.065]), R)
        self.arm(q)
        self.checkpoint("lifted")
        if self.d.xpos[self.sample, 2] < 0.7763:
            raise RuntimeError("Sample did not lift at least 50 mm")
        self.phase = "reverse_clear_pickup"
        self.go((-5.25, 2.5), 0, reverse=True)
        self.phase = "traverse_aisle_1"
        self.go((-5.25, 3.6))
        self.phase = "traverse_aisle_2"
        self.go((-4.6, 4.6), math.pi / 2)
        self.phase = "approach_inspection_station"
        self.go((-4.6, 5.38), math.pi / 2)
        self.checkpoint("stage_approach")
        if (
            np.linalg.norm(self.d.xpos[self.sample] - self.d.site_xpos[self.tcp])
            > 0.025
        ):
            raise RuntimeError("Sample was not retained through traversal")
        goal = self.d.site_xpos[self.m.site("nasa_task__stage_target").id].copy()
        R = self.d.site_xmat[self.tcp].reshape(3, 3).copy()
        self.phase = "position_over_stage"
        q, _ = self.ik(goal + np.array([0, 0.00124, 0.045]), R)
        self.arm(q)
        self.checkpoint("over_stage")
        self.phase = "lower_to_seat"
        q, _ = self.ik(goal + np.array([0, 0.00124, 0.0001]), R)
        self.arm(q)
        seat = self.m.geom("nasa_task__stage_seat").id
        for _ in range(24):
            if self._sample_contact_force(seat) > 0.03:
                break
            q, _ = self.ik(self.d.site_xpos[self.tcp] + np.array([0, 0, -0.00025]), R)
            self.arm(q, t=0.2)
        else:
            raise RuntimeError("Sample did not physically contact stage seat")
        self.wait(0.5)
        self.checkpoint("seat_contact")
        self.phase = "open_and_release"
        self.hand(0.015)
        self.wait(1)
        self.checkpoint("released")
        self.phase = "withdraw_tool"
        q, _ = self.ik(self.d.site_xpos[self.tcp] + np.array([0, -0.03, 0.08]), R)
        self.arm(q)
        self.wait(2)
        self.checkpoint("retreated")
        error = self.d.xpos[self.sample] - goal
        force = self._sample_contact_force(seat)
        other_contacts = []
        for c in self.d.contact:
            if self.sg in (c.geom1, c.geom2) and seat not in (c.geom1, c.geom2):
                other_contacts.append(
                    [self.m.geom(c.geom1).name, self.m.geom(c.geom2).name]
                )
        if (
            np.linalg.norm(error[:2]) > 0.003
            or abs(error[2]) > 0.00005
            or abs(force - 0.025 * 9.81) > 0.002
            or other_contacts
        ):
            raise RuntimeError("Final sample staging/support gate failed")
        if np.linalg.norm(self.d.site_xpos[self.tcp] - self.d.xpos[self.sample]) < 0.07:
            raise RuntimeError("Tool did not withdraw clear")
        self._direct_check()
        _require_binding(self.session)
        self.completed = True
        return self.report()

    def report(self, failure=None):
        reset = (
            combine_reset_reports(self.physics_events)
            if self.physics_events
            else dict(
                reset_performed=False,
                explicit_reset_performed=False,
                automatic_reset_detected=False,
            )
        )
        return dict(
            operation=self.operation,
            mast_captures=self.mast_captures,
            max_parked_mast_drift_m=self.max_parked_drift,
            success=self.completed and failure is None,
            failure=failure,
            version=VERSION,
            scene_sha256=self.binding["scene_sha256"],
            physics_steps=self.steps,
            completed_checks=self.completed_checks,
            unchecked_steps=self.steps - self.completed_checks,
            coverage_complete=self.steps > 0 and self.steps == self.completed_checks,
            physics_events=list(self.physics_events),
            time_discontinuity_detected=any(
                not event["time_continuous"] for event in self.physics_events
            ),
            **reset,
            simulated_seconds=float(self.d.time - self.initial_time),
            timestep_s=float(self.m.opt.timestep),
            no_live_pose_writes=True,
            no_resets=(
                None if reset["reset_performed"] is None else not reset["reset_performed"]
            ),
            external_forces=False,
            gravity_compensation="joint actuator feedforward; all rover bodies gravcomp=0",
            geometry_checks=self.geometry_checks,
            direct_queries=self.direct_queries,
            minimum_direct_m=self.minimum_direct,
            direct_worst=self.direct_worst,
            worst_native_contact=self.worst_contact,
            max_wheel_torque_nm=self.max_wheel_torque,
            max_arm_endpoint_error_rad=self.max_arm_tracking_error,
            max_retention_error_m=self.max_retention_error,
            minimum_carried_height_m=self.minimum_carried_height,
            ik=self.ik_records,
            phases=self.events,
            final_sample_xyz_m=self.d.xpos[self.sample].tolist(),
            final_stage_force_n=self._sample_contact_force(
                self.m.geom("nasa_task__stage_seat").id
            ),
            execution_provenance=dict(self.execution_provenance),
            qualification_scope="Executed steps of the named operation only; no whole-scene qualification",
            scope="Authored rigid flat Earth-gravity bay traversal and inert-box staging; no soil, flight rover or OEM calibration",
        )

    def save(self, destination, report):
        """Write to an already-created, exclusively owned experiment directory."""
        p = Path(destination)
        fields = np.array(["time", "qpos", "qvel", "ctrl", "actuator_force"])
        if self.record:
            np.savez_compressed(
                p / "states.npz",
                states=np.array(self.raw),
                nq=self.m.nq,
                nv=self.m.nv,
                nu=self.m.nu,
                fields=fields,
            )
        np.savez_compressed(
            p / "initial.npz", **self.initial_state, time=self.initial_time
        )
        (p / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (p / "trace.json").write_text(json.dumps(self.rows, allow_nan=False) + "\n")
        (p / "events.json").write_text(
            json.dumps(self.events, indent=2, allow_nan=False) + "\n"
        )
        (p / "nasa_rover_task.py").write_bytes(Path(__file__).read_bytes())


def inspect_grounded_nasa_mast(session):
    """The parked moving-base replacement for the historical stand protocol."""
    task = NasaRoverTask(session, record=False, require_start=False)
    task.operation = "parked_mast_inspection"
    try:
        task.phase = "settle"
        task.wait(2)
        task.inspect_mast()
        task.completed = True
        report = task.report()
    except (RuntimeError, ValueError) as exc:
        report = task.report(f"{type(exc).__name__}: {exc}")
    report["scope"] = (
        "Parked grounded-base mast/camera pan and return only; no sample transfer in this call"
    )
    report["captures"] = task.mast_captures
    return report


def traverse_and_stage_nasa(session, *, output=None):
    """Execute the full task without resetting or replacing the caller's state.

    A supplied output directory must be fresh. Failure raises RuntimeError and
    preserves the observed failure report/trace there. Call session.reset()
    explicitly for a new episode; reset is not counted as task completion.
    """
    destination = Path(output) if output is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    task = NasaRoverTask(session, record=destination is not None)
    try:
        report = task.perform()
    except Exception as exc:
        if destination is not None:
            task.save(destination, task.report(f"{type(exc).__name__}: {exc}"))
        raise
    if destination is not None:
        task.save(destination, report)
    return report


def main():
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession
    from real_labs.export import _portable_resources
    import shutil, time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output directory must be fresh")
    root, manifest = build_scene("nasa_planetary")
    resources = _portable_resources(root)
    assets = {a["file"]: Path(a["source"]).read_bytes() for a in resources}
    xml = ET.tostring(root, encoding="unicode")
    manifest["scene_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
    session = InstrumentSession(mujoco.MjModel.from_xml_string(xml, assets), manifest)
    started = time.perf_counter()
    try:
        report = traverse_and_stage_nasa(session, output=args.output)
    finally:
        if args.output.exists():
            (args.output / "scene.xml").write_text(xml)
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            for asset in resources:
                dest = args.output / asset["file"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset["source"], dest)
            (args.output / "wall_time.json").write_text(
                json.dumps(
                    dict(
                        seconds=time.perf_counter() - started,
                        cpu_workers=1,
                        gpu_used=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "success",
                    "physics_steps",
                    "simulated_seconds",
                    "minimum_direct_m",
                    "max_wheel_torque_nm",
                    "final_stage_force_n",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
