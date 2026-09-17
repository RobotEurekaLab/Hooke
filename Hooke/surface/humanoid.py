"""G1 joint control with a recurrent policy and gravity-scaled timing.

The Unitree BSD-3-Clause policy is prepared separately in ignored storage.
Only actuator targets are written; the floating base evolves in physics.
"""

import math

import mujoco
import numpy as np

from surface.artifacts import GAIT_POLICY as POLICY
from surface.recurrent import RecurrentPolicy

PREFIX = "/g1:"
LEG_NAMES = tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
)
DEFAULT_LEGS = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
KP = np.array([100, 100, 100, 150, 40, 40] * 2)
KD = np.array([2, 2, 2, 4, 2, 2] * 2)
HOME = np.array([-0.5, -0.75])


class HumanoidNavigation:
    def __init__(self, model, gravity):
        self.model = model
        self.body = model.body(PREFIX + "pelvis").id
        self.root_va = int(model.joint(PREFIX + "floating_base_joint").dofadr[0])
        self.qa = np.array([int(model.joint(PREFIX + n).qposadr[0]) for n in LEG_NAMES])
        self.va = np.array([int(model.joint(PREFIX + n).dofadr[0]) for n in LEG_NAMES])
        self.actuators = np.array([model.actuator(PREFIX + n).id for n in LEG_NAMES])
        if not np.isfinite(gravity) or gravity <= 0:
            raise ValueError("G1 walking requires positive finite gravity")
        self.rate = math.sqrt(gravity / 9.81)
        self.target = HOME.copy()
        self.enabled = False
        self.policy = None
        self.next_update = 0.0
        self.action = np.zeros(12, np.float32)
        self.command = np.zeros(3)

    def command_joints(self, data):
        if not self.enabled or data.time + 1e-9 < self.next_update:
            return
        if self.policy is None:
            self.policy = RecurrentPolicy(POLICY)
        rotation = data.xmat[self.body].reshape(3, 3)
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = self.target - data.xpos[self.body, :2]
        local = rotation[:2, :2].T @ delta
        command = np.r_[
            np.clip(2.5 * local / self.rate, [-0.55, -0.22], [0.55, 0.22]),
            np.clip(-2 * heading / self.rate, -0.35, 0.35),
        ]
        self.command = command
        phase = data.time * self.rate * math.tau / 0.8
        observation = np.r_[
            data.qvel[self.root_va + 3 : self.root_va + 6] / self.rate * 0.25,
            rotation.T @ [0, 0, -1],
            command * [2, 2, 0.25],
            data.qpos[self.qa] - DEFAULT_LEGS,
            data.qvel[self.va] / self.rate * 0.05,
            self.action,
            math.sin(phase),
            math.cos(phase),
        ].astype(np.float32)
        self.action = self.policy(observation)
        if not np.isfinite(self.action).all():
            raise FloatingPointError("G1 policy produced nonfinite controls")
        data.ctrl[self.actuators] = np.clip(
            DEFAULT_LEGS + 0.25 * self.action,
            self.model.actuator_ctrlrange[self.actuators, 0],
            self.model.actuator_ctrlrange[self.actuators, 1],
        )
        self.next_update = data.time + 0.02 / self.rate


class IndexFingerControl:
    """Seven arm joints track a contact target using scratch-data Jacobians."""

    def __init__(self, model, rate=1.0):
        self.model = model
        self.rate = rate
        names = [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
        joints = [model.joint(PREFIX + "left_" + n + "_joint") for n in names]
        self.qa = np.array([int(j.qposadr[0]) for j in joints])
        self.va = np.array([int(j.dofadr[0]) for j in joints])
        self.actuators = np.array([model.actuator(j.name).id for j in joints])
        self.bounds = np.array([j.range for j in joints])
        self.site = model.site(PREFIX + "press_tip").id
        self.scratch = mujoco.MjData(model)
        self.target = None
        self.next_update = 0.0
        self.jacp = np.zeros((3, model.nv))
        self.jacr = self.jacp.copy()

    def command_joints(self, data):
        if self.target is None or data.time + 1e-9 < self.next_update:
            return
        scratch = self.scratch
        scratch.qpos[:] = data.qpos
        q = data.ctrl[self.actuators].copy()
        scratch.qpos[self.qa] = q
        for _ in range(12):
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            current = scratch.site_xmat[self.site].reshape(3, 3)[:, 0]
            error = np.r_[
                self.target - scratch.site_xpos[self.site],
                0.1 * np.cross(current, [0, 1, 0]),
            ]
            if np.linalg.norm(error) < 0.001:
                break
            mujoco.mj_jacSite(self.model, scratch, self.jacp, self.jacr, self.site)
            projection = np.eye(3) - np.outer(current, current)
            jac = np.vstack(
                (self.jacp[:, self.va], 0.1 * projection @ self.jacr[:, self.va])
            )
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.001 * np.eye(6), error)
            q = np.clip(
                q + np.clip(dq, -0.12, 0.12), self.bounds[:, 0], self.bounds[:, 1]
            )
            scratch.qpos[self.qa] = q
        data.ctrl[self.actuators] += np.clip(
            q - data.ctrl[self.actuators], -0.012, 0.012
        )
        self.next_update = data.time + 0.04 / self.rate
