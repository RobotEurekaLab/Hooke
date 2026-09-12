"""Shared robot-arm and motion-primitive scaffolding for AutoBio/Hooke expert scripts.

This module exists because every `mani_*.py` task file in the original AutoBio
release re-declared an identical `UR5eArm` class and an identical set of
motion-primitive helpers (`interpolate`, `interpolate2`, `path_follow`,
`move_to`, `gripper_control`, `wait`) on its `*Expert` class, verbatim,
file after file. That duplication is the main practical obstacle to adding new
instrument/task variants cheaply: every new task paid the cost of re-deriving
(and re-testing) this plumbing. Extracting it once here means new task
archetypes only need to author the instrument-specific interaction logic.

`UR5eArm` and `ExpertMotionMixin` are verified to reproduce the original
per-file implementations exactly (same inputs -> same outputs); see
`private/technical-log.md` for the equivalence check used when this module
was introduced.
"""
from __future__ import annotations

import numpy as np
import mujoco

from kinematics import Pose, slerp


def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)


class UR5eArm:
    """6-DOF UR5e + Robotiq 2F-85 gripper, addressed via a MJCF name prefix."""

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}shoulder_pan'
        self.act_name = f'{prefix}shoulder_pan'
        self.site_name = f'{prefix}2f85:pinch'
        self.base_name = f'{prefix}base'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f'{prefix}2f85:fingers_actuator').id
        self.gripper_jnt_adr = model.joint(f'{prefix}2f85:right_driver_joint').qposadr.item()
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik = None

    def register_ik(self, data: mujoco.MjData):
        from kinematics import IK
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self, lows=(-0.1, 0.0, -0.2, -0.1, 0.0, -0.2), highs=(0.1, 0.3, 0.2, 0.1, 0.3, 0.2)):
        return np.random.uniform(lows, highs)


def qpos_interpolate(qpos_list: list[np.ndarray], num_steps: list[int]) -> list[np.ndarray]:
    """Piecewise-linear interpolation between successive qpos waypoints."""
    interpolated_qpos = []
    for i in range(len(num_steps)):
        start_qpos = qpos_list[i]
        end_qpos = qpos_list[i + 1]
        step_qpos = (end_qpos - start_qpos) / num_steps[i]
        for step in range(num_steps[i]):
            interpolated_qpos.append(start_qpos + step * step_qpos)
    interpolated_qpos.append(qpos_list[-1])
    return interpolated_qpos


class ExpertMotionMixin:
    """Cartesian-space motion primitives shared by every AutoBio/Hooke expert script.

    Assumes the including class provides: `self.arm` (a `UR5eArm`), `self.planner`
    (a `Topp` instance), `self.period`, `self.dt`, and `self.step_and_log`.
    """

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def interpolate2(self, start: Pose, end: Pose, num_steps: int, height: float | None = None) -> list[Pose]:
        """Like `interpolate`, but arcs through a parabolic waypoint `height` above
        the straight line between `start` and `end` (used for approach/retreat
        moves that should not clip through the instrument body)."""
        path = []
        p1 = start.pos
        p2 = end.pos
        horizon_vec = np.array([p2[0] - p1[0], p2[1] - p1[1], 0.0])
        horizon_dis = np.linalg.norm(horizon_vec)
        origin = p1.copy()
        origin[2] = 0.0
        basis1 = horizon_vec / horizon_dis
        basis2 = np.array([0.0, 0.0, 1.0])
        p1_ = np.array([0.0, p1[2]])
        p2_ = np.array([horizon_dis, p2[2]])
        if height is None:
            height = horizon_dis / 4.0
        p3_ = (p1_ + p2_) / 2.0
        p3_[1] += height
        x = np.array([p1_[0], p3_[0], p2_[0]])
        y = np.array([p1_[1], p3_[1], p2_[1]])
        coef = np.polyfit(x, y, 2)
        x_eval = np.linspace(p1_[0], p2_[0], num_steps + 1)
        y_eval = np.polyval(coef, x_eval)
        for i in range(num_steps + 1):
            t = i / num_steps
            quat = slerp(start.quat, end.quat, t)
            pos = x_eval[i] * basis1 + y_eval[i] * basis2 + origin
            path.append(Pose(pos, quat))
        return path

    def path_follow(self, path: list[Pose]):
        trajectory = self.planner.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = self.planner.query(trajectory, t)
                self.data.ctrl[self.arm.act_span] = ctrl
            self.step_and_log({})

    def move_to(self, pose: Pose, num_steps: int = 2):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(delay):
            self.step_and_log({})

    def wait(self, wait_time: float, info: dict):
        wait_steps = int(wait_time / self.dt)
        for _ in range(wait_steps):
            mujoco.mj_step(self.model, self.data)
            self.serializer.record(info)


def make_topp_planner(dof: int, ik_solve):
    from topp import Topp
    return Topp(dof=dof, qc_vel=1.5, qc_acc=1.0, ik=ik_solve)
