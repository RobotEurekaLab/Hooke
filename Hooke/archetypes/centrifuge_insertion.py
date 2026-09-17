"""Shared rotor-frame targets and contact-driven tube insertion recipe.

The host provides the existing arm, tube, instrument and motion primitives.
This module issues robot controls through that host and has no backend imports.
"""
import mujoco
import numpy as np
from kinematics import Pose, mul_pose, neg_pose


class RotorSlotPoses:
    """Pose helpers for instruments exposing num_slots and slot_sites."""

    def get_slot_pose(self, data, slot_id: int) -> Pose:
        if not 0 <= slot_id < self.num_slots:
            raise ValueError(f'Invalid slot id {slot_id}')
        site = self.slot_sites[slot_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        return Pose(data.site_xpos[site].copy(), quat)

    def get_tube_pose(self, data, slot_id: int, mode: str = 'distal') -> Pose:
        offsets = {'distal': .005, 'proximal': -.03}
        if mode not in offsets:
            raise ValueError(f'Unknown insertion target mode {mode!r}')
        offset = Pose(np.array([0., 0., offsets[mode]]),
                      np.array([1., 0., 0., -1.]) / np.sqrt(2.))
        return mul_pose(self.get_slot_pose(data, slot_id), offset)

    def rotor_perturb(self):
        return np.random.uniform(-.1, .1)


def insertion_target(task) -> Pose:
    """The selected slot moves with the free rotor during contact."""
    return task.instrument.get_tube_pose(task.data, task.slot_id, 'proximal')


def insertion_geometry_passes(task) -> bool:
    position = task.tube.get_body_pose(task.data).pos
    return bool(.955 < position[2] < .961
                and np.linalg.norm(position - insertion_target(task).pos) < .005)


def execute_insertion(expert):
    """Grasp, transfer, remeasure the grasp, seat, release and clear the tube."""
    def site_pose():
        return expert.arm.get_site_pose(expert.data)

    expert.path_follow(expert.interpolate(site_pose(), expert.tube.get_eef_pose(expert.data), 10))
    expert.gripper_control(240)
    current = site_pose()
    expert.move_to(Pose(current.pos + np.array([0., 0., .1]), current.quat), 20)

    grasp = mul_pose(neg_pose(expert.tube.get_body_pose(expert.data)), site_pose())
    distal = expert.instrument.get_tube_pose(expert.data, expert.slot_id, 'distal')
    expert.path_follow(expert.interpolate2(site_pose(), mul_pose(distal, grasp), 20))

    # A transferred tube can slip slightly in the gripper. Both the grasp
    # transform and rotor target must be measured again before lowering.
    grasp = mul_pose(neg_pose(expert.tube.get_body_pose(expert.data)), site_pose())
    expert.move_to(mul_pose(insertion_target(expert), grasp), 20)
    expert.gripper_control(0)
    current = site_pose()
    expert.move_to(Pose(current.pos + np.array([0., 0., .06]), current.quat), 10)
    # Allow contact oscillations to decay before observing a stable seat.
    for _ in range(round(1.2 / expert.dt)):
        expert.step_and_log({})
