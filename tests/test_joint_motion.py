"""Joint approach commands stay smooth and never assign simulated joint state."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from expert_common import ExpertMotionMixin


class MotionHost(ExpertMotionMixin):
    def __init__(self, dof=7):
        self.arm = SimpleNamespace(dof=dof, jnt_span=range(1, dof+1), act_span=range(1, dof+1))
        self.data = SimpleNamespace(qpos=np.zeros(dof+2), ctrl=np.full(dof+2, .123))
        self.dt = .002
        self.commands = [self.data.qpos[self.arm.jnt_span].copy()]

    def step_and_log(self, info):
        self.commands.append(self.data.ctrl[self.arm.act_span].copy())


class JointMotionTests(unittest.TestCase):
    def test_seven_joint_commands_obey_speed_and_acceleration_bounds(self):
        host = MotionHost()
        target = np.array([1., -.5, .25, -.75, .5, -.25, .1])
        original = host.data.qpos.copy()
        host.move_joints(target, velocity=.8, acceleration=.6, settle_seconds=0)
        commands = np.asarray(host.commands)
        speed = np.diff(commands, axis=0) / host.dt
        acceleration = np.diff(speed, axis=0) / host.dt
        self.assertLessEqual(np.max(abs(speed)), .8)
        self.assertLessEqual(np.max(abs(acceleration)), .6)
        self.assertLess(np.max(abs(speed[[0, -1]])), 1e-5)
        np.testing.assert_array_equal(host.data.qpos, original)
        np.testing.assert_array_equal(host.data.ctrl[host.arm.act_span], target)
        np.testing.assert_array_equal(host.data.ctrl[[0, -1]], [.123, .123])

    def test_stationary_target_does_not_create_joint_motion(self):
        host = MotionHost()
        host.move_joints(np.zeros(7), settle_seconds=.006)
        self.assertEqual(len(host.commands), 5)
        np.testing.assert_array_equal(host.commands, np.zeros((5, 7)))

    def test_invalid_targets_do_not_change_controls(self):
        for target in (np.zeros(6), np.full(7, np.nan), np.full(7, np.inf)):
            host = MotionHost()
            with self.subTest(target=target), self.assertRaises(ValueError):
                host.move_joints(target)
            np.testing.assert_array_equal(host.data.ctrl, np.full(9, .123))

    def test_invalid_motion_limits_do_not_send_commands(self):
        for options in ({'velocity': 0}, {'acceleration': -1}, {'settle_seconds': -1},
                        {'velocity': np.nan}, {'acceleration': np.inf}):
            host = MotionHost()
            with self.subTest(options=options), self.assertRaises(ValueError):
                host.move_joints(np.ones(7), **options)
            self.assertEqual(len(host.commands), 1)


if __name__ == '__main__':
    unittest.main()
