"""Zero Cartesian movement must hold; actual translation/rotation must move."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from topp import Topp
from kinematics import Pose


class ZeroPathTests(unittest.TestCase):
    def test_identical_pose_and_quaternion_sign_do_not_accumulate_ik_drift(self):
        calls = []
        def drifting_ik(pos, quat):
            calls.append(1)
            return np.array([.2 + len(calls) * 1e-5])
        planner = Topp(1, 1., 1., drifting_ik)
        path = [Pose(np.zeros(3), np.array([sign, 0., 0., 0.])) for sign in [1., -1., 1.]]
        trajectory = planner.jnt_traj(path)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trajectory.duration, 0.)
        np.testing.assert_array_equal(trajectory.evald([0., .1]), [[0.], [0.]])
        np.testing.assert_array_equal(trajectory.evaldd(0.), [0.])
        np.testing.assert_allclose(planner.query(trajectory, 10.), [.20001])
        np.testing.assert_allclose(trajectory.eval([0., .1]), [[.20001], [.20001]])
        trajectory.eval(0.)[0] = 99.
        np.testing.assert_allclose(trajectory.eval(0.), [.20001])

    def test_real_translation_is_not_silently_replaced_by_a_hold(self):
        planner = Topp(1, 1., 1., lambda pos, quat: np.array([pos[0]]))
        path = [Pose(np.array([x, 0., 0.]), np.array([1., 0., 0., 0.])) for x in [0., .05, .1]]
        trajectory = planner.jnt_traj(path)
        self.assertGreater(trajectory.duration, 0.)
        np.testing.assert_allclose(trajectory.eval(trajectory.duration), [.1], atol=1e-8)
        times = np.linspace(0, trajectory.duration, 100)
        self.assertLessEqual(np.max(np.abs(trajectory.evald(times))), 1. + 1e-7)

    def test_pure_rotation_is_not_mistaken_for_zero_movement(self):
        planner = Topp(1, 1., 1., lambda pos, quat: np.array([2*np.arctan2(quat[3], quat[0])]))
        path = [Pose(np.zeros(3), np.array([np.cos(a/2), 0., 0., np.sin(a/2)])) for a in [0., .1, .2]]
        trajectory = planner.jnt_traj(path)
        self.assertGreater(trajectory.duration, 0.)
        np.testing.assert_allclose(trajectory.eval(trajectory.duration), [.2], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
