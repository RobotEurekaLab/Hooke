"""Parallel closure and independent upstream float32 reference poses."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.parallel_kinematics import closure_residuals, forward, inverse, jacobian


class ParallelKinematicsTests(unittest.TestCase):
    def test_upstream_cpp_reference_poses(self):
        # Unmodified v4 C++ compiled on a host; millimetres converted to metres.
        # The tolerance covers float32 arithmetic, not instrument accuracy.
        angles = ((45, 45, 45), (30, 40, 50), (55, 33, 48), (20, 20, 20), (60, 60, 60))
        expected = ((.000737329483, .0007373371124, .000737323761),
                    (-.003186508179, -.0006608829498, .001980783463),
                    (.003290353775, -.002478712082, .001407003403),
                    (-.006088117599, -.006088132858, -.006088140488),
                    (.004293203354, .004293201447, .004293197632))
        for q, p in zip(angles, expected):
            with self.subTest(angles=q):
                np.testing.assert_allclose(forward(np.deg2rad(q)), p, rtol=0, atol=5e-8)

    def test_workspace_roundtrips_and_actual_rod_closure(self):
        points = np.random.default_rng(0).uniform(-.004, .004, (1000, 3))
        for point in points:
            angles = inverse(point)
            np.testing.assert_allclose(forward(angles), point, rtol=0, atol=2e-12)
            np.testing.assert_allclose(closure_residuals(angles, point), 0, rtol=0, atol=2e-12)
        for bad in ([1, 1, 1], [0, 0], [0, np.nan, 0], [0, 0, np.inf]):
            with self.subTest(position=bad), self.assertRaises(ValueError):
                inverse(bad)

    def test_differentiated_constraints_match_observed_motion(self):
        for point in ([0, 0, 0], [.002, -.001, .003]):
            angles = inverse(point)
            h = 1e-6
            observed = np.column_stack([(forward(angles+h*axis)-forward(angles-h*axis))/(2*h)
                                        for axis in np.eye(3)])
            np.testing.assert_allclose(jacobian(angles), observed, rtol=0, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
