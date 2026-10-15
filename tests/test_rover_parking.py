"""Bounded wheel commands hold a parked pose without modifying physical state."""
import math
import unittest

import numpy as np

from real_labs.rover_parking import parking_twist


def twist(position=(0, 0, 0), heading=0, velocity=(0, 0, 0), yaw_rate=0,
          anchor=(0, 0, 0), anchor_heading=0):
    return parking_twist(position, heading, velocity, yaw_rate, anchor, anchor_heading)


class RoverParkingTests(unittest.TestCase):
    def test_rest_and_small_heading_noise_have_zero_command(self):
        self.assertEqual(twist(), (0, 0))
        self.assertEqual(twist(heading=.001, yaw_rate=.005)[1], 0)

    def test_longitudinal_error_and_velocity_have_restoring_signs(self):
        self.assertLess(twist(position=(.001, 0, 0))[0], 0)
        self.assertGreater(twist(position=(-.001, 0, 0))[0], 0)
        self.assertLess(twist(velocity=(.001, 0, 0))[0], 0)

    def test_heading_and_yaw_rate_have_restoring_signs(self):
        self.assertLess(twist(heading=.003)[1], 0)
        self.assertGreater(twist(heading=-.003)[1], 0)
        self.assertLess(twist(yaw_rate=.02)[1], 0)

    def test_rotation_invariant_longitudinal_control(self):
        expected = twist(position=(.001, 0, 0), velocity=(.001, 0, 0))
        actual = twist(position=(0, .001, 0), heading=math.pi / 2,
                       velocity=(0, .001, 0), anchor_heading=math.pi / 2)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_heading_wrap_uses_shortest_rotation(self):
        self.assertAlmostEqual(twist(heading=math.pi - .0011,
                                     anchor_heading=-math.pi + .0011)[1], .0008)

    def test_bounded_commands_do_not_mutate_observed_position(self):
        position = np.array([100., 100., 100.])
        before = position.copy()
        velocity, yaw = twist(position=position, heading=2, yaw_rate=100)
        self.assertLessEqual(abs(velocity), .02)
        self.assertLessEqual(abs(yaw), .05)
        np.testing.assert_array_equal(position, before)

    def test_lateral_error_does_not_create_unavailable_crab_motion(self):
        self.assertEqual(twist(position=(0, .001, 0)), (0, 0))

    def test_invalid_observation_rejected(self):
        for field, value in [('position', (0, 0)), ('velocity', (0, math.nan, 0)),
                             ('yaw_rate', math.inf)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                twist(**{field: value})


if __name__ == '__main__':
    unittest.main()
