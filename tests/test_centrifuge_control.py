"""Measured speed, safety interlocks and torque limits govern the programme."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from centrifuge_control import SpinController, SpinProgram


class CentrifugeControl(unittest.TestCase):
    def test_complete_requires_physical_speed_hold_and_standstill(self):
        controller = SpinController()
        controller.start(0.0, True, True, 0.0)
        speed = 0.0
        for _ in range(10000):
            torque = controller.update(0.002, speed, True, True, 0.0)
            self.assertLessEqual(abs(torque), 0.1)
            # Independent inertial response, not the controller's target.
            speed += 0.002 * (torque - 0.0001 * speed) / 0.0033
            if controller.state == "COMPLETE":
                break
        self.assertEqual(controller.state, "COMPLETE")
        self.assertGreater(controller.rotation_rad, 6.283)
        self.assertGreaterEqual(controller.hold_s, 1.0)
        self.assertTrue(controller.can_unlock(speed))

    def test_nonmoving_motor_never_accrues_hold_or_success(self):
        controller = SpinController()
        controller.start(0.0, True, True, 0.0)
        for _ in range(10000):
            controller.update(0.002, 0.0, True, True, 0.0)
        self.assertEqual(controller.state, "ACCELERATING")
        self.assertEqual(controller.hold_s, 0.0)
        self.assertFalse(controller.can_unlock(0.0))

    def test_unsafe_start_or_unlock_are_rejected(self):
        for speed, closed, locked, balance in [
            (0.0, False, True, 0.0),
            (0.0, True, False, 0.0),
            (0.0, True, True, 0.06),
            (0.1, True, True, 0.0),
        ]:
            with self.subTest(speed=speed, balance=balance), self.assertRaises(
                RuntimeError
            ):
                SpinController().start(speed, closed, locked, balance)
        controller = SpinController()
        controller.start(0.0, True, True, 0.0)
        controller.update(0.002, 1.0, True, False, 0.0)
        self.assertEqual(controller.fault, "lid_unlocked")
        self.assertEqual(controller.state, "BRAKING")
        self.assertFalse(controller.can_unlock(1.0))
        for _ in range(1000):
            controller.update(0.002, 0.0, True, False, 0.0)
        self.assertEqual(controller.state, "FAULT")
        self.assertTrue(controller.can_unlock(0.0))

    def test_interrupted_hold_and_restarted_motion_do_not_reuse_old_dwell(self):
        controller = SpinController()
        controller.start(0.0, True, True, 0.0)
        for _ in range(10):
            for _ in range(250):
                controller.update(0.002, 6.283185307, True, True, 0.0)
            controller.update(0.002, 0.0, True, True, 0.0)
        self.assertEqual(controller.state, "HOLDING")
        self.assertEqual(controller.hold_s, 0.0)
        controller.state = "COMPLETE"
        controller.stopped_s = 0.5
        controller.update(0.002, 1.0, True, True, 0.0)
        controller.update(0.002, 0.0, True, True, 0.0)
        self.assertFalse(controller.can_unlock(0.0))
        for _ in range(250):
            controller.update(0.002, 0.0, True, True, 0.0)
        self.assertTrue(controller.can_unlock(0.0))

    def test_invalid_program_or_sensor_is_rejected(self):
        with self.assertRaises(ValueError):
            SpinProgram(rpm=float("nan"))
        with self.assertRaises(ValueError):
            SpinController().start(float("nan"), True, True, 0.0)


if __name__ == "__main__":
    unittest.main()
