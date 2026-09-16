"""Motor and holders must not inherit unrelated model defaults."""

from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from archetypes.task_catalog import CATALOG
from backends.centrifuge_assessment import CentrifugeCycleSequence
from backends.assessment import LidClosureSequence
from centrifuge_control import SpinProgram


class CentrifugeCycleTests(unittest.TestCase):
    def test_original_assets_compile_with_inactive_motor_and_mounts(self):
        task = CATALOG["centrifuge_5430_cycle"].make_expert()
        task.reset(0)
        p = task.rotor_program
        self.assertEqual(
            task.model.actuator_gaintype[p.motor_id], mujoco.mjtGain.mjGAIN_FIXED
        )
        self.assertEqual(
            task.model.actuator_biastype[p.motor_id], mujoco.mjtBias.mjBIAS_NONE
        )
        np.testing.assert_array_equal(task.model.eq_data[p.mount_ids, :3], 0.0)
        np.testing.assert_array_equal(task.data.eq_active[p.mount_ids], False)
        np.testing.assert_array_equal(
            task.model.eq_obj2id[p.mount_ids], [task.tube.body_id, task.tube2.body_id]
        )
        self.assertFalse(task.check())
        with self.assertRaises(RuntimeError):
            p.start(task.data)

    def test_observer_rejects_static_and_early_unlock(self):
        p = SpinProgram()
        static = CentrifugeCycleSequence()
        for _ in range(1000):
            static.update(0.002, 0.0, 0.0, True, True, True, p)
        self.assertFalse(all(static.checks().values()))
        state = CentrifugeCycleSequence()
        for _ in range(300):
            state.update(0.002, 0.0, 0.0, True, True, True, p)
        for _ in range(600):
            state.update(0.002, 6.283185307, 0.0, True, True, True, p)
        state.update(0.002, 6.0, 0.0, True, False, True, p)
        self.assertTrue(state.safety_violation)
        self.assertFalse(state.unlocked)
        self.assertFalse(all(state.checks().values()))

    def test_lid_observer_requires_robot_motion_and_standstill(self):
        passive = LidClosureSequence(1.0, 0.0)
        driven = LidClosureSequence(1.0, 0.0)
        for i in range(100):
            position = 1.0 - (i + 1) / 100
            passive.update(0.01, position, -1.0, False)
            driven.update(0.01, position, -1.0, True)
        for _ in range(10):
            passive.update(0.01, 0.0, 0.0, False)
            driven.update(0.01, 0.0, 0.0, False)
        self.assertFalse(all(passive.checks().values()))
        self.assertTrue(all(driven.checks().values()))
        driven.update(0.01, 0.0, 0.1, False)
        self.assertFalse(driven.checks()["lid_standstill_50ms"])


if __name__ == "__main__":
    unittest.main()
