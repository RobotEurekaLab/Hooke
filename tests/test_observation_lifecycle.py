"""An explicit source reset is required after native observation ownership ends."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from backends.closed_loop import PhysXTaskAdapter
from real_labs.runtime import InstrumentSession


def session():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><body><joint name="axis" type="slide" range="-.2 .2"/>
      <geom type="sphere" size=".02" mass="1"/></body></worldbody>
      <actuator><position name="drive" joint="axis" kp="10" ctrlrange="-.2 .2"/></actuator>
    </mujoco>''')
    return InstrumentSession(model, {"equipment": [{
        "id": "fixture", "joints": {"axis": "axis"}, "actuators": {"axis": "drive"}
    }]})


def adapter_for(target):
    adapter = PhysXTaskAdapter.__new__(PhysXTaskAdapter)
    adapter.task = SimpleNamespace(model=target.model, data=target.data)
    adapter.observation_target = target
    adapter.max_fk_position_error = adapter.max_fk_rotation_error = 0.0
    adapter.apply_state({
        "qpos": [.04], "qvel": [0.0], "time": .002, "contacts": [],
        "actuator_effort": {
            "available": True, "quantity": "applied_explicit_actuator_input",
            "native_input_verified": True, "last": [.7], "mean": [.7],
            "peak_abs": [.8], "signed_peak": [.8], "sample_count": 1,
            "interval_s": .002,
        },
    })
    return adapter


class ObservationLifecycleTests(unittest.TestCase):
    def test_active_owner_rejects_source_reset_before_mutation(self):
        target = session()
        adapter = adapter_for(target)
        before = (target.data.qpos.copy(), target.data.qvel.copy(), target.data.ctrl.copy(), target.data.time)
        with adapter:
            with self.assertRaisesRegex(RuntimeError, "owning physics backend"):
                target.reset()
            for name, expected in zip(("qpos", "qvel", "ctrl", "time"), before):
                np.testing.assert_array_equal(getattr(target.data, name), expected)
            self.assertIs(target._physics_context_owner, adapter)

    def test_exit_invalidates_peaks_and_explicit_reset_restores_source(self):
        target = session()
        adapter = adapter_for(target)
        native_force = target.data.actuator_force.copy()
        with adapter:
            self.assertTrue(target.physics_observations["actuator_force"])
            self.assertEqual(target.physics_observations["actuator_force_peak_abs"], [.8])
        self.assertFalse(target.physics_observations["actuator_force"])
        self.assertNotIn("actuator_force_peak_abs", target.physics_observations)
        self.assertFalse(hasattr(target, "_physics_context_owner"))
        np.testing.assert_array_equal(target.data.actuator_force, native_force)
        target.reset()
        self.assertTrue(target.physics_observations["actuator_force"])
        self.assertNotIn("actuator_force_peak_abs", target.physics_observations)
        self.assertEqual(target.data.time, 0)
        self.assertTrue(np.isfinite(target.data.actuator_force).all())
        self.assertFalse(np.array_equal(target.data.actuator_force, native_force))

    def test_exception_exit_releases_owner_and_restores_original_functions(self):
        target = session()
        adapter = adapter_for(target)
        original_step, original_forward = mujoco.mj_step, mujoco.mj_forward
        with self.assertRaisesRegex(RuntimeError, "task failure"):
            with adapter:
                raise RuntimeError("task failure")
        self.assertIs(mujoco.mj_step, original_step)
        self.assertIs(mujoco.mj_forward, original_forward)
        self.assertFalse(target.physics_observations["actuator_force"])
        self.assertFalse(hasattr(target, "_physics_context_owner"))

    def test_other_session_can_reset_while_native_task_is_owned(self):
        target, other = session(), session()
        adapter = adapter_for(target)
        other.data.qpos[:] = .1
        with adapter:
            other.reset()
            self.assertEqual(other.data.qpos[0], 0)
            self.assertTrue(other.physics_observations["actuator_force"])
            self.assertTrue(target.physics_observations["actuator_force"])

    def test_failed_source_reset_does_not_restore_observation_claim(self):
        target = session()
        with adapter_for(target):
            pass
        with patch.object(mujoco, "mj_forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                target.reset()
        self.assertFalse(target.physics_observations["actuator_force"])
        self.assertNotIn("actuator_force_peak_abs", target.physics_observations)

    def test_second_owner_rejects_before_loading_or_patching_functions(self):
        target = session()
        adapter = adapter_for(target)
        second = PhysXTaskAdapter.__new__(PhysXTaskAdapter)
        second.task, second.observation_target = adapter.task, target
        with adapter:
            original_step = mujoco.mj_step
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                second.__enter__()
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                PhysXTaskAdapter(adapter.task, None, Path("unused"), observation_target=target)
            self.assertIs(mujoco.mj_step, original_step)
            self.assertIs(target._physics_context_owner, adapter)


if __name__ == "__main__":
    unittest.main()
