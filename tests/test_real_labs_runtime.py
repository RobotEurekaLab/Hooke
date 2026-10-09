"""Observed movement, control bounds and sample gates for real-lab mechanisms."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.runtime import InstrumentSession


MODEL = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <site name="sample_target" pos="0.5 0 0" size="0.005"/>
    <body name="door">
      <joint name="door_joint" type="slide" axis="1 0 0" range="0 0.2" damping="2"/>
      <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
    </body>
    <body name="sample" pos="0.5 0 0">
      <freejoint/>
      <geom type="sphere" size="0.003" mass="0.001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="door_servo" joint="door_joint" kp="50" ctrlrange="0 0.2"/>
  </actuator>
</mujoco>
"""
MANIFEST = {
    "id": "test_lab",
    "equipment": [
        {
            "id": "reader",
            "kind": "reader",
            "joints": {"door": "door_joint"},
            "actuators": {"door": "door_servo"},
            "process": {
                "access_control": "door",
                "sample_site": "sample_target",
                "sample_body": "sample",
            },
        }
    ],
}


class InstrumentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL)
        self.session = InstrumentSession(self.model, deepcopy(MANIFEST))

    def test_command_moves_physical_joint_and_does_not_teleport_sample(self):
        sample_before = self.session.data.qpos[1:].copy()
        result = self.session.command("reader", "door", 0.1)
        self.assertGreater(result["measured"], 0.09)
        self.assertAlmostEqual(result["measured"], self.session.data.qpos[0])
        self.assertLess(abs(result["error"]), 0.001)
        self.assertTrue((sample_before == self.session.data.qpos[1:]).all())

    def test_invalid_targets_and_durations_leave_state_untouched(self):
        for target in [-0.1, 0.3, float("nan"), float("inf")]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.session.command("reader", "door", target)
        for duration in [0, -1, 31, float("nan")]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.session.command("reader", "door", 0.1, duration=duration)
        self.assertEqual(self.session.data.time, 0)
        self.assertEqual(self.session.data.ctrl[0], 0)

    def test_smoke_tracks_measured_motion_and_resets(self):
        report = self.session.smoke_test()
        self.assertTrue(report["passed"])
        self.assertEqual(report["control_count"], 1)
        self.assertGreater(report["controls"][0]["measured"], 0.09)
        self.assertEqual(self.session.data.time, 0)
        self.assertEqual(self.session.data.qpos[0], 0)

    def test_stalled_actuator_does_not_count_as_success(self):
        # Remove the motor's force after construction, keeping requested ctrl.
        # This independently represents power loss despite a valid command.
        self.model.actuator_gainprm[0, 0] = 0
        self.model.actuator_biasprm[0, :] = 0
        report = self.session.smoke_test()
        self.assertFalse(report["passed"])
        self.assertEqual(report["controls"][0]["measured"], 0)

    def test_velocity_servo_reports_velocity_and_distinct_joint_position(self):
        xml = MODEL.replace(
            'range="0 0.2" damping="2"', 'limited="false" damping="0.1"'
        )
        xml = xml.replace(
            '<position name="door_servo" joint="door_joint" kp="50" ctrlrange="0 0.2"/>',
            '<velocity name="door_servo" joint="door_joint" kv="10" ctrlrange="-1 1"/>',
        )
        manifest = deepcopy(MANIFEST)
        manifest["equipment"][0]["actuator_modes"] = {"door": "velocity"}
        del manifest["equipment"][0]["process"]
        session = InstrumentSession(mujoco.MjModel.from_xml_string(xml), manifest)
        result = session.command("reader", "door", 0.1)
        self.assertAlmostEqual(result["measured"], 0.1, delta=0.002)
        self.assertAlmostEqual(result["joint_position"], 0.2, delta=0.01)
        self.assertEqual(result["mode"], "velocity")
        self.assertEqual(result["units"], "m/s")
        session.reset()
        self.assertEqual(session.data.ctrl[0], 0)

    def test_process_gates_use_observed_door_not_its_target(self):
        self.session.command("reader", "door", 0.1)
        self.session.data.ctrl[0] = 0  # A close request is not a closed door.
        with self.assertRaisesRegex(RuntimeError, "access_open"):
            self.session.measure("reader")
        self.session.command("reader", "door", 0)
        result = self.session.measure("reader")
        self.assertEqual(result["measurement_kind"], "synthetic_geometry_proxy")
        self.assertFalse(result["robot_task_completed"])

    def test_sample_must_be_observed_at_target_and_stationary(self):
        # Alter initial physical state externally; the runtime has no load/teleport API.
        self.session.data.qpos[1] += 0.1
        with self.assertRaisesRegex(RuntimeError, "sample_not_at_target"):
            self.session.measure("reader")
        self.session.reset()
        self.session.data.qvel[1] = 0.2
        with self.assertRaisesRegex(RuntimeError, "sample_moving"):
            self.session.measure("reader")

    def test_static_sample_or_mechanism_only_does_not_enable_measurement(self):
        manifest = deepcopy(MANIFEST)
        manifest["equipment"][0]["process"]["sample_body"] = "door"
        session = InstrumentSession(self.model, manifest)
        with self.assertRaisesRegex(RuntimeError, "sample_is_not_free_body"):
            session.measure("reader")
        del manifest["equipment"][0]["process"]
        session = InstrumentSession(self.model, manifest)
        with self.assertRaisesRegex(RuntimeError, "no_process_model"):
            session.measure("reader")

    def test_invalid_observed_state_cannot_become_ready(self):
        self.session.data.qpos[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite_physics_state"):
            self.session.measure("reader")

    def test_invalid_manifest_and_motor_controls_fail_early(self):
        manifest = deepcopy(MANIFEST)
        manifest["equipment"].append(deepcopy(manifest["equipment"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            InstrumentSession(self.model, manifest)
        manifest = deepcopy(MANIFEST)
        manifest["equipment"][0]["joints"]["door"] = "absent"
        with self.assertRaisesRegex(ValueError, "Unknown model object"):
            InstrumentSession(self.model, manifest)
        self.model.actuator_biasprm[0, 1] = 0
        with self.assertRaisesRegex(ValueError, "position servo"):
            InstrumentSession(self.model, deepcopy(MANIFEST))


if __name__ == "__main__":
    unittest.main()
