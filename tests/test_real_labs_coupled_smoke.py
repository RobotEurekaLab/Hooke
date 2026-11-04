"""Complete authored vectors must pass every measured control, with reset."""

import unittest

import mujoco
import numpy as np

from real_labs.runtime import InstrumentSession


def make(targets=None, *, blocked=False, velocity=False):
    second = (
        '<velocity name="drive_b" joint="b" kv="100" ctrllimited="true" ctrlrange="-.15 .15"/>'
        if velocity
        else '<position name="drive_b" joint="b" kp="500" kv="45" ctrllimited="true" ctrlrange="-.15 .15"/>'
    )
    xml = f"""<mujoco><option gravity="0 0 0" timestep=".002" integrator="implicitfast"/>
    <worldbody><body pos="0 0 .5"><joint name="a" type="slide" axis="1 0 0" range="-.15 .15"/><geom type="box" size=".02 .02 .02" mass="1"/></body>
    <body pos="0 1 .5"><joint name="b" type="slide" axis="1 0 0" range="-.15 .15" stiffness="{1000000 if blocked else 0}"/><geom type="box" size=".02 .02 .02" mass="1"/></body></worldbody>
    <actuator><position name="drive_a" joint="a" kp="500" kv="45" ctrllimited="true" ctrlrange="-.15 .15"/>{second}</actuator></mujoco>"""
    item = dict(
        id="coupled",
        joints={"first": "a", "second": "b"},
        actuators={"first": "drive_a", "second": "drive_b"},
        actuator_modes={
            "first": "position",
            "second": "velocity" if velocity else "position",
        },
    )
    if targets is not None:
        item["coordinated_inspection_targets"] = targets
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(xml), {"id": "synthetic", "equipment": [item]}
    )


def state(s):
    return (
        s.data.time,
        s.data.qpos.tobytes(),
        s.data.qvel.tobytes(),
        s.data.ctrl.tobytes(),
        s.data.qacc_warmstart.tobytes(),
    )


class CoupledSmokeTests(unittest.TestCase):
    def test_simultaneous_vector_measures_both_axes_and_restores_reset(self):
        s = make({"first": 0.06, "second": -0.04})
        report = s.smoke_test(duration=1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["control_count"], 2)
        self.assertEqual(
            {r["control"] for r in report["controls"]}, {"first", "second"}
        )
        for result in report["controls"]:
            self.assertTrue(result["moved"])
            self.assertEqual(result["command_mode"], "simultaneous_position_vector")
            self.assertEqual(
                result["simulated_seconds"], report["controls"][0]["simulated_seconds"]
            )
            self.assertEqual(result["tracking_tolerance"], 0.0075)
            self.assertEqual(
                result["coordinated_targets"], {"first": 0.06, "second": -0.04}
            )
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)
        np.testing.assert_array_equal(s.data.ctrl, [0, 0])
        self.assertEqual(s.data.time, 0)

    def test_one_blocked_axis_invalidates_the_whole_vector(self):
        s = make({"first": 0.06, "second": -0.04}, blocked=True)
        result = s.smoke_test()
        self.assertFalse(result["passed"])
        self.assertTrue(result["controls"][0]["passed"])
        self.assertFalse(result["controls"][1]["passed"])
        self.assertFalse(result["controls"][1]["moved"])
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)

    def test_missing_foreign_nonfinite_out_of_range_or_velocity_targets_reject(self):
        for bad in (
            {"first": 0.04},
            {"first": 0.04, "second": 0.05, "foreign": 0.03},
            {"first": 0.04, "other.second": 0.05},
            {"first": 0.04, "second": float("nan")},
            {"first": 0.04, "second": 0.15001},
            [],
            {},
        ):
            with self.subTest(target=bad), self.assertRaises(ValueError):
                make(bad)
        with self.assertRaises(ValueError):
            make({"first": 0.04, "second": 0.05}, velocity=True)

    def test_changed_metadata_rejects_before_reset_or_actuation(self):
        s = make({"first": 0.06, "second": -0.04})
        s.data.qpos[:] = [0.012, -0.015]
        s.data.ctrl[:] = [0.01, -0.01]
        s.data.time = 0.25
        original = state(s)
        s.equipment["coupled"]["coordinated_inspection_targets"].pop("second")
        with self.assertRaises(ValueError):
            s.smoke_test()
        self.assertEqual(state(s), original)

    def test_no_motion_target_does_not_receive_credit(self):
        s = make({"first": 0.06, "second": 0})
        result = s.smoke_test()
        self.assertFalse(result["passed"])
        self.assertFalse(result["controls"][1]["moved"])

    def test_first_step_physics_reset_warning_is_not_credited(self):
        s = make({"first": 0.06, "second": -0.04})
        # A real excessive acceleration triggers MuJoCo's BADQACC reset on
        # the first step. Its elapsed time can still equal one timestep and
        # the reset state is finite; the native warning must reject the run.
        s.model.actuator_gainprm[0, 0] = 1e30
        warnings = []
        previous = mujoco.get_mju_user_warning()
        try:
            mujoco.set_mju_user_warning(warnings.append)
            with self.assertRaisesRegex(RuntimeError, "Native physics warning"):
                s.smoke_test(duration=s.model.opt.timestep)
        finally:
            mujoco.set_mju_user_warning(previous)
        self.assertTrue(any("QACC" in warning for warning in warnings))
        self.assertEqual(s.data.time, 0)
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)

    def test_absent_metadata_retains_independent_command_path(self):
        a = make()
        results = a.smoke_test()["controls"]
        for result in results:
            b = make()
            actual = b.command("coupled", result["control"], result["target"])
            for key, value in actual.items():
                self.assertEqual(result[key], value)
            self.assertNotIn("coordinated_targets", result)


if __name__ == "__main__":
    unittest.main()
