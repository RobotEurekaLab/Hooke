"""Failed verification must not expose stale efforts or reusable native state."""

import ast
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

import test_native_substeps as boundary


def additional_methods():
    tree = ast.parse(boundary.RUNTIME_PATH.read_text())
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "NativeScene")
    methods = [node for node in scene.body if isinstance(node, ast.FunctionDef)
               and node.name in ("observe", "render_frame", "_configure_drives")]
    class Attribute:
        def Set(self, value): pass
    class Drive:
        def __init__(self, *args): pass
        @staticmethod
        def Apply(*args): return Drive()
        def __getattr__(self, name): return lambda *args: Attribute()
    namespace = {"np": np, "UsdPhysics": SimpleNamespace(DriveAPI=Drive)}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(boundary.RUNTIME_PATH), "exec"), namespace)
    return {node.name: namespace[node.name] for node in methods}


def fixture(explicit=True):
    runtime, submitted, gravity = boundary.fixture(1, explicit=explicit)
    # Give the real production observer access to the fake world's state. The
    # original closure reads that state without stepping or manufacturing effort.
    original_observe = runtime.observe
    item = runtime.views[0]
    item["body_ids"] = []
    item["view"].get_dof_positions = lambda: np.asarray([original_observe()["qpos"]])
    item["view"].get_dof_velocities = lambda: np.asarray([original_observe()["qvel"]])
    item["view"].get_link_transforms = lambda: np.empty((1, 0, 7))
    runtime.m.update(eq_type=np.array([], int), eq_obj1id=np.array([], int),
                     eq_obj2id=np.array([], int))
    for name, method in additional_methods().items():
        setattr(runtime, name, method.__get__(runtime))
    runtime.observe()
    return runtime, submitted, gravity


class NativeEffortBiasTests(unittest.TestCase):
    def test_none_bias_matches_compiled_mujoco_joint_actuator(self):
        runtime, submitted, _ = fixture()
        runtime.m["actuator_biastype"][0] = 0
        result = runtime.step([7.])
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint name="j" type="slide"/><geom size=".1" mass="1"/>
          </body></worldbody><actuator><general joint="j" biastype="none"
          biasprm="0 2 3" gainprm="1"/></actuator></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[0] = 1.
        data.ctrl[0] = 7.
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(result["actuator_effort"]["last"], data.actuator_force)
        np.testing.assert_array_equal(submitted, [7.])

    def test_none_bias_is_ignored_in_legacy_explicit_tendon_force(self):
        runtime, submitted, _ = fixture(explicit=False)
        runtime.force_actuators = {}
        runtime.m["actuator_biastype"][0] = 0
        runtime.m["actuator_trntype"][0] = 3
        runtime.m.update(tendon_adr=np.array([0]), tendon_num=np.array([1]),
                         wrap_type=np.array([1]), wrap_objid=np.array([0]), wrap_prm=np.array([1.]))
        runtime.step([7.])
        np.testing.assert_array_equal(submitted, [7.])

    def test_ignored_bias_does_not_create_an_implicit_position_drive(self):
        runtime, _, _ = fixture(explicit=False)
        runtime.explicit_affine_actuators = False
        runtime.bridge.joint_paths = {0: "/Joint"}
        runtime.world.stage = SimpleNamespace(GetPrimAtPath=lambda path: object())
        runtime.m.update(jnt_stiffness=np.array([0.]), dof_damping=np.array([0.]))
        runtime.m["actuator_biastype"][0] = 0
        runtime.m["actuator_biasprm"][0] = [100., -10., -20.]
        runtime._configure_drives()
        self.assertEqual(runtime.force_actuators, {0: 0})
        self.assertEqual(runtime.actuators, {})


class NativeEffortFailureTests(unittest.TestCase):
    def test_invalid_force_rejected_before_native_writes_or_step_then_reset_recovers(self):
        for gain, control in ((1., 1e100), (1e300, 1e300)):
            with self.subTest(gain=gain, control=control):
                runtime, submitted, _ = fixture()
                runtime.reset()
                runtime.m["actuator_gainprm"][0, 0] = gain
                view = runtime.views[0]["view"]
                writes = []
                view.set_dof_position_targets = lambda *args: writes.append("position")
                view.set_dof_velocity_targets = lambda *args: writes.append("velocity")
                before = len(submitted)
                with np.errstate(over="ignore", invalid="ignore"):
                    with self.assertRaisesRegex(ValueError, "force inputs.*float32 range"):
                        runtime.step([control])
                self.assertEqual(writes, [])
                self.assertEqual(len(submitted), before)
                self.assertEqual((runtime.steps, runtime.native_steps, runtime.time), (0, 0, 0.))
                with self.assertRaisesRegex(RuntimeError, "reset required"):
                    runtime.observe()
                runtime.m["actuator_gainprm"][0, 0] = 1.
                runtime.reset()
                runtime.step([7.])
                self.assertEqual(runtime.native_steps, 1)

    def test_invalid_position_and_velocity_targets_rejected_before_native_writes(self):
        for bias in ([0., -1., -1.], [0., 0., -1.]):
            with self.subTest(bias=bias):
                runtime, submitted, _ = fixture(explicit=False)
                runtime.reset()
                runtime.force_actuators = {}
                runtime.actuators = {0: 0}
                runtime.m["actuator_biasprm"][0] = bias
                runtime.m.update(jnt_stiffness=np.array([0.]), dof_damping=np.array([0.]))
                view = runtime.views[0]["view"]
                writes = []
                view.set_dof_position_targets = lambda *args: writes.append("position")
                view.set_dof_velocity_targets = lambda *args: writes.append("velocity")
                before = len(submitted)
                with np.errstate(over="ignore", invalid="ignore"):
                    with self.assertRaisesRegex(ValueError, "position and velocity targets must be finite"):
                        runtime.step([1e100])
                self.assertEqual(writes, [])
                self.assertEqual(len(submitted), before)
                self.assertEqual((runtime.steps, runtime.native_steps, runtime.time), (0, 0, 0.))
                with self.assertRaisesRegex(RuntimeError, "reset required"):
                    runtime.observe()
                runtime.reset()
                runtime.step([7.])
                self.assertEqual(runtime.native_steps, 1)

    def test_successful_reset_restores_disabled_effort_reason(self):
        runtime, _, _ = fixture(explicit=False)
        result = runtime.reset()
        self.assertFalse(result["actuator_effort"]["available"])
        self.assertEqual(result["actuator_effort"]["reason"],
                         "Implicit drive effort is not independently observable")
        runtime.step([7.])
        self.assertEqual(runtime.observe()["actuator_effort"], result["actuator_effort"])

    def test_begin_interval_invalidates_previous_verified_effort(self):
        runtime, _, _ = fixture()
        runtime.step([7.])
        self.assertTrue(runtime.effort_recorder.state["available"])
        runtime.effort_recorder.begin_interval()
        self.assertFalse(runtime.effort_recorder.state["available"])

    def test_poststep_readback_failure_counts_step_and_blocks_reuse_until_reset(self):
        runtime, submitted, _ = fixture()
        runtime.step([7.])
        view = runtime.views[0]["view"]
        view.get_dof_actuation_forces = lambda: np.zeros((1, 1))
        with self.assertRaisesRegex(RuntimeError, "does not match submission"):
            runtime.step([11.])
        self.assertEqual(runtime.native_steps, 2)
        self.assertEqual(runtime.steps, 1)
        self.assertEqual(runtime.time, .004)
        self.assertFalse(runtime.effort_recorder.state["available"])
        before = len(submitted)
        for operation in (runtime.observe, lambda: runtime.step([3.]), lambda: runtime.render_frame({})):
            with self.assertRaisesRegex(RuntimeError, "reset required"):
                operation()
        self.assertEqual(len(submitted), before)
        view.get_dof_actuation_forces = lambda: np.array([[submitted[-1]]], dtype=np.float32)
        reset = runtime.reset()
        self.assertEqual(reset["time"], 0.)
        self.assertEqual(reset["actuator_effort"]["last"], [0.])
        self.assertEqual(reset["actuator_effort"]["sample_count"], 0)
        self.assertTrue(reset["actuator_effort"]["native_input_verified"])
        runtime.step([3.])
        self.assertEqual((runtime.steps, runtime.native_steps, runtime.time), (1, 1, .002))

    def test_failed_reset_keeps_scene_invalid_and_effort_unavailable(self):
        runtime, submitted, _ = fixture()
        runtime.step([7.])
        view = runtime.views[0]["view"]
        view.get_dof_actuation_forces = lambda: np.ones((1, 1))
        with self.assertRaisesRegex(RuntimeError, "does not match submission"):
            runtime.reset()
        self.assertFalse(runtime.effort_recorder.state["available"])
        with self.assertRaisesRegex(RuntimeError, "reset required"):
            runtime.observe()
        view.get_dof_actuation_forces = lambda: np.array([[submitted[-1]]], dtype=np.float32)
        state = runtime.reset()
        self.assertTrue(state["actuator_effort"]["available"])
        self.assertEqual(state["actuator_effort"]["last"], [0.])

    def test_rejected_control_before_native_step_does_not_poison_good_state(self):
        runtime, _, _ = fixture()
        runtime.step([7.])
        with self.assertRaises(ValueError):
            runtime.step([np.nan])
        self.assertTrue(runtime.observe()["actuator_effort"]["available"])
        self.assertEqual((runtime.steps, runtime.native_steps, runtime.time), (1, 1, .002))


if __name__ == "__main__":
    unittest.main()
