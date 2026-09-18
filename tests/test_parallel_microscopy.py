"""Physical closure, Cartesian command units and force transmission at the needle."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.control import apply_request
from microscopy.parallel_kinematics import CAD_GEOMETRY, inverse
from microscopy.parallel_scene import build_xml
from microscopy.tasks import MicroscopyExpert, make_task


class FeedbackPolicyTests(unittest.TestCase):
    """Synthetic sensor cases verify the stop policy, not physical accuracy."""

    def controller(self, amplitude):
        data = SimpleNamespace(time=0.)
        sensor = SimpleNamespace(feedback=lambda _: {"injector_x": amplitude * (-1 if round(data.time * 1000) % 2 else 1)},
                                 targets=lambda _: {"injector_x": 0.})
        def step(_):
            data.time += .001
        return SimpleNamespace(data=data, dt=.001, instrument_drives=sensor, motion_feedback=[],
                               wait=lambda _: None, step_and_log=step, command=lambda *_: None)

    def test_mean_feedback_retains_instantaneous_error_and_jitter(self):
        controller = self.controller(2e-8)
        MicroscopyExpert.settle_microposition(controller, {"injector_x": 0.})
        record = controller.motion_feedback[0]
        self.assertEqual(record["iterations"], 0)
        self.assertAlmostEqual(record["final_error_m"], 0.)
        self.assertAlmostEqual(record["samples"][0]["rms_jitter_m"], 2e-8)
        self.assertAlmostEqual(record["samples"][0]["instantaneous_error_m"], 2e-8)

    def test_large_oscillation_cannot_pass_by_cancelling_its_mean(self):
        controller = self.controller(1e-6)
        with self.assertRaisesRegex(RuntimeError, "RMS jitter.*motion stopped"):
            MicroscopyExpert.settle_microposition(controller, {"injector_x": 0.})
        record = controller.motion_feedback[0]
        self.assertEqual(record["iterations"], 10)
        self.assertAlmostEqual(record["final_error_m"], 0.)
        self.assertGreater(record["samples"][-1]["rms_jitter_m"], record["rms_jitter_limit_m"])


class ParallelMotionTests(unittest.TestCase):
    def test_motor_actuation_closes_rods_and_returns_platform_without_state_assignment(self):
        model = mujoco.MjModel.from_xml_string(build_xml())
        data = mujoco.MjData(model)
        for position in ((.001, -.001, .0005), (-.002, .001, .0015), (0., 0., 0.)):
            data.ctrl[:] = inverse(position, CAD_GEOMETRY) - np.deg2rad([42.] * 3)
            mujoco.mj_step(model, data, nstep=2000)
            np.testing.assert_allclose(data.xpos[model.body("parallel_platform").id] - [0., 0., .20], position, atol=1e-6)
            self.assertFalse(any(warning.number for warning in data.warning))


class ParallelWorkstationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.environ.get("HOOKE_TEST_PARALLEL_MESH_ROOT")
        if not cls.root:
            raise unittest.SkipTest("Set HOOKE_TEST_PARALLEL_MESH_ROOT to the licensed, inspected CAD conversion")

    def setUp(self):
        self.settings = patch.dict(os.environ, {
            "HOOKE_MICROSCOPY_ASSETS": "reference", "HOOKE_MICROSCOPY_STAND": "te2000-s-reference",
            "HOOKE_MICROSCOPY_STAGE": "reference", "HOOKE_MICROSCOPY_OPTICS": "estimated",
            "HOOKE_MICROSCOPY_MANIPULATOR": "parallel-v4", "HOOKE_MICROSCOPY_PARALLEL_ROOT": self.root,
        })
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_xyz_commands_remain_metres_and_raw_motor_radians_cannot_modify_pressure(self):
        task = make_task("cell_injection")
        task.reset(0)
        self.assertEqual(task.model.neq, 6)
        self.assertEqual(set(task.instrument_drives.parallel), {"injector"})
        self.assertEqual(task.public_state()["manipulator_profile"], "parallel-v4")
        from microscopy.assembly_audit import geometry_group
        self.assertEqual(geometry_group(task.model, task.model.geom("injector_foot").id), "injector")
        self.assertEqual(geometry_group(task.model, task.model.geom("injector_holder_adapter").id), "injector")
        np.testing.assert_allclose([task.instrument_drives.feedback(task.data)["injector_" + axis] for axis in "xyz"], 0.)
        ctrl, pressure = task.data.ctrl.copy(), task.data.userdata.copy()
        with self.assertRaises(ValueError):
            apply_request(task, dict(command="step", pressure_pa=5000., actuators={"focus": 1e-6, "injector_motor_0": .01}))
        np.testing.assert_array_equal(task.data.ctrl, ctrl)
        np.testing.assert_array_equal(task.data.userdata, pressure)
        task.command({"injector_x": 1e-3, "injector_y": -1e-3, "injector_z": .0005}, .4)
        values = task.instrument_drives.feedback(task.data)
        np.testing.assert_allclose([values["injector_" + axis] for axis in "xyz"], [1e-3, -1e-3, .0005], atol=1e-6)
        self.assertFalse(any(warning.number for warning in task.data.warning))

    def test_needle_load_transmits_translation_and_lever_torque_through_free_platform(self):
        task = make_task("cell_injection")
        task.reset(0)
        force = np.array([20., -30., 40.]) * 1e-9
        task.mechanics.apply_reaction(task.data, force)
        joint = task.model.joint("injector_platform_free")
        start = int(joint.dofadr[0])
        generalized = task.data.qfrc_applied[start:start + 6]
        lever = task.data.site_xpos[task.model.site("injector_tcp").id] - task.data.xpos[task.model.body("injector_platform").id]
        rotation = task.data.xmat[task.model.body("injector_platform").id].reshape(3, 3)
        np.testing.assert_allclose(generalized[:3], force, atol=1e-20)
        np.testing.assert_allclose(rotation @ generalized[3:], np.cross(lever, force), atol=1e-20)
        self.assertGreater(np.linalg.norm(generalized[3:]), 0.)

    def test_subthreshold_approach_settles_without_exciting_the_closed_mechanism(self):
        from microscopy.cell_scene import NEEDLE_BACK
        from microscopy.scene import ORIGIN

        task = make_task("cell_injection")
        task.reset(0)
        center = np.asarray(task.mechanics.image_state(task.data)["centres_m"][0])
        target = center + NEEDLE_BACK * (task.mechanics.ray_radius_m - .4e-6)
        task.move("injector", target, .7)
        tip = task.data.site_xpos[task.model.site("injector_tcp").id]
        np.testing.assert_allclose(tip, ORIGIN + target, atol=1e-7, rtol=0.)
        self.assertGreater(task.mechanics.maximum_indentation_m, .2e-6)
        self.assertLess(task.mechanics.maximum_indentation_m, .8e-6)
        self.assertFalse(task.mechanics.punctured)
        self.assertFalse(any(warning.number for warning in task.data.warning))


if __name__ == "__main__":
    unittest.main()
