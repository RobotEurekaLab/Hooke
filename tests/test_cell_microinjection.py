"""Cell-phantom gates, actual motor reactions, volume accounting and units."""

import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.cell_scene import CELL_CENTER, NEEDLE_BACK
from microscopy.cell_tasks import CellInjection
from microscopy.worker import apply_request


class CellInjectionTests(unittest.TestCase):
    def task(self, seed=0):
        task = CellInjection.Expert(CellInjection.load())
        task.reset(seed)
        return task

    def approach(self, task, depth):
        # Test setup uses truth; the expert itself only observes its image.
        centre = task.mechanics.image_state(task.data)["centres_m"][0]
        task.move("injector", np.asarray(centre)+NEEDLE_BACK*(task.mechanics.ray_radius_m-depth), .5)

    def test_complete_image_guided_injection_across_three_layouts(self):
        for seed in (0, 1, 4):
            with self.subTest(seed=seed):
                task = self.task(seed)
                task.execute()
                self.assertTrue(task.check(), task.microscopy_report())
                self.assertFalse(np.any(task.data.warning.number))
                self.assertAlmostEqual(task.mechanics.ledger.state("cell").volume_m3*1e15, .5)
                self.assertGreater(task.mechanics.puncture_time_s, 0)
                self.assertGreater(task.mechanics.withdrawal_time_s, task.mechanics.puncture_time_s)
                self.assertEqual(task.model.neq, 0)
                json.dumps(task.microscopy_report(), allow_nan=False)

    def test_pressure_away_from_cell_never_becomes_intracellular_delivery(self):
        task = self.task()
        self.assertFalse(task.check())
        task.data.userdata[0] = 5000
        task.wait(.2)
        self.assertEqual(task.mechanics.ledger.state("cell").volume_m3, 0)
        self.assertGreater(task.mechanics.ledger.state("environment").volume_m3, 0)
        self.assertFalse(task.mechanics.punctured)
        self.assertLess(abs(task.mechanics.ledger.total_m3-task.mechanics.ledger.initial_total_m3), 2e-25)

    def test_subthreshold_indentation_does_not_unlock_injection(self):
        task = self.task()
        self.approach(task, .4e-6)
        task.data.userdata[0] = 5000
        task.wait(.3)
        self.assertGreater(task.mechanics.maximum_indentation_m, .2e-6)
        self.assertFalse(task.mechanics.punctured)
        self.assertEqual(task.mechanics.ledger.state("cell").volume_m3, 0)
        reaction = task.data.qfrc_applied[task.mechanics.axis_dofs]
        self.assertGreater(np.linalg.norm(reaction), 0)
        # Motor forces are in their own axes; the CAD mount rotates those axes.
        # Check action/reaction through independent world-space site Jacobians.
        tip_jacobian = np.zeros((3, task.model.nv))
        cell_jacobian = np.zeros_like(tip_jacobian)
        mujoco.mj_jacSite(task.model, task.data, tip_jacobian, None, task.mechanics.tip)
        mujoco.mj_jacSite(task.model, task.data, cell_jacobian, None, task.mechanics.centre)
        world_reaction = np.linalg.solve(tip_jacobian[:, task.mechanics.axis_dofs].T, reaction)
        expected = -cell_jacobian[:, task.mechanics.stage_dofs].T @ world_reaction
        np.testing.assert_allclose(task.data.qfrc_applied[task.mechanics.stage_dofs], expected, atol=1e-16)

    def test_puncture_latch_alone_cannot_inject_outside_cell(self):
        task = self.task()
        self.approach(task, 2.5e-6)
        self.assertTrue(task.mechanics.punctured)
        self.approach(task, -20e-6)
        self.assertTrue(task.mechanics.withdrawn)
        before = task.mechanics.ledger.state("cell").volume_m3
        task.data.userdata[0] = 5000
        task.wait(.2)
        self.assertEqual(task.mechanics.ledger.state("cell").volume_m3, before)

    def test_overdepth_is_a_failure_even_if_other_flags_are_set(self):
        task = self.task()
        self.approach(task, task.mechanics.parameters.maximum_depth_m+.2e-6)
        self.assertTrue(task.mechanics.overdepth)
        self.assertTrue(task.mechanics.nuclear_contact)
        task.completed = task.focused = True
        self.assertFalse(task.microscopy_checks()["no_overdepth"])
        self.assertFalse(task.check())

    def test_clogged_needle_has_pressure_but_no_fluid_transfer(self):
        task = self.task()
        self.approach(task, 2.5e-6)
        self.assertTrue(task.mechanics.punctured)
        task.mechanics.needle_clogged = True
        task.data.userdata[0] = 5000
        task.wait(.3)
        self.assertGreater(task.mechanics.actual_pressure_pa, 4000)
        self.assertEqual(task.mechanics.ledger.state("cell").volume_m3, 0)
        self.assertEqual(task.mechanics.ledger.state("environment").volume_m3, 0)
        self.assertFalse(task.microscopy_checks()["intracellular_dose"])

    def test_cell_volume_units_and_batch_validation_are_atomic(self):
        task = self.task()
        for body in ({"target_nl": .5}, {"target_pl": 100}, {"target_pl": float("nan")},
                     {"target_pl": .5, "actuators": {"injector_x": .1}}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                apply_request(task, dict(command="step", pressure_pa=5000, **body))
            self.assertEqual(task.data.time, 0)
            self.assertEqual(task.data.userdata[0], 0)
        state = task.public_state()
        self.assertEqual(state["volume_unit"], "pL")
        self.assertNotIn("centres_m", state)
        self.assertNotIn("delivered_nl", state)
        self.assertAlmostEqual(state["calibration"]["pixel_size_m"], 160e-6/768)


if __name__ == "__main__":
    unittest.main()
