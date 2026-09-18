"""Cell scale, image feedback and force-based manipulation success/failures."""

import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from archetypes.task_catalog import CATALOG
from microscopy import OPERATIONS, model_family
from microscopy.cell_handling_scene import CELL_CENTER, JAW_CLOSED_M, JAW_OPEN_M, LIFT_M
from microscopy.control import apply_request
from microscopy.scene import ORIGIN
from microscopy.tasks import make_task


class CellManipulation(unittest.TestCase):
    def task(self, operation):
        task = make_task(operation)
        task.reset(0)
        return task

    def test_every_public_experiment_has_cell_geometry_scale_and_phase_contrast(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                task = self.task(operation)
                state = task.public_state()
                self.assertEqual(state["volume_unit"], "pL")
                self.assertLess(max(state["sample_scale"]["diameters_m"]), 100e-6)
                self.assertEqual(state["calibration"]["display_channels"], ["phase_contrast", "fluorescence"])
                self.assertAlmostEqual(state["calibration"]["scale_bar_m"], 20e-6)
                if operation == "push":
                    self.assertAlmostEqual(state["sample_scale"]["probe_tip_outer_diameter_m"], 4e-6)
                elif operation == "pick_place":
                    np.testing.assert_allclose(state["sample_scale"]["forceps_pad_dimensions_m"],
                                               np.array([16, 4, 16])*1e-6)
                for body in ("bead_push", "bead_pick"):
                    self.assertEqual(mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, body), -1)
                cls, _ = CATALOG["microscopy_"+operation].load_classes()
                self.assertEqual(cls.operation, operation)
                self.assertEqual(len(set(model_family(op) for op in OPERATIONS)), 5)

    def test_three_replacement_experts_complete_using_cells_and_safe_contact(self):
        for operation in ("push", "pick_place", "injection"):
            with self.subTest(operation=operation):
                task = self.task(operation)
                task.execute()
                self.assertTrue(task.check(), task.microscopy_report())
                json.dumps(task.microscopy_report(), allow_nan=False)
                self.assertFalse(np.any(task.data.warning.number))
                if operation == "injection":
                    self.assertAlmostEqual(task.mechanics.ledger.state("cell").volume_m3*1e15, .5)
                else:
                    self.assertGreater(np.linalg.norm(task.mechanics.position_m-task.mechanics.initial_position_m), 20e-6)
                    self.assertEqual(task.model.neq, 0, "No cell weld or pose attachment")
                    state = task.mechanics.image_state(task.data)
                    np.testing.assert_allclose(state["centres_m"][0], task.mechanics.position_m-ORIGIN)
                    image = task.microscope.render(state, annotate=False, channel="fluorescence")
                    phase = task.microscope.render(state, annotate=False)
                    self.assertFalse(np.array_equal(np.asarray(image), np.asarray(phase)))
                    np.testing.assert_allclose(task.microscope.locate(image, "cell"),
                                               state["centres_m"][0][:2], atol=.5e-6)

    def test_idle_cell_cannot_move_or_fabricate_contacts(self):
        task = self.task("push")
        initial = task.mechanics.position_m.copy()
        task.wait(.3)
        np.testing.assert_allclose(task.mechanics.position_m, initial, atol=1e-12)
        self.assertEqual(task.mechanics.probe_contact_s, 0.)
        self.assertFalse(task.check())

    def test_one_jaw_cannot_establish_grasp_or_lift(self):
        task = self.task("pick_place")
        xy = task.mechanics.initial_position_m[:2]-ORIGIN[:2]
        task.move("gripper", [*xy, CELL_CENTER[2]], .5)
        task.command({"jaw_a": JAW_CLOSED_M, "jaw_b": JAW_OPEN_M}, .3)
        task.move("gripper", [*xy, CELL_CENTER[2]+LIFT_M], .5)
        self.assertFalse(task.mechanics.ever_grasped)
        self.assertEqual(task.mechanics.bilateral_contact_s, 0.)
        self.assertEqual(task.mechanics.maximum_lift_m, 0.)
        self.assertFalse(task.microscopy_checks()["cell_lift_20um"])

    def test_nanoliter_requests_are_rejected_without_partial_progress(self):
        for operation in ("push", "pick_place", "injection"):
            task = self.task(operation)
            before = task.data.ctrl.copy()
            with self.assertRaisesRegex(ValueError, "volume unit"):
                apply_request(task, dict(command="step", target_nl=100, pressure_pa=5000))
            np.testing.assert_array_equal(task.data.ctrl, before)
            self.assertEqual(task.data.time, 0.)
            self.assertEqual(task.data.userdata[0], 0.)

    def test_pressure_without_cell_puncture_does_not_deliver_intracellular_dose(self):
        task = self.task("injection")
        apply_request(task, dict(command="step", target_pl=.5, pressure_pa=5000, seconds=.15))
        self.assertEqual(task.mechanics.ledger.state("cell").volume_m3, 0.)
        self.assertFalse(task.microscopy_checks()["intracellular_dose"])


if __name__ == "__main__":
    unittest.main()
