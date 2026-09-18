"""Bilateral suction depends on actual stages, vacuum and load history."""

import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.visual_state import LiveVisuals
from backends.ipc import read_message, write_message
from microscopy import model_family
from microscopy.cell_scene import NEEDLE_BACK
from microscopy.scene import ORIGIN
from microscopy.suction_scene import CELL_RADII, HOLDER_BACK
from microscopy.tasks import make_task
from microscopy.worker import apply_request


class SuctionInjectionTests(unittest.TestCase):
    def task(self, seed=0):
        task = make_task("suction_injection")
        task.reset(seed)
        return task

    def hold(self, task, *, occluded=False):
        centre = task.mechanics.position_m-ORIGIN
        task.move("holder", centre+HOLDER_BACK*CELL_RADII[0], .7)
        task.mechanics.holding_pressure_pa = -1500.
        task.mechanics.holding_occluded = occluded
        task.wait(.2)

    def approach(self, task, depth=2.5e-6):
        centre = task.mechanics.position_m-ORIGIN
        task.move("injector", centre+NEEDLE_BACK*(CELL_RADII[0]-depth), .7)

    def test_complete_three_image_guided_layouts_and_release_order(self):
        for seed in (0, 1, 4):
            with self.subTest(seed=seed):
                task = self.task(seed)
                task.execute()
                m = task.mechanics
                self.assertTrue(task.check(), task.microscopy_report())
                self.assertFalse(np.any(task.data.warning.number))
                self.assertEqual(task.model.neq, 0)
                self.assertEqual(task.model.nmocap, 0)
                self.assertAlmostEqual(m.ledger.state("cell").volume_m3*1e15, .5)
                self.assertLess(m.seal_time_s, m.puncture_time_s)
                self.assertLess(m.withdrawal_time_s, m.release_time_s)
                self.assertGreater(m.maximum_holding_force_n, 20e-9)
                self.assertLess(m.maximum_displacement_m, 1e-6)
                json.dumps(task.microscopy_report(), allow_nan=False)

    def test_unheld_cell_moves_under_needle_load_without_puncture_or_dose(self):
        task = self.task()
        self.approach(task)
        task.data.userdata[0] = 5000.
        task.wait(.3)
        m = task.mechanics
        self.assertGreater(m.maximum_displacement_m, 1e-6)
        self.assertFalse(m.punctured)
        self.assertEqual(m.ledger.state("cell").volume_m3, 0)
        self.assertFalse(task.microscopy_checks()["holding_established"])
        self.assertFalse(task.check())

    def test_occluded_holding_channel_has_upstream_vacuum_without_seal(self):
        task = self.task()
        self.hold(task, occluded=True)
        m = task.mechanics
        self.assertLess(m.actual_holding_pressure_pa, -1400)
        self.assertEqual(m.holding_capacity_n, 0)
        self.assertFalse(m.sealed)
        self.assertFalse(m.established)
        self.assertEqual(m.hold_s, 0)

    def test_insufficient_capacity_loses_seal_and_blocks_supported_puncture(self):
        task = self.task()
        self.hold(task)
        self.assertTrue(task.mechanics.sealed)
        task.mechanics.holding_pressure_pa = -350.
        task.wait(.3)
        self.approach(task)
        m = task.mechanics
        self.assertFalse(m.sealed)
        self.assertGreater(m.seal_losses, 0)
        self.assertTrue(m.premature_release)
        self.assertFalse(m.punctured)
        self.assertFalse(task.microscopy_checks()["no_premature_release"])

    def test_release_during_injection_is_latched_as_failure(self):
        task = self.task()
        self.hold(task)
        self.approach(task)
        self.assertTrue(task.mechanics.punctured)
        task.mechanics.holding_pressure_pa = 0.
        task.data.userdata[0] = 5000.
        task.wait(.3)
        m = task.mechanics
        self.assertTrue(m.premature_release)
        self.assertTrue(m.delivery_without_hold)
        self.assertFalse(task.microscopy_checks()["held_during_injection"])
        self.assertFalse(task.microscopy_checks()["no_premature_release"])

    def test_actual_holding_reaction_and_both_visual_paths_follow_cell_model(self):
        task = self.task()
        self.hold(task)
        self.approach(task, .5e-6)
        m = task.mechanics
        np.testing.assert_allclose(task.data.qfrc_applied[m.holder_dofs],
                                   m.holder_axes.T @ -m.holding_force_n, atol=1e-16)
        self.assertGreater(np.linalg.norm(task.data.qfrc_applied[m.holder_dofs]), 0)
        np.testing.assert_array_equal(task.data.qfrc_applied[m.stage_dofs], 0.)
        np.testing.assert_allclose(task.runtime_visuals()[0]["pos"], m.position_m)
        np.testing.assert_allclose(m.image_state(task.data)["centres_m"][0], m.position_m-ORIGIN)
        visuals = LiveVisuals(task, Path("unused"))
        try:
            self.assertEqual(visuals.snapshot()["geometry"], task.runtime_visuals())
        finally:
            visuals.close()

    def test_partial_dose_tracer_survives_native_binary_transport(self):
        task = self.task()
        self.hold(task)
        self.approach(task)
        self.assertTrue(task.mechanics.punctured)
        task.data.userdata[0] = 5000.
        task.wait(.08)
        dose = task.mechanics.ledger.state("cell").volume_m3
        self.assertGreater(dose, 0.)
        self.assertLess(dose, .5e-15)
        visuals = LiveVisuals(task, Path("unused"))
        try:
            state = visuals.snapshot()
        finally:
            visuals.close()
        stream = io.BytesIO()
        write_message(stream, state)
        stream.seek(0)
        decoded = read_message(stream)
        for original, received in zip(state["geometry"], decoded["geometry"]):
            self.assertEqual(original["rgba"], received["rgba"])
            self.assertTrue(all(type(value) is float for value in received["rgba"]))

    def test_holding_batch_validation_and_distinct_cell_models(self):
        task = self.task()
        for body in ({"holding_pressure_pa": 1}, {"holding_pressure_pa": float("nan")},
                     {"holding_pressure_pa": -6000}, {"holding_occluded": "yes"},
                     {"holding_pressure_pa": -1500, "actuators": {"holder_x": .1}}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                apply_request(task, dict(command="step", pressure_pa=5000, **body))
            self.assertEqual(task.data.time, 0)
            self.assertEqual(task.data.userdata[0], 0)
            self.assertEqual(task.mechanics.holding_pressure_pa, 0)
        self.assertNotEqual(model_family("cell_injection"), model_family("suction_injection"))
        self.assertNotEqual(model_family("push"), model_family("injection"))
        with self.assertRaises(ValueError):
            apply_request(task, {"command": "reset", "operation": "cell_injection"})
        other = make_task("cell_injection")
        other.reset(0)
        with self.assertRaises(ValueError):
            apply_request(other, {"command": "step", "holding_pressure_pa": -1500})


if __name__ == "__main__":
    unittest.main()
