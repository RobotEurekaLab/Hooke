"""Objective focus and archived interaction evidence follow observed physics."""

from collections import deque
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from microscopy.scene import ORIGIN
from microscopy.tasks import make_task


class InjectionEvidenceTests(unittest.TestCase):
    def task(self, operation):
        task = make_task(operation)
        task.reset(0)
        return task

    def test_both_experts_focus_the_injection_then_restore_the_cell_plane(self):
        for operation in ("cell_injection", "suction_injection"):
            with self.subTest(operation=operation):
                task = self.task(operation)
                task.execute()
                self.assertTrue(task.check(), task.microscopy_report())
                focus, review = task.observation_focus
                self.assertGreater(focus["target_focus_m"], 2e-6)
                self.assertAlmostEqual(focus["observed_focus_m"], focus["target_focus_m"], delta=1e-8)
                self.assertAlmostEqual(review["observed_focus_m"], 0., delta=1e-8)
                self.assertTrue(task.focused)  # Historical cell-localisation autofocus.
                trace = task.mechanics.interaction_trace()
                delivering = (trace["command_pressure_pa"] > 0) & trace["inside"]
                self.assertTrue(delivering.any())
                axial_error = trace["tip_world_m"][delivering, 2]-trace["focus_plane_z_m"][delivering]
                self.assertLess(np.abs(axial_error).max(), .45e-6)
                np.testing.assert_allclose(trace["normalized_distance"],
                    np.linalg.norm((trace["tip_world_m"]-trace["contact_cell_world_m"])
                                   /task.mechanics.cell_radii, axis=1), atol=1e-12)
                np.testing.assert_array_equal(trace["inside"], trace["normalized_distance"] < 1.)
                puncture = np.flatnonzero(trace["punctured"])[0]
                self.assertAlmostEqual(trace["time_s"][puncture], task.mechanics.puncture_time_s)
                self.assertGreater(trace["penetration_m"][puncture], 0.)
                self.assertAlmostEqual(trace["delivered_m3"][-1]*1e15, .5)
                self.assertEqual(task.mechanics.interaction_trace_metadata()["dropped_rows"], 0)

    def test_trace_is_bounded_records_actual_pose_and_clears_on_reset(self):
        task = self.task("suction_injection")
        task.mechanics._interaction_rows = deque(maxlen=4)
        task.wait(.012)
        trace = task.mechanics.interaction_trace()
        metadata = task.mechanics.interaction_trace_metadata()
        self.assertEqual(metadata["rows"], 4)
        self.assertGreater(metadata["dropped_rows"], 0)
        self.assertEqual(metadata["observed_steps"], metadata["rows"]+metadata["dropped_rows"])
        np.testing.assert_allclose(trace["tip_world_m"][-1], task.data.site("injector_tcp").xpos)
        np.testing.assert_allclose(trace["holding_tip_world_m"][-1], task.data.site("holder_tcp").xpos)
        np.testing.assert_allclose(trace["cell_world_m"][-1], task.mechanics.position_m)
        expected_focus = ORIGIN[2]+task.mechanics.optical_reference_height_m+float(
            task.data.qpos[task.model.joint("focus").qposadr[0]])
        self.assertEqual(trace["focus_plane_z_m"][-1], expected_focus)
        task.mechanics.update(task.data)
        self.assertEqual(task.mechanics.interaction_trace_metadata(), metadata)
        task.reset(0)
        self.assertEqual(task.mechanics.interaction_trace_metadata()["observed_steps"], 0)
        self.assertEqual(task.mechanics.interaction_trace()["tip_world_m"].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
