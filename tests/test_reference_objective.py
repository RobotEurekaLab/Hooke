"""Dimension facts, bored support topology and focus/glass separation."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))
from microscopy.cad_assets import asset_evidence
from microscopy.reference_objective import BODY_LENGTH_M
from microscopy.tasks import CalibrationPush, make_task


class ReferenceObjectiveTests(unittest.TestCase):
    def task(self, operation):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="reference",
                HOOKE_MICROSCOPY_STAND="te2000-s-reference", HOOKE_MICROSCOPY_OPTICS="estimated",
                HOOKE_MICROSCOPY_STAGE="reference")):
            task = make_task(operation)
            task.reset(0)
        return task

    def shape(self, task, name):
        geom = task.model.geom(name)
        mesh = task.model.mesh(int(geom.dataid[0]))
        v, n = int(mesh.vertadr[0]), int(mesh.vertnum[0])
        f, nf = int(mesh.faceadr[0]), int(mesh.facenum[0])
        vertices = (task.model.mesh_vert[v:v+n]@task.data.geom_xmat[geom.id].reshape(3, 3).T
                    +task.data.geom_xpos[geom.id])
        return trimesh.Trimesh(vertices, task.model.mesh_face[f:f+nf], process=False)

    def test_objective_exterior_matches_selected_drawing_datums(self):
        task = self.task("cell_injection")
        objective = asset_evidence(task.model)["objective"]
        self.assertFalse(objective["official_cad"])
        self.assertFalse(objective["one_to_one_verified"])
        self.assertAlmostEqual(BODY_LENGTH_M, .05611)
        collar = self.shape(task, "reference_40x_correction")
        np.testing.assert_allclose(np.ptp(collar.vertices[:, :2], axis=0), [.034, .034], atol=2e-9)
        self.assertAlmostEqual(float(np.ptp(collar.vertices[:, 2])), .005, delta=2e-9)
        for part in ("thread", "rear", "correction", "barrel", "lip", "nose", "band"):
            shape = self.shape(task, "reference_40x_"+part)
            self.assertTrue(shape.is_watertight)
            self.assertTrue(shape.is_winding_consistent)
            self.assertGreater(shape.volume, 0.)

    def test_six_through_bores_remove_their_full_volume(self):
        task = self.task("cell_injection")
        shape = self.shape(task, "nosepiece")
        self.assertTrue(shape.is_watertight)
        self.assertTrue(shape.is_winding_consistent)
        expected = (.055**2*192*np.sin(2*np.pi/192)/2
                    -6*.0125**2*96*np.sin(2*np.pi/96)/2)*.005
        self.assertAlmostEqual(shape.volume/expected, 1., delta=3e-7)
        slots = np.array(asset_evidence(task.model)["objective"]["mount_slots_m"])
        self.assertEqual(len(slots), 6)
        np.testing.assert_allclose(np.linalg.norm(np.diff(np.vstack((slots, slots[:1])), axis=0), axis=1),
                                   .035, atol=1e-12)
        self.assertEqual(sum(task.model.geom(i).name.startswith("objective_")
                             for i in range(task.model.ngeom)), 15)
        plate_top = float(shape.vertices[:, 2].max())
        for index in range(3):
            barrel = task.model.geom(f"objective_{index}_barrel")
            shoulder = task.data.geom_xpos[barrel.id, 2]-barrel.size[1]
            self.assertAlmostEqual(float(shoulder), plate_top, delta=2e-9)

    def test_focus_changes_geometric_working_distance_without_changing_sample_height(self):
        for operation, reference in (("cell_injection", 4e-6), ("suction_injection", 26e-6)):
            task = self.task(operation)
            sample = task.data.site_xpos[task.model.site("sample_plane").id].copy()
            for focus in (-.001, 0., .001):
                task.data.qpos[task.model.joint("focus").qposadr[0]] = focus
                mujoco.mj_forward(task.model, task.data)
                state = task.public_state()
                calibration = state["calibration"]
                objective = calibration["reference_objective"]
                self.assertAlmostEqual(objective["glass_thickness_m"], .0012)
                self.assertAlmostEqual(objective["geometric_working_distance_m"], .0031-reference-focus,
                                       delta=2e-9)
                self.assertAlmostEqual(calibration["virtual_sensor_width_m"], .0064)
                self.assertFalse(calibration["physical_optics_calibrated"])
                np.testing.assert_allclose(task.data.site_xpos[task.model.site("sample_plane").id], sample)

    def test_calibration_bead_scene_keeps_its_separate_objective_and_glass(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="reference",
                HOOKE_MICROSCOPY_STAND="te2000-s-reference", HOOKE_MICROSCOPY_OPTICS="estimated",
                HOOKE_MICROSCOPY_STAGE="reference")):
            task = CalibrationPush.Expert(CalibrationPush.load())
            task.reset(0)
        self.assertNotIn("objective", asset_evidence(task.model))
        self.assertAlmostEqual(task.model.geom("nosepiece").size[0], .040)
        self.assertAlmostEqual(task.model.geom("sample_glass").size[2]*2, .0005)


if __name__ == "__main__":
    unittest.main()
