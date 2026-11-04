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
from microscopy.scene import ORIGIN
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

    def test_six_oblique_through_bores_match_the_seated_objectives(self):
        task = self.task("cell_injection")
        shape = self.shape(task, "nosepiece")
        self.assertTrue(shape.is_watertight)
        self.assertTrue(shape.is_winding_consistent)
        self.assertGreater(shape.volume, 0.)
        self.assertLess(shape.volume, np.pi*.055**2*.005)
        evidence = asset_evidence(task.model)["objective"]
        slots = np.asarray(evidence["mount_socket_centres_m"])
        axes = np.asarray(evidence["mount_socket_axes"])
        datum = ORIGIN+[0., 0., evidence["legacy_shoulder_offset_m"]]
        datum += task.data.xpos[task.model.body("focus").id]
        self.assertEqual(len(slots), 6)
        np.testing.assert_allclose(slots[0], 0., atol=1e-15)
        np.testing.assert_allclose(axes[0], [0, 0, 1], atol=1e-15)

        def intersections(start, direction):
            # Direct ray/triangle intersections avoid an optional spatial-index
            # package. Probe actual compiled triangles, not the mesh authoring
            # function or only its reported socket metadata.
            triangles = shape.triangles
            first, second = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
            h = np.cross(direction, second)
            determinant = np.einsum("ij,ij->i", first, h)
            valid = np.abs(determinant) > 1e-14
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            offset = start-triangles[:, 0]
            u = inverse*np.einsum("ij,ij->i", offset, h)
            q = np.cross(offset, first)
            v = inverse*(q@direction)
            distance = inverse*np.einsum("ij,ij->i", second, q)
            return distance[valid & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
                            & (distance >= 0.) & (distance <= .080)]

        for point, axis in zip(datum+slots, axes):
            tangent = np.cross(axis, [0, 0, 1] if abs(axis[2]) < .95 else [1, 0, 0])
            tangent /= np.linalg.norm(tangent)
            for offset in (0., .010, -.010):
                self.assertEqual(len(intersections(point+tangent*offset-axis*.040, axis)), 0)
            # Material remains just outside each bore; an empty or omitted
            # carousel would incorrectly pass only the open-axis assertions.
            self.assertGreaterEqual(len(intersections(point+tangent*.014-axis*.040, axis)), 2)
        self.assertEqual(sum(task.model.geom(i).name.startswith("objective_")
                             for i in range(task.model.ngeom)), 15)
        for index, slot in enumerate(evidence["legacy_occupied_slots"]):
            barrel = task.model.geom(f"objective_{index}_barrel")
            shoulder = task.data.geom_xpos[barrel.id]-axes[slot]*barrel.size[1]
            np.testing.assert_allclose(shoulder, datum+slots[slot], atol=2e-9)

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
