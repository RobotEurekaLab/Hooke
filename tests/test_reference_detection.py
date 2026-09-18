"""Verify installed nominal datums and finite collection-bundle clearance."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))

from microscopy.reference_detection import installed_dimensions
from microscopy.tasks import make_task
from test_detection_shell import segment_hits


def world_shape(task, gid):
    model, data = task.model, task.data
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        shape = model.mesh(int(model.geom_dataid[gid]))
        va, vn = int(shape.vertadr[0]), int(shape.vertnum[0])
        fa, fn = int(shape.faceadr[0]), int(shape.facenum[0])
        vertices = model.mesh_vert[va:va+vn]
        mesh = trimesh.Trimesh(vertices, model.mesh_face[fa:fa+fn], process=False)
    elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(model.geom_size[gid, 0], 2*model.geom_size[gid, 1], sections=64)
    elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(2*model.geom_size[gid])
    else:
        raise AssertionError(f"Unsupported collection geometry {model.geom(gid).name}")
    mesh.vertices = mesh.vertices@data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
    return mesh


class ReferenceDetectionTests(unittest.TestCase):
    def task(self, operation):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="reference",
                HOOKE_MICROSCOPY_STAND="te2000-s-reference", HOOKE_MICROSCOPY_STAGE="reference",
                HOOKE_MICROSCOPY_OPTICS="estimated")):
            task = make_task(operation); task.reset(0)
        return task

    def test_measured_datums_and_virtual_field_follow_the_correct_reference_planes(self):
        for operation in ("cell_injection", "suction_injection"):
            task = self.task(operation)
            focus = int(task.model.joint("focus").qposadr[0])
            sensor_before = task.data.site_xpos[task.model.site("collection_nominal_sensor_plane").id].copy()
            paths = []
            for encoder in (-.001, 0., .001):
                task.data.qpos[focus] = encoder; mujoco.mj_kinematics(task.model, task.data)
                dimensions = installed_dimensions(task.model, task.data)
                self.assertAlmostEqual(dimensions["tube_image_datum_to_nominal_sensor_m"], .148, delta=1e-10)
                distance = dimensions["objective_shoulder_to_unit_entry_air_path_m"]
                self.assertGreater(distance, .070); self.assertLess(distance, .170); paths.append(distance)
                np.testing.assert_allclose(task.data.site_xpos[task.model.site("collection_nominal_sensor_plane").id], sensor_before, atol=1e-12, rtol=0)
                self.assertFalse(dimensions["physical_optics_calibrated"])
                self.assertFalse(dimensions["actual_vendor_sensor_plane_verified"])
                self.assertAlmostEqual(task.public_state()["calibration"]["field_width_m"], .0064/40)
                tube = world_shape(task, task.model.geom("collection_vertical_tube").id)
                exit = task.data.site_xpos[task.model.site("collection_objective_exit").id]
                self.assertAlmostEqual(float(tube.vertices[:, 2].max()), exit[2], delta=5e-8)
            np.testing.assert_allclose(np.diff(paths), [.001, .001], atol=1e-12, rtol=0)

    def test_compiled_original_housings_are_closed_and_preserve_camera_body_dimensions(self):
        task = self.task("suction_injection")
        for name in ("te_lower_shell", "scope_camera", "collection_port_seat", "collection_unit_body", "collection_camera_tube"):
            with self.subTest(name=name):
                shape = world_shape(task, task.model.geom(name).id)
                self.assertTrue(shape.is_watertight)
                self.assertTrue(shape.is_winding_consistent)
                self.assertGreater(shape.volume, 0)
        shape = world_shape(task, task.model.geom("scope_camera").id)
        np.testing.assert_allclose(np.ptp(shape.vertices, axis=0), [.030, .029, .029], atol=5e-8, rtol=0)

    def test_sampled_fourteen_mm_bundle_clears_the_compiled_opaque_collection_assembly(self):
        working_optics = {"collection_fold_mirror", "collection_tube_lens_envelope", "scope_camera_nominal_sensor"}
        for operation in ("cell_injection", "suction_injection"):
            task = self.task(operation)
            opaque = []
            for gid in range(task.model.ngeom):
                name = task.model.geom(gid).name
                bid = int(task.model.geom_bodyid[gid]); ancestors = set()
                while bid:
                    ancestors.add(task.model.body(bid).name); bid = int(task.model.body_parentid[bid])
                if ancestors.intersection(("microscope", "focus")) and name not in working_optics:
                    opaque.append((name, gid))
            for encoder in (-.001, 0., .001):
                task.data.qpos[int(task.model.joint("focus").qposadr[0])] = encoder
                mujoco.mj_kinematics(task.model, task.data)
                origin = task.data.site_xpos[task.model.site("collection_objective_exit").id]
                fold = task.data.site_xpos[task.model.site("collection_fold_axis").id]
                sensor = task.data.site_xpos[task.model.site("collection_nominal_sensor_plane").id]
                shapes = [(name, world_shape(task, gid)) for name, gid in opaque]
                samples = [(0., 0.)]+[(radius*np.cos(angle), radius*np.sin(angle))
                    for radius in (.0035, .007) for angle in np.linspace(0, 2*np.pi, 16, endpoint=False)]
                for x, y in samples:
                    # The 45 degree mirror has z-fold_z=x, rather than a
                    # shared Z endpoint for all incoming bundle rays.
                    reflection = fold+[x, y, x]
                    incoming = origin+[x, y, 0]
                    outgoing = sensor+[0, y, x]
                    for name, shape in shapes:
                        self.assertFalse(segment_hits(shape, incoming, reflection).any(), (operation, encoder, name, "incoming"))
                        self.assertFalse(segment_hits(shape, reflection, outgoing).any(), (operation, encoder, name, "outgoing"))
