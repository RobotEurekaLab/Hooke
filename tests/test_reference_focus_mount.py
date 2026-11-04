"""Verify the visible mounting chain follows the actual focus encoder."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Hooke"))

from microscopy.cad_assets import asset_evidence
from microscopy.tasks import make_task

CAD_ROOT = ROOT/"temp/microscopy_research/cad_meshes_surface_normals"


class ReferenceFocusMountTests(unittest.TestCase):
    def task(self, operation="suction_injection", assets="reference"):
        with patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS=assets,
                HOOKE_MICROSCOPY_STAND="te2000-s-reference", HOOKE_MICROSCOPY_STAGE="reference",
                HOOKE_MICROSCOPY_OPTICS="estimated", HOOKE_MICROSCOPY_ASSET_ROOT=str(CAD_ROOT)):
            task = make_task(operation)
            task.reset(0)
        task.data.qpos[task.model.joint("focus").qposadr[0]] = 0
        mujoco.mj_kinematics(task.model, task.data)
        return task

    def shape(self, task, name):
        geom = task.model.geom(name)
        shape = task.model.mesh(int(geom.dataid[0]))
        start, count = int(shape.vertadr[0]), int(shape.vertnum[0])
        face, nface = int(shape.faceadr[0]), int(shape.facenum[0])
        vertices = (task.model.mesh_vert[start:start+count]
                    @ task.data.geom_xmat[geom.id].reshape(3, 3).T+task.data.geom_xpos[geom.id])
        return trimesh.Trimesh(vertices, task.model.mesh_face[face:face+nface], process=False)

    def test_mounting_faces_stay_seated_and_fixed_parts_do_not_follow_focus(self):
        for operation in ("cell_injection", "suction_injection"):
            task = self.task(operation)
            model, data = task.model, task.data
            foot = model.geom("focus_mount_foot")
            mount = asset_evidence(model)["objective"]["focus_mount"]
            self.assertAlmostEqual(data.geom_xpos[foot.id, 2]-foot.size[2],
                                   mount["stand_attachment_plane_world_z_m"], delta=2e-9)
            fixed = data.geom_xpos[model.geom("focus_mount_fixed_outline").id].copy()
            sample = data.site_xpos[model.site("sample_plane").id].copy()
            for encoder in (-.001, 0., .001):
                data.qpos[model.joint("focus").qposadr[0]] = encoder
                mujoco.mj_kinematics(model, data)
                cap = model.geom("focus_mount_carrier_cap")
                cap_top = data.geom_xpos[cap.id, 2]+cap.size[2]
                spindle = model.geom("focus_mount_spindle")
                self.assertAlmostEqual(data.geom_xpos[spindle.id, 2]-spindle.size[1], cap_top, delta=2e-9)
                hub = self.shape(task, "focus_mount_hub")
                spindle_top = data.geom_xpos[spindle.id, 2]+spindle.size[1]
                self.assertGreater(spindle_top, hub.bounds[0, 2])
                self.assertLess(spindle_top, hub.bounds[1, 2])
                np.testing.assert_allclose(data.geom_xpos[spindle.id, :2], hub.bounds[:, :2].mean(axis=0), atol=2e-9)
                self.assertGreater(float(self.shape(task, "nosepiece").vertices[:, 2].min()), cap_top)
                moving = model.geom("focus_mount_moving_outline")
                self.assertAlmostEqual(float(data.geom_xpos[moving.id, 2]-moving.size[2]),
                                       float(cap_top), delta=2e-9)
                self.assertGreater(data.geom_xpos[moving.id, 2]-moving.size[2]
                                   -mount["stand_attachment_plane_world_z_m"], 0)
                np.testing.assert_allclose(data.geom_xpos[model.geom("focus_mount_fixed_outline").id], fixed)
                np.testing.assert_allclose(data.site_xpos[model.site("sample_plane").id], sample)
            self.assertFalse(mount["load_capacity_validated"])
            self.assertFalse(mount["official_cad"])

    def test_support_and_carriage_bores_are_open_watertight_meshes(self):
        task = self.task()
        for name in ("focus_mount_backplate", "focus_mount_carrier_web"):
            shape = self.shape(task, name)
            self.assertTrue(shape.is_watertight, name)
            self.assertTrue(shape.is_winding_consistent, name)
            self.assertGreater(shape.volume, 0, name)
        for name, width, height, thickness, radius in (
                ("focus_mount_backplate", .050, .052, .008, .0017),
                ("focus_mount_carrier_web", .032, .042, .002, .0011)):
            hole_area = 48*np.sin(2*np.pi/48)/2*radius**2
            expected = (width*height-4*hole_area)*thickness
            self.assertAlmostEqual(self.shape(task, name).volume/expected, 1, delta=1e-6)

    @unittest.skipUnless((CAD_ROOT/"independent-q545-mounting/manifest.json").is_file(),
                         "Optional independent focus CAD fixture")
    def test_cad_asset_profile_keeps_the_original_focus_support_and_encoder_ownership(self):
        task = self.task(assets="cad")
        model, data = task.model, task.data
        parts = {role: model.geom(f"focus_mount_{role}_outline")
                 for role in ("fixed", "moving")}
        before = {role: data.geom_xpos[part.id].copy() for role, part in parts.items()}
        task.command({"focus": 20e-6}, .2)
        actual = float(data.qpos[model.joint("focus").qposadr[0]])
        self.assertGreater(actual, 0)
        np.testing.assert_allclose(data.geom_xpos[parts["moving"].id]-before["moving"], [0, 0, actual], atol=1e-9)
        np.testing.assert_allclose(data.geom_xpos[parts["fixed"].id], before["fixed"], atol=1e-9)
        evidence = asset_evidence(model)["objective"]["focus_mount"]
        self.assertEqual(evidence["profile"], "original-inclined-carousel-focus-support")
        self.assertIsNone(evidence["source_sha256"])
        self.assertFalse(evidence["official_cad"])
        self.assertFalse(evidence["one_to_one_verified"])


if __name__ == "__main__":
    unittest.main()
