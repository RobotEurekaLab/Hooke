"""Validate the mounted camera geometry and fail closed on missing metrology."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))

from microscopy.cad_assets import CadAssets, asset_evidence, requested_assets
from microscopy.scene import BENCH_TOP, ORIGIN
from microscopy.tasks import make_task

FIXTURE = ROOT/'temp/microscopy_research/cad_meshes_surface_normals'


class OpticalProfile(unittest.TestCase):
    def test_invalid_and_incompatible_profiles_are_rejected(self):
        for assets, optics in (("reference", "mechanical"), ("reference", "assembled"), ("cad", "misspelled")):
            with patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS=assets, HOOKE_MICROSCOPY_OPTICS=optics):
                with self.assertRaises(ValueError):
                    requested_assets()


@unittest.skipUnless((FIXTURE/'openframe/OF-CL-SM2/manifest.json').is_file(),
                     'Prepared open hardware optics are an optional local fixture')
class MountedCamera(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS="cad",
                                 HOOKE_MICROSCOPY_OPTICS="mechanical",
                                 HOOKE_MICROSCOPY_ASSET_ROOT=str(FIXTURE))
        environment.start()
        self.addCleanup(environment.stop)

    def test_actual_detector_pose_faces_inwards_and_preserves_all_drive_dynamics(self):
        task = make_task('cell_injection')
        task.reset(0)
        evidence = asset_evidence(task.model)['microscope']['camera_assembly']
        camera = task.model.body('scope_camera_reference').id
        position = task.data.xpos[camera].copy()
        np.testing.assert_allclose(position, [ORIGIN[0]+.10565, ORIGIN[1], BENCH_TOP+.117], atol=1e-8)
        np.testing.assert_allclose(task.data.xmat[camera].reshape(3, 3)[:, 2], [-1, 0, 0], atol=1e-12)
        self.assertLess(evidence['mounting_hole_alignment_error_m'], 1e-9)
        self.assertAlmostEqual(evidence['clamp_tube_engagement_m'], .022)
        self.assertAlmostEqual(evidence['standoff_length_m'], .0065)
        self.assertEqual(len(evidence['original_standoffs']), 3)
        self.assertAlmostEqual(evidence['adapter_frame_engagement_m'], .008, places=8)
        # Inspect compiled mesh transforms too: metadata alone cannot show
        # that the visible standoffs actually bridge the recessed CAD face.
        for name, reference in zip(("scope_camera_standoff_-25_-25",
                                    "scope_camera_standoff_-25_25",
                                    "scope_camera_standoff_25_-25"), evidence['original_standoffs']):
            geom = task.model.geom(name).id
            mesh = task.model.geom_dataid[geom]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, task.model.mesh_quat[mesh])
            for axial in (0., .0065):
                point = rotation.reshape(3, 3).T@(np.array([0., 0., axial])-task.model.mesh_pos[mesh])
                actual = task.data.geom_xpos[geom]+task.data.geom_xmat[geom].reshape(3, 3)@point
                expected = np.array([ORIGIN[0], ORIGIN[1], BENCH_TOP])+reference['position_m']+[axial, 0., 0.]
                np.testing.assert_allclose(actual, expected, atol=1e-8)
        self.assertFalse(evidence['camera']['dimensional_reproduction_verified'])
        self.assertFalse(evidence['camera']['sensor_plane_verified'])
        task.command({'focus': 20e-6}, .2)
        np.testing.assert_allclose(task.data.xpos[camera], position, atol=1e-12)
        with patch.dict(os.environ, HOOKE_MICROSCOPY_OPTICS='estimated'):
            previous = make_task('cell_injection')
        for attribute in ('jnt_range', 'jnt_axis', 'dof_damping', 'actuator_gainprm', 'actuator_biasprm'):
            np.testing.assert_array_equal(getattr(task.model, attribute), getattr(previous.model, attribute))
        for joint in range(task.model.njnt):
            name = task.model.joint(joint).name
            current_body = task.model.jnt_bodyid[joint]
            previous_body = previous.model.jnt_bodyid[previous.model.joint(name).id]
            self.assertEqual(task.model.body_mass[current_body], previous.model.body_mass[previous_body])

    def test_cached_asset_read_cannot_bypass_source_hash_binding(self):
        assets = CadAssets(FIXTURE)
        assets.read('openframe/OF-LL-CORE')
        with self.assertRaises(ValueError):
            assets.read('openframe/OF-LL-CORE', '0'*64)

    def test_missing_finite_planes_fail_before_simulation(self):
        original = CadAssets.interfaces

        def without_planes(assets, name):
            metrology = original(assets, name)
            return {**metrology, 'planes': []}

        with patch.object(CadAssets, 'interfaces', without_planes):
            with self.assertRaisesRegex(ValueError, 'installation plane'):
                make_task('cell_injection')


if __name__ == '__main__':
    unittest.main()
