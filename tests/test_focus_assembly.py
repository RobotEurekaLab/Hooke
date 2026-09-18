"""Verify the actual objective joint, focal reference and fixed installation."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))

from microscopy.cad_assets import asset_evidence
from microscopy.scene import ORIGIN
from microscopy.tasks import make_task

FIXTURE = ROOT/'temp/microscopy_research/cad_meshes_surface_normals'


@unittest.skipUnless((FIXTURE/'independent-q545-mounting/manifest.json').is_file(),
                     'Dimensioned optical assembly is an optional local CAD fixture')
class FocusAssembly(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, HOOKE_MICROSCOPY_ASSETS='cad',
                                 HOOKE_MICROSCOPY_STAND='openframe',
                                 HOOKE_MICROSCOPY_STAGE='reference',
                                 HOOKE_MICROSCOPY_OPTICS='assembled',
                                 HOOKE_MICROSCOPY_ASSET_ROOT=str(FIXTURE))
        environment.start()
        self.addCleanup(environment.stop)

    def test_zero_planes_and_encoder_travel_are_consistent_for_all_sample_families(self):
        for operation, height in (('push', 18e-6), ('pick_place', 18e-6),
                                  ('injection', 4e-6), ('cell_injection', 4e-6),
                                  ('suction_injection', 26e-6)):
            with self.subTest(operation=operation):
                task = make_task(operation)
                task.reset(0)
                model, data = task.model, task.data
                evidence = asset_evidence(model)['microscope']['focus_assembly']
                focus = model.joint('focus').qposadr[0]
                camera = data.xpos[model.body('scope_camera_reference').id].copy()
                for encoder in (-20e-6, 0., 20e-6):
                    data.qpos[focus] = encoder
                    mujoco.mj_forward(model, data)
                    focal = data.site_xpos[model.site('objective_nominal_focal_plane').id]
                    front = data.site_xpos[model.site('objective_front').id]
                    np.testing.assert_allclose(focal, ORIGIN+[0, 0, height+encoder], atol=1e-12)
                    self.assertAlmostEqual(focal[2]-front[2], .01066)
                    state = task.mechanics.image_state(data)
                    self.assertAlmostEqual(state['focus_m'], encoder)
                    self.assertAlmostEqual(focal[2]-ORIGIN[2], state['optical_reference_height_m']+state['focus_m'])
                    np.testing.assert_allclose(data.xpos[model.body('scope_camera_reference').id], camera, atol=1e-12)
                self.assertLess(evidence['fixed_to_moving_hole_alignment_error_m'], 1e-9)
                self.assertAlmostEqual(evidence['original_spacer_length_m'], .00814)
                self.assertFalse(evidence['derived_adapter']['thread_fit_verified'])
                self.assertFalse(evidence['derived_adapter']['fabrication_validated'])
                self.assertNotIn('nosepiece', [model.geom(i).name for i in range(model.ngeom)])

    def test_command_moves_entire_optical_carriage_and_preserves_servo_dynamics(self):
        task = make_task('cell_injection')
        task.reset(0)
        joint = task.model.joint('focus').qposadr[0]
        before = task.data.site_xpos[task.model.site('objective_front').id].copy()
        previous_q = task.data.qpos[joint]
        task.command({'focus': 20e-6}, .2)
        after = task.data.site_xpos[task.model.site('objective_front').id]
        np.testing.assert_allclose(after-before, [0, 0, task.data.qpos[joint]-previous_q], atol=1e-10)
        self.assertGreater(task.data.qpos[joint], previous_q)
        with patch.dict(os.environ, HOOKE_MICROSCOPY_OPTICS='mechanical'):
            previous = make_task('cell_injection')
        for attribute in ('jnt_range', 'jnt_axis', 'dof_damping', 'actuator_gainprm', 'actuator_biasprm'):
            np.testing.assert_array_equal(getattr(task.model, attribute), getattr(previous.model, attribute))
        for name in (task.model.joint(i).name for i in range(task.model.njnt)):
            current_body = task.model.jnt_bodyid[task.model.joint(name).id]
            previous_body = previous.model.jnt_bodyid[previous.model.joint(name).id]
            for attribute in ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia'):
                np.testing.assert_array_equal(getattr(task.model, attribute)[current_body],
                                              getattr(previous.model, attribute)[previous_body])


if __name__ == '__main__':
    unittest.main()
