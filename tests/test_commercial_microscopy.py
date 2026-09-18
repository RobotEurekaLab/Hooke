"""Compiled reference geometry, preserved SI sample scale and mounted axes."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))

from microscopy.cad_assets import DEFAULT_ROOT, asset_evidence
from microscopy.kinematics import tool_axes
from microscopy.scene import ORIGIN
from microscopy.stand import stand_profile
from microscopy.tasks import make_task

ASSETS = Path(os.environ.get('HOOKE_TEST_CAD_ASSET_ROOT', DEFAULT_ROOT))


class ReferenceStand(unittest.TestCase):
    def test_stand_keeps_the_target_scale_and_declares_reference_boundaries(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS='reference',
                                        HOOKE_MICROSCOPY_STAND='te2000-s-reference',
                                        HOOKE_MICROSCOPY_STAGE='reference',
                                        HOOKE_MICROSCOPY_OPTICS='estimated')):
            t = make_task('cell_injection')
            t.reset(0)
        np.testing.assert_allclose(t.model.geom('cell_0_shell').size*2, [36e-6, 28e-6, 8e-6])
        stand = asset_evidence(t.model)['microscope']
        self.assertFalse(stand['official_cad'])
        self.assertFalse(stand['one_to_one_verified'])
        # Bare base dimensions come from Nikon's side/front drawings. The
        # 135.4 mm dimension describes the stage front, not the optical axis.
        base = t.model.geom('te_lower_shell')
        base_mesh = t.model.mesh(int(base.dataid[0]))
        start, count = int(base_mesh.vertadr[0]), int(base_mesh.vertnum[0])
        base_points = (t.model.mesh_vert[start:start+count]
                       @ t.data.geom_xmat[base.id].reshape(3, 3).T + t.data.geom_xpos[base.id])
        np.testing.assert_allclose(np.ptp(base_points[:, :2], axis=0), [.195, .4764], atol=1e-7, rtol=0)
        np.testing.assert_allclose([base_points[:, 1].min()-ORIGIN[1],
                                   base_points[:, 1].max()-ORIGIN[1]], [-.2382, .2382], atol=1e-7, rtol=0)
        self.assertEqual(stand['stage_front_from_base_front_reference_m'], .1354)
        self.assertIn('estimate', next(key for key in stand if key.startswith('axis_from_base_front')))
        height = t.data.geom_xpos[t.model.geom('te_lamp_arm').id, 2]
        mesh = t.model.mesh(int(t.model.geom('te_lamp_arm').dataid[0]))
        vertices = t.model.mesh_vert[int(mesh.vertadr[0]):int(mesh.vertadr[0])+int(mesh.vertnum[0])]
        rotation = t.data.geom_xmat[t.model.geom('te_lamp_arm').id].reshape(3, 3)
        self.assertAlmostEqual(height+float((vertices@rotation.T)[:, 2].max())-.705, .611, delta=1e-7)

    def test_openframe_mounts_cannot_silently_use_the_commercial_stand(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_STAND='te2000-s-reference',
                                        HOOKE_MICROSCOPY_OPTICS='assembled')):
            with self.assertRaises(ValueError):
                stand_profile(True)


@unittest.skipUnless((ASSETS/'zaber-rhf/manifest.json').is_file(),
                     'Manufacturer CAD is an optional private fixture')
class CompactTools(unittest.TestCase):
    def test_new_mount_reaches_world_targets_and_needle_holder_joins_the_glass(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS='cad',
                                        HOOKE_MICROSCOPY_ASSET_ROOT=str(ASSETS),
                                        HOOKE_MICROSCOPY_STAND='te2000-s-reference',
                                        HOOKE_MICROSCOPY_STAGE='reference',
                                        HOOKE_MICROSCOPY_OPTICS='estimated')):
            t = make_task('suction_injection')
            t.reset(0)
        for tool in ('holder', 'injector'):
            # Axis metrology happens away from the cell, glass and opposite
            # needle; coincident TCPs would measure contact, not free motion.
            target = np.array([-.0008 if tool == 'holder' else .0008, -.0006, .0009])
            t.move(tool, target, .4)
            tcp = t.data.site_xpos[t.model.site(tool+'_tcp').id]
            np.testing.assert_allclose(tcp, ORIGIN+target, atol=2e-7)
            np.testing.assert_allclose(tool_axes(t.model, tool).T@tool_axes(t.model, tool), np.eye(3), atol=1e-12)
            node = t.model.body(tool+'_needle_holder').id
            direction = t.data.xmat[node].reshape(3, 3)[:, 2]
            np.testing.assert_allclose(t.data.xpos[node], tcp+direction*.012, atol=1e-10)
            shaft = t.model.geom(tool+'_holder_shaft')
            self.assertAlmostEqual(float(shaft.size[0])*2, .004)
            tail = t.data.geom_xpos[t.model.geom(tool+'_holder_tail').id]+direction*.003
            self.assertAlmostEqual(float((tail-t.data.xpos[node])@direction), .140, delta=1e-9)


if __name__ == '__main__':
    unittest.main()
