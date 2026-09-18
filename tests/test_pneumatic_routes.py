"""Visual hose continuity at the actual moving holder, without added physics."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))
from microscopy.cad_assets import DEFAULT_ROOT
from microscopy.pneumatics import hose_visuals
from microscopy.tasks import make_task

ASSETS = Path(os.environ.get('HOOKE_TEST_CAD_ASSET_ROOT', DEFAULT_ROOT))


def segment_end(shape, sign):
    return np.asarray(shape['pos'])+sign*shape['size'][1]*np.asarray(shape['mat']).reshape(3, 3)[:, 2]


@unittest.skipUnless((ASSETS/'zaber-rhf/manifest.json').is_file(),
                     'Manufacturer CAD is an optional private fixture')
class MovingHoses(unittest.TestCase):
    def test_both_routes_follow_encoder_poses_and_join_the_actual_tail(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS='cad',
                                        HOOKE_MICROSCOPY_ASSET_ROOT=str(ASSETS),
                                        HOOKE_MICROSCOPY_STAND='te2000-s-reference',
                                        HOOKE_MICROSCOPY_STAGE='reference',
                                        HOOKE_MICROSCOPY_OPTICS='estimated')):
            task = make_task('suction_injection')
            task.reset(0)
        model, data = task.model, task.data
        initial = hose_visuals(model, data)
        qpos = data.qpos.copy()
        masses = model.body_mass.copy()
        controls = data.ctrl.copy()
        for axis, value in zip('xyz', (.005, -.003, .002)):
            data.qpos[model.joint('injector_'+axis).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        before = data.qpos.copy()
        shapes = hose_visuals(model, data)
        self.assertEqual(len(shapes), 96)
        for index, tool in enumerate(('injector', 'holder')):
            route = shapes[index*48:(index+1)*48]
            start = data.site_xpos[model.site(f'pneumatic_{tool}_anchor0').id]
            tail = model.geom(tool+'_holder_tail').id
            half_axis = data.geom_xmat[tail].reshape(3, 3)[:, 2]*model.geom_size[tail, 1]
            ends = data.geom_xpos[tail]+np.array([-1., 1.])[:, None]*half_axis
            holder = model.body(tool+'_needle_holder').id
            back = data.xmat[holder].reshape(3, 3)[:, 2]
            end = ends[int(np.argmax((ends-data.xpos[holder])@back))]
            np.testing.assert_allclose(segment_end(route[0], -1), start, atol=1e-12)
            np.testing.assert_allclose(segment_end(route[-1], 1), end, atol=1e-12)
            for a, b in zip(route, route[1:]):
                np.testing.assert_allclose(segment_end(a, 1), segment_end(b, -1), atol=1e-12)
                rotation = np.asarray(a['mat']).reshape(3, 3)
                np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.norm(segment_end(initial[47], 1)-segment_end(shapes[47], 1)), .003)
        np.testing.assert_array_equal(initial[48:], shapes[48:])
        np.testing.assert_array_equal(data.qpos, before)
        np.testing.assert_array_equal(model.body_mass, masses)
        np.testing.assert_array_equal(data.ctrl, controls)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        self.assertEqual(hose_visuals(model, data), initial)
        self.assertEqual([shape['role'] for shape in task.runtime_visuals()[:2]], ['cell_shell', 'cell_nucleus'])


if __name__ == '__main__':
    unittest.main()
