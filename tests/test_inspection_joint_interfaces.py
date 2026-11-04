"""Dry positioning joints have real support geometry without interpenetration."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.rigid_access import RigidAccess
from real_labs.runtime import InstrumentSession


CASES = (
    ('hkust_aaf_wind_tunnel', 'suspended_model_section',
     {'model_yaw': (-.65, .65), 'model_pitch': (-.26, .26)}),
    ('kth_l2000_wind_tunnel', 'wind_tunnel',
     {'model_yaw': (-.28, .28), 'model_pitch': (-.17, .17)}),
    ('uwa_ngcf_c72', 'primary_apparatus',
     {'beam_azimuth': (-.16, .16), 'basket_tilt': (-.12, .12)}),
    ('warwick_highfield_nmr_1ghz', 'nmr_cryostat',
     {'inspection_lift': (0, .20), 'holder_rotation': (-.35, .35)}),
)


class InspectionJointInterfaceTests(unittest.TestCase):
    def test_combined_full_travel_preserves_original_limits_without_interference(self):
        # Touching planar/rotary support interfaces are permitted. Their direct
        # signed distance still must stay nonnegative to numerical precision.
        for scene, owner, limits in CASES:
            with self.subTest(scene=scene):
                root, manifest = build_scene(scene)
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
                session = InstrumentSession(model, manifest)
                access = RigidAccess(session, scene, owner, limits, minimum_m=1e-7)
                controls = [session.controls[owner, key] for key in limits]
                for control, bounds in zip(controls, limits.values()):
                    np.testing.assert_array_equal(model.jnt_range[control.joint], bounds)
                a, b = access.pairs.T
                for values in itertools.product(*(np.linspace(*bound, 9) for bound in limits.values())):
                    for control, value in zip(controls, values):
                        session.data.qpos[model.jnt_qposadr[control.joint]] = value
                    mujoco.mj_forward(model, session.data)
                    rotation = session.data.geom_xmat.reshape(-1, 3, 3)
                    center = session.data.geom_xpos + np.einsum('nij,nj->ni', rotation, model.geom_aabb[:, :3])
                    half = np.einsum('nij,nj->ni', np.abs(rotation), model.geom_aabb[:, 3:])
                    near = np.linalg.norm(np.maximum(np.abs(center[a]-center[b])-half[a]-half[b], 0), axis=1) < .002
                    for first, second in access.pairs[near]:
                        gap = mujoco.mj_geomDistance(model, session.data, int(first), int(second), .002, None)
                        self.assertGreaterEqual(gap, -1e-7, (scene, values, model.geom(int(first)).name, model.geom(int(second)).name, gap))
                    self.assertTrue(np.isfinite(session.data.qpos).all())
                    self.assertTrue(all(contact.dist >= -1e-7 for contact in session.data.contact))


if __name__ == '__main__':
    unittest.main()
