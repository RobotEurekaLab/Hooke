"""Native welded-body compensation must preserve source virtual work."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.gravity import gravity_wrench, needs_gravity_wrench
from backends.mass_properties import compose_welded_mass, rotation


class WeldedGravityTests(unittest.TestCase):
    def model(self, coefficients, *, articulated=False):
        joint = '<joint axis="0 1 0"/>' if articulated else '<freejoint/>'
        children = ''.join(
            f'<body pos="{x} {y} {z}" gravcomp="{coefficient}">'
            f'<inertial pos=".02 -.01 .03" mass="{mass}" diaginertia=".03 .04 .05"/></body>'
            for coefficient, (x, y, z, mass) in zip(coefficients, [(.3, .1, .2, 2), (-.2, .3, .1, 1)])
        )
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><option gravity="1.7 -2.1 -7.4"/><worldbody>'
            f'<body pos=".4 -.2 .8" quat=".9238795 0 .3826834 0">{joint}{children}</body>'
            '</worldbody></mujoco>')
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        arrays = {name: getattr(model, name) for name in
                  ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_gravcomp')}
        arrays.update(reference_xpos=reference.xpos, reference_xquat=reference.xquat)
        properties = compose_welded_mass(arrays, 1, [1, 2, 3])
        return model, properties

    def test_mixed_compensation_preserves_force_and_torque_at_rotated_pose(self):
        for coefficients in ([0, 1], [.25, .75], [1.25, .5], [1, 1], [0, 0]):
            for articulated in (False, True):
                with self.subTest(coefficients=coefficients, articulated=articulated):
                    model, properties = self.model(coefficients, articulated=articulated)
                    data = mujoco.MjData(model)
                    if articulated:
                        data.qpos[0] = .63
                    else:
                        data.qpos[3:7] = np.array([.7, -.2, .4, .3]) / np.linalg.norm([.7, -.2, .4, .3])
                    mujoco.mj_forward(model, data)
                    observed = np.zeros(model.nv)
                    if not properties['disable_gravity']:
                        center = data.xpos[1] + rotation(data.xquat[1]) @ properties['center']
                        mujoco.mj_applyFT(model, data, properties['mass'] * model.opt.gravity,
                                         np.zeros(3), center, 1, observed)
                    force, torque = gravity_wrench(properties, data.xquat[1], model.opt.gravity)
                    mujoco.mj_applyFT(model, data, force, torque, data.xpos[1], 1, observed)
                    np.testing.assert_allclose(observed, data.qfrc_gravcomp - data.qfrc_bias,
                                               atol=2e-14, rtol=2e-14)

    def test_average_one_does_not_discard_asymmetric_compensation_moment(self):
        _, properties = self.model([1.25, .5])
        self.assertEqual(properties['gravcomp'], 1.)
        self.assertFalse(properties['disable_gravity'])
        self.assertTrue(needs_gravity_wrench(properties))
        force, moment = gravity_wrench(properties, [1, 0, 0, 0], [0, 0, -9.81])
        self.assertGreater(np.linalg.norm(moment - np.cross(properties['center'], force)), .1)

    def test_fully_compensated_massless_owner_uses_native_disabled_gravity(self):
        _, properties = self.model([1, 1])
        self.assertTrue(properties['disable_gravity'])
        self.assertFalse(needs_gravity_wrench(properties))
        force, torque = gravity_wrench(properties, [1, 0, 0, 0], [0, 0, -9.81])
        np.testing.assert_array_equal(force, [0, 0, 0])
        np.testing.assert_array_equal(torque, [0, 0, 0])

    def test_near_full_compensation_is_not_rounded(self):
        _, properties = self.model([1 - 1e-7, 1])
        self.assertFalse(properties['disable_gravity'])
        self.assertTrue(needs_gravity_wrench(properties))


if __name__ == '__main__':
    unittest.main()
