"""Physical axes, CAD component ownership and mounted force consistency."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Hooke'))

from microscopy.cad_assets import DEFAULT_ROOT, asset_evidence
from microscopy.cell_scene import CELL_CENTER, NEEDLE_BACK
from microscopy.kinematics import tool_axes, tool_commands
from microscopy.scene import HOMES, ORIGIN
from microscopy.tasks import make_task, CalibrationPickPlace


class MountedToolAxes(unittest.TestCase):
    def test_rotated_slides_reach_world_displacement_and_preserve_virtual_work(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><joint name="tool_x" type="slide" axis=".707106781 .707106781 0"/>
            <inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/>
            <body><joint name="tool_y" type="slide" axis="-.707106781 .707106781 0"/>
              <inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/>
              <body><joint name="tool_z" type="slide" axis="0 0 1"/>
                <geom type="sphere" size=".01" mass="1"/><site name="tip"/>
              </body></body></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        target = np.array([.0003, -.0002, .0001])
        values = tool_commands(model, 'tool', target)
        for name, value in values.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.site_xpos[0], target, atol=1e-12)
        force = np.array([10e-9, -20e-9, 30e-9])
        local_force = tool_axes(model, 'tool').T @ force
        self.assertAlmostEqual(float(local_force @ data.qpos), float(force @ target), delta=1e-22)


@unittest.skipUnless((DEFAULT_ROOT/'zaber-rhf/manifest.json').is_file()
                     and (DEFAULT_ROOT/'openframe/OF-LL-CORE/manifest.json').is_file(),
                     'Private manufacturer CAD is an optional local research fixture')
class ManufacturerAssembly(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'HOOKE_MICROSCOPY_ASSETS': 'cad',
                                                  'HOOKE_MICROSCOPY_ASSET_ROOT': str(DEFAULT_ROOT)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.task = make_task('cell_injection')
        self.task.reset(0)

    def test_only_correct_slide_components_follow_an_axis(self):
        t = self.task
        indices = [t.model.geom(f'injector_cad_zaber_rhf_{i}').id for i in (0, 18, 84, 42, 143)]
        initial = t.data.geom_xpos[indices].copy()
        t.command({'injector_y': .0004}, .4)
        delta = t.data.geom_xpos[indices]-initial
        np.testing.assert_allclose(delta[0], 0, atol=1e-12)
        expected = tool_axes(t.model, 'injector')[:, 1]*.0004
        np.testing.assert_allclose(delta[1:], np.tile(expected, (4, 1)), atol=2e-7)
        t.move('injector', HOMES['injector']+[.0003, -.0002, .0001], .3)
        tip = t.data.site_xpos[t.model.site('injector_tcp').id]
        np.testing.assert_allclose(tip, ORIGIN+HOMES['injector']+[.0003, -.0002, .0001], atol=2e-7)
        evidence = asset_evidence(t.model)
        self.assertEqual([x['tool'] for x in evidence['assemblies']], ['injector'])
        self.assertLess(evidence['microscope']['stage_pillar_hole_alignment_error_m'], 1e-9)

    def test_membrane_reaction_is_projected_to_actual_mounted_axes(self):
        t = self.task
        ray = t.mechanics.ray_radius_m
        t.move('injector', CELL_CENTER+NEEDLE_BACK*(ray-.7e-6), .4)
        world_force = tool_axes(t.model, 'injector') @ t.data.qfrc_applied[t.mechanics.axis_dofs]
        self.assertGreater(np.linalg.norm(world_force), 1e-9)
        np.testing.assert_allclose(world_force[:2]+t.data.qfrc_applied[t.mechanics.stage_dofs],
                                   0, atol=1e-18)
        self.assertFalse(t.mechanics.punctured)

    def test_unknown_or_incomplete_asset_profile_fails_before_simulation(self):
        with patch.dict(os.environ, {'HOOKE_MICROSCOPY_ASSETS': 'typo'}):
            with self.assertRaises(ValueError):
                make_task('cell_injection')
        with patch.dict(os.environ, {'HOOKE_MICROSCOPY_ASSET_ROOT': str(ROOT/'temp/missing-cad-fixture')}):
            with self.assertRaises(FileNotFoundError):
                make_task('cell_injection')


@unittest.skipUnless((DEFAULT_ROOT/'smaract-sgp17f/manifest.json').is_file(),
                     'Private parallel-gripper CAD is an optional research fixture')
class ParallelGripperAssembly(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'HOOKE_MICROSCOPY_ASSETS': 'cad',
                                                   'HOOKE_MICROSCOPY_ASSET_ROOT': str(DEFAULT_ROOT)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_fixed_jaw_stays_while_the_actual_carriage_and_finger_translate(self):
        t = CalibrationPickPlace.Expert(CalibrationPickPlace.load())
        t.reset(0)
        ids = [t.model.geom(f'gripper_cad_smaract_sgp17f_{i}').id for i in (0, 31, 36, 40, 41)]
        initial = t.data.geom_xpos[ids].copy()
        t.command({'jaw_a': .0009}, .4)
        delta = t.data.geom_xpos[ids]-initial
        # Compare to the mount's tiny servo reaction, not an immovable world.
        np.testing.assert_allclose(delta[0], delta[4], atol=1e-12)
        self.assertLess(np.linalg.norm(delta[0]), 1e-8)
        np.testing.assert_allclose(delta[[1, 2, 3]], np.tile([0, -.0004, 0], (3, 1)), atol=2e-6)
        self.assertNotIn('jaw_b', t.drives)
        self.assertEqual(asset_evidence(t.model)['gripper']['parts'], 56)

    def test_parallel_gripper_lifts_and_releases_through_actual_bilateral_contact(self):
        t = CalibrationPickPlace.Expert(CalibrationPickPlace.load())
        t.reset(1)
        t.execute()
        self.assertTrue(t.check(), t.microscopy_checks())
        self.assertGreater(t.mechanics.bilateral_contact_s, .05)
        self.assertGreater(t.mechanics.lift_m, .0008)
        self.assertFalse(t.mechanics.grasped)


if __name__ == '__main__':
    unittest.main()
