"""Prevent cap-only grasps and moving rotor frames from producing false results."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.assessment import EpisodeAssessment, InsertionSequence
from archetypes.centrifuge_insertion import RotorSlotPoses


class ProcessAssessmentTests(unittest.TestCase):
    def test_a_grasp_on_the_cap_is_not_a_released_tube(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <site name="origin1"/>
          <body name="1/centrifuge_15ml_body" pos="0 0 1"><freejoint/><geom name="tube" type="sphere" size=".01"/></body>
          <body name="2/centrifuge_15ml_cap" pos="0 0 2"><freejoint/><geom name="cap" type="sphere" size=".01"/></body>
          <body name="/vortex_mixer_genie_2:platform" pos="1 0 0"><joint name="/vortex_mixer_genie_2:platform/pivot"/><geom type="sphere" size=".01"/></body>
          <body name="1/aloha:finger" pos="2 0 0"><geom name="finger" type="sphere" size=".01"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        def historical_check():
            return False
        task = SimpleNamespace(model=model, data=data, time_limit=30., check=historical_check)
        observer = EpisodeAssessment(task, 'vortex_mixer')
        contact = mujoco.MjContact()
        contact.geom[:] = [model.geom('cap').id, model.geom('finger').id]
        mujoco.mj_addContact(model, data, contact)
        data.time = .002
        observer.update()
        self.assertFalse(observer.report()['checks']['tube_released'])
        data.ncon = 0
        data.time += .002
        observer.update()
        self.assertTrue(observer.report()['checks']['tube_released'])
        json.dumps(observer.report(), allow_nan=False)

    def test_insertion_needs_release_orientation_and_stability(self):
        state = InsertionSequence()
        offset = np.array([.001, 0., 0.])
        for _ in range(400):
            state.update(.002, .959, offset, 0., True)
        self.assertFalse(state.checks()['seated_stable_500ms'])
        for _ in range(260):
            state.update(.002, .959, offset, 0., False)
        self.assertTrue(all(state.checks().values()))
        json.dumps(state.checks(), allow_nan=False)
        state.update(.002, .959, offset + [.001, 0., 0.], 0., False)
        self.assertFalse(state.checks()['seated_stable_500ms'])
        state.update(.002, .959, offset, .2, False)
        self.assertFalse(state.checks()['tube_axis_within_5deg'])

    def test_five_millimetre_insertion_tolerance_is_not_relaxed(self):
        state = InsertionSequence()
        for _ in range(400):
            state.update(.002, .959, np.array([.0051, 0., 0.]), 0., False)
        self.assertFalse(state.checks()['target_within_5mm'])
        self.assertFalse(state.checks()['seated_stable_500ms'])

    def test_slot_target_follows_actual_rotor_rotation(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="0 0 1"><joint name="rotor"/><geom type="sphere" size=".1"/>
            <site name="slot" pos=".1 0 0"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        instrument = RotorSlotPoses()
        instrument.num_slots = 1
        instrument.slot_sites = [model.site('slot').id]
        mujoco.mj_forward(model, data)
        old = instrument.get_tube_pose(data, 0, 'proximal')
        data.qpos[0] = np.pi / 2
        mujoco.mj_forward(model, data)
        current = instrument.get_tube_pose(data, 0, 'proximal')
        np.testing.assert_allclose(current.pos, [0., .1, .97], atol=1e-12)
        np.testing.assert_allclose(old.pos, [.1, 0., .97], atol=1e-12)
        with self.assertRaises(ValueError):
            instrument.get_tube_pose(data, 0, 'unknown')


if __name__ == '__main__':
    unittest.main()
