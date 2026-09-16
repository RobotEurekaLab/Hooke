"""Check frequency, duplicate observations and resets cannot change success."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from mani_pipette import Pipette
from process_progress import ProcessProgressSystem, PipetteTransferSequence
from simulation import Manager
from archetypes.task_catalog import CATALOG
from process_progress import ProcessProgress


class ProcessProgressTests(unittest.TestCase):
    def fixture(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
          <body name="tube"><geom type="cylinder" size=".01 .03"/></body>
          <body pos="1 0 0"><joint name="thumb" range="0 1"/>
            <geom type="sphere" size=".001"/></body>
          <body mocap="true"><site name="tip"/></body>
        </worldbody><actuator><position joint="thumb"/></actuator></mujoco>''')
        progress = ProcessProgressSystem()
        manager = Manager.from_model(model, [progress])
        manager.reload()
        task = SimpleNamespace(model=model, data=manager.data, default_task='pipette', progress=progress,
            object=SimpleNamespace(body_id=model.body('tube').id),
            arm1=SimpleNamespace(site_id=model.site('tip').id, thj3_qposadr=0, thj3_id=0),
            container=SimpleNamespace(container=SimpleNamespace(position=np.zeros(3), rotation_matrix=np.eye(3),
                liquid=SimpleNamespace(surface=SimpleNamespace(distance=.03), surface_normal=np.array([0., 0., 1.])))))
        task.check = MethodType(Pipette.check, task)
        progress.bind(task)
        manager.reset()
        return task, progress, manager

    def observe(self, task, progress, z, thumb):
        task.data.mocap_pos[0] = [0., 0., z]
        task.data.qpos[0] = thumb
        mujoco.mj_kinematics(task.model, task.data)
        task.data.time += .002
        progress.update(task.data)

    def test_sparse_and_frequent_checks_have_identical_history(self):
        histories = []
        for reads in (0, 100):
            task, progress, manager = self.fixture()
            self.assertFalse(task.check())
            for z, thumb in ((.02, .8), (.02, .4), (.091, .4)):
                self.observe(task, progress, z, thumb)
                before = deepcopy(progress.observer.state)
                clock = task.data.time
                for _ in range(reads):
                    task.check()
                self.assertEqual(before, progress.observer.state)
                self.assertEqual(clock, task.data.time)
            self.assertTrue(task.check())
            histories.append(deepcopy(progress.observer.state))
        self.assertEqual(*histories)

    def test_duplicate_update_and_reset_cannot_retain_success(self):
        task, progress, manager = self.fixture()
        for z, thumb in ((.02, .8), (.02, .4), (.091, .4)):
            self.observe(task, progress, z, thumb)
        self.assertTrue(task.check())
        samples = progress.observer.samples
        progress.update(task.data)
        self.assertEqual(samples, progress.observer.samples)
        manager.reset()
        self.assertFalse(task.check())
        self.assertEqual(progress.observer.state.events, {})

    def test_delivered_volume_without_aspiration_history_is_not_a_completed_transfer(self):
        sequence = PipetteTransferSequence(200e-9)
        state = dict(reservoirs={'destination': {'volume_m3': 200e-9}, 'tip': {'volume_m3': 0.},
                                'environment': {'volume_m3': 0.}},
                     initial_volumes_m3={'destination': 0.}, tip_reservoir='destination')
        sequence.update(1., state, 'destination')
        state['tip_reservoir'] = None
        sequence.update(2., state, 'destination')
        self.assertTrue(sequence.checks()['delivered_target_within_5pct'])
        self.assertFalse(sequence.checks()['ordered_aspirate_transfer_dispense_withdraw'])
        json.dumps(sequence.__dict__, default=lambda obj: obj.__dict__, allow_nan=False)

    def test_vortex_anchors_follow_the_randomized_reset_pose(self):
        task = CATALOG['vortex_mixer'].make_expert()
        for seed in (0, 4):
            task.reset(seed)
            mujoco.mj_forward(task.model, task.data)
            independent = ProcessProgress(task, 'vortex_mixer')
            observer = task.progress.observer
            np.testing.assert_array_equal(observer.return_position, independent.return_position)
            self.assertEqual(observer.initial_height, independent.initial_height)
            self.assertFalse(task.check())


if __name__ == '__main__':
    unittest.main()
