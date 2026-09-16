"""Geometry selects a reservoir; only an observed piston stroke moves volume."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from pipetting import PipetteTransferSystem
from liquid import ContainerSystem


class PipettingGeometryTests(unittest.TestCase):
    def test_volume_update_does_not_advance_surface_dynamics_or_add_a_log_tick(self):
        system = ContainerSystem.__new__(ContainerSystem)
        system.definition = SimpleNamespace(interior=SimpleNamespace(volume=1e-6))
        updates = []
        liquid = SimpleNamespace(volume=300e-9, surface_normal=np.array([0., 0., 1.]),
                                 surface=SimpleNamespace(distance=.003, valid=True), update_level=lambda: updates.append(1))
        def fail(*args, **kwargs):raise AssertionError('Surface dynamics advanced twice')
        class Container:
            @property
            def volume(self):return self.liquid.volume if self.liquid is not None else 0.
        system.container = Container()
        system.container.liquid = liquid
        system.container.update = fail
        system.log = SimpleNamespace(normal=[liquid.surface_normal], distance=[.003], present=[True], volume_m3=[300e-9])
        system.set_volume(None, 200e-9)
        self.assertEqual(liquid.volume, 200e-9)
        self.assertEqual(updates, [1])
        self.assertEqual(len(system.log.normal), 1)
        system.set_volume(None, 0.)
        self.assertIsNone(system.container.liquid)
        self.assertEqual(system.log.distance, [0.])
        self.assertEqual(system.log.present, [False])
        self.assertEqual(system.log.volume_m3, [0.])

    def container(self, position, volume):
        mesh = trimesh.creation.box(extents=[.01, .01, .01])
        state = SimpleNamespace(position=np.asarray(position), rotation_matrix=np.eye(3), volume=volume)
        state.liquid = SimpleNamespace(surface_normal=np.array([0., 0., 1.]),
                                       surface=SimpleNamespace(distance=.004)) if volume else None
        system = SimpleNamespace(container=state, definition=SimpleNamespace(_interior=mesh,
                                 interior=SimpleNamespace(volume=mesh.volume)))
        def set_volume(data, value):state.volume = value
        system.set_volume = set_volume
        return system

    def fixture(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="1 0 0">
          <joint name="plunger" type="slide" range="-.008 0"/>
          <geom type="sphere" size=".001"/>
        </body><body mocap="true"><site name="tip"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        source, target = self.container([0., 0., 0.], 500e-9), self.container([.03, 0., 0.], 0.)
        system = PipetteTransferSystem(source=source, destinations={'target': target},
                                       tip_site='tip', plunger_joint='plunger', tip_capacity_m3=200e-9)
        system.reload(model)
        mujoco.mj_forward(model, data)
        system.reset(data)
        return model, data, system, source, target

    def test_submerged_aspiration_and_dispense_to_an_empty_destination(self):
        model, data, system, source, target = self.fixture()
        data.qpos[0] = -.008
        mujoco.mj_kinematics(model, data)
        system.update(data)
        data.qpos[0] = 0.
        mujoco.mj_kinematics(model, data)
        system.update(data)
        self.assertAlmostEqual(source.container.volume, 300e-9)
        data.mocap_pos[0] = [.03, 0., 0.]
        data.qpos[0] = -.004
        mujoco.mj_kinematics(model, data)
        system.update(data)
        self.assertAlmostEqual(target.container.volume, 100e-9)
        self.assertAlmostEqual(system.pipette.ledger.total_m3, 500e-9)

    def test_releasing_outside_the_interior_does_not_aspirate(self):
        model, data, system, source, _ = self.fixture()
        data.mocap_pos[0] = [.1, 0., 0.]
        data.qpos[0] = -.008
        mujoco.mj_kinematics(model, data)
        system.update(data)
        data.qpos[0] = 0.
        mujoco.mj_kinematics(model, data)
        system.update(data)
        data.mocap_pos[0] = [0., 0., 0.]
        mujoco.mj_kinematics(model, data)
        system.update(data)
        self.assertEqual(source.container.volume, 500e-9)
        self.assertEqual(system.pipette.ledger.state('tip').volume_m3, 0.)
