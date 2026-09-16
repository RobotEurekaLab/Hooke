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
from liquid import ContainerSystem, Container, LiquidState
from meshplane import Mesh
from backends.volume_assessment import PipetteVolumeAssessment


class PipettingGeometryTests(unittest.TestCase):
    def test_small_aspiration_strokes_change_the_geometric_surface_volume(self):
        mesh = trimesh.creation.cylinder(radius=.02, height=.05, sections=64)
        boundary = np.flatnonzero(mesh.vertices[:, 2] > .024)
        definition = Mesh(mesh.vertices, mesh.faces.astype(np.uint64), boundary.astype(np.uint64))
        liquid = LiquidState.create(definition, definition.volume*.5, np.array([0., 0., 9.81]), .002)
        liquid.update_level()
        initial = liquid.volume
        for step in range(1, 201):
            liquid.volume = initial-step*1e-9
            liquid.update_level()
            geometric_volume = liquid.meshplane.calculate_volume(liquid.surface.distance)[0]
            self.assertAlmostEqual(geometric_volume, liquid.volume, delta=1e-12)

        for normal in (np.array([0., 0., 9.81]), np.array([9.81, 0., 0.])):
            for volume in (1e-9, definition.volume*.8):
                with self.subTest(normal=normal, volume=volume):
                    liquid = LiquidState.create(definition, volume, normal, .002)
                    liquid.update_level()
                    geometric_volume = liquid.meshplane.calculate_volume(liquid.surface.distance)[0]
                    self.assertAlmostEqual(geometric_volume, volume, delta=1e-12)

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
        definition = SimpleNamespace(_interior=mesh, interior=Mesh(mesh.vertices, mesh.faces.astype(np.uint64),
            np.flatnonzero(mesh.vertices[:, 2] > .004).astype(np.uint64)))
        system = ContainerSystem.__new__(ContainerSystem)
        system.definition = definition
        system.container = Container(definition, None, None, None, False, None, .002)
        system.container.update(np.asarray(position), np.eye(3), np.array([0., 0., 9.81]), volume)
        system.log = SimpleNamespace(normal=[], distance=[], present=[], volume_m3=[])
        system._acceleration = lambda data: np.array([0., 0., 9.81])
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
        data.mocap_pos[0] = [0., 0., -.001]
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
        geometric_volume = target.container.liquid.meshplane.calculate_volume(target.container.liquid.surface.distance)[0]
        self.assertAlmostEqual(geometric_volume, 100e-9, delta=1e-12)
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

    def test_complete_transfer_goal_checks_both_container_surfaces(self):
        model, data, system, source, target = self.fixture()
        system.target_reservoir = 'target'
        observer = PipetteVolumeAssessment(system)
        for position, stroke in (([0., 0., -.001], -.008), ([0., 0., -.001], 0.),
                                 ([.03, 0., 0.], -.008), ([.1, 0., 0.], 0.)):
            data.mocap_pos[0] = position
            data.qpos[0] = stroke
            mujoco.mj_kinematics(model, data)
            system.update(data)
            observer.update()
        report = observer.report()
        self.assertTrue(report['success'])
        self.assertEqual(report['version'], 'hooke-ideal-volume-v3')
        self.assertAlmostEqual(target.container.volume, 200e-9)
        self.assertEqual(system.pipette.ledger.state('tip').volume_m3, 0.)
        # Corrupt the actual receiver plane while leaving its cached volume
        # and the source plane unchanged: independent geometry must catch it.
        target.container.liquid.surface.distance += .001
        observer.update()
        report = observer.report()
        self.assertTrue(report['checks']['surface_volume_matches_ledger'])
        self.assertFalse(report['checks']['geometric_surface_volume_matches_ledger'])
        self.assertLessEqual(report['metrics']['reservoir_errors_m3']['source']['geometric_m3'], 1e-12)
        self.assertGreater(report['metrics']['reservoir_errors_m3']['target']['geometric_m3'], 1e-12)

    def test_receiver_button_release_reaspirates_and_fails_transfer_goal(self):
        model, data, system, source, target = self.fixture()
        system.target_reservoir = 'target'
        for position, stroke in (([0., 0., -.001], -.008), ([0., 0., -.001], 0.),
                                 ([.03, 0., 0.], -.008), ([.03, 0., -.004], 0.)):
            data.mocap_pos[0] = position
            data.qpos[0] = stroke
            mujoco.mj_kinematics(model, data)
            system.update(data)
        observer = PipetteVolumeAssessment(system)
        observer.update()
        self.assertFalse(observer.report()['checks']['tip_residual_within_1nl'])
        self.assertFalse(observer.report()['success'])
