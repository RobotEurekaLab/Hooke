"""CPU execution of the native stepping boundary, without importing Isaac."""
import ast
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np
from backends.effort_runtime import AppliedEffortRecorder


RUNTIME_PATH = Path(__file__).resolve().parents[1] / 'Hooke/backends/isaac_runtime.py'


def runtime_methods():
    # Run production methods with a tiny observable world. This checks public
    # command and contact units; it makes no claim about the PhysX solver.
    tree = ast.parse(RUNTIME_PATH.read_text())
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NativeScene')
    names = {'step', '_step_physics', '_contact_report', '_contact_sample', 'reset', '_require_valid_state'}
    methods = [node for node in scene.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        'np': np, 'time': time,
        'PhysicsSchemaTools': SimpleNamespace(intToSdfPath=lambda value: value),
        'get_physx_interface': lambda: SimpleNamespace(update_transformations=lambda *args: None),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(RUNTIME_PATH), 'exec'), namespace)
    return type('RuntimeMethods', (), {node.name: namespace[node.name] for node in methods})


def fixture(substeps, explicit=False):
    runtime = runtime_methods()()
    runtime.dt = .002
    runtime.native_substeps = substeps
    runtime.physics_dt = runtime.dt / substeps
    runtime.steps = runtime.native_steps = 0
    runtime.time = runtime.physics_wall = 0.
    runtime.render_enabled = False
    runtime.managed_render = True
    runtime.contacts = []
    runtime.friction_contacts = []
    runtime.force_actuators = {0: 0}
    runtime.actuators = {}
    runtime.tendon_drives = {}
    runtime.friction_drives = {}
    runtime.dof_map = {0: (0, 0)}
    runtime.free = {}
    runtime.eq_active = np.array([], bool)
    runtime.m = {
        'eq_data': np.empty((0, 11)), 'reset_ctrl': np.zeros(1),
        'reset_qpos': np.zeros(1), 'reset_qvel': np.zeros(1),
        'qpos_spring': np.zeros(1), 'qpos0': np.zeros(1),
        'actuator_ctrlrange': np.array([[-100, 100]]), 'actuator_ctrllimited': np.array([False]),
        'actuator_gainprm': np.array([[1.]]), 'actuator_biasprm': np.array([[0., 2., 3.]]),
        'actuator_forcelimited': np.array([False]), 'actuator_trntype': np.array([0]),
        'jnt_qposadr': np.array([0]), 'jnt_dofadr': np.array([0]),
        'jnt_actfrclimited': np.array([False]), 'dof_frictionloss': np.array([0.]),
        'actuator_dyntype': np.array([0]), 'actuator_gaintype': np.array([0]),
        'actuator_biastype': np.array([1]), 'actuator_gear': np.array([[1., 0, 0, 0, 0, 0]]),
        'actuator_trnid': np.array([[0, -1]]), 'jnt_type': np.array([2]),
    }
    runtime.effort_recorder = AppliedEffortRecorder(runtime.m, explicit)
    runtime.bridge = SimpleNamespace(frame=SimpleNamespace(to_source=np.asarray))
    runtime.geom_ids = {'a': 0, 'b': 1}
    forces, gravity = [], []
    state = {'q': 0., 'v': 1.}
    view = SimpleNamespace(
        set_dof_position_targets=lambda *args: None,
        set_dof_velocity_targets=lambda *args: None,
        set_dof_actuation_forces=lambda values, ids: forces.append(float(values[0, 0])),
        get_dof_actuation_forces=lambda: np.array([[forces[-1]]], dtype=np.float32),
        set_dof_positions=lambda values, ids: state.update(q=float(values[0, 0])),
        set_dof_velocities=lambda values, ids: state.update(v=float(values[0, 0])),
    )
    runtime.views = [{'qa': np.array([0]), 'va': np.array([0]), 'ids': np.array([0]), 'view': view}]

    def observe():
        runtime._observed = {'qpos': [state['q']], 'qvel': [state['v']],
                             'time': runtime.time, 'contacts': runtime.contacts,
                             'friction_contacts': runtime.friction_contacts}
        runtime._observed['actuator_effort'] = runtime.effort_recorder.state
        return runtime._observed

    runtime.observe = observe
    runtime._apply_articulation_gravity = lambda: gravity.append(runtime._observed['qpos'][0])
    point = SimpleNamespace(
        position=SimpleNamespace(x=0, y=0, z=0), normal=SimpleNamespace(x=0, y=0, z=1),
        impulse=SimpleNamespace(x=0, y=0, z=2 * runtime.physics_dt), separation=0.,
    )
    header = SimpleNamespace(collider0='a', collider1='b', contact_data_offset=0, num_contact_data=1,
                             friction_anchors_offset=0, num_friction_anchors_data=1)
    anchor = SimpleNamespace(position=SimpleNamespace(x=.1, y=0, z=0),
                             impulse=SimpleNamespace(x=-runtime.physics_dt, y=0, z=0))

    def step_world(render):
        if render:
            raise AssertionError('Internal physics steps must not trigger rendering')
        runtime._contact_report([header], [point], [anchor])
        state['q'] += 1
        state['v'] += 1

    runtime.world = SimpleNamespace(
        step=step_world,
        physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None),
    )
    observe()
    return runtime, forces, gravity


class NativeSubstepContractTests(unittest.TestCase):
    def test_held_command_refreshes_state_dependent_forces_each_native_step(self):
        for count in (1, 4):
            with self.subTest(native_substeps=count):
                runtime, forces, gravity = fixture(count)
                command = np.array([7.])
                extra = np.array([5.])
                runtime.step(command, extra)
                np.testing.assert_array_equal(forces, 15 + 5 * np.arange(count))
                np.testing.assert_array_equal(gravity, np.arange(count))
                np.testing.assert_array_equal(command, [7.])
                np.testing.assert_array_equal(extra, [5.])
                self.assertEqual(runtime.steps, 1)
                self.assertEqual(runtime.native_steps, count)
                self.assertEqual(runtime.time, .002)

    def test_contact_mean_force_impulse_and_sample_time_are_substep_invariant(self):
        for count in (1, 4):
            with self.subTest(native_substeps=count):
                runtime, _, _ = fixture(count)
                for public_step in (1, 2):
                    result = runtime.step([7.])
                    contacts = result['contacts']
                    self.assertEqual(len(contacts), count)
                    self.assertAlmostEqual(sum(row['force'][2] for row in contacts), 2.)
                    self.assertAlmostEqual(sum(row['impulse_Ns'][2] for row in contacts), .004)
                    self.assertAlmostEqual(sum(row['force'][2] * row['force_interval_s'] for row in contacts), .004)
                    for index, row in enumerate(contacts):
                        self.assertEqual(row['substep_index'], index)
                        self.assertEqual(row['public_step'], public_step)
                        self.assertAlmostEqual(row['time_s'], (public_step - 1) * .002 + (index + 1) * runtime.physics_dt)
                        self.assertEqual(row['substep_force'], [0., 0., 2.])
                        self.assertEqual(row['physics_timestep_s'], runtime.physics_dt)
                    anchors = result['friction_contacts']
                    self.assertEqual(len(anchors), count)
                    self.assertAlmostEqual(sum(row['force'][0] for row in anchors), -1.)
                    self.assertAlmostEqual(sum(row['impulse_Ns'][0] for row in anchors), -.002)
                    for index, row in enumerate(anchors):
                        self.assertEqual((row['geom1'], row['geom2']), (1, 0))
                        self.assertEqual((row['collider1'], row['collider2']), ('b', 'a'))
                        self.assertEqual(row['pos'], [.1, 0., 0.])
                        self.assertEqual(row['substep_force'], [-1., 0., 0.])
                        self.assertEqual(row['time_s'], contacts[index]['time_s'])
                        self.assertNotIn('distance', row)
                self.assertEqual(runtime.native_steps, 2 * count)
                self.assertEqual(runtime.time, .004)

    def test_reset_clears_public_native_time_and_contacts_without_stepping(self):
        runtime, forces, _ = fixture(4)
        runtime.step([7.])
        count = len(forces)
        result = runtime.reset()
        self.assertEqual(runtime.steps, 0)
        self.assertEqual(runtime.native_steps, 0)
        self.assertEqual(runtime._substep_index, 0)
        self.assertEqual(result['time'], 0.)
        self.assertEqual(result['contacts'], [])
        self.assertEqual(result['friction_contacts'], [])
        self.assertEqual(result['qpos'], [0.])
        self.assertEqual(result['qvel'], [0.])
        self.assertEqual(len(forces), count + 1)
        self.assertEqual(forces[-1], 0.)

    def test_explicit_effort_separates_caller_input_and_retains_substep_peak(self):
        runtime, forces, _ = fixture(4, explicit=True)
        runtime.m['actuator_forcelimited'][0] = True
        runtime.m['actuator_forcerange'] = np.array([[-21., 21.]])
        runtime.m['jnt_actfrclimited'][0] = True
        runtime.m['jnt_actfrcrange'] = np.array([[-18., 18.]])
        result = runtime.step([7.], [5.])
        np.testing.assert_allclose(forces, [15., 20., 23., 23.])
        effort = result['actuator_effort']
        self.assertTrue(effort['native_input_verified'])
        self.assertEqual(effort['last'], [18.])
        self.assertEqual(effort['peak_abs'], [18.])
        self.assertEqual(effort['mean'], [15.25])
        self.assertEqual(effort['sample_count'], 4)
        self.assertEqual(effort['interval_s'], .002)
        reset = runtime.reset()['actuator_effort']
        self.assertEqual(reset['sample_count'], 0)
        self.assertEqual(reset['last'], [0.])
        self.assertTrue(reset['native_input_verified'])

    def test_explicit_effort_rejects_dropped_native_force_submission(self):
        runtime, _, _ = fixture(1, explicit=True)
        runtime.views[0]['view'].get_dof_actuation_forces = lambda: np.zeros((1, 1))
        with self.assertRaisesRegex(RuntimeError, 'does not match submission'):
            runtime.step([7.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
