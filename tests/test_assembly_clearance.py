"""A clear initial pose cannot establish clearance for a colliding motion."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import mujoco
except ImportError:
    mujoco = None

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Hooke'))
from microscopy.assembly_audit import recorded_native_state, resolve_motion_candidates


@unittest.skipUnless(mujoco is not None, 'MuJoCo provides archived-model kinematics; it is absent in the offline CAD environment')
class NativeArchiveValidation(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'
        source.mkdir()
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
            <worldbody><body><joint type="slide" axis="1 0 0"/>
            <geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>''')
        mujoco.mj_saveModel(model, str(source/'model.mjb'))
        np.savez_compressed(source/'model.npz', reset_qpos=np.array([0.]))
        hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ('model.mjb','model.npz')}
        (source/'scene.json').write_text(json.dumps(dict(archive_sha256=hashes)))
        np.savez_compressed(root/'trajectory.npz', qpos=np.array([[.001],[.002]]), time=np.array([.001,.002]))
        result = dict(backend='isaac', task='microscopy_push', status='TASK_SUCCEEDED', steps=2,
                      runtime=dict(steps=2, physics_events_since_load=2),
                      max_fk_position_error_m=1e-7, max_fk_rotation_error_rad=0.)
        (root/'result.json').write_text(json.dumps(result))
        return result

    def test_observed_joint_archive_is_loaded_without_source_physics(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            root = Path(folder)
            self.fixture(root)
            with patch.object(mujoco, 'mj_step', side_effect=AssertionError('Source physics must not run')):
                model, data, qpos, evidence = recorded_native_state(root, 'push')
                data.qpos[:] = qpos[-1]
                mujoco.mj_kinematics(model, data)
            self.assertAlmostEqual(data.geom_xpos[0,0], .002)
            self.assertEqual(evidence['actual_physics_steps'], 2)
            self.assertAlmostEqual(evidence['native_pair_pose_error_allowance_m'], 2e-7)

    def test_live_native_trace_binds_every_step_and_rejects_tampering(self):
        from types import SimpleNamespace
        from microscopy.native_trace import archive_native_trace

        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            root = Path(folder)
            cli = self.fixture(root)
            adapter = SimpleNamespace(rows=[dict(qpos=np.array([.001]), time=.001),
                                            dict(qpos=np.array([.002]), time=.002)],
                                      max_fk_position_error=1e-7, max_fk_rotation_error=0.)
            adapter.loaded = {"conversion": {"physics_options": {"world_origin_m": [0., -.07, .995]}}}
            trace = archive_native_trace(adapter, cli['runtime'], root)
            self.assertEqual(trace["world_origin_m"], [0., -.07, .995])
            self.assertEqual(trace["observation_coordinates"], "source world, metres")
            report = dict(operation='push', checks=dict(complete=True), native_trace=trace,
                          execution=dict(backend='isaac', native_runtime=cli['runtime']))
            (root/'result.json').write_text(json.dumps(report))
            with patch.object(mujoco, 'mj_step', side_effect=AssertionError('Source physics must not run')):
                _, _, qpos, evidence = recorded_native_state(root, 'push')
            np.testing.assert_allclose(qpos[:,0], [.001,.002])
            self.assertEqual(evidence['actual_physics_steps'], 2)
            with (root/'trajectory.npz').open('ab') as archive:
                archive.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'archived hash'):
                recorded_native_state(root, 'push')
            with self.assertRaisesRegex(ValueError, 'physics events differ'):
                archive_native_trace(adapter, dict(steps=2, physics_events_since_load=1), root)

    def test_modified_model_and_incomplete_physics_events_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            root = Path(folder)
            result = self.fixture(root)
            result['runtime']['physics_events_since_load'] = 1
            (root/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                recorded_native_state(root, 'push')
            result['runtime']['physics_events_since_load'] = 2
            (root/'result.json').write_text(json.dumps(result))
            np.savez_compressed(root/'source/model.npz', reset_qpos=np.array([.5]))
            with self.assertRaises(ValueError):
                recorded_native_state(root, 'push')

    def test_missing_ticks_and_wrong_task_cannot_certify_a_native_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as folder:
            root = Path(folder)
            self.fixture(root)
            with self.assertRaises(ValueError):
                recorded_native_state(root, 'injection')
            np.savez_compressed(root/'trajectory.npz', qpos=np.array([[.001],[.002]]), time=np.array([.001,.003]))
            with self.assertRaises(ValueError):
                recorded_native_state(root, 'push')


@unittest.skipUnless(importlib.util.find_spec('fcl'), 'FCL is an optional offline CAD inspection dependency')
class TranslationalClearance(unittest.TestCase):
    def report(self, displacement, correlated=False):
        with tempfile.TemporaryDirectory(dir=ROOT/'temp') as directory:
            root = Path(directory)
            positions = np.array([[0., 0., 0.], [.05, 0., 0.]])
            names = ['moving', 'fixed']
            segments = {}
            if correlated:
                positions = np.vstack((positions, [.015, 0., 0.], [.075, 0., 0.]))
                names += ['moving_rep', 'fixed_rep']
                starts = np.array([positions[[2, 3]]+[i*displacement/20, 0., 0.] for i in range(20)])
                segments = dict(segment_map=np.array([0, 1, 0, 1]), segment_geoms=np.array([2, 3]),
                                segment_positions=starts, segment_min=starts,
                                segment_max=starts+[displacement/20, 0., 0.])
            maximum = positions.copy()
            if correlated:
                maximum[:, 0] += displacement
            else:
                maximum[0, 0] = displacement
            path = root/'geometry.npz'
            np.savez_compressed(path, names=np.array(names),
                geom_type=np.full(len(names), 6), sizes=np.full((len(names), 3), .005),
                rotations=np.tile(np.eye(3), (len(names), 1, 1)), positions=positions,
                positions_min=positions, positions_max=maximum, **segments)
            report = dict(geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                motion_steps=50, skipped=[], motion_scope='Synthetic rigid translation fixture',
                motion_envelopes={'moving vs fixed':dict(candidate_pairs=[['moving', 'fixed']])})
            (root/'report.json').write_text(json.dumps(report))
            return resolve_motion_candidates(root)

    def test_distance_margin_establishes_clearance_for_a_short_motion(self):
        report = self.report(.01)
        self.assertTrue(report['recorded_motion_mesh_clearance_established'])
        evidence = report['motion_envelopes']['moving vs fixed']['cleared_by_distance_bound'][0]
        self.assertAlmostEqual(evidence['clearance_lower_bound_m'], .03, places=8)

    def test_clear_reset_pose_does_not_pass_when_the_object_moves_into_an_obstacle(self):
        report = self.report(.05)
        self.assertFalse(report['recorded_motion_mesh_clearance_established'])
        self.assertEqual(len(report['motion_envelopes']['moving vs fixed']['unresolved']), 1)

    def test_segmented_motion_uses_body_representatives_and_their_constant_geom_offsets(self):
        report = self.report(.1, correlated=True)
        self.assertTrue(report['recorded_motion_mesh_clearance_established'])
        evidence = report['motion_envelopes']['moving vs fixed']['cleared_by_distance_bound'][0]
        self.assertLess(evidence['clearance_lower_bound_m'], 0)
        self.assertAlmostEqual(evidence['segmented_clearance_lower_bound_m'], .03, places=8)


if __name__ == '__main__':
    unittest.main()
