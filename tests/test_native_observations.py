"""Native body-pose boundary tests with valid finite joint state, on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


RUNTIME_PATH = Path(__file__).resolve().parents[1] / 'Hooke/backends/isaac_runtime.py'


def runtime_with_transform(transform):
    tree = ast.parse(RUNTIME_PATH.read_text())
    scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NativeScene')
    methods = [node for node in scene.body if isinstance(node, ast.FunctionDef)
               and node.name in ('observe', '_require_valid_state')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(RUNTIME_PATH), 'exec'), namespace)
    runtime = SimpleNamespace()
    runtime._require_valid_state = namespace['_require_valid_state'].__get__(runtime)
    runtime.m = {'reset_qpos': np.zeros(1), 'reset_qvel': np.zeros(1), 'qpos0': np.zeros(1),
                 'eq_type': np.array([], dtype=int)}
    runtime.eq_active = np.array([], dtype=bool)
    runtime.bridge = SimpleNamespace(frame=SimpleNamespace(to_source=np.asarray))
    view = SimpleNamespace(get_dof_positions=lambda: np.zeros((1, 1)),
                           get_dof_velocities=lambda: np.zeros((1, 1)),
                           get_link_transforms=lambda: np.asarray([[transform]], dtype=float))
    runtime.views = [{'qa': np.array([0]), 'va': np.array([0]), 'view': view, 'body_ids': [17]}]
    runtime.free = {}
    runtime.time = 0.
    runtime.contacts = []
    runtime.friction_contacts = []
    runtime.effort_recorder = SimpleNamespace(state={'available': False})
    runtime._observed = {'sentinel': 'last valid observation'}
    return runtime, namespace['observe']


class NativeObservationTests(unittest.TestCase):
    def test_rejects_nonfinite_body_pose_even_when_joint_state_is_finite(self):
        for field in range(7):
            for bad_value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(field=field, bad_value=bad_value):
                    transform = [.1, .2, .3, 0., 0., 0., 1.]
                    transform[field] = bad_value
                    runtime, observe = runtime_with_transform(transform)
                    with self.assertRaisesRegex(RuntimeError, 'body pose.*17'):
                        observe(runtime)
                    self.assertEqual(runtime._observed, {'sentinel': 'last valid observation'})

    def test_rejects_exact_zero_quaternion(self):
        runtime, observe = runtime_with_transform([.1, .2, .3, 0., 0., 0., 0.])
        with self.assertRaisesRegex(RuntimeError, 'zero quaternion'):
            observe(runtime)
        self.assertEqual(runtime._observed, {'sentinel': 'last valid observation'})

    def test_preserves_finite_nonzero_quaternion_without_precision_threshold(self):
        for scalar in (1., 1.000002, 1e-250):
            with self.subTest(quaternion_scalar=scalar):
                runtime, observe = runtime_with_transform([.1, .2, .3, 0., 0., 0., scalar])
                result = observe(runtime)
                self.assertEqual(result['body_poses'][17], [.1, .2, .3, scalar, 0., 0., 0.])
                self.assertEqual(result['qpos'], [0.])
                self.assertEqual(result['qvel'], [0.])
                self.assertIs(runtime._observed, result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
