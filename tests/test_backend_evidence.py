from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Hooke'))
from backends.evidence import compare_archives, compare_sources, file_sha256


class EvidenceTests(unittest.TestCase):
    def test_missing_state_fields_and_changed_precision_are_not_equal(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder)/'left.npz', Path(folder)/'right.npz'
            np.savez(left, qpos=np.array([1.], dtype=np.float64), qvel=[0.])
            np.savez(right, qpos=np.array([1.], dtype=np.float32))
            result = compare_archives(left, right)
            self.assertFalse(result['equal'])
            self.assertFalse(result['equal_fields']['qpos'])
            self.assertEqual(result['missing_fields']['right'], ['qvel'])

    def test_saved_nan_locations_compare_but_changed_values_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder)/'left.npz', Path(folder)/'right.npz'
            np.savez(left, qpos=[1., np.nan])
            np.savez(right, qpos=[1., np.nan])
            self.assertTrue(compare_archives(left, right)['equal'])
            np.savez(right, qpos=[2., np.nan])
            self.assertFalse(compare_archives(left, right)['equal'])

    def test_actual_asset_corruption_is_detected_even_with_matching_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder)/'left', Path(folder)/'right'
            for path in (left, right):
                path.mkdir()
                np.savez(path/'model.npz', body_mass=[1.])
                (path/'scene.xml').write_text('<mujoco/>')
                (path/'mesh.obj').write_text('v 0 0 0\n')
                (path/'scene.json').write_text(json.dumps(dict(names={'body': ['world']},
                    source_assets={'mesh.obj': {'sha256': file_sha256(path/'mesh.obj')}})))
            self.assertTrue(compare_sources(left, right)['equal'])
            (right/'mesh.obj').write_text('v 1 0 0\n')
            result = compare_sources(left, right)
            self.assertFalse(result['equal'])
            self.assertTrue(result['asset_manifests_equal'])
            self.assertEqual(result['invalid_asset_files']['right'], ['mesh.obj'])
            self.assertFalse(result['parity_qualified'])
