"""Recorded phase replay must use actual samples and reject incomplete evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.keyframes import phase_indices, render_episode


class RecordedKeyframeTests(unittest.TestCase):
    def test_phase_endpoint_selects_a_recorded_sample_without_interpolation(self):
        times = np.array([.002, .004, .006])
        phases = [{'phase': 'aspirate', 'end_s': .0051}, {'phase': 'dispense', 'end_s': .006}]
        original = deepcopy(phases)
        np.testing.assert_array_equal(phase_indices(times, phases, ['aspirate', 'dispense']),
                                      [('aspirate', 2), ('dispense', 2)])
        np.testing.assert_array_equal(times, [.002, .004, .006])
        self.assertEqual(phases, original)

    def test_invalid_trajectory_times_cannot_produce_a_keyframe(self):
        for times in ([], [.004, .002], [.002, .002], [.002, np.nan], [[.002]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                phase_indices(times, [{'phase': 'aspirate', 'end_s': .002}], ['aspirate'])

    def test_missing_ambiguous_or_duplicate_phases_are_rejected(self):
        phase = {'phase': 'aspirate', 'end_s': .004}
        for phases, requested in (([], ['aspirate']), ([phase, phase], ['aspirate']),
                                  ([phase], ['aspirate', 'aspirate']), ([phase], [])):
            with self.subTest(phases=phases, requested=requested), self.assertRaises(ValueError):
                phase_indices([.002, .004], phases, requested)

    def test_phase_after_interrupted_recording_is_rejected(self):
        for end in (.001, .008, np.nan):
            with self.subTest(end=end), self.assertRaises(ValueError):
                phase_indices([.002, .004], [{'phase': 'return_source', 'end_s': end}], ['return_source'])

    def test_unrecorded_control_or_failed_startup_cannot_create_images(self):
        with TemporaryDirectory() as directory:
            episode = Path(directory)
            for mode, status in (('no_action', 'CONTROL_COMPLETE'), ('expert', 'ERROR'), ('expert', 'RUNNING')):
                (episode / 'result.json').write_text(json.dumps(dict(mode=mode, status=status)))
                with self.subTest(mode=mode, status=status), self.assertRaises(ValueError):
                    render_episode(episode, episode / 'images', ['aspirate'], 6)
                self.assertFalse((episode / 'images').exists())


if __name__ == '__main__':
    unittest.main()
