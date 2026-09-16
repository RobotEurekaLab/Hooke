"""Prevent false positives from gravity, action ordering and missing evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.assessment import EpisodeAssessment, PipetteSequence, SashManipulation, VortexSequence, literal_predicate
from backends.matrix import summarize


class AssessmentTests(unittest.TestCase):
    def test_pipette_order_and_report_are_pure(self):
        state = PipetteSequence()
        state.update(1., .01, .002, True, .8)
        state.update(2., .01, .002, True, .4)
        state.update(3., -.06, .002, True, .4)
        self.assertTrue(all(state.checks().values()))
        task = SimpleNamespace(data=SimpleNamespace(time=0), model=None, time_limit=30,
                               check=lambda: (_ for _ in ()).throw(AssertionError('must not call check')))
        observer = EpisodeAssessment(task, 'fixture')
        observer.state, observer.samples = state, 3
        before = deepcopy(state)
        self.assertEqual(observer.report(), observer.report())
        self.assertEqual(state, before)

    def test_release_in_air_or_outside_tube_is_not_aspiration(self):
        for radial, depth in ((.03, .01), (.002, -.01)):
            state = PipetteSequence()
            state.update(1., .01, .002, True, .8)
            state.update(2., depth, radial, True, .4)
            state.update(3., .01, .002, True, .4)
            state.update(4., -.06, .002, True, .4)
            self.assertFalse(all(state.checks().values()))

    def test_empty_container_cannot_complete_an_aspiration_sequence(self):
        state = PipetteSequence()
        state.update(1., .01, .002, True, .8)
        state.update(2., .01, .002, True, .4)
        state.update(3., -.06, .002, True, .4, liquid_available=False)
        self.assertFalse(all(state.checks().values()))
        self.assertEqual(state.events, {})

    def test_gravity_closed_sash_and_late_contact_fail(self):
        state = SashManipulation(.18, 0.)
        state.update(0., False, .3)
        for _ in range(100):
            state.update(0., True, .002)
        self.assertTrue(state.checks()['target_within_20mm'])
        self.assertFalse(state.checks()['travel_in_contact_50mm'])

    def test_contact_motion_passes_but_jitter_does_not_accumulate(self):
        state = SashManipulation(.18, 0.)
        for i in range(101):
            state.update(.18 * (1 - i / 100), True, .002)
        self.assertTrue(all(state.checks().values()))
        state = SashManipulation(.01, 0.)
        for _ in range(100):
            state.update(.01, True, .002)
            state.update(0., True, .002)
        self.assertLess(state.travel_m, .011)

    def test_vortex_requires_continuous_contact_with_moving_platform(self):
        for contact, speed in ((False, 10.), (True, 0.)):
            state = VortexSequence()
            for _ in range(600):
                state.update(.002, .1, contact, speed, .1, True)
            state.update(.002, 0., False, 0., 0., False)
            self.assertFalse(all(state.checks().values()))
        state = VortexSequence()
        for _ in range(300):
            state.update(.002, .1, True, 10., .1, True)
        state.update(.002, 0., False, 0., 0., False)
        self.assertTrue(all(state.checks().values()))
        state = VortexSequence()
        for _ in range(10):
            state.update(.1, .1, True, 10., .1, True)
            state.update(.1, .1, False, 10., .1, True)
        self.assertLess(state.longest_active_s, .5)

    def test_constant_predicate_detection_does_not_guess_branches(self):
        def constant():
            """A placeholder."""
            return True
        def conditional():
            if True:
                return True
            return False
        self.assertIs(literal_predicate(constant), True)
        self.assertIsNone(literal_predicate(conditional))

    def test_missing_and_crashed_pairs_are_not_agreement(self):
        parameters = {'tasks': ['fixture'], 'seeds': [0, 1],
                      'modes': ['expert'], 'backends': ['mujoco', 'isaac']}
        rows = [{'task': 'fixture', 'seed': 0, 'mode': 'expert', 'backend': 'mujoco',
                 'status': 'TASK_FAILED', 'source_success': False},
                {'task': 'fixture', 'seed': 0, 'mode': 'expert', 'backend': 'isaac',
                 'status': 'PROCESS_ERROR'}]
        report = summarize(rows, parameters)
        self.assertEqual(report['requested_episodes'], 4)
        self.assertTrue(all(p['legacy_agreement'] is None for p in report['pairs']))
        self.assertEqual(report['groups'][1]['completed'], 0)
        self.assertEqual(report['groups'][1]['requested'], 2)

    def test_display_completion_is_not_a_crash_or_valid_predicate_agreement(self):
        parameters = {'tasks': ['display'], 'seeds': [0],
                      'modes': ['expert'], 'backends': ['mujoco', 'isaac']}
        rows = [{'task': 'display', 'seed': 0, 'mode': 'expert', 'backend': backend,
                 'status': 'DISPLAY_COMPLETE', 'source_success': True,
                 'assessment': {'legacy_constant': True, 'success': None}}
                for backend in parameters['backends']]
        report = summarize(rows, parameters)
        self.assertTrue(report['pairs'][0]['completed'])
        self.assertIsNone(report['pairs'][0]['legacy_agreement'])
        self.assertIsNone(report['pairs'][0]['assessment_agreement'])
        self.assertEqual(report['groups'][0]['legacy_constant_count'], 1)


if __name__ == '__main__':
    unittest.main()
