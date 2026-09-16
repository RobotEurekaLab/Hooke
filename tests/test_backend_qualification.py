"""Regression tests: incomplete evidence must never expose a backend as qualified."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Hooke'))
from backends.qualification import REQUIRED, evaluate


class QualificationTests(unittest.TestCase):
    def test_import_only_cannot_qualify(self):
        result = evaluate({'checks': {'geometry_and_units': {'passed': True, 'artifact': 'geometry.json'}}})
        self.assertEqual(result['status'], 'UNQUALIFIED')
        self.assertIn('source_task_action_is_necessary', result['unresolved'])

    def test_every_requirement_is_mandatory(self):
        complete = {key: {'passed': True, 'artifact': f'{key}.json'} for key in REQUIRED}
        self.assertEqual(evaluate({'checks': complete})['status'], 'QUALIFIED')
        for key in REQUIRED:
            partial = {k: dict(v) for k, v in complete.items()}
            partial[key]['passed'] = False
            self.assertEqual(evaluate({'checks': partial})['unresolved'], [key])
            partial[key] = {'passed': True}
            self.assertEqual(evaluate({'checks': partial})['status'], 'UNQUALIFIED')

    def test_truthy_strings_are_not_passes(self):
        self.assertEqual(evaluate({'checks': {key: {'passed': 'false', 'artifact': 'a'} for key in REQUIRED}})['status'], 'UNQUALIFIED')


if __name__ == '__main__':
    unittest.main()
