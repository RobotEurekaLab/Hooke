"""Concurrent arm recipes share one tick and cannot silently overwrite commands."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from control_streams import run_control_streams


class ControlStreamTests(unittest.TestCase):
    def test_unequal_stream_lengths_hold_targets_and_share_ticks(self):
        control = np.zeros(2)
        history = []
        run_control_streams(control, lambda: history.append(control.copy()),
                            iter([{0: 1.}, {}, {0: 2.}]), iter([{1: 3.}]))
        np.testing.assert_array_equal(history, [[1., 3.], [1., 3.], [2., 3.]])

    def test_conflicting_commands_fail_before_control_or_physics_changes(self):
        control = np.zeros(1)
        calls = []
        with self.assertRaises(ValueError):
            run_control_streams(control, lambda: calls.append(1), iter([{0: 1.}]), iter([{0: 2.}]))
        np.testing.assert_array_equal(control, [0.])
        self.assertEqual(calls, [])

    def test_empty_streams_do_not_advance_physics(self):
        calls = []
        run_control_streams(np.zeros(1), lambda: calls.append(1), iter([]), iter([]))
        self.assertEqual(calls, [])

    def test_commands_on_alternating_ticks_still_require_distinct_actuators(self):
        control = np.zeros(1)
        history = []
        with self.assertRaises(ValueError):
            run_control_streams(control, lambda: history.append(control.copy()),
                                iter([{0: 1.}, {}]), iter([{}, {0: 2.}]))
        np.testing.assert_array_equal(history, [[1.]])
        np.testing.assert_array_equal(control, [1.])

    def test_physics_failure_closes_all_streams(self):
        closed = []
        def actions(actuator):
            try:
                while True:yield {actuator: 1.}
            finally:
                closed.append(actuator)
        def fail():raise RuntimeError('physics failed')
        with self.assertRaises(RuntimeError):
            run_control_streams(np.zeros(2), fail, actions(0), actions(1))
        self.assertEqual(closed, [0, 1])
