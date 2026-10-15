"""Keep failed isolated commands visible while testing their public interlocks.

Passing this assertion does not qualify the raw diagnostic or a full operation.
The latter requires its own measured, complete trajectory.
"""

import numpy as np

from real_labs.operations import available_operations, control_policy


BLOCKED_RAW_CONTROLS = {
    "nasa_planetary": {
        *(('rover', f'wheel_{side}_{index}') for side in ('left', 'right') for index in (1, 2, 3)),
        ('rover', 'arm_gripper_left'),
        ('rover', 'arm_gripper_right'),
    },
    "rochester_genomics": {
        ('liquid_handler', 'loader_x'), ('liquid_handler', 'loader_y'),
    },
}


def assert_raw_diagnostics(test, session, report):
    expected = BLOCKED_RAW_CONTROLS.get(session.manifest['id'], set())
    failed = {(row['equipment'], row['control']) for row in report['controls'] if not row['passed']}
    test.assertEqual(failed, expected, report)
    test.assertEqual(report['passed'], not bool(expected))
    test.assertEqual(report['control_count'], session.model.nu)
    policies = {(p['equipment_id'], p['control']): p for p in control_policy(session)}
    operations = {item['id'] for item in available_operations(session)}
    for row in report['controls']:
        key = row['equipment'], row['control']
        if key not in expected:
            continue
        test.assertGreater(abs(row['error']), row['tracking_tolerance'])
        test.assertEqual(policies[key]['mode'], 'operation_only')
        test.assertIn(policies[key]['operation'], operations)
        before = session.data.qpos.copy(), session.data.qvel.copy(), session.data.ctrl.copy(), session.data.time
        with test.assertRaises(ValueError):
            session.command(*key, row['target'])
        for actual, value in zip((session.data.qpos, session.data.qvel, session.data.ctrl), before[:3]):
            np.testing.assert_array_equal(actual, value)
        test.assertEqual(session.data.time, before[3])
