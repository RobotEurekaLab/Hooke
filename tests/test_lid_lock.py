"""An activated constraint cannot count as success while the lid is open."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from archetypes.lid_lock import lid_lock_passes
from archetypes.task_catalog import CATALOG


class LidLockTests(unittest.TestCase):
    def test_force_lock_without_actual_closure_fails(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <worldbody><body><joint name="lid" range="-1.8 0"/>
            <geom type="sphere" size=".01"/></body></worldbody>
          <equality><joint name="lock" joint1="lid" polycoef="-1.7" active="false"/></equality>
        </mujoco>''')
        data = mujoco.MjData(model)
        instrument = SimpleNamespace(model=model, lid_lock=model.equality('lock').id, lid_qposadr=0)
        self.assertFalse(lid_lock_passes(data, instrument))
        data.eq_active[:] = True
        self.assertFalse(lid_lock_passes(data, instrument))
        data.qpos[0] = -1.7
        self.assertTrue(lid_lock_passes(data, instrument))
        data.eq_active[:] = False
        self.assertFalse(lid_lock_passes(data, instrument))

    def test_catalogue_lid_tasks_start_unsuccessful(self):
        for name in ('centrifuge_5430_close_lid', 'centrifuge_5910_lid_close',
                     'composite_centrifuge_5430_close_lid','centrifuge_mini_close_lid'):
            task_cls, _ = CATALOG[name].load_classes()
            task = task_cls(task_cls.load())
            if CATALOG[name].task_override:
                task.task = CATALOG[name].task_override
            task.reset(0)
            self.assertFalse(task.check(), name)


if __name__ == '__main__':
    unittest.main()
