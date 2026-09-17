"""Changing model ownership must retain live state and reject duplicate systems."""

from pathlib import Path
import sys
import unittest
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from simulation import Manager, System


class OwnedSystem(System):
    def _configure(self):
        self.resets = 0

    def _reset(self, data):
        self.resets += 1


class SystemReplacementTests(unittest.TestCase):
    def test_replacement_updates_ownership_without_resetting_scene(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        old, new = OwnedSystem(), OwnedSystem()
        manager = Manager.from_model(model, [old])
        manager.reload()
        manager.reset()
        manager.data.qpos[0] = 0.7
        manager.set_systems([new])
        self.assertEqual(manager.data.qpos[0], 0.7)
        self.assertEqual(old.resets, 1)
        self.assertEqual(new.resets, 0)
        self.assertIs(new.manager, manager)
        self.assertEqual(manager.systems_by_type[OwnedSystem], [new])
        self.assertNotIn(old, manager.systems_by_type[System])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manager.set_systems([new, new])
        self.assertEqual(manager.systems, (new,))


if __name__ == "__main__":
    unittest.main()
