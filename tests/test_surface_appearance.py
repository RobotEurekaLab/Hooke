"""Appearance refresh may not inherit success from a changed physics model."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from backends.baseline import write_snapshot
from surface.render_contract import check_physics


class AppearanceContract(unittest.TestCase):
    def snapshot(self, directory, *, mass=1, friction=1, gravity=1.62, scenery=""):
        directory.mkdir()
        source = directory / "input.xml"
        source.write_text(
            f'<mujoco><option gravity="0 0 -{gravity}"/>'
            f'<worldbody>{scenery}<body name="robot" pos="0 0 1">'
            '<freejoint name="root"/>'
            f'<geom name="foot" type="sphere" size=".1" mass="{mass}" friction="{friction} .01 .001"/>'
            "</body></worldbody></mujoco>"
        )
        spec = mujoco.MjSpec.from_file(str(source))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        expert = SimpleNamespace(
            spec=spec,
            model=model,
            data=data,
            task="test",
            task_info={},
            default_scene=source,
        )
        write_snapshot(expert, directory)
        return directory

    def test_visual_scenery_accepts_changed_geometry_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.snapshot(root / "before")
            right = self.snapshot(
                root / "after",
                scenery='<geom name="scenery" type="box" size="1 1 1" contype="0" conaffinity="0" mass="0"/>',
            )
            self.assertTrue(check_physics(left, right)["passed"])

    def test_changed_mass_cannot_reuse_the_recorded_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.snapshot(root / "before")
            right = self.snapshot(root / "after", mass=2)
            with self.assertRaisesRegex(ValueError, "changed physics"):
                check_physics(left, right)

    def test_changed_contact_friction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.snapshot(root / "before")
            right = self.snapshot(root / "after", friction=0.2)
            with self.assertRaisesRegex(ValueError, "geom_friction"):
                check_physics(left, right)

    def test_changed_gravity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.snapshot(root / "before")
            right = self.snapshot(root / "after", gravity=3.73)
            with self.assertRaisesRegex(ValueError, "gravity"):
                check_physics(left, right)


if __name__ == "__main__":
    unittest.main()
