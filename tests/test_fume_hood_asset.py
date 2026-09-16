"""The counterbalanced sash must stay put freely and still respond to force."""
from pathlib import Path
import unittest
import mujoco

ASSET = Path(__file__).resolve().parents[1] / 'Hooke/model/instrument/fume_hood.xml'


class FumeHoodPhysicsTests(unittest.TestCase):
    def test_sash_holds_midstroke_but_is_not_locked(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET))
        data = mujoco.MjData(model)
        joint = model.joint('sash_slide')
        qa, va = int(joint.qposadr[0]), int(joint.dofadr[0])
        data.qpos[qa] = .09
        for _ in range(round(15 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        self.assertAlmostEqual(data.qpos[qa], .09, places=8)
        for force in (-5., 5.):
            mujoco.mj_resetData(model, data)
            data.qpos[qa] = .09
            data.qfrc_applied[va] = force
            for _ in range(round(.2 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            self.assertGreater(force * (data.qpos[qa] - .09), .005)
        # A control with compensation disabled must fall under the same
        # gravity; the hold above cannot be attributed to a frozen scene.
        model.body_gravcomp[:] = 0.
        mujoco.mj_resetData(model, data)
        data.qpos[qa] = .09
        for _ in range(round(1 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        self.assertLess(data.qpos[qa], .02)


if __name__ == '__main__':
    unittest.main()
