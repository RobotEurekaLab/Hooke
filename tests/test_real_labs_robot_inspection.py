"""A reachable inspection path must not hide the robot's collision boundaries."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class RobotInspectionTests(unittest.TestCase):
    def make_session(self):
        root, manifest = build_scene("waterloo_robohub")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        return InstrumentSession(model, manifest)

    def test_shoulder_inspection_moves_and_returns_without_robot_penetration(self):
        session = self.make_session()
        control = session.controls["robot_arm", "shoulder"]
        np.testing.assert_allclose([control.lower, control.upper], [-1.65, 1.65])
        target = session.equipment["robot_arm"]["inspection_targets"]["shoulder"]
        robot_geoms = {
            i
            for i in range(session.model.ngeom)
            if session.model.geom(i).name.startswith("robot_arm__")
        }
        for command in (target, 0.0, target, 0.0):
            session.data.ctrl[control.actuator] = command
            for _ in range(100):
                mujoco.mj_step(session.model, session.data, nstep=10)
                mujoco.mj_forward(session.model, session.data)
                penetrations = [
                    contact.dist
                    for contact in session.data.contact
                    if contact.dist < -1e-4
                    and (contact.geom1 in robot_geoms or contact.geom2 in robot_geoms)
                ]
                self.assertEqual(penetrations, [])
            measured = session.data.qpos[session.model.jnt_qposadr[control.joint]]
            self.assertLess(abs(measured - command), 1e-4)

    def test_bench_contact_still_blocks_an_unsafe_command(self):
        session = self.make_session()
        result = session.command("robot_arm", "shoulder", 0.825, duration=2)
        self.assertGreater(abs(result["error"]), 0.08)
        self.assertTrue(
            any(
                contact.dist < -1e-4
                and "handoff_station_top"
                in (
                    session.model.geom(contact.geom1).name,
                    session.model.geom(contact.geom2).name,
                )
                for contact in session.data.contact
            )
        )


if __name__ == "__main__":
    unittest.main()
