"""Continuous inspection rotation, distinct from centrifugation performance."""

import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.reuse import MODEL_ROOT, attach_asset
from real_labs.reuse_drives import MINI_CENTRIFUGE, add_inspection_drive
from real_labs.runtime import InstrumentSession


def make_session():
    root = ET.fromstring(
        '<mujoco><compiler angle="radian" autolimits="true"/>'
        '<option gravity="0 0 -9.81" timestep=".002" integrator="implicitfast"/>'
        '<default><joint damping="2" armature=".005"/></default>'
        "<asset/><worldbody/><actuator/></mujoco>"
    )
    record = attach_asset(
        root, root.find("worldbody"), MINI_CENTRIFUGE, "centrifuge", [0, 0, 0]
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, {"equipment": [record]})


class ReusedRotorDriveTests(unittest.TestCase):
    def test_speed_semantics_are_explicit_and_lid_stays_position_controlled(self):
        session = make_session()
        rotor = session.controls["centrifuge", "rotor"]
        lid = session.controls["centrifuge", "lid"]
        self.assertEqual((rotor.mode, rotor.units), ("velocity", "rad/s"))
        self.assertEqual((lid.mode, lid.units), ("position", "rad"))
        np.testing.assert_allclose([rotor.lower, rotor.upper], [-math.tau, math.tau])
        self.assertFalse(session.model.jnt_limited[rotor.joint])
        contract = session.equipment["centrifuge"]["control_contracts"]["rotor"]
        self.assertEqual(contract["source_joint_range_rad"], [-180, 180])
        self.assertIn("not manufacturer ratings", contract["parameter_basis"])
        self.assertIn("scientific centrifugation", contract["limitations"])

    def test_full_speed_reversals_stop_and_reset_without_large_torque(self):
        session = make_session()
        rotor = session.controls["centrifuge", "rotor"]
        np.testing.assert_allclose(
            session.model.actuator_forcerange[rotor.actuator], [-0.05, 0.05]
        )
        for target in (-math.tau, math.tau, -math.pi, 0, math.tau, 0):
            result = session.command("centrifuge", "rotor", target, duration=2)
            self.assertLess(abs(result["error"]), 0.001, result)
            self.assertLessEqual(
                abs(session.data.actuator_force[rotor.actuator]), 0.05 + 1e-12
            )
        session.reset()
        self.assertEqual(session.data.qpos[session.model.jnt_qposadr[rotor.joint]], 0)
        self.assertEqual(session.data.qvel[session.model.jnt_dofadr[rotor.joint]], 0)
        self.assertEqual(session.data.ctrl[rotor.actuator], 0)

    def test_continuous_rotation_crosses_both_old_angle_stops(self):
        session = make_session()
        for direction in (-1, 1):
            session.reset()
            result = session.command(
                "centrifuge", "rotor", direction * math.tau, duration=30
            )
            self.assertGreater(direction * result["joint_position"], 180)
            self.assertLess(abs(result["error"]), 1e-5)
            stopped = session.command("centrifuge", "rotor", 0, duration=2)
            self.assertLess(abs(stopped["measured"]), 1e-5)

    def test_original_asset_and_other_manual_joints_are_unchanged(self):
        source = MODEL_ROOT / MINI_CENTRIFUGE
        original = source.read_bytes()
        session = make_session()
        self.assertEqual(source.read_bytes(), original)
        original_joint = ET.fromstring(original).find(".//joint[@name='rotor']")
        self.assertEqual(original_joint.get("range"), "-180 180")
        self.assertIsNone(original_joint.get("limited"))
        lid = session.controls["centrifuge", "lid"]
        np.testing.assert_allclose([lid.lower, lid.upper], [0, 1.712])
        actuator = ET.Element("actuator")
        joint = ET.Element("joint", name="other__rotor", range="-180 180")
        before = ET.tostring(joint)
        self.assertIsNone(
            add_inspection_drive(
                actuator, "instrument/other.xml", "rotor", joint, "drive"
            )
        )
        self.assertEqual(ET.tostring(joint), before)
        self.assertEqual(len(actuator), 0)


if __name__ == "__main__":
    unittest.main()
