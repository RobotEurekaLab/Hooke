"""Wheel inspection drives under viscous, brake and rolling-contact loads."""

import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.equipment import _Builder, _wheel
from real_labs.runtime import InstrumentSession


def wheel_fixture(*, brake=0, rolling=False):
    root = ET.fromstring(
        '<mujoco><compiler angle="radian" autolimits="true"/>'
        '<option timestep=".002" integrator="implicitfast" gravity="0 0 -9.81" iterations="80"/>'
        '<default><geom solref=".008 1" friction=".8 .01 .001"/></default>'
        "<asset/><worldbody/><actuator/></mujoco>"
    )
    world = root.find("worldbody")
    carrier = ET.SubElement(world, "body", name="fixture", pos="0 0 .145")
    if rolling:
        # A guided 2 kg load can roll horizontally and settle vertically onto
        # the floor. The dampers model fixture drag, not a surveyed robot.
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="20 20 .1",
            contype="1",
            conaffinity="1",
        )
        ET.SubElement(
            carrier,
            "joint",
            name="load_vertical",
            type="slide",
            axis="0 0 1",
            damping="20",
        )
        ET.SubElement(
            carrier,
            "joint",
            name="load_horizontal",
            type="slide",
            axis="1 0 0",
            damping="1",
        )
        ET.SubElement(
            carrier, "inertial", pos="0 0 0", mass="2", diaginertia=".01 .01 .01"
        )
    builder = _Builder(root, "wheel_fixture")
    _wheel(builder, carrier, "wheel", (0, 0, 0), 0.14, 0.108)
    root.find(".//joint[@name='wheel_fixture__joint_wheel']").set(
        "frictionloss", str(brake)
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    session = InstrumentSession(
        model, {"equipment": [dict(id="wheel_fixture", **builder.metadata)]}
    )
    if rolling:
        mujoco.mj_step(model, session.data, nstep=500)
    return session


class WheelInspectionDriveTests(unittest.TestCase):
    def test_wheel_limits_and_drag_are_retained_with_grounded_control_interlocks(self):
        count = 0
        for identifier in ("nasa_planetary", "waterloo_robohub"):
            root, manifest = build_scene(identifier)
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            session = InstrumentSession(model, manifest)
            for key, control in session.controls.items():
                if control.mode != "velocity" or not key[1].startswith("wheel_"):
                    continue
                count += 1
                np.testing.assert_allclose([control.lower, control.upper], [-8, 8])
                np.testing.assert_allclose(
                    model.actuator_forcerange[control.actuator], [-12, 12]
                )
                self.assertEqual(
                    model.dof_damping[model.jnt_dofadr[control.joint]], 0.8
                )
                self.assertEqual(model.actuator_gainprm[control.actuator, 0], 40)
                session.reset()
                for target in (-5.6, 5.6, -4, 8, -8, 0):
                    if identifier == "nasa_planetary":
                        # Six grounded wheels must be steered/driven together.
                        # Raw tracking failures remain in the catalogue audit;
                        # a public rejection is not a successful wheel motion.
                        before = session.data.qpos.copy(), session.data.qvel.copy(), session.data.ctrl.copy(), session.data.time
                        with self.assertRaisesRegex(ValueError, "rover_traverse_and_stage"):
                            session.command(*key, target, duration=2)
                        for actual, value in zip((session.data.qpos, session.data.qvel, session.data.ctrl), before[:3]):
                            np.testing.assert_array_equal(actual, value)
                        self.assertEqual(session.data.time, before[3])
                        continue
                    result = session.command(*key, target, duration=2)
                    self.assertLessEqual(
                        abs(result["error"]), 0.25, (identifier, key, result)
                    )
        self.assertEqual(count, 8)

    def test_three_nm_brake_load_retains_the_predicted_speed_droop(self):
        session = wheel_fixture(brake=3)
        for target in (-8, 8, -4, 0):
            result = session.command("wheel_fixture", "wheel", target, duration=2)
            expected = (40 * target - math.copysign(3, target)) / 40.8 if target else 0
            self.assertAlmostEqual(result["measured"], expected, places=5)
            self.assertLess(abs(result["error"]), 0.25)

    def test_two_kg_rolling_contact_stays_loaded_and_reverses(self):
        session = wheel_fixture(rolling=True)
        model, data = session.model, session.data
        translation = model.jnt_dofadr[model.joint("load_horizontal").id]
        for target in (-8, 8, -4, 0):
            result = session.command("wheel_fixture", "wheel", target, duration=2)
            self.assertLess(abs(result["error"]), 0.25, result)
            self.assertGreater(data.ncon, 0)
            force = np.zeros(6)
            normal_force = 0.0
            for index in range(data.ncon):
                mujoco.mj_contactForce(model, data, index, force)
                normal_force += force[0]
            self.assertGreater(normal_force, 15)
            self.assertLess(normal_force, 25)
            self.assertLess(
                abs(data.qvel[translation] - 0.14 * result["measured"]), 0.02
            )

    def test_overload_remains_observable_with_torque_saturated(self):
        session = wheel_fixture(brake=20)
        control = session.controls["wheel_fixture", "wheel"]
        for target in (-8, 8):
            result = session.command("wheel_fixture", "wheel", target, duration=2)
            # A brake stronger than the 12 Nm drive must not be hidden by
            # overwriting velocity, disabling its constraint or adding torque.
            self.assertGreater(abs(result["error"]), 0.4)
            self.assertLess(abs(result["measured"]), 2)
            self.assertAlmostEqual(
                abs(session.data.actuator_force[control.actuator]), 12
            )

    def test_other_velocity_joints_keep_the_existing_default_gain(self):
        root = ET.fromstring(
            '<mujoco><compiler angle="radian"/><asset/><worldbody/><actuator/></mujoco>'
        )
        builder = _Builder(root, "other")
        builder.moving(
            root.find("worldbody"),
            "axis",
            (0, 0, 0),
            (0, 0, 1),
            None,
            kind="hinge",
            mode="velocity",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual(model.actuator_gainprm[0, 0], 10)


if __name__ == "__main__":
    unittest.main()
