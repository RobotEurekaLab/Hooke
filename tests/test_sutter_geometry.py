"""Articulated carriage ownership and SI tool interfaces for the Sutter reference."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))

from microscopy.geometry import geom
from microscopy.kinematics import tool_commands
from microscopy.sutter_manipulator import mount_sutter


def model_for(tool="injector"):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".001", integrator="implicitfast")
    asset, world, actuators = [ET.SubElement(root, name) for name in ("asset", "worldbody", "actuator")]
    for name in ("metal", "graphite", "rubber"):
        ET.SubElement(asset, "material", name=name, rgba=".5 .5 .5 1")
    angle = np.deg2rad(35)
    back = np.array([np.cos(angle)/2**.5, np.cos(angle)/2**.5, np.sin(angle)])
    base = np.array([.010, .010, .005])  # Deliberately not 12 mm along the shaft.
    origin = np.array([0., 0., .995])
    tcp, evidence = mount_sutter(world, asset, actuators, tool, np.zeros(3), origin,
                                 .705, back, capillary_base=base, terminal_back=[1., 0., 0.])
    geom(tcp, tool+"_tip", "sphere", (.000001,), material="metal", mass=".000001")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), evidence, origin


class SutterGeometryTests(unittest.TestCase):
    def test_commands_move_only_the_real_downstream_carriages(self):
        model, evidence, origin = model_for()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        names = ["fixed_base", "x_carriage", "y_carriage", "z_carriage"]
        ids = [model.geom("injector_sutter_"+name).id for name in names]
        start = data.geom_xpos[ids].copy()
        np.testing.assert_allclose(data.site_xpos[model.site("injector_tcp").id], origin, atol=1e-12)
        displacement = np.array([.0003, -.0002, .0001])
        for name, value in tool_commands(model, "injector", displacement).items():
            data.ctrl[model.actuator(name+"_drive").id] = value
        mujoco.mj_step(model, data, nstep=500)
        np.testing.assert_allclose(data.site_xpos[model.site("injector_tcp").id],
                                   origin+displacement, atol=2e-8, rtol=0.)
        delta = data.geom_xpos[ids]-start
        np.testing.assert_allclose(delta[0], 0., atol=1e-12)
        self.assertAlmostEqual(delta[1, 2], 0., delta=1e-12)
        self.assertAlmostEqual(delta[2, 2], 0., delta=1e-12)
        np.testing.assert_allclose(delta[3], displacement, atol=2e-8, rtol=0.)
        self.assertFalse(any(warning.number for warning in data.warning))
        self.assertEqual(evidence["travel_m"], .025)
        self.assertFalse(evidence["official_cad"])

    def test_holder_starts_at_bent_glass_base_and_reports_a_real_pressure_port(self):
        from microscopy.sutter_holder import HOLDER_LENGTH_M

        model, evidence, origin = model_for("holder")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        holder = model.body("holder_needle_holder").id
        base = np.asarray(evidence["capillary_base_m"])
        back = np.asarray(evidence["tool_back_direction"])
        np.testing.assert_allclose(data.xpos[holder], origin+base, atol=2e-10)
        port = model.site("holder_pneumatic_port").id
        np.testing.assert_allclose(data.site_xpos[port], origin+base+back*HOLDER_LENGTH_M, atol=2e-10)
        np.testing.assert_equal(evidence["terminal_back_direction"], [1., 0., 0.])
        self.assertNotEqual(evidence["terminal_back_direction"], evidence["tool_back_direction"])


if __name__ == "__main__":
    unittest.main()
