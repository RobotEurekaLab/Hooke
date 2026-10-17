"""Legacy accessory adapters retain collision while clearing their full travel."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.reuse import attach_asset
from real_labs.runtime import InstrumentSession


def make_session(asset, obstacle=False):
    root = ET.fromstring(
        '<mujoco><compiler angle="radian" autolimits="true"/>'
        '<option gravity="0 0 -9.81" timestep=".002"/>'
        '<default><joint armature=".005"/></default><asset/><worldbody/><actuator/></mujoco>'
    )
    world = root.find("worldbody")
    row = attach_asset(root, world, f"instrument/{asset}.xml", asset, [0, 0, 0])
    if obstacle:
        # External obstruction must remain visible to the collision engine.
        position = "-.075 .003 .12" if asset == "analytical_balance" else ".03 .09 .14"
        ET.SubElement(
            world, "geom", name="obstruction", type="sphere", size=".01", pos=position
        )
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
        {"id": asset, "equipment": [row]},
    )


class ReusedAccessoryClearanceTests(unittest.TestCase):
    def test_reset_and_full_travel_clear_while_original_ranges_remain(self):
        for asset, bounds in (
            ("analytical_balance", (-0.1, 0)),
            ("ph_meter", (-0.6, 0.6)),
        ):
            session = make_session(asset)
            control = next(iter(session.controls.values()))
            np.testing.assert_allclose([control.lower, control.upper], bounds)
            for value in np.linspace(*bounds, 101):
                session.reset()
                session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
                mujoco.mj_forward(session.model, session.data)
                self.assertFalse(
                    any(contact.dist < -1e-6 for contact in session.data.contact),
                    (asset, value),
                )

    def test_direction_reversal_commands_settle(self):
        for asset in ("analytical_balance", "ph_meter"):
            session = make_session(asset)
            key, control = next(iter(session.controls.items()))
            span = control.upper - control.lower
            for fraction in (0.15, 0.85, 0.25):
                result = session.command(
                    *key, control.lower + fraction * span, duration=2
                )
                self.assertLessEqual(
                    abs(result["error"]), span * 0.025, (asset, result)
                )

    def test_repairs_do_not_disable_contact_with_external_obstacles(self):
        for asset in ("analytical_balance", "ph_meter"):
            session = make_session(asset, obstacle=True)
            obstruction = session.model.geom("obstruction").id
            self.assertTrue(
                any(
                    obstruction in (c.geom1, c.geom2) and c.dist < -1e-4
                    for c in session.data.contact
                ),
                asset,
            )


if __name__ == "__main__":
    unittest.main()
