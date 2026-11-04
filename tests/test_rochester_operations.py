"""Rochester public preflight preserves real loading and rejects raw collisions."""

from copy import copy, deepcopy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.rochester_operations import (
    OperationRequired,
    command,
    run_plate_loading,
    validate_command,
)
from real_labs.runtime import InstrumentSession


class RochesterOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("rochester_genomics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def assert_no_mutation(self, session, action, error=ValueError):
        fields = ("qpos", "qvel", "ctrl", "qacc", "qfrc_applied", "xfrc_applied")
        before = {key: getattr(session.data, key).copy() for key in fields}
        time = session.data.time
        with self.assertRaises(error) as raised:
            action()
        self.assertEqual(session.data.time, time)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(session.data, key), value)
        return raised.exception

    def test_known_colliding_xy_requests_reject_before_command(self):
        session = self.session()
        for axis, target in (("loader_x", -0.35), ("loader_y", -0.41)):
            with patch.object(session, "command") as raw:
                error = self.assert_no_mutation(
                    session,
                    lambda: command(session, "liquid_handler", axis, target),
                    OperationRequired,
                )
            raw.assert_not_called()
            self.assertEqual(error.operation, "plate_loading")
        self.assertEqual(session.controls["liquid_handler", "loader_x"].lower, -0.70)
        self.assertEqual(session.controls["liquid_handler", "loader_y"].lower, -0.82)

    def test_full_vertical_inspection_and_hinge_remain_available(self):
        session = self.session()
        for axis, target in (
            ("loader_z", 0.105),
            ("loader_z", 0),
            ("front_door_hinge", 1.75),
            ("front_door_hinge", 0),
        ):
            result = command(session, "liquid_handler", axis, target, duration=3.6)
            self.assertAlmostEqual(result["measured"], target, delta=0.0007)
            self.assertFalse(any(c.dist < -0.0001 for c in session.data.contact))
        self.assertAlmostEqual(session.data.time, 14.4)

    def test_unparked_and_displaced_states_cannot_bypass_access(self):
        for state in ("fork", "plate", "pipette", "force"):
            session = self.session()
            if state in ("fork", "pipette"):
                key = "loader_x" if state == "fork" else "gantry_x"
                spec = session.controls["liquid_handler", key]
                session.data.qpos[session.model.jnt_qposadr[spec.joint]] = -0.05
                session.data.ctrl[spec.actuator] = -0.05
            elif state == "plate":
                address = session.model.jnt_qposadr[
                    session.model.joint("rochester_loading__plate_free").id
                ]
                session.data.qpos[address] += 0.05
            else:
                session.data.qfrc_applied[:] = 1
            self.assert_no_mutation(
                session,
                lambda: command(session, "liquid_handler", "front_door_hinge", 1.75),
            )

    def test_invalid_targets_and_binding_reject_without_mutation(self):
        for target in (float("nan"), float("inf"), -0.001, 0.106):
            session = self.session()
            self.assert_no_mutation(
                session, lambda: command(session, "liquid_handler", "loader_z", target)
            )
        session = self.session()
        session.model.geom_contype[
            session.model.geom("rochester_loading__plate_contact").id
        ] = 0
        self.assert_no_mutation(session, lambda: run_plate_loading(session))

    def test_other_equipment_is_not_restricted_by_loading_policy(self):
        session = self.session()
        validate_command(session, "sequencer", "flowcell_tray", 0.125)
        result = command(session, "sequencer", "flowcell_tray", 0.125)
        self.assertAlmostEqual(result["measured"], 0.125, delta=0.0007)

    def test_complete_physical_loading_route_remains_available(self):
        session = self.session()
        result = run_plate_loading(session)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["steps"], 55800)
        self.assertFalse(result["reset_called"])
        self.assertFalse(result["pose_assignment"])
        self.assertFalse(result["equality_switching"])
        self.assertLess(result["max_plate_position_error_m"], 0.0001)
        phases = {row["phase"]: row for row in result["phases"]}
        self.assertTrue(all("deck_pad" in n for n in phases["close"]["support_geoms"]))
        self.assertTrue(
            all("stage_pad" in n for n in phases["close_finish"]["support_geoms"])
        )
        self.assertAlmostEqual(session.data.time, 111.6)


if __name__ == "__main__":
    unittest.main()
