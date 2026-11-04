"""Real free-plate and versioned access contracts for Rochester loading."""

from copy import copy, deepcopy
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.rochester_loading_motion import run_rochester_plate_loading


class RochesterPlateLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.meta = build_scene("rochester_genomics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.meta))

    def test_plate_is_free_and_access_semantics_are_versioned(self):
        s = self.session()
        m = s.model
        self.assertNotIn(("liquid_handler", "door"), s.controls)
        self.assertEqual(s.controls["liquid_handler", "front_door_hinge"].units, "rad")
        j = m.joint("rochester_loading__plate_free").id
        self.assertEqual(m.jnt_type[j], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.body("rochester_loading__plate").jntnum[0], 1)
        plate = m.geom("rochester_loading__plate_contact").id
        np.testing.assert_allclose(m.geom_size[plate], [0.0639, 0.0428, 0.011])
        door = m.jnt_bodyid[s.controls["liquid_handler", "front_door_hinge"].joint]
        geoms = np.flatnonzero(m.geom_bodyid == door)
        self.assertTrue(np.all(m.geom_contype[geoms]))

    def test_new_hinge_controls_original_carrier_readiness(self):
        s = self.session()
        self.assertTrue(s.process_readiness("liquid_handler")["ready"])
        c = s.controls["liquid_handler", "front_door_hinge"]
        s.data.qpos[s.model.jnt_qposadr[c.joint]] = 0.1
        self.assertIn("access_open", s.process_readiness("liquid_handler")["reasons"])
        self.assertEqual(s.data.time, 0)

    def test_direct_parent_filtered_fork_guide_is_clear_over_full_z_travel(self):
        s = self.session()
        m, d = s.model, s.data
        c = s.controls["liquid_handler", "loader_z"]
        adr = m.jnt_qposadr[c.joint]
        for z in np.linspace(c.lower, c.upper, 15):
            d.qpos[adr] = z
            mujoco.mj_forward(m, d)
            for suffix in ("-0.076", "0.076"):
                g = m.geom("rochester_loading__z_guide_" + suffix).id
                for name in ("fork_back", "fork_rear_floor", "z_pad_" + suffix):
                    other = m.geom("rochester_loading__" + name).id
                    self.assertGreaterEqual(
                        mujoco.mj_geomDistance(m, d, g, other, 0.02, None), 0.000999999
                    )

    def test_rejects_missing_plate_and_nonhome_control_before_any_step(self):
        for disturbed in ("plate", "control"):
            s = self.session()
            if disturbed == "plate":
                s.data.qpos[
                    s.model.jnt_qposadr[
                        s.model.joint("rochester_loading__plate_free").id
                    ]
                ] += 0.03
            else:
                s.data.ctrl[s.controls["liquid_handler", "loader_x"].actuator] = -0.1
            with self.assertRaises(ValueError):
                run_rochester_plate_loading(s)
            self.assertEqual(s.data.time, 0)

    def test_unexpected_control_mutation_stops_and_holds_without_reset(self):
        s = self.session()
        original = mujoco.mj_step
        calls = []

        def changed(m, d):
            original(m, d)
            calls.append(1)
            d.ctrl[s.controls["liquid_handler", "loader_y"].actuator] = 0.1

        with patch(
            "real_labs.rochester_loading_motion.mujoco.mj_step", side_effect=changed
        ):
            with self.assertRaisesRegex(RuntimeError, "control was changed"):
                run_rochester_plate_loading(s)
        self.assertEqual(len(calls), 1)
        self.assertGreater(s.data.time, 0)
        c = s.controls["liquid_handler", "loader_y"]
        self.assertAlmostEqual(
            s.data.ctrl[c.actuator], s.data.qpos[s.model.jnt_qposadr[c.joint]]
        )

    def test_observer_cannot_reset_live_state_silently(self):
        s = self.session()

        def altered(phase, m, d):
            d.time = 0

        with self.assertRaisesRegex(RuntimeError, "callback mutated"):
            run_rochester_plate_loading(s, observe=altered)


if __name__ == "__main__":
    unittest.main()
