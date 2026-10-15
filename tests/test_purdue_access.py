"""Supported source-informed AAPF mechanics and measured inert-target imaging."""

from copy import copy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.motion import _model_signature
from real_labs.purdue_motion import CAMERA, CONTROLS, EQUIPMENT, SCENE, PurdueInspection
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def geometry(model, key):
    return model.geom(EQUIPMENT + "__purdue_" + key).id


class PurdueAccessTests(unittest.TestCase):
    def test_original_contracts_and_registered_actual_camera(self):
        s = session()
        m = s.model
        for semantic, frame, limits, mass, gain in (
            ("conveyor", (0, -0.54, 0.66), (0, 0.8), 3, 300),
            ("plant_turntable", (0, 0, 0.047), (-np.pi, np.pi), 1, 180),
            ("camera_height", (0.46, 0.05, 1.13), (-0.18, 0.52), 0.6, 180),
        ):
            c = s.controls[EQUIPMENT, semantic]
            np.testing.assert_allclose(m.body_pos[m.jnt_bodyid[c.joint]], frame)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
            self.assertAlmostEqual(m.body_mass[m.jnt_bodyid[c.joint]], mass)
            self.assertAlmostEqual(m.actuator_gainprm[c.actuator, 0], gain)
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
        np.testing.assert_allclose(
            m.site(EQUIPMENT + "__site_plant_pot").pos, [0, 0, 0.025]
        )
        self.assertIn(CAMERA, s.manifest["camera_names"])
        camera = m.camera(CAMERA).id
        site = m.site(EQUIPMENT + "__site_camera_pinhole").id
        np.testing.assert_allclose(m.cam_pos[camera], [-0.118, 0, 0])
        np.testing.assert_array_equal(m.cam_pos[camera], m.site_pos[site])
        self.assertAlmostEqual(m.cam_fovy[camera], 65)
        self.assertFalse(screen_contact_coverage(m)["issues"])
        for name, position in {
            "overview": [13.8, -17.5, 12.5],
            "interior": [1.05, -3.45, 1.68],
            "workstation": [4.5, -3.8, 2.9],
            "reference": [-5.98, -3.18, 1.85],
        }.items():
            self.assertEqual(s.manifest["camera"][name]["position"], position)

    def test_real_camera_lens_and_bracket_connections(self):
        s = session()
        m, d = s.model, s.data
        body = m.body(EQUIPMENT + "__body_camera_height").id
        ids = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body]
        self.assertTrue(all(m.geom_contype[i] and m.geom_conaffinity[i] for i in ids))
        reached, pending = {ids[0]}, set(ids[1:])
        while pending:
            new = {
                i
                for i in pending
                if any(
                    mujoco.mj_geomDistance(m, d, i, j, 0.00001, None) <= 1e-7
                    for j in reached
                )
            }
            if not new:
                break
            reached |= new
            pending -= new
        self.assertFalse(pending, [m.geom(i).name for i in pending])
        for a, b in (
            ("lens", "barrel"),
            ("camera_rail", "camera_rail_anchor_0.76"),
            ("camera_rail_anchor_0.76", "side_panel_0.66"),
            ("chain_backbone", "chain_anchor_0.76"),
        ):
            self.assertLessEqual(
                mujoco.mj_geomDistance(
                    m, d, geometry(m, a), geometry(m, b), 0.0001, None
                ),
                1e-7,
            )

    def test_carrier_rotor_and_rigid_pot_support_are_real(self):
        s = session()
        m, d = s.model, s.data
        for a, b in (
            ("carrier_plate", "spindle"),
            ("carrier_plate", "thrust_support"),
            ("carrier_plate", "carrier_arm_0.45_0.18"),
            ("carrier_rail_0.45", "rail_pedestal_0.45_0.87"),
            ("roller_5", "roller_journal_5_1"),
            ("roller_journal_5_1", "transport_frame_0.4"),
        ):
            self.assertLessEqual(
                mujoco.mj_geomDistance(
                    m, d, geometry(m, a), geometry(m, b), 0.0001, None
                ),
                1e-7,
                (a, b),
            )
        pot = m.geom("facility_imaging_specimen_pot").id
        self.assertTrue(m.geom_contype[pot] and m.geom_conaffinity[pot])
        supports = [
            i
            for i in range(m.ngeom)
            if m.geom(i).name.startswith(EQUIPMENT + "__purdue_pot_seat_proxy_")
        ]
        self.assertLessEqual(
            min(mujoco.mj_geomDistance(m, d, pot, i, 0.0001, None) for i in supports),
            1e-7,
        )
        gap = mujoco.mj_geomDistance(
            m, d, geometry(m, "carrier_plate"), geometry(m, "roller_5"), 0.01, None
        )
        self.assertGreater(gap, 0.00059)

    def test_full_original_travel_and_unculled_filtered_pairs(self):
        s = session()
        m, d = s.model, s.data
        t = PurdueInspection(s)
        moving = {int(m.body_weldid[b]) for b in t.motion.moving_bodies}
        ids = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        expected = {
            (int(a), int(b))
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in moving
                or m.body_weldid[m.geom_bodyid[b]] in moving
            )
        }
        self.assertEqual(set(map(tuple, t.pairs)), expected)
        for pose in product(
            np.linspace(0, 0.8, 5),
            np.linspace(-np.pi, np.pi, 7),
            np.linspace(-0.18, 0.52, 5),
        ):
            d.qpos[:3] = pose
            mujoco.mj_forward(m, d)
            t.check(m, d)
        self.assertEqual(t.checked, 175)
        self.assertGreater(t.minimum, 0.00059)
        for pose in ((0.173, 0.417, 0.093), (0.789, -2.731, 0.499), (0, 0, -0.18)):
            d.qpos[:3] = pose
            mujoco.mj_forward(m, d)
            t.reset_counts()
            t.check(m, d)
            direct = min(
                min(float(mujoco.mj_geomDistance(m, d, a, b, 0.002, None)), 0.002)
                for a, b in expected
            )
            self.assertAlmostEqual(direct, t.minimum, places=8)

    def test_open_guides_remain_engaged_at_limits(self):
        s = session()
        m, d = s.model, s.data
        for q, h in product((0, 0.8), (-0.18, 0.52)):
            d.qpos[:3] = [q, 0, h]
            mujoco.mj_forward(m, d)
            for a, b, axis in (
                ("carrier_rail_0.45", "carrier_shoe_0.45_0.18_0_1", 1),
                ("camera_rail", "camera_shoe_0.1_0_1", 2),
            ):
                rail, shoe = geometry(m, a), geometry(m, b)
                self.assertLess(
                    abs(d.geom_xpos[rail, axis] - d.geom_xpos[shoe, axis])
                    + m.geom_size[shoe, axis],
                    m.geom_size[rail, axis],
                )
                self.assertGreater(
                    mujoco.mj_geomDistance(m, d, rail, shoe, 0.002, None), 0.00059
                )

    def test_disabled_contacts_metadata_and_foreign_data_reject_without_motion(self):
        changes = (
            lambda s: s.model.geom_contype.__setitem__(
                geometry(s.model, "camera_main"), 0
            ),
            lambda s: s.model.geom_conaffinity.__setitem__(
                geometry(s.model, "witness_0"), 0
            ),
            lambda s: setattr(
                s.model.opt, "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda s: s.model.cam_pos.__setitem__((s.model.camera(CAMERA).id, 0), -0.1),
            lambda s: s.equipment[EQUIPMENT]["inspection_interface"].__setitem__(
                "guide_clearance_m", 0.1
            ),
            lambda s: setattr(s, "data", mujoco.MjData(copy(s.model))),
        )
        for mutate in changes:
            s = session()
            t = PurdueInspection(s)
            mutate(s)
            q = s.data.qpos.copy()
            u = s.data.ctrl.copy()
            with self.assertRaises(ValueError):
                t.move((0.01, 0, 0))
            np.testing.assert_array_equal(s.data.qpos, q)
            np.testing.assert_array_equal(s.data.ctrl, u)
            self.assertEqual(s.data.time, 0)
        s = session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaises(ValueError):
            PurdueInspection(s)
        s = session()
        t = PurdueInspection(s)
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            t.move((0.01, 0, 0))

    def test_actual_two_witness_inspection_and_return(self):
        s = session()
        signature = _model_signature(s.model)
        r = PurdueInspection(s).run()
        self.assertEqual(
            r["status"],
            "succeeded",
            [(p["phase"], p.get("failure")) for p in r["phases"]],
        )
        self.assertEqual(r["checked_steps"], r["executed_steps"])
        self.assertGreater(r["executed_steps"], 1000)
        self.assertEqual(len(r["observations"]), 2)
        for o in r["observations"]:
            self.assertTrue(o["registered"])
            self.assertLess(o["centre_error_px"], 1)
            self.assertAlmostEqual(o["optical_standoff_m"], 0.13, places=3)
            self.assertGreater(o["actual_state"]["time_s"], 0)
        self.assertLess(max(r["measured_return_error"]), 1e-4)
        self.assertFalse(r["reset_performed"])
        self.assertEqual(signature, _model_signature(s.model))

    def test_disturbed_held_camera_fails_with_partial_coverage_and_no_reset(self):
        s = session()
        t = PurdueInspection(s, CONTROLS[:2])
        c = s.controls[EQUIPMENT, "camera_height"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 500
        result = t.move((0.03, 0))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(
            result["physics_steps"], result["direct_clearance"]["checked_steps"]
        )
        self.assertIsNone(
            result["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        self.assertGreater(s.data.time, 0)

    def test_read_only_observer_failure_does_not_claim_complete_leg(self):
        s = session()
        t = PurdueInspection(s)

        def stop(model, data):
            raise RuntimeError("independent_observer_stop")

        r = t.move((0.01, 0, 0), step_observer=stop)
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertEqual(r["direct_clearance"]["checked_steps"], 0)
        self.assertIsNone(r["direct_clearance"]["whole_leg_clearance_lower_bound_m"])


if __name__ == "__main__":
    unittest.main()
