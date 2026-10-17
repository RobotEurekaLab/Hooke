"""Enabled open supports and actual unloaded drive/lift motion, not gait claims."""

from copy import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.stanford_gait_motion import GaitInspection, inspect_stanford_gait


def session():
    root, manifest = build_scene("stanford_biomechatronics_gait")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return tuple(x.copy() for x in (s.data.qpos, s.data.qvel, s.data.ctrl)), float(
        s.data.time
    )


class StanfordGaitTests(unittest.TestCase):
    def unchanged(self, s, before):
        for a, b in zip(before[0], (s.data.qpos, s.data.qvel, s.data.ctrl)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(before[1], s.data.time)

    def test_original_drive_and_lift_contract(self):
        s = session()
        m = s.model
        for name, x in (("drive_left", -0.325), ("drive_right", 0.325)):
            c = s.controls["gait_test_rig", name]
            np.testing.assert_array_equal(
                m.body_pos[m.jnt_bodyid[c.joint]], [x, -1, 0.188]
            )
            np.testing.assert_array_equal(m.jnt_axis[c.joint], [1, 0, 0])
            self.assertFalse(m.jnt_limited[c.joint])
            self.assertEqual(c.mode, "velocity")
            np.testing.assert_array_equal(m.actuator_ctrlrange[c.actuator], [-8, 8])
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], 10)
            self.assertEqual(m.dof_damping[m.jnt_dofadr[c.joint]], 0.1)
        c = s.controls["gait_test_rig", "fixture_lift"]
        np.testing.assert_array_equal(
            m.body_pos[m.jnt_bodyid[c.joint]], [0, 0.18, 2.54]
        )
        np.testing.assert_array_equal(m.jnt_range[c.joint], [-0.14, 0.14])
        self.assertEqual(m.actuator_gainprm[c.actuator, 0], 600)
        self.assertIn(
            "Fixed belt decks",
            s.equipment["gait_test_rig"]["limitations"][-1].replace(
                "Belt decks stay fixed", "Fixed belt decks"
            ),
        )

    def test_visible_moving_shells_have_enabled_contacts_or_open_proxies(self):
        s = session()
        m = s.model
        t = GaitInspection(s)
        groups = {m.body_weldid[m.jnt_bodyid[c.joint]] for c in t.controls}
        for g in range(m.ngeom):
            if m.body_weldid[m.geom_bodyid[g]] not in groups:
                continue
            name = m.geom(g).name
            if name.endswith("_shell"):
                self.assertTrue(
                    any(
                        m.geom(k).name.startswith(name[:-6] + "_proxy_")
                        and m.geom_contype[k]
                        for k in range(m.ngeom)
                    )
                )
            else:
                self.assertEqual(
                    (m.geom_contype[g], m.geom_conaffinity[g]), (1, 1), name
                )
        shaft = m.geom("gait_test_rig__gait_left_drive_shaft_-1").id
        bearing = m.geom("gait_test_rig__gait_left_drive_bearing_-1_proxy_0").id
        self.assertTrue(any(set(p) == {shaft, bearing} for p in t.pairs))

    def test_independent_unculled_full_pair_boundary_sweep(self):
        s = session()
        m, d = s.model, s.data
        t = GaitInspection(s)
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        groups = {m.body_weldid[m.jnt_bodyid[c.joint]] for c in t.controls}
        pairs = [
            (int(a), int(b))
            for i, a in enumerate(enabled)
            for b in enabled[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in groups
                or m.body_weldid[m.geom_bodyid[b]] in groups
            )
        ]
        self.assertEqual(set(pairs), set(map(tuple, t.pairs)))
        poses = [
            (0, 0, -0.14),
            (np.pi / 6, -np.pi / 6, 0.14),
            (np.pi, 0, 0),
            (0, np.pi, 0.14),
        ]
        poses.extend(
            np.random.default_rng(611).uniform(
                [-np.pi, -np.pi, -0.14], [np.pi, np.pi, 0.14], (9, 3)
            )
        )
        for pose in poses:
            d.qpos[t.q] = pose
            mujoco.mj_forward(m, d)
            exact = min(
                mujoco.mj_geomDistance(m, d, a, b, 0.003, None) for a, b in pairs
            )
            screened, _ = t.distances()
            self.assertGreaterEqual(exact, 0.0005998, (pose, exact))
            self.assertAlmostEqual(exact, float(min(screened)), places=9)

    def test_real_support_chain_and_hangers(self):
        s = session()
        m, d = s.model, s.data
        pairs = [
            ("bottom_pan", "lower_rail_0"),
            ("lower_rail_0", "bearing_base_-1.0"),
            ("bearing_base_-1.0", "left_drive_pedestal_-1"),
            ("lift_top_bridge", "lift_guide_-0.24"),
            ("lift_crossbar", "hanger_-0.15"),
            ("hanger_-0.15", "hanging_crossbar"),
            ("frame_post_-0.8_-1.15", "fixture_side_support_-0.8"),
            ("fixture_side_support_-0.8", "fixture_crossbar"),
            ("ankle_-0.29", "foot_adapter_-0.29"),
            ("foot_adapter_-0.29", "foot_-0.29"),
            ("service_crossbar", "fixed_service_-0.29_1"),
        ]
        for a, b in pairs:
            ga, gb = [m.geom("gait_test_rig__gait_" + n).id for n in (a, b)]
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, ga, gb, 1, None), 1e-8, (a, b)
            )
        foot = m.geom("gait_test_rig__gait_frame_foot_-0.8_-1.15").id
        self.assertAlmostEqual(d.geom_xpos[foot, 2] - m.geom_size[foot, 2], 0)

    def test_changed_source_collision_physics_and_metadata_rejected_unchanged(self):
        for change in ("guide", "mask", "gain", "metadata", "obstacle"):
            s = session()
            m = s.model
            g = m.geom("gait_test_rig__gait_lift_guide_-0.24").id
            if change == "guide":
                m.geom_pos[g, 0] += 0.25
            elif change == "mask":
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            elif change == "gain":
                m.actuator_gainprm[0, 0] *= 2
            elif change == "metadata":
                s.equipment["gait_test_rig"]["gait_inspection"][
                    "bearing_radial_clearance_m"
                ] = 0.1
            else:
                # Retain an unsafe obstacle by moving an existing enabled room obstacle
                # into the rig; source binding must reject rather than ignore it.
                gid = next(
                    i
                    for i in range(m.ngeom)
                    if not m.geom(i).name.startswith("gait_test_rig__")
                    and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                    and m.geom_contype[i]
                )
                m.geom_pos[gid] = [-0.55, -1, 0.188]
            before = state(s)
            with self.assertRaises(ValueError):
                GaitInspection(s)
            self.unchanged(s, before)

    def test_foreign_and_replaced_data_rejected_without_step(self):
        s = session()
        other = copy(s.model)
        s.data = mujoco.MjData(other)
        before = state(s)
        with self.assertRaises(ValueError):
            GaitInspection(s)
        self.unchanged(s, before)
        s = session()
        task = GaitInspection(s)
        old = task.d
        s.data = mujoco.MjData(s.model)
        before = state(s)
        with self.assertRaises(ValueError):
            task.execute()
        self.unchanged(s, before)
        self.assertEqual(old.time, 0)

    def test_force_failure_has_partial_coverage_and_no_reset(self):
        s = session()
        t = GaitInspection(s)
        s.data.qfrc_applied[t.v[0]] = 100
        result = t.execute()
        self.assertFalse(result["success"])
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertTrue(result["model_unchanged"])
        self.assertGreater(result["physics_steps"], result["completed_checks"])
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(result["whole_run_clearance_lower_bound_m"])
        self.assertGreater(s.data.time, 0)
        self.assertGreater(abs(s.data.qvel[t.v[0]]), 0)
        self.assertEqual(s.data.ctrl[t.a[0]], 0)

    def test_nonfinite_preflight_rejects_unchanged(self):
        s = session()
        t = GaitInspection(s)
        s.data.qfrc_applied[t.v[0]] = np.nan
        before = state(s)
        with self.assertRaises(ValueError):
            t.execute()
        self.unchanged(s, before)

    def test_actual_unloaded_registration_lift_and_return(self):
        s = session()
        r = inspect_stanford_gait(s)
        self.assertTrue(r["success"], r["failure"])
        self.assertTrue(r["coverage_complete"])
        self.assertEqual(r["physics_steps"], r["completed_checks"])
        self.assertTrue(r["model_unchanged"])
        self.assertFalse(r["reset_performed"])
        self.assertEqual(len(r["phases"]), 5)
        self.assertAlmostEqual(
            r["phases"][0]["measured_displacement"][0], 1.2, delta=0.0002
        )
        self.assertAlmostEqual(
            r["phases"][1]["measured_displacement"][1], -0.8, delta=0.0002
        )
        self.assertAlmostEqual(
            r["phases"][2]["measured_displacement"][2], 0.1, delta=0.0001
        )
        self.assertLess(max(abs(np.array(r["return_error"]))), 0.0002)
        for p in r["phases"]:
            self.assertTrue(p["success"])
            self.assertTrue(p["coverage_complete"])
            self.assertEqual(len(p["state"]["qpos"]), s.model.nq)
        self.assertIn("no belt transport", r["scope"])


if __name__ == "__main__":
    unittest.main()
