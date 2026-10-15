"""Eight-leg source contract, real open bearings and checked inert motion."""

import itertools
import unittest
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from real_labs.bristol_table_geometry import CONTROLS, ID, LEGS, PREFIX, SCENE
from real_labs.bristol_table_motion import (
    TableInspection,
    commanded_pose,
    diagnostic_pose,
)
from real_labs.bristol_table_geometry import leg_lengths
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


def load():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(session):
    return (
        session.data.time,
        *(
            getattr(session.data, name).tobytes()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "qfrc_applied",
                "xfrc_applied",
                "qacc_warmstart",
            )
        ),
    )


class BristolTableTests(unittest.TestCase):
    def test_boundary_command_recovery_preserves_actual_compliance_and_strict_targets(
        self,
    ):
        s = load()
        before = state(s)
        # Actual failed-attempt endpoint: finite servo compliance is physical,
        # while -0.15000000000000002 was an arithmetic endpoint artefact.
        measured = np.array(
            [
                0.15054211442736698,
                2.8374624965043288e-6,
                -0.00042980225227126634,
                -5.104172276349228e-6,
                4.407246301801796e-5,
                -1.1433326071980652e-5,
            ]
        )
        command = commanded_pose(leg_lengths((0.15, 0, 0)), measured)
        self.assertEqual(command[0], 0.15)
        np.testing.assert_allclose(command[1:], 0, atol=1e-12)
        self.assertEqual(state(s), before)
        incompatible = np.zeros(8)
        incompatible[0] = 0.05
        with self.assertRaises(ValueError):
            commanded_pose(incompatible, np.zeros(6))
        with self.assertRaises(ValueError):
            diagnostic_pose(s.model, s.data, [0.15000000000000002, 0, 0, 0, 0, 0])
        self.assertEqual(state(s), before)

    def test_explicit_migration_free_platform_source_stroke_and_real_drives(self):
        s = load()
        m = s.model
        item = s.equipment[ID]
        self.assertEqual(set(item["joints"]), set(CONTROLS))
        self.assertEqual(item["mechanical_interface"]["version"], 2)
        self.assertNotIn((ID, "table_x"), s.controls)
        self.assertEqual(
            m.joint(PREFIX + "platform_free").type, mujoco.mjtJoint.mjJNT_FREE
        )
        self.assertEqual(m.body(PREFIX + "platform_body").mass, 2000)
        self.assertEqual(m.body_gravcomp[m.body(PREFIX + "platform_body").id], 0)
        self.assertEqual(m.neq, 8)
        self.assertTrue(np.all(m.eq_type == mujoco.mjtEq.mjEQ_CONNECT))
        self.assertEqual(m.nexclude, 0)
        np.testing.assert_array_equal(
            m.geom(PREFIX + "platform").size, [1.5, 1.5, 0.065]
        )
        for key in CONTROLS:
            c = s.controls[ID, key]
            np.testing.assert_array_equal(m.jnt_range[c.joint], [-0.15, 0.15])
            np.testing.assert_array_equal(
                m.actuator_forcerange[c.actuator], [-70000, 70000]
            )
        # The declared combined smoke pose moves and measures all eight rods.
        smoke = s.smoke_test()
        self.assertTrue(smoke["passed"])
        self.assertEqual(len(smoke["controls"]), 8)
        self.assertTrue(
            all(
                r["command_mode"] == "simultaneous_position_vector"
                for r in smoke["controls"]
            )
        )

    def test_full_axis_travel_and_extra_coupled_poses_have_open_contact_clearance(self):
        s = load()
        t = TableInspection(s)
        d = mujoco.MjData(s.model)
        poses = [np.zeros(6)]
        for axis in range(3):
            for value in np.linspace(-0.15, 0.15, 11):
                pose = np.zeros(6)
                pose[axis] = value
                poses.append(pose)
        poses.extend(
            np.array(v)
            for v in (
                (0.06, -0.04, 0.025, 0.01, -0.008, 0.012),
                (-0.052, 0.033, -0.019, -0.014, 0.009, -0.011),
            )
        )
        for pose in poses:
            diagnostic_pose(s.model, d, pose)
            minimum, _ = t.direct_clearance(d)
            self.assertGreater(minimum, 0.0009)
            self.assertLess(max(abs(d.efc_pos[:24])), 1e-12)
            self.assertFalse(any(c.dist < -1e-7 for c in d.contact[: d.ncon]))

    def test_pruner_matches_unculled_parent_inclusive_pairs_off_grid(self):
        s = load()
        t = TableInspection(s)
        d = mujoco.MjData(s.model)
        self.assertLess(len(t.pairs), len(t.all_pairs))
        for pose in (
            (0.117, 0, 0.019, 0, 0.005, -0.008),
            (-0.091, 0.027, -0.041, -0.006, 0.004, 0.007),
        ):
            diagnostic_pose(s.model, d, pose)
            screened, _ = t.direct_clearance(d)
            direct = min(
                t.cap,
                min(
                    mujoco.mj_geomDistance(s.model, d, int(a), int(b), t.cap, None)
                    for a, b in t.all_pairs
                ),
            )
            self.assertAlmostEqual(screened, direct, places=12)

    def test_all_moving_parts_and_fixed_bearings_have_continuous_assemblies(self):
        s = load()
        m, d = s.model, s.data
        enabled = [
            i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]
        ]
        weld = m.body_weldid[m.geom_bodyid]
        for group in set(weld[enabled]):
            if group == 0:
                ids = [
                    i
                    for i in enabled
                    if weld[i] == 0 and m.geom(i).name.startswith(PREFIX)
                ]
            else:
                ids = [i for i in enabled if weld[i] == group]
            reached = {ids[0]}
            remaining = set(ids[1:])
            while remaining:
                added = {
                    b
                    for b in remaining
                    if any(
                        mujoco.mj_geomDistance(m, d, a, b, 0.005, None) <= 0.00005
                        for a in reached
                    )
                }
                if not added:
                    break
                reached.update(added)
                remaining -= added
            self.assertFalse(remaining, [m.geom(i).name for i in sorted(remaining)])

    def test_support_query_cap_is_not_a_connection_and_frame_endpoints_meet(self):
        s = load()
        m, d = s.model, s.data
        for x in (-1.05, 1.05):
            for y in (-1.05, 1.05):
                foot = m.geom(PREFIX + f"foot_{x}_{y}").id
                post = m.geom(PREFIX + f"post_{x}_{y}").id
                foot_top = m.geom_pos[foot, 2] + m.geom_size[foot, 2]
                post_bottom = m.geom_pos[post, 2] - m.geom_size[post, 2]
                self.assertAlmostEqual(foot_top, post_bottom, places=12)
                self.assertLessEqual(
                    mujoco.mj_geomDistance(m, d, foot, post, 0.005, None), 0.00005
                )
        # Reproduce the former flaw: beyond distmax MuJoCo may return the cap.
        # That cap must exceed the physical attachment acceptance threshold.
        first = m.geom(PREFIX + "foot_-1.05_-1.05").id
        second = m.geom(PREFIX + "foot_1.05_1.05").id
        gap = mujoco.mj_geomDistance(m, d, first, second, 0.005, None)
        self.assertGreater(gap, 0.00005)

    def test_nonfinite_time_rejects_before_any_state_mutation(self):
        for value in (float("nan"), float("inf")):
            s = load()
            s.data.time = value
            before = state(s)[1:]
            with self.assertRaisesRegex(ValueError, "finite"):
                TableInspection(s)
            self.assertEqual(state(s)[1:], before)
            s = load()
            t = TableInspection(s)
            s.data.time = value
            before = state(s)[1:]
            with self.assertRaisesRegex(ValueError, "finite"):
                t.move(np.zeros(6), ramp_s=0.05, settle_s=0)
            self.assertEqual(state(s)[1:], before)
            self.assertEqual(t.executed, 0)
            self.assertTrue(
                np.isnan(s.data.time) if np.isnan(value) else np.isinf(s.data.time)
            )

    def test_native_reset_is_reported_at_initial_and_nonzero_clock(self):
        for initial_time in (0.0, 0.7):
            s = load()
            t = TableInspection(s)
            s.data.time = initial_time
            s.data.xfrc_applied[t.platform, 2] = -1e25
            warnings = []
            previous = mujoco.get_mju_user_warning()
            mujoco.set_mju_user_warning(warnings.append)
            try:
                report = t.move(np.zeros(6), ramp_s=0.05, settle_s=0)
            finally:
                mujoco.set_mju_user_warning(previous)
            self.assertTrue(warnings)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["direct_clearance"]["executed_steps"], 1)
            self.assertEqual(report["direct_clearance"]["checked_steps"], 0)
            self.assertIsNone(
                report["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
            )
            self.assertTrue(report["automatic_reset_detected"])
            self.assertTrue(report["reset_used"])
            self.assertTrue(report["reset_performed"])
            self.assertFalse(report["explicit_reset_performed"])
            self.assertTrue(report["hold_requested"])
            self.assertEqual(s.data.time, s.model.opt.timestep)
            self.assertEqual(report["time_discontinuity_detected"], initial_time != 0)
            self.assertEqual(len(report["physics_events"]), 1)

    def test_actual_loaded_transit_and_return_without_reset(self):
        s = load()
        t = TableInspection(s)
        # Supplemental 2 kN downward load is an applied wrench, not a robot
        # or a new payload's inertial dynamics. The rigid test frame has mass.
        s.data.xfrc_applied[t.platform, 2] = -2000
        initial = s.data.time
        report = t.move(
            (0.025, -0.018, 0.012, 0.003, -0.002, 0.004), ramp_s=1.5, settle_s=0.5
        )
        self.assertEqual(report["status"], "succeeded", report["reason"])
        self.assertEqual(report["direct_clearance"]["checked_steps"], 1000)
        self.assertTrue(report["direct_clearance"]["coverage_complete"])
        self.assertGreater(s.data.time, initial)
        out = t.move((0, 0, 0, 0, 0, 0), ramp_s=1.5, settle_s=0.5)
        self.assertEqual(out["status"], "succeeded", out["reason"])
        self.assertLess(
            max(abs(np.asarray(out["observation"]["platform_translation_m"]))), 0.001
        )
        self.assertFalse(out["reset_used"])

    def test_unsafe_coupled_targets_reject_without_live_mutation(self):
        s = load()
        t = TableInspection(s)
        before = state(s)
        for pose in (
            (0.15, 0.15, 0, 0, 0, 0),
            (0.15001, 0, 0, 0, 0, 0),
            (0, 0, 0, 0.05, 0, 0),
        ):
            with self.assertRaises(ValueError):
                t.move(pose)
            self.assertEqual(state(s), before)

    def test_foreign_state_disabled_collider_flags_and_constraint_reject(self):
        for change in ("foreign", "mask", "flags", "equality", "active", "gain"):
            s = load()
            if change == "foreign":
                s.data = mujoco.MjData(load().model)
            elif change == "mask":
                s.model.geom(PREFIX + "vertical_0_shaft").contype = 0
                s.model.geom(PREFIX + "vertical_0_shaft").conaffinity = 0
            elif change == "flags":
                s.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            elif change == "equality":
                s.model.eq_data[0, 0] += 0.01
            elif change == "active":
                s.data.eq_active[0] = False
            else:
                s.model.actuator_gainprm[0, 0] *= 0.5
            before = state(s)
            with self.subTest(change=change), self.assertRaises(ValueError):
                TableInspection(s)
            self.assertEqual(state(s), before)

    def test_adverse_force_and_mutating_observer_fail_with_honest_coverage(self):
        s = load()
        t = TableInspection(s)
        s.data.xfrc_applied[t.platform, 0] = 10000000
        report = t.move((0.02, 0, 0, 0, 0, 0))
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_used"])
        self.assertGreater(
            report["direct_clearance"]["executed_steps"],
            report["direct_clearance"]["checked_steps"],
        )
        self.assertIsNone(
            report["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        s = load()
        t = TableInspection(s)

        def mutate(task):
            task.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        report = t.move((0, 0, 0, 0, 0, 0), step_observer=mutate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("Observer mutated", report["reason"])


if __name__ == "__main__":
    unittest.main()
