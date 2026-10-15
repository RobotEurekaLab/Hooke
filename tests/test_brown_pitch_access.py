"""Connected Brown positioner supports, preserved ranges and measured motion."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.brown_pitch_geometry import PREFIX
from real_labs.brown_pitch_motion import SCENE, EQUIPMENT, CouponIndexing
from real_labs.runtime import InstrumentSession


def load():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return (
        s.data.time,
        *(
            getattr(s.data, n).tobytes()
            for n in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            )
        ),
    )


class BrownPitchAccessTests(unittest.TestCase):
    def test_public_failure_preserves_native_reset_or_clock_uncertainty(self):
        for case in ("native_reset", "unknown_clock"):
            with self.subTest(case=case):
                s = load()
                if case == "native_reset":
                    s.data.time = 1.0
                    s.data.qfrc_applied[1] = 1e25
                else:
                    s.data.time = 1e30
                report = CouponIndexing(s).run()
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["executed_steps"], 1)
                self.assertEqual(report["checked_steps"], 0)
                self.assertFalse(report["coverage_complete"])
                self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
                self.assertIs(
                    report["reset_performed"], True if case == "native_reset" else None
                )
                self.assertEqual(
                    report["reset_performed"], report["phases"][0]["reset_performed"]
                )
                self.assertFalse(report["explicit_reset_performed"])
                self.assertTrue(report["phases"][0]["hold_requested"])

    def test_original_frames_inertias_drives_and_coupon_are_preserved(self):
        s = load()
        m = s.model
        for name, pos, axis, limits, mass, kp in (
            ("model_yaw", [0, 0, 1.345], [0, 0, 1], [-0.45, 0.45], 0.20, 160),
            ("model_pitch", [0, 0, 0.100], [0, 1, 0], [-0.32, 0.32], 0.08, 100),
        ):
            c = s.controls[EQUIPMENT, name]
            body = m.jnt_bodyid[c.joint]
            np.testing.assert_array_equal(m.body_pos[body], pos)
            np.testing.assert_array_equal(m.jnt_axis[c.joint], axis)
            np.testing.assert_array_equal(m.jnt_range[c.joint], limits)
            self.assertEqual(m.body_mass[body], mass)
            np.testing.assert_allclose(
                m.body_inertia[body], [mass * 0.006] * 3, rtol=0, atol=1e-18
            )
            self.assertEqual(
                m.body_mass[m.body_weldid == m.body_weldid[body]].sum(), mass
            )
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], kp)
            np.testing.assert_array_equal(
                m.actuator_forcerange[c.actuator], [-120, 120]
            )
        np.testing.assert_array_equal(
            m.geom(PREFIX + "inert_coupon").size, [0.28, 0.055, 0.014]
        )
        np.testing.assert_array_equal(
            m.geom(PREFIX + "inert_coupon").pos, [0, 0, 0.040]
        )
        np.testing.assert_array_equal(
            m.site(EQUIPMENT + "__site_aerodynamic_coupon").pos, [0, 0, 0.055]
        )
        self.assertIn(
            "None:",
            s.equipment[EQUIPMENT]["positioner_interface"]["coordinate_migration"],
        )

    def test_real_fixed_load_path_and_connected_moving_assemblies(self):
        s = load()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(
                m, d, m.geom(PREFIX + a).id, m.geom(PREFIX + b).id, 0.1, None
            )

        self.assertLessEqual(gap("pedestal_foot", "pedestal_column"), 1e-8)
        floor = next(g for g in range(m.ngeom) if "__test_floor_" in m.geom(g).name)
        self.assertAlmostEqual(
            mujoco.mj_geomDistance(
                m, d, m.geom(PREFIX + "pedestal_foot").id, floor, 0.1, None
            ),
            0,
            delta=1e-8,
        )
        proxies = [
            g for g in range(m.ngeom) if "brown_yaw_bearing_proxy_" in m.geom(g).name
        ]
        self.assertEqual(len(proxies), 48)
        self.assertLessEqual(
            min(
                mujoco.mj_geomDistance(
                    m, d, g, m.geom(PREFIX + "pedestal_column").id, 0.1, None
                )
                for g in proxies
            ),
            1e-8,
        )
        for sign in (-1, 1):
            self.assertLessEqual(gap(f"fork_post_{sign}", "yaw_plate"), 1e-8)
            bearing = [
                g
                for g in range(m.ngeom)
                if f"brown_pitch_bearing_{sign}_proxy_" in m.geom(g).name
            ]
            self.assertEqual(len(bearing), 48)
            self.assertLessEqual(
                min(
                    mujoco.mj_geomDistance(
                        m, d, g, m.geom(PREFIX + f"fork_post_{sign}").id, 0.1, None
                    )
                    for g in bearing
                ),
                1e-8,
            )
        for control in s.controls.values():
            weld = m.body_weldid[m.jnt_bodyid[control.joint]]
            remaining = set(
                int(g)
                for g in np.flatnonzero(
                    (m.geom_contype | m.geom_conaffinity)
                    & (m.body_weldid[m.geom_bodyid] == weld)
                )
            )
            reached = {remaining.pop()}
            while remaining:
                linked = {
                    g
                    for g in remaining
                    if any(
                        mujoco.mj_geomDistance(m, d, g, h, 1e-6, None) <= 1e-8
                        for h in reached
                    )
                }
                self.assertTrue(linked, [m.geom(g).name for g in remaining])
                remaining -= linked
                reached |= linked

    def test_full_coupled_travel_open_bores_and_physical_orientation(self):
        s = load()
        m, d = s.model, s.data
        t = CouponIndexing(s)
        pairset = {frozenset((m.geom(a).name, m.geom(b).name)) for a, b in t.pairs}
        self.assertIn(
            frozenset((PREFIX + "pitch_shaft", PREFIX + "pitch_bearing_1_proxy_0")),
            pairset,
        )
        self.assertIn(
            frozenset((PREFIX + "yaw_spindle", PREFIX + "yaw_bearing_proxy_0")), pairset
        )
        for yaw, pitch in itertools.product(
            np.linspace(-0.45, 0.45, 13), np.linspace(-0.32, 0.32, 13)
        ):
            d.qpos[:] = [yaw, pitch]
            mujoco.mj_forward(m, d)
            t.check(m, d)
            self.assertFalse(any(c.dist < -1e-8 for c in d.contact))
            o = t.observe()
            np.testing.assert_allclose(
                o["measured_orientation_rad"], [yaw, pitch], atol=1e-12
            )
            expected = [
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
            ]
            np.testing.assert_allclose(o["coupon_chord_world"], expected, atol=1e-12)
        self.assertGreaterEqual(t.minimum, 0.0005999)
        # Original shaft endpoints and open sleeves overlap fully, with real retainers outside them.
        self.assertAlmostEqual(0.100 - (0.087 + 0.012), 0.001)

    def test_bounded_actual_motion_and_return_have_complete_checks(self):
        s = load()
        t = CouponIndexing(s)
        for target in ([0.06, 0.04], [0, 0]):
            r = t.move(target)
            self.assertEqual(r["status"], "succeeded", r)
            self.assertTrue(r["direct_clearance"]["coverage_complete"])
            self.assertEqual(r["physics_steps"], r["direct_clearance"]["checked_steps"])
            self.assertGreaterEqual(
                r["direct_clearance"]["whole_leg_clearance_lower_bound_m"], 0.0005999
            )
            self.assertFalse(r["reset_performed"])
            np.testing.assert_allclose(
                t.observe()["measured_orientation_rad"], target, atol=1e-4
            )
        self.assertGreater(s.data.time, 0)

    def test_source_state_and_bad_start_reject_without_mutation(self):
        cases = (
            "mask",
            "support",
            "physics",
            "metadata",
            "foreign",
            "cached_data",
            "invalid_start",
        )
        for case in cases:
            s = load()
            t = CouponIndexing(s)
            m, d = s.model, s.data
            if case == "mask":
                m.geom_contype[m.geom(PREFIX + "pitch_shaft").id] = 0
            elif case == "support":
                m.geom_pos[m.geom(PREFIX + "pedestal_column").id, 0] += 0.2
            elif case == "physics":
                m.opt.timestep = 0.001
            elif case == "metadata":
                s.equipment[EQUIPMENT]["positioner_interface"]["revision"] = 1
            elif case == "foreign":
                s.data = mujoco.MjData(copy(m))
            elif case == "cached_data":
                s.data = mujoco.MjData(m)
            elif case == "invalid_start":
                d.qpos[0] = 0.1
                mujoco.mj_forward(m, d)
            before = state(s)
            with self.assertRaises(ValueError):
                t.run()
            self.assertEqual(state(s), before)

    def test_disturbance_fails_with_honest_partial_coverage_and_hold(self):
        s = load()
        t = CouponIndexing(s)
        s.data.qfrc_applied[1] = 10
        start = s.data.qpos.copy()
        r = t.move([0.04, 0])
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertGreater(r["physics_steps"], r["direct_clearance"]["checked_steps"])
        self.assertIsNone(r["direct_clearance"]["whole_leg_clearance_lower_bound_m"])
        self.assertFalse(np.array_equal(start, s.data.qpos))
        self.assertGreater(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
