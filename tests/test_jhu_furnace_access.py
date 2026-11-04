"""Cold furnace feed supports, retained surface and measured motion boundaries."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.jhu_furnace_motion import FurnaceInspection, inspect_jhu_furnace, SCENE
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return (
        s.data.time,
        *(
            getattr(s.data, k).tobytes()
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "act",
                "eq_active",
            )
        ),
    )


def gid(s, key):
    return s.model.geom("mirror_furnace__jhu_" + key).id


class FurnaceAccessTests(unittest.TestCase):
    def test_original_frames_drives_and_sample_scale(self):
        s = session()
        m = s.model
        for key, pos, limits, kp, mass in (
            ("feed_rod_height", [0, -0.025, 1.92], [-0.06, 0.06], 360, 0.22),
            ("feed_rod_rotation", [0, 0, -0.005], [-1, 1], 90, 0.06),
        ):
            c = s.controls["mirror_furnace", key]
            body = m.jnt_bodyid[c.joint]
            np.testing.assert_allclose(m.body_pos[body], pos)
            np.testing.assert_allclose(m.jnt_axis[c.joint], [0, 0, 1])
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
            self.assertEqual(m.body_mass[body], mass)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], kp)
            self.assertEqual(
                m.dof_damping[m.jnt_dofadr[c.joint]],
                2.5 if key == "feed_rod_height" else 0.8,
            )
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
        np.testing.assert_allclose(
            m.geom_size[gid(s, "inert_feed_coupon"), :2], [0.006, 0.02]
        )
        site = m.site("mirror_furnace__site_crystal_coupon").id
        np.testing.assert_allclose(m.site_pos[site], [0, -0.008, -0.470])
        witness = gid(s, "coupon_witness")
        normal = s.data.geom_xmat[witness].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(
            s.data.site_xpos[site],
            s.data.geom_xpos[witness] + normal * 0.0015,
            atol=1e-10,
        )

    def test_enabled_contact_covers_each_moving_weld_group(self):
        s = session()
        r = screen_contact_coverage(s.model)
        self.assertFalse(r["issues"])
        self.assertEqual(r["zero_contact_visible_group_count"], 0)
        for c in s.controls.values():
            group = s.model.body_weldid[s.model.jnt_bodyid[c.joint]]
            enabled = [
                g
                for g in range(s.model.ngeom)
                if s.model.body_weldid[s.model.geom_bodyid[g]] == group
                and (s.model.geom_contype[g] or s.model.geom_conaffinity[g])
            ]
            self.assertGreater(len(enabled), 4)

    def test_full_coupled_travel_checks_parent_filtered_interfaces(self):
        s = session()
        t = FurnaceInspection(s)
        m, d = s.model, s.data
        pairset = {
            frozenset((m.geom(int(a)).name, m.geom(int(b)).name)) for a, b in t.pairs
        }
        self.assertIn(
            frozenset(
                (
                    "mirror_furnace__jhu_guide_rod_0.085",
                    "mirror_furnace__jhu_guide_sleeve_0.085_proxy_0",
                )
            ),
            pairset,
        )
        self.assertIn(
            frozenset(
                (
                    "mirror_furnace__jhu_rotor_journal",
                    "mirror_furnace__jhu_spindle_bearing_proxy_0",
                )
            ),
            pairset,
        )
        for h, a in itertools.product(
            np.linspace(-0.06, 0.06, 13), np.linspace(-1, 1, 17)
        ):
            d.qpos[:] = [h, a]
            mujoco.mj_forward(m, d)
            t.check(m, d)
            self.assertFalse(any(c.dist < -1e-7 for c in d.contact), (h, a))
            obs = t.observe()
            self.assertGreaterEqual(obs["cold_axial_gap_m"], 0.0249999)
        self.assertGreaterEqual(t.minimum, 0.0005999)

    def test_fullstroke_guide_engagement_and_mount_load_paths(self):
        s = session()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(m, d, gid(s, a), gid(s, b), 0.1, None)

        for a, b in (
            ("cabinet", "lower_rim"),
            ("lower_rim", "furnace_optical_deck"),
            ("furnace_optical_deck", "lower_pedestal"),
            ("lower_pedestal", "lower_spindle"),
            ("datum_foot", "datum_post"),
            ("datum_post", "datum_upright"),
            ("datum_upright", "datum_tick"),
            ("carriage_bridge", "carriage_arm"),
            ("rotor_journal", "rotor_cap"),
            ("rotor_journal", "rotor_chuck"),
            ("rotor_chuck", "feed_shaft"),
            ("feed_shaft", "inert_feed_coupon"),
            ("inert_feed_coupon", "coupon_witness"),
        ):
            self.assertLessEqual(gap(a, b), 2e-7, (a, b, gap(a, b)))
        for x in (-0.085, 0.085):
            for a, b in (
                (f"guide_lower_mount_{x}", "guide_bridge"),
                (f"guide_rod_{x}", f"guide_lower_mount_{x}"),
                (f"guide_top_mount_{x}", "roof"),
            ):
                self.assertLessEqual(gap(a, b), 2e-7, (a, b))
            sleeve = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name.startswith(
                    f"mirror_furnace__jhu_guide_sleeve_{x}_proxy_"
                )
            ]
            bridge = gid(s, "carriage_bridge")
            self.assertLessEqual(
                min(mujoco.mj_geomDistance(m, d, g, bridge, 0.1, None) for g in sleeve),
                2e-7,
            )
            for h in (-0.06, 0, 0.06):
                d.qpos[0] = h
                mujoco.mj_forward(m, d)
                self.assertGreaterEqual(
                    min(d.geom_xpos[g, 2] - m.geom_size[g, 2] for g in sleeve), 1.78
                )
                self.assertLessEqual(
                    max(d.geom_xpos[g, 2] + m.geom_size[g, 2] for g in sleeve), 2.20
                )
        proxies = [
            g for g in range(m.ngeom) if "jhu_spindle_bearing_proxy_" in m.geom(g).name
        ]
        arm = gid(s, "carriage_arm")
        self.assertLessEqual(
            min(mujoco.mj_geomDistance(m, d, g, arm, 0.1, None) for g in proxies), 2e-7
        )

    def test_source_context_mounts_and_parked_services_connect(self):
        s = session()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(m, d, gid(s, a), gid(s, b), 0.25, None)

        for side in (-1, 1):
            for a, b in (
                (f"reflector_{side}", f"reflector_strap_{side}_1.31"),
                (f"reflector_strap_{side}_1.31", f"front_rod_{side}"),
                (f"front_rod_{side}", "furnace_optical_deck"),
                (f"fan_mount_{side}", f"fan_case_{side}"),
                (f"fan_mount_{side}", f"reflector_{side}"),
                (f"blower_plinth_{side}", "furnace_optical_deck"),
                (f"blower_plinth_{side}", f"blower_motor_support_{side}"),
                (f"blower_motor_support_{side}", f"blower_body_{side}"),
                (f"blower_body_{side}", f"blower_outlet_{side}"),
                (f"fixed_cooling_lead_{side}_0", f"cooling_upper_anchor_{side}"),
                (f"cooling_upper_anchor_{side}", "roof"),
                (f"fixed_cooling_lead_{side}_2", f"fan_case_{side}"),
            ):
                self.assertLessEqual(gap(a, b), 2e-7, (a, b, gap(a, b)))

    def test_actual_height_index_observe_and_return_no_reset(self):
        s = session()
        home = s.data.qpos.copy()
        report = inspect_jhu_furnace(s)
        self.assertEqual(
            report["status"], "succeeded", report["phases"][-1].get("failure")
        )
        self.assertTrue(report["coverage_complete"])
        self.assertGreater(report["executed_steps"], 1000)
        self.assertEqual(report["checked_steps"], report["executed_steps"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 1)
        np.testing.assert_allclose(s.data.qpos, home, atol=1e-4)
        for o in report["observations"]:
            self.assertTrue(o["visible"])
            self.assertEqual(
                o["camera_first_hit"], "mirror_furnace__jhu_coupon_witness"
            )
            if o["phase"] != "lower_to_home":
                self.assertLess(abs(o["height_error_m"]), 1e-4)
        self.assertGreater(report["whole_run_clearance_lower_bound_m"], 0.00059)
        for phase in report["phases"]:
            self.assertEqual(
                phase["physics_steps"], phase["direct_clearance"]["checked_steps"]
            )

    def test_disabled_essential_colliders_or_changed_source_reject_before_mutation(
        self,
    ):
        for case in (
            "guide",
            "witness",
            "coupon",
            "geometry",
            "mass",
            "contact",
            "constraint",
            "limit",
            "actuation",
            "metadata",
        ):
            s = session()
            if case in ("guide", "witness", "coupon"):
                key = {
                    "guide": "guide_sleeve_0.085_proxy_0",
                    "witness": "coupon_witness",
                    "coupon": "inert_feed_coupon",
                }[case]
                i = gid(s, key)
                s.model.geom_contype[i] = s.model.geom_conaffinity[i] = 0
            elif case == "geometry":
                s.model.geom_pos[gid(s, "guide_rod_0.085"), 0] += 0.01
            elif case == "mass":
                s.model.body_mass[
                    s.model.jnt_bodyid[
                        s.controls["mirror_furnace", "feed_rod_height"].joint
                    ]
                ] *= 2
            elif case == "metadata":
                s.equipment["mirror_furnace"]["feed_interface"][
                    "inspection_height_m"
                ] = 0
            else:
                s.model.opt.disableflags |= int(
                    getattr(
                        mujoco.mjtDisableBit,
                        "mjDSBL_"
                        + {
                            "contact": "CONTACT",
                            "constraint": "CONSTRAINT",
                            "limit": "LIMIT",
                            "actuation": "ACTUATION",
                        }[case],
                    )
                )
            before = state(s)
            with self.assertRaises(ValueError, msg=case):
                FurnaceInspection(s)
            self.assertEqual(before, state(s))

    def test_foreign_replaced_state_and_postconstruction_geometry_reject(self):
        s = session()
        s.data = mujoco.MjData(copy(s.model))
        before = state(s)
        with self.assertRaises(ValueError):
            FurnaceInspection(s)
        self.assertEqual(before, state(s))
        for case in ("state", "geometry"):
            s = session()
            t = FurnaceInspection(s)
            if case == "state":
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_size[gid(s, "inert_feed_coupon"), 0] *= 2
            before = state(s)
            with self.assertRaises(ValueError):
                t.move([0.04, 0])
            self.assertEqual(before, state(s))

    def test_held_axis_force_failure_has_partial_checks_and_measured_hold(self):
        s = session()
        t = FurnaceInspection(s)
        c = s.controls["mirror_furnace", "feed_rod_rotation"]
        v = s.model.jnt_dofadr[c.joint]
        q = s.model.jnt_qposadr[c.joint]
        s.data.qfrc_applied[v] = 100
        before = s.data.qpos.copy()
        r = t.move([0.04, 0])
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertGreater(r["physics_steps"], r["direct_clearance"]["checked_steps"])
        self.assertIsNone(r["direct_clearance"]["whole_leg_clearance_lower_bound_m"])
        self.assertFalse(np.array_equal(before, s.data.qpos))
        self.assertGreater(s.data.time, 0)
        self.assertAlmostEqual(
            s.data.ctrl[c.actuator], np.clip(s.data.qpos[q], c.lower, c.upper)
        )

    def test_existing_fullspan_commands_and_reset_remain_available(self):
        s = session()
        for key, c in s.controls.items():
            s.reset()
            for fraction in (0.15, 0.85, 0.25):
                r = s.command(
                    *key, c.lower + fraction * (c.upper - c.lower), duration=2
                )
                self.assertLessEqual(abs(r["error"]), 0.025 * (c.upper - c.lower))
        s.reset()
        np.testing.assert_allclose(s.data.qpos, s.model.qpos0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
