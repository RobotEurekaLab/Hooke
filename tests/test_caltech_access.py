"""Continuous support, real access aperture and guarded measured hood motion."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.caltech_motion import HoodAccess
from real_labs.motion import PlanningError, _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.usability import camera_target_screening


PREFIX = "hood_primary__access_"


def session(obstacle=False):
    root, manifest = build_scene("caltech_stoltz_schlinger")
    if obstacle:
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="forbidden_obstacle",
            type="box",
            pos="-3.5 2.282 2.0",
            size=".15 .035 .025",
            contype="1",
            conaffinity="1",
        )
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def geom(s, key):
    return s.model.geom(PREFIX + key).id


def gap(s, first, second):
    return float(mujoco.mj_geomDistance(s.model, s.data, first, second, 2, None))


class CaltechAccessTests(unittest.TestCase):
    def test_supports_retain_cassette_and_ground_the_deck(self):
        s = session()
        for side in (-1, 1):
            for first, second in (
                ("cabinet", f"deck_support_{side}"),
                (f"deck_support_{side}", "deck"),
                ("cabinet", f"rail_pedestal_{side}"),
                (f"rail_pedestal_{side}", f"drawer_rail_{side}"),
            ):
                self.assertAlmostEqual(
                    gap(s, geom(s, first), geom(s, second)), 0, places=7
                )
            for axis in ("x", "y"):
                self.assertLessEqual(
                    gap(
                        s,
                        geom(s, "cassette"),
                        geom(s, f"cassette_retainer_{axis}_{side}"),
                    ),
                    1e-7,
                )
        for first, second in (
            ("cassette", "cassette_riser"),
            ("cassette_riser", "drawer_plate"),
        ):
            self.assertAlmostEqual(gap(s, geom(s, first), geom(s, second)), 0, places=7)
        for index, x in enumerate((-0.47, 0, 0.44)):
            for dx, dy in itertools.product((-0.083, 0.083), (-0.087, 0.087)):
                foot = geom(s, f"apparatus_{index}_foot_{dx}_{dy}")
                self.assertAlmostEqual(gap(s, geom(s, "deck"), foot), 0, places=7)
                self.assertAlmostEqual(
                    gap(s, geom(s, f"apparatus_{index}_hotplate"), foot), 0, places=7
                )

    def test_glovebox_foundation_and_transfer_saddle_connect(self):
        s = session()
        m = s.model

        def by(key):
            return [
                i
                for i in range(m.ngeom)
                if m.geom(i).name.startswith("facility_caltech_" + key + "_")
            ]

        cabinet = by("glovebox_support")[0]
        floor = by("glovebox_floor")[0]
        for runner in by("glovebox_support_runner"):
            self.assertAlmostEqual(gap(s, cabinet, runner), 0, places=7)
            self.assertAlmostEqual(gap(s, floor, runner), 0, places=7)
        saddle = by("glovebox_transfer_saddle")[0]
        barrel = by("glovebox_antechamber")[0]
        bracket = by("glovebox_transfer_bracket")[0]
        # Compiled cylinder/box faces are analytically tangent. MuJoCo GJK
        # reports a small negative distance at this coplanar support contact.
        self.assertLessEqual(gap(s, saddle, barrel), 0)
        self.assertAlmostEqual(
            s.data.geom_xpos[barrel, 2] - m.geom_size[barrel, 0],
            s.data.geom_xpos[saddle, 2] + m.geom_size[saddle, 2],
            places=9,
        )
        self.assertLessEqual(gap(s, saddle, bracket), 0)
        self.assertLessEqual(gap(s, floor, bracket), 0)

    def test_open_bearing_and_channel_clearances_over_original_travel(self):
        s = session()
        m = s.model
        np.testing.assert_allclose(
            [s.controls["hood_primary", k].upper for k in ("supply_drawer", "sash")],
            [0.23, 0.4],
        )
        for key, maximum in (("supply_drawer", 0.23), ("sash", 0.4)):
            index = m.jnt_qposadr[s.controls["hood_primary", key].joint]
            for q in np.linspace(0, maximum, 41):
                s.data.qpos[:] = 0
                s.data.qpos[index] = q
                mujoco.mj_forward(m, s.data)
                for side in (-1, 1):
                    if key == "supply_drawer":
                        rail = geom(s, f"drawer_rail_{side}")
                        for station in range(2):
                            roof = geom(s, f"shoe_roof_{side}_{station}")
                            self.assertAlmostEqual(gap(s, rail, roof), 0.001, places=7)
                            self.assertAlmostEqual(
                                gap(s, roof, geom(s, "drawer_plate")), 0, places=7
                            )
                            for flank in (-1, 1):
                                self.assertAlmostEqual(
                                    gap(
                                        s,
                                        rail,
                                        geom(s, f"shoe_flank_{side}_{station}_{flank}"),
                                    ),
                                    0.001,
                                    places=7,
                                )
                    else:
                        stile = geom(s, f"sash_stile_{side}")
                        for face in (-1, 1):
                            self.assertAlmostEqual(
                                gap(
                                    s,
                                    stile,
                                    geom(s, f"sash_channel_face_{side}_{face}"),
                                ),
                                0.001,
                                places=7,
                            )
                        self.assertAlmostEqual(
                            gap(s, stile, geom(s, f"sash_channel_side_{side}")),
                            0.001,
                            places=7,
                        )
        self.assertEqual(
            set(m.geom_contype[[geom(s, "sash_glass"), geom(s, "cassette")]]), {1}
        )

    def test_direct_parent_filtered_pairs_remain_clear_at_combined_extremes(self):
        s = session()
        a = HoodAccess(s)
        for q in itertools.product(np.linspace(0, 0.23, 9), np.linspace(0, 0.4, 9)):
            s.data.qpos[a.motion.qadr] = q
            mujoco.mj_forward(s.model, s.data)
            a._observe(s.model, s.data)
        self.assertGreaterEqual(a.minimum_gap, 0.001 - 1e-9)
        self.assertEqual(a.checked_steps, 81)

    def test_sash_is_an_actual_movable_barrier_and_open_aperture(self):
        s = session()
        m = s.model
        glass = geom(s, "sash_glass")
        # A ray through the working opening hits the enabled glass when closed.
        origin = np.array([-3.5, 2.0, 1.10])
        direction = np.array([0.0, 1.0, 0.0])
        hit = np.array([-1], np.int32)

        def ray():
            return mujoco.mj_ray(m, s.data, origin, direction, None, 1, -1, hit)

        self.assertGreater(ray(), 0)
        self.assertEqual(hit[0], glass)
        s.data.qpos[m.jnt_qposadr[s.controls["hood_primary", "sash"].joint]] = 0.4
        mujoco.mj_forward(m, s.data)
        self.assertGreater(ray(), 0.3)
        self.assertNotEqual(hit[0], glass)
        for i in range(m.ngeom):
            if (
                m.geom_bodyid[i]
                == m.jnt_bodyid[s.controls["hood_primary", "sash"].joint]
            ):
                self.assertEqual(int(m.geom_contype[i]), 1)

    def test_original_axes_track_full_range_commands(self):
        s = session()
        for equipment in s.equipment:
            for key in ("supply_drawer", "sash"):
                s.reset()
                control = s.controls[equipment, key]
                for fraction in (0.15, 0.85, 0.25, 1.0, 0.0):
                    report = s.command(equipment, key, fraction * control.upper)
                    self.assertLess(abs(report["error"]), 0.025 * control.upper)
                    self.assertTrue(np.isfinite(s.data.qpos).all())

    def test_measured_open_inspect_return_has_full_coverage_and_visible_target(self):
        s = session()
        a = HoodAccess(s)
        signature = _model_signature(s.model)
        report = a.open_inspect_return()
        self.assertEqual(
            report["status"],
            "succeeded",
            [(x["phase"], x.get("failure")) for x in report["legs"]],
        )
        self.assertEqual(report["physics_steps"], report["fully_checked_steps"])
        self.assertGreater(report["physics_steps"], 10000)
        self.assertFalse(report["reset_performed"])
        self.assertFalse(report["hold_requested"])
        self.assertLess(max(report["return_error_m"]), 0.0002)
        self.assertEqual(signature, _model_signature(s.model))
        exposed = report["legs"][1]
        self.assertGreater(exposed["aperture_height_m"], 0.4228)
        for leg in report["legs"]:
            self.assertTrue(leg["whole_leg_clearance_observed"])
            self.assertGreaterEqual(
                leg["minimum_clearance_lower_bound_m"], 0.001 - 1e-8
            )
            self.assertEqual(len(leg["after"]["qpos"]), s.model.nq)
        # Inspection copy only: task itself never writes qpos or resets.
        copy = mujoco.MjData(s.model)
        copy.qpos[:] = exposed["after"]["qpos"]
        mujoco.mj_forward(s.model, copy)
        row = next(
            x
            for x in camera_target_screening(s.model, copy, s.manifest)
            if x["equipment"] == "hood_primary"
        )
        self.assertIn("access", row["candidate_views"])

    def test_disabled_or_rebound_essential_interfaces_are_rejected(self):
        for key in ("sash_glass", "drawer_rail_1", "cassette"):
            s = session()
            i = geom(s, key)
            s.model.geom_contype[i] = s.model.geom_conaffinity[i] = 0
            with self.assertRaises(ValueError):
                HoodAccess(s)
        s = session()
        s.equipment["hood_primary"]["sample_site"] = s.equipment["hood_middle"][
            "sample_site"
        ]
        with self.assertRaises(ValueError):
            HoodAccess(s)
        for bit in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            s = session()
            s.model.opt.disableflags |= int(bit)
            with self.assertRaises(PlanningError):
                HoodAccess(s)

    def test_foreign_state_rejects_before_constructor_mutation(self):
        s = session()
        other = copy(s.model)
        other.geom_pos[geom(s, "sash_glass"), 2] += 0.5
        s.data = mujoco.MjData(other)
        s.data.time = 0.123
        s.data.ctrl[:] = 0.01
        before = (
            s.data.qpos.copy(),
            s.data.qvel.copy(),
            s.data.ctrl.copy(),
            s.data.time,
        )
        with self.assertRaises(ValueError):
            HoodAccess(s)
        for actual, expected in zip(
            (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time), before
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_replaced_state_rejects_before_move_or_hold_mutation(self):
        s = session()
        task = HoodAccess(s)
        other = copy(s.model)
        other.geom_pos[geom(s, "sash_glass"), 2] += 0.5
        s.data = mujoco.MjData(other)
        s.data.time = 0.321
        s.data.ctrl[:] = 0.01
        before = (
            s.data.qpos.copy(),
            s.data.qvel.copy(),
            s.data.ctrl.copy(),
            s.data.time,
        )
        with self.assertRaises(ValueError):
            task.move((0, 0.005))
        for actual, expected in zip(
            (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time), before
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(task.observed_steps, 0)

    def test_model_mutation_is_blocked_before_stepping(self):
        s = session()
        a = HoodAccess(s)
        s.model.geom_contype[geom(s, "sash_glass")] = 0
        result = a.move((0, 0.03))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(s.data.time, 0)
        self.assertFalse(result["whole_leg_clearance_observed"])
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])

    def test_forbidden_obstacle_does_not_become_a_successful_access(self):
        s = session(obstacle=True)
        a = HoodAccess(s)
        initial = s.data.qpos.copy()
        result = a.move((0, 0.4))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["physics_steps"], 0)
        np.testing.assert_array_equal(s.data.qpos, initial)
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["whole_leg_clearance_observed"])

    def test_loaded_failure_holds_actual_state_without_claiming_complete_coverage(self):
        s = session()
        a = HoodAccess(s)
        index = s.model.jnt_dofadr[s.controls["hood_primary", "supply_drawer"].joint]
        s.data.qfrc_applied[index] = 100
        result = a.move((0.03, 0))
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["physics_steps"], result["fully_checked_steps"])
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertFalse(result["whole_leg_clearance_observed"])
        self.assertGreater(s.data.time, 0)
        self.assertNotEqual(float(s.data.qpos[a.motion.qadr[0]]), 0)
        np.testing.assert_allclose(
            s.data.ctrl[a.motion.actuators],
            np.clip(a.motion.measured(), a.motion.lower, a.motion.upper),
        )
        s.reset()
        self.assertEqual(s.data.time, 0)
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)


if __name__ == "__main__":
    unittest.main()
