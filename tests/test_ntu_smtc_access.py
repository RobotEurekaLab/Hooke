"""Supported open guides and measured cold access on both membrane skids."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.ntu_smtc_motion import EQUIPMENT, SCENE, SkidInspection
from real_labs.runtime import InstrumentSession


class MembraneSkidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_controls_frames_mass_and_sample_scale(self):
        m = self.model
        self.assertEqual(m.njnt, 4)
        np.testing.assert_array_equal(m.actuator_gainprm[:, 0], [50, 180, 50, 180])
        np.testing.assert_allclose(m.jnt_range, [[0, 1.5707963], [0, 0.15]] * 2)
        np.testing.assert_array_equal(m.jnt_axis, [[0, 1, 0], [0, -1, 0]] * 2)
        for equipment in EQUIPMENT:
            np.testing.assert_allclose(
                m.body(equipment + "__body_isolation_valve").pos, [0.30, -0.405, 0.86]
            )
            np.testing.assert_allclose(
                m.body(equipment + "__body_sample_carriage").pos, [0, -0.16, 0.40]
            )
            self.assertAlmostEqual(
                m.body(equipment + "__body_isolation_valve").mass[0], 0.08
            )
            self.assertAlmostEqual(
                m.body(equipment + "__body_sample_carriage").mass[0], 0.28
            )
            np.testing.assert_allclose(
                m.site(equipment + "__site_water_vial").pos, [0, 0, 0.065]
            )
            np.testing.assert_allclose(
                m.geom(equipment + "__smtc_vial_1").size[:2], [0.032, 0.045]
            )

    def test_combined_full_travel_keeps_enabled_clearance(self):
        s = self.session()
        tasks = [SkidInspection(s, e) for e in EQUIPMENT]
        for angle, travel in product(
            np.linspace(0, self.model.jnt_range[0, 1], 5), np.linspace(0, 0.15, 5)
        ):
            s.data.qpos[:] = [
                angle,
                travel,
                self.model.jnt_range[0, 1] - angle,
                0.15 - travel,
            ]
            mujoco.mj_forward(s.model, s.data)
            for task in tasks:
                task.check(s.model, s.data)
                self.assertGreater(task.minimum, 0.00059)

    def test_unculled_enabled_pairs_match_broadphase(self):
        s = self.session()
        m, d = s.model, s.data
        task = SkidInspection(s)
        moving = {int(m.body_weldid[b]) for b in task.motion.moving_bodies}
        ids = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        pairs = [
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in moving
                or m.body_weldid[m.geom_bodyid[b]] in moving
            )
        ]
        self.assertEqual(set(pairs), {tuple(p) for p in task.pairs})
        for q in (
            (0, 0, 0, 0.15),
            (0.617, 0.091, 1.233, 0.037),
            (1.5707963, 0.15, 0, 0),
        ):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            direct = min(
                [0.002]
                + [
                    float(mujoco.mj_geomDistance(m, d, a, b, 0.002, None))
                    for a, b in pairs
                ]
            )
            task.reset_counts()
            task.check(m, d)
            self.assertGreater(direct, 0.00059)
            self.assertAlmostEqual(direct, task.minimum, places=8)

    def test_frame_modules_and_guides_have_connected_supports(self):
        s = self.session()
        m, d = s.model, s.data
        for equipment in EQUIPMENT:
            prefix = equipment + "__smtc_"
            pairs = [
                (f"foot_{i}_{j}", f"post_{i}_{j}") for i in range(2) for j in range(2)
            ]
            pairs += [
                (f"module_arm_{k}_{i}", f"module_saddle_{k}_{i}")
                for k in range(2)
                for i in range(2)
            ]
            pairs += [
                (f"module_arm_{k}_{i}", f"module_crossbar_{k}")
                for k in range(2)
                for i in range(2)
            ]
            pairs += [
                (f"rail_pedestal_{i}_{j}", f"rail_footplate_{i}")
                for i in range(2)
                for j in range(2)
            ]
            pairs += [
                ("monitor_stalk", "monitor_support"),
                ("monitor_support", "crossbar_2_1"),
                ("screen_bezel", "monitor_housing"),
                ("inactive_display", "screen_bezel"),
            ]
            pairs += [("bottom_deck", f"crossbar_0_{j}") for j in range(2)]
            pairs += [("bottom_deck", f"sidebar_0_{i}") for i in range(2)]
            pairs += [
                (f"module_{k}", f"module_flange_{k}_{i}")
                for k in range(2)
                for i in range(2)
            ]
            pairs += [
                (f"gauge_tick_{i}_{j}", f"gauge_face_{i}")
                for i in range(2)
                for j in range(9)
            ]
            for a, b in pairs:
                self.assertLessEqual(
                    mujoco.mj_geomDistance(
                        m, d, m.geom(prefix + a).id, m.geom(prefix + b).id, 0.001, None
                    ),
                    1e-7,
                    (equipment, a, b),
                )
        # Bearings remain away from the end bosses throughout the complete stroke.
        for travel in (0, 0.15):
            d.qpos[[1, 3]] = travel
            mujoco.mj_forward(m, d)
            for task in (SkidInspection(s, e) for e in EQUIPMENT):
                task.check(m, d)

    def test_actual_move_return_and_observed_vial_displacement(self):
        s = self.session()
        task = SkidInspection(s, EQUIPMENT[1])
        initial = task.observe()
        outward = task.move((0.1, 0.015))
        self.assertEqual(outward["status"], "succeeded", outward)
        actual = task.observe()
        np.testing.assert_allclose(
            np.subtract(actual["vial_center_world_m"], initial["vial_center_world_m"]),
            [0, -0.015, 0],
            atol=0.0002,
        )
        returned = task.move((0, 0))
        for report in (outward, returned):
            self.assertEqual(report["status"], "succeeded", report)
            self.assertTrue(report["direct_clearance"]["coverage_complete"])
            self.assertEqual(
                report["physics_steps"], report["direct_clearance"]["checked_steps"]
            )
        np.testing.assert_allclose(task.motion.measured(), [0, 0], atol=0.0002)

    def test_changed_geometry_contacts_drive_and_metadata_reject_before_step(self):
        mutations = (
            lambda s: s.model.geom_size.__setitem__(
                (s.model.geom(EQUIPMENT[0] + "__smtc_valve_stem").id, 0), 0.009
            ),
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom(EQUIPMENT[0] + "__smtc_valve_lever").id, 0
            ),
            lambda s: s.model.actuator_gainprm.__setitem__((0, 0), 51),
            lambda s: s.manifest.__setitem__("scene_sha256", "foreign"),
            lambda s: s.manifest["camera"]["skid_detail"].__setitem__("fovy", 50),
        )
        for mutate in mutations:
            s = self.session()
            mutate(s)
            before = s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time
            with self.assertRaises(ValueError):
                SkidInspection(s)
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, before[2])

    def test_held_other_skid_disturbance_reports_incomplete_coverage(self):
        s = self.session()
        task = SkidInspection(s)
        s.data.qfrc_applied[3] = 1000
        report = task.move((0.1, 0.015))
        self.assertEqual(report["status"], "failed", report)
        self.assertGreater(s.data.time, 0)
        self.assertTrue(report["hold_requested"])
        coverage = report["direct_clearance"]
        self.assertLess(coverage["checked_steps"], coverage["executed_steps"])
        self.assertIsNone(coverage["whole_leg_clearance_lower_bound_m"])
        self.assertNotEqual(s.data.qpos[3], 0)

    def test_replaced_state_rejects_without_mutation(self):
        s = self.session()
        task = SkidInspection(s)
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            task.move((0.1, 0.01))
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
