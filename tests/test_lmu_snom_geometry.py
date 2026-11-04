"""Physical clearances and measured coarse-to-fine LMU operation contracts."""

import itertools
from dataclasses import replace
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.lmu_snom_geometry import CANTILEVER_SIZE, TIP_RADIUS
from real_labs.lmu_snom_motion import SurfaceObserver, inspect_lmu
from real_labs.runtime import InstrumentSession


class LmuSnomGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("lmu_hybrid_snom")
        cls.xml = ET.tostring(root, encoding="unicode")

    def setUp(self):
        self.s = InstrumentSession(
            mujoco.MjModel.from_xml_string(self.xml), self.manifest
        )
        self.m, self.d = self.s.model, self.s.data

    def test_true_probe_scale_and_surface_registration(self):
        beam = self.m.geom("snom_table__cantilever").id
        np.testing.assert_allclose(
            self.m.geom_size[beam] * 2, CANTILEVER_SIZE, rtol=0, atol=1e-12
        )
        self.assertEqual(
            self.m.geom_size[self.m.geom("snom_table__snom_probe_tip").id, 0],
            TIP_RADIUS,
        )
        observer = SurfaceObserver(self.s)
        self.assertTrue(observer.read()["over_coupon"])
        self.assertAlmostEqual(observer.read()["clearance_m"], 0.0201, places=11)
        for name in (
            "cantilever",
            "fine_tip_cone",
            "snom_probe_tip",
            "probe_chip",
            "coarse_head",
        ):
            i = self.m.geom("snom_table__" + name).id
            self.assertEqual(int(self.m.geom_contype[i]), 1)
        for name in ("overview", "interior", "workstation", "source", "probe_detail"):
            self.assertGreaterEqual(self.m.camera(name).id, 0)

    def test_all_combined_extremes_and_intermediate_filtered_pairs(self):
        observer = SurfaceObserver(self.s)
        controls = list(self.s.controls.values())
        rows = set(itertools.product((0.0, 0.5, 1.0), repeat=4))
        for axis in range(4):
            for side in (0.0, 1.0):
                for fraction in np.linspace(0, 1, 21):
                    values = [side] * 4
                    values[axis] = float(fraction)
                    rows.add(tuple(values))
        minimum = float("inf")
        for row in sorted(rows):
            self.s.reset()
            for c, f in zip(controls, row):
                self.d.qpos[self.m.jnt_qposadr[c.joint]] = (
                    c.lower + (c.upper - c.lower) * f
                )
            mujoco.mj_forward(self.m, self.d)
            measured = observer.check()
            minimum = min(minimum, measured["clearance_m"])
        self.assertAlmostEqual(minimum, 5e-6, delta=1e-9)

    def test_open_collar_bores_and_support_connections(self):
        head = self.s.controls["snom_table", "inspection_head_height"]
        for value in (head.lower, 0.0, head.upper):
            self.s.reset()
            self.d.qpos[self.m.jnt_qposadr[head.joint]] = value
            mujoco.mj_forward(self.m, self.d)
            for side, x in ((-0.05, 0.84), (0.05, 0.94)):
                post = self.m.geom("snom_table__coarse_guide_post_" + str(x)).id
                for i in range(40):
                    wall = self.m.geom(
                        "snom_table__coarse_collar_" + str(side) + "_contact_" + str(i)
                    ).id
                    gap = mujoco.mj_geomDistance(self.m, self.d, post, wall, 0.1, None)
                    self.assertGreaterEqual(gap, 0.000599)
                # Fixed guides remain longer than the moving collar at both ends.
                centre = self.d.geom_xpos[post, 2]
                self.assertLessEqual(centre - 0.09, 1.14 + value - 0.026)
                self.assertGreaterEqual(centre + 0.09, 1.14 + value + 0.026)

        # Continuous chip-to-holder support, not a floating microscopic asset.
        def bounds(name):
            g = self.m.geom("snom_table__" + name).id
            return (
                self.d.geom_xpos[g] - self.m.geom_size[g],
                self.d.geom_xpos[g] + self.m.geom_size[g],
            )

        chip_low, chip_high = bounds("probe_chip")
        mount_low, mount_high = bounds("probe_chip_mount")
        self.assertAlmostEqual(chip_high[2], mount_low[2], places=11)
        beam_low, beam_high = bounds("cantilever")
        self.assertGreater(
            np.minimum(chip_high, beam_high)[0] - np.maximum(chip_low, beam_low)[0], 0
        )
        self.assertAlmostEqual(chip_low[2], beam_low[2], places=11)

    def test_enclosure_and_hinge_have_continuous_static_support(self):
        def bounds(name):
            i = self.m.geom("snom_table__" + name).id
            center = self.d.geom_xpos[i]
            return center - self.m.geom_size[i], center + self.m.geom_size[i]

        low, high = bounds("station_foundation")
        for name in ("left_instrument_housing", "right_instrument_housing"):
            shell_low, shell_high = bounds(name)
            self.assertAlmostEqual(shell_low[2], high[2], places=11)
            self.assertTrue(np.all(shell_low[:2] >= low[:2]))
            self.assertTrue(np.all(shell_high[:2] <= high[:2]))
        hinge_low, hinge_high = bounds("lid_hinge_support")
        for name in ("left_instrument_housing", "right_instrument_housing"):
            shell_low, shell_high = bounds(name)
            self.assertTrue(
                np.all(
                    np.minimum(hinge_high, shell_high)
                    - np.maximum(hinge_low, shell_low)
                    > 0
                )
            )

    def test_all_four_controls_reverse_and_return(self):
        for key, c in self.s.controls.items():
            self.s.reset()
            for f in (0.15, 0.85, 0.25):
                result = self.s.command(
                    *key, c.lower + (c.upper - c.lower) * f, duration=2
                )
                self.assertLessEqual(abs(result["error"]), (c.upper - c.lower) * 0.025)
            self.s.command(*key, 0, duration=2)
            self.assertAlmostEqual(
                self.d.qpos[self.m.jnt_qposadr[c.joint]], 0, delta=1e-8
            )

    def test_registered_line_scan_and_physical_return(self):
        self.s.command("snom_table", "specimen_translation", 0.015, duration=1)
        initial = self.d.qpos.copy()
        report = inspect_lmu(self.s)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["returned"])
        self.assertFalse(report["reset_performed"])
        self.assertEqual(report["executed_steps"], report["checked_steps"])
        self.assertGreater(report["minimum_checked_clearance_m"], 3e-6)
        for row in report["scan"]:
            self.assertAlmostEqual(row["x_m"], row["requested_x_m"], delta=3e-8)
            self.assertAlmostEqual(row["clearance_m"], 10e-6, delta=3e-8)
        np.testing.assert_allclose(self.d.qpos, initial, atol=2e-7, rtol=0)
        self.assertEqual(report["science_model"], "not_simulated")

    def test_invalid_requests_and_registration_do_not_move_state(self):
        q = self.d.qpos.copy()
        ctrl = self.d.ctrl.copy()
        with self.assertRaises(ValueError):
            inspect_lmu(self.s, clearance_m=1e-9)
        with self.assertRaises(ValueError):
            inspect_lmu(self.s, offsets_m=(0.002,))
        np.testing.assert_array_equal(self.d.qpos, q)
        np.testing.assert_array_equal(self.d.ctrl, ctrl)
        surface = self.m.site("snom_table__site_snom_coupon").id
        self.m.site_pos[surface, 1] = -0.01
        with self.assertRaises(ValueError):
            inspect_lmu(self.s)
        np.testing.assert_array_equal(self.d.qpos, q)

    def test_failure_keeps_actual_state_and_reports_incomplete_coverage(self):
        fine = self.s.controls["snom_table", "fine_approach_z"]
        self.d.qfrc_applied[self.m.jnt_dofadr[fine.joint]] = -3.0
        report = inspect_lmu(self.s)
        self.assertFalse(report["passed"])
        self.assertFalse(report["reset_performed"])
        self.assertFalse(report["return_attempted"])
        self.assertGreater(report["executed_steps"], report["checked_steps"])
        self.assertFalse(report["coverage_complete"])
        self.assertGreater(np.linalg.norm(self.d.qpos), 1e-8)
        self.s.reset()
        np.testing.assert_array_equal(self.d.qpos, self.m.qpos0)

    def test_disabled_essential_colliders_reject_before_state_mutation(self):
        names = [
            "snom_probe_tip",
            "fine_tip_cone",
            "cantilever",
            "snom_inert_coupon",
            "coarse_guide_post_0.84",
            "coarse_collar_-0.05_contact_0",
            "fine_x_carriage_side_-1",
        ]
        for name in names:
            for mask in ("geom_contype", "geom_conaffinity"):
                with self.subTest(geometry=name, mask=mask):
                    s = InstrumentSession(
                        mujoco.MjModel.from_xml_string(self.xml), self.manifest
                    )
                    i = s.model.geom("snom_table__" + name).id
                    getattr(s.model, mask)[i] = 0
                    q, v, ctrl, time = (
                        s.data.qpos.copy(),
                        s.data.qvel.copy(),
                        s.data.ctrl.copy(),
                        s.data.time,
                    )
                    with self.assertRaises(ValueError):
                        inspect_lmu(s)
                    np.testing.assert_array_equal(s.data.qpos, q)
                    np.testing.assert_array_equal(s.data.qvel, v)
                    np.testing.assert_array_equal(s.data.ctrl, ctrl)
                    self.assertEqual(s.data.time, time)

    def test_physics_flags_and_stale_control_bindings_reject_without_movement(self):
        flags = (
            "mjDSBL_CONTACT",
            "mjDSBL_CONSTRAINT",
            "mjDSBL_ACTUATION",
            "mjDSBL_LIMIT",
        )
        for flag in flags:
            s = InstrumentSession(
                mujoco.MjModel.from_xml_string(self.xml), self.manifest
            )
            s.model.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, flag))
            with self.assertRaises(ValueError):
                inspect_lmu(s)
            self.assertEqual(s.data.time, 0)
            np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)
            np.testing.assert_array_equal(s.data.ctrl, np.zeros(4))
        key = ("snom_table", "fine_scan_x")
        self.s.controls[key] = replace(self.s.controls[key], actuator=0)
        with self.assertRaises(ValueError):
            inspect_lmu(self.s)
        self.assertEqual(self.d.time, 0)

    def test_wrong_model_geometry_state_and_metadata_are_rejected(self):
        variants = (
            "tip_body",
            "tip_size",
            "actuator_force",
            "surface_position",
            "foreign_data",
            "scene_hash",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                s = InstrumentSession(
                    mujoco.MjModel.from_xml_string(self.xml), dict(self.manifest)
                )
                tip = s.model.geom("snom_table__snom_probe_tip").id
                if variant == "tip_body":
                    s.model.geom_bodyid[tip] = 0
                elif variant == "tip_size":
                    s.model.geom_size[tip, 0] *= 10
                elif variant == "actuator_force":
                    s.model.actuator_forcelimited[0] = 0
                elif variant == "surface_position":
                    s.model.site_pos[
                        s.model.site("snom_table__site_snom_coupon").id, 0
                    ] += 0.001
                elif variant == "foreign_data":
                    s.data = mujoco.MjData(mujoco.MjModel.from_xml_string(self.xml))
                else:
                    s.manifest["scene_sha256"] = "incorrect"
                with self.assertRaises(ValueError):
                    inspect_lmu(s)
                self.assertEqual(s.data.time, 0)

    def test_tracking_failure_is_not_scan_success(self):
        fine = self.s.controls["snom_table", "fine_approach_z"]
        self.d.qfrc_applied[self.m.jnt_dofadr[fine.joint]] = -0.05
        report = inspect_lmu(self.s)
        self.assertFalse(report["passed"])
        self.assertIn("tracking_failure", report["failure"])
        self.assertEqual(report["scan"], [])
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["returned"])


if __name__ == "__main__":
    unittest.main()
