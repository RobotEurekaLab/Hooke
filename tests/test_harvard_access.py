"""Source-informed aligner supports and measured retained-wafer inspection."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.harvard_motion import CONTROLS, WaferInspection
from real_labs.runtime import InstrumentSession


STATIONS = ("lithography_aligner", "inspection_aligner")


def session():
    root, manifest = build_scene("harvard_cns_cambridge")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def geom(s, key, station=STATIONS[0]):
    return s.model.geom(station + "__cns_" + key).id


def gap(s, first, second):
    return float(mujoco.mj_geomDistance(s.model, s.data, first, second, 2, None))


class HarvardAccessTests(unittest.TestCase):
    def test_original_six_control_frames_limits_and_sample_dimensions(self):
        s = session()
        self.assertEqual(len(s.controls), 6)
        for equipment in STATIONS:
            for key, pos, axis, limits in (
                ("wafer_y", [0, -0.01, 0.30], [0, 1, 0], [-0.09, 0.09]),
                ("wafer_x", [0, 0, 0.067], [1, 0, 0], [-0.065, 0.065]),
                ("alignment_head", [0, 0.115, 0.735], [0, 0, 1], [0, 0.11]),
            ):
                control = s.controls[equipment, key]
                body = s.model.jnt_bodyid[control.joint]
                np.testing.assert_allclose(s.model.body_pos[body], pos)
                np.testing.assert_allclose(s.model.jnt_axis[control.joint], axis)
                np.testing.assert_allclose(s.model.jnt_range[control.joint], limits)
                np.testing.assert_array_equal(
                    s.model.actuator_forcerange[control.actuator], [-120, 120]
                )
            wafer = geom(s, "wafer", equipment)
            np.testing.assert_allclose(s.model.geom_size[wafer, :2], [0.074, 0.0015])
            self.assertEqual(
                s.model.geom_bodyid[wafer],
                s.model.jnt_bodyid[s.controls[equipment, "wafer_x"].joint],
            )
            self.assertIn(
                "Rigid clamp", s.equipment[equipment]["harvard_access"]["attachment"]
            )

    def test_fixed_foundation_and_wafer_support_chains(self):
        s = session()
        for equipment in STATIONS:
            for first, second in (
                ("core", "deck"),
                ("deck", "y_rail_mount_0_0"),
                ("y_rail_mount_0_0", "y_rail_0"),
                ("core", "column_foot_0"),
                ("column_foot_0", "column_0"),
                ("column_0", "top_crossmember"),
                ("y_shoe_0_0_roof", "y_plate"),
                ("y_plate", "x_rail_mount_0_0"),
                ("x_rail_mount_0_0", "x_rail_0"),
                ("x_shoe_0_0_roof", "x_plate"),
                ("x_plate", "chuck"),
                ("chuck", "wafer_shim"),
                ("wafer_shim", "wafer"),
                ("head_shell", "head_neck"),
                ("head_neck", "head_bridge"),
                ("head_shell", "objective_nose"),
                ("objective_nose", "objective_0"),
                ("objective_0", "objective_lens_0"),
            ):
                self.assertLessEqual(
                    gap(s, geom(s, first, equipment), geom(s, second, equipment)),
                    1e-7,
                    (equipment, first, second),
                )
            for i in range(3):
                for first, second in (
                    ("chuck", f"clamp_foot_{i}"),
                    (f"clamp_foot_{i}", f"clamp_cap_{i}"),
                    (f"clamp_cap_{i}", "wafer"),
                ):
                    self.assertLessEqual(
                        gap(s, geom(s, first, equipment), geom(s, second, equipment)),
                        1e-7,
                    )
        for first, second in (
            ("core", "tower_pedestal"),
            ("tower_pedestal", "bellows_core"),
            ("bellows_core", "tower_shell"),
            ("core", "handle_post"),
            ("handle_post", "handle_link_0"),
            ("handle_link_0", "handle_link_1"),
            ("handle_link_1", "handle_grip"),
        ):
            self.assertLessEqual(
                gap(s, geom(s, first), geom(s, second)), 1e-7, (first, second)
            )
        for first, second in (
            ("core", "rear_foundation"),
            ("rear_foundation", "rear_tower_pedestal"),
            ("rear_tower_pedestal", "rear_light_enclosure"),
        ):
            self.assertLessEqual(
                gap(s, geom(s, first, STATIONS[1]), geom(s, second, STATIONS[1])), 1e-7
            )

    def test_full_combined_travel_has_direct_parent_filtered_clearance(self):
        s = session()
        for equipment in STATIONS:
            task = WaferInspection(s, equipment)
            # Offline inspection data only; production motion never assigns qpos.
            for q in itertools.product(
                np.linspace(-0.09, 0.09, 9),
                np.linspace(-0.065, 0.065, 5),
                np.linspace(0, 0.11, 5),
            ):
                s.data.qpos[task.motion.qadr] = q
                mujoco.mj_forward(s.model, s.data)
                task._observe(s.model, s.data)
            self.assertEqual(task.checked, 225)
            self.assertGreaterEqual(task.minimum, 0.00059999)
            self.assertGreater(task.queries, 0)
            s.reset()

    def test_old_crossbeam_and_panel_collision_commands_really_move(self):
        s = session()
        for equipment in STATIONS:
            for key, target in (
                ("alignment_head", 0.11),
                ("wafer_y", -0.09),
                ("wafer_x", 0.065),
                ("wafer_x", -0.065),
                ("wafer_y", 0.09),
            ):
                c = s.controls[equipment, key]
                s.data.ctrl[c.actuator] = target
                for _ in range(round(2 / s.model.opt.timestep)):
                    mujoco.mj_step(s.model, s.data)
                self.assertLess(
                    abs(s.data.qpos[s.model.jnt_qposadr[c.joint]] - target), 0.0002
                )
                self.assertEqual(s.data.ncon, 0)
            shell = geom(s, "head_shell", equipment)
            self.assertEqual(s.model.geom_contype[shell], 1)
            self.assertGreater(
                gap(s, shell, geom(s, "top_crossmember", equipment)), 0.05
            )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)

    def test_measured_projected_axis_inspection_returns_without_reset(self):
        s = session()
        result = WaferInspection(s).inspect_and_return()
        self.assertEqual(
            result["status"], "succeeded", [r.get("failure") for r in result["legs"]]
        )
        self.assertEqual(len(result["legs"]), 6)
        self.assertEqual(result["physics_steps"], result["fully_checked_steps"])
        self.assertGreater(result["physics_steps"], 1000)
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 1)
        np.testing.assert_allclose(
            result["derived_target_y_x_head_m"], [-0.015, 0.055, 0.015], atol=1e-9
        )
        observation = result["observation"]
        self.assertTrue(observation["axis_reaches_wafer"])
        self.assertTrue(observation["visible"])
        self.assertLess(observation["projected_axis_error_m"], 0.0002)
        self.assertLess(observation["objective_to_surface_height_m"], 0.04)
        self.assertLess(max(result["return_error_m"]), 0.0002)
        for leg in result["legs"]:
            self.assertTrue(leg["direct_checks"]["coverage_complete"])
            self.assertGreaterEqual(
                leg["direct_checks"]["whole_leg_clearance_lower_bound_m"], 0.00059999
            )

    def test_foreign_and_replaced_data_reject_before_any_mutation(self):
        for after_constructor in (False, True):
            s = session()
            task = WaferInspection(s) if after_constructor else None
            other = copy(s.model)
            other.geom_pos[geom(s, "wafer"), 2] += 0.1
            s.data = mujoco.MjData(other)
            s.data.ctrl[:] = 0.012
            before = s.data.ctrl.copy()
            with self.assertRaises(ValueError):
                task.move((0, 0, 0.01)) if task else WaferInspection(s)
            np.testing.assert_array_equal(s.data.ctrl, before)
            self.assertEqual(s.data.time, 0)

    def test_essential_colliders_and_global_options_cannot_be_disabled(self):
        for key in (
            "head_shell",
            "y_rail_0",
            "wafer",
            "objective_0",
            "head_bearing_0_contact_0",
        ):
            s = session()
            task = WaferInspection(s)
            g = geom(s, key)
            s.model.geom_contype[g] = s.model.geom_conaffinity[g] = 0
            with self.assertRaises(ValueError):
                task.move((0, 0, 0.01))
            self.assertEqual(s.data.time, 0)
        for bit in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            s = session()
            s.model.opt.disableflags |= int(bit)
            with self.assertRaises(ValueError):
                WaferInspection(s)
            self.assertEqual(s.data.time, 0)

    def test_model_or_manifest_rebinding_is_not_a_valid_operation(self):
        for what in ("mesh", "geometry", "manifest"):
            s = session()
            task = WaferInspection(s)
            if what == "mesh":
                s.model.mesh_vert[0, 0] += 0.01
            elif what == "geometry":
                s.model.geom_pos[geom(s, "y_rail_0"), 2] += 0.01
            else:
                s.manifest["camera"]["workstation"]["fovy"] += 1
            with self.assertRaises(ValueError):
                task.move((0, 0, 0.01))
            self.assertEqual(s.data.time, 0)

    def test_nonzero_neighbour_controls_are_held(self):
        s = session()
        for key, target in (("wafer_y", 0.04), ("alignment_head", 0.03)):
            s.data.ctrl[s.controls[STATIONS[1], key].actuator] = target
        for _ in range(1500):
            mujoco.mj_step(s.model, s.data)
        task = WaferInspection(s)
        before = s.data.qpos[3:].copy()
        result = task.move((0.005, 0, 0.005))
        self.assertEqual(result["status"], "succeeded")
        np.testing.assert_allclose(s.data.qpos[3:], before, atol=1e-7)

    def test_loaded_failure_holds_measured_state_and_has_partial_coverage(self):
        s = session()
        task = WaferInspection(s)
        s.data.qfrc_applied[
            s.model.jnt_dofadr[s.controls[STATIONS[0], "wafer_x"].joint]
        ] = 100
        result = task.move((0, 0.01, 0.01))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        direct = result["direct_checks"]
        self.assertGreater(direct["executed_steps"], direct["fully_checked_steps"])
        self.assertIsNone(direct["whole_leg_clearance_lower_bound_m"])
        np.testing.assert_allclose(
            s.data.ctrl[task.motion.actuators],
            np.clip(task.motion.measured(), task.motion.lower, task.motion.upper),
        )
        self.assertGreater(s.data.time, 0)
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)

    def test_read_only_observer_failure_holds_without_task_success(self):
        s = session()
        task = WaferInspection(s)
        entries = 0

        def fail(m, d):
            nonlocal entries
            entries += 1
            if entries == 4:
                raise RuntimeError("external_inspection_stop")

        result = task.move((0, 0, 0.01), step_observer=fail)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["physics_steps"], 4)
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
