"""Physical assembly, loaded free-coupon motion and truthful failure coverage."""

from copy import copy
from pathlib import Path
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.export import _portable_resources
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.uva_motion import ChamberAccess


PREFIX = "deposition_chamber__uva_"


def session():
    root, manifest = build_scene("uva_deposition")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def geom(s, key):
    return s.model.geom(PREFIX + key).id


def gap(s, first, second):
    return float(mujoco.mj_geomDistance(s.model, s.data, first, second, 2, None))


class UvaAssemblyTests(unittest.TestCase):
    def test_grounded_vessel_controller_pump_hinge_and_branch(self):
        s = session()
        pairs = [
            ("foundation", "vessel_saddle_-1"),
            ("vessel_saddle_-1", "floor"),
            ("foundation", "vessel_saddle_1"),
            ("vessel_saddle_1", "floor"),
            ("foundation", "pump_foot"),
            ("pump_foot", "pump_body"),
            ("foundation", "controller_pedestal"),
            ("controller_pedestal", "controller"),
            ("lateral_support_foot", "lateral_support"),
            ("lateral_support", "lateral_manifold"),
            ("foundation", "valve_stand"),
            ("valve_stand", "valve_body"),
        ]
        for a, b in pairs:
            self.assertAlmostEqual(
                gap(s, geom(s, a), geom(s, b)), 0, places=7, msg=(a, b)
            )
        for side in (-1, 1):
            self.assertAlmostEqual(
                gap(s, geom(s, "foundation"), geom(s, f"hinge_post_{side}")),
                0,
                places=7,
            )
            self.assertLessEqual(
                gap(
                    s,
                    geom(s, f"hinge_post_{side}"),
                    geom(s, f"hinge_fixed_knuckle_{side}"),
                ),
                0,
            )
            self.assertLessEqual(
                gap(s, geom(s, f"hinge_fixed_knuckle_{side}"), geom(s, "hinge_pin")), 0
            )
            self.assertAlmostEqual(
                gap(s, geom(s, "lid_disc"), geom(s, f"handle_standoff_{side}")),
                0,
                places=7,
            )
            self.assertLessEqual(
                gap(s, geom(s, "handle"), geom(s, f"handle_standoff_{side}")), 0
            )
        self.assertGreater(
            gap(s, geom(s, "controller"), geom(s, "front_cover_flange")), 0
        )

    def test_actual_hollow_bearings_and_original_frames(self):
        s = session()
        m = s.model
        for key, pos, axis, limits in (
            ("chamber_lid", [0, 0.285, 0.617], [1, 0, 0], [-1.7, 0]),
            ("platen", [0, 0, 0.341], [0, 0, 1], [-np.pi, np.pi]),
            ("vacuum_valve", [0.29, -0.08, 0.485], [0, 0, 1], [0, np.pi / 2]),
        ):
            c = s.controls["deposition_chamber", key]
            b = m.jnt_bodyid[c.joint]
            np.testing.assert_allclose(m.body_pos[b], pos, atol=1e-10)
            np.testing.assert_allclose(m.jnt_axis[c.joint], axis, atol=1e-10)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits, atol=5e-8)
        for q in np.linspace(-np.pi, np.pi, 25):
            c = s.controls["deposition_chamber", "platen"]
            s.data.qpos[m.jnt_qposadr[c.joint]] = q
            mujoco.mj_forward(m, s.data)
            for index in range(48):
                self.assertAlmostEqual(
                    gap(
                        s,
                        geom(s, "platen_spindle"),
                        geom(s, f"platen_bearing_proxy_{index}"),
                    ),
                    0.0006,
                    places=7,
                )
        self.assertEqual(m.body_jntnum[m.body("deposition_chamber__specimen").id], 1)
        self.assertEqual(
            m.jnt_type[m.body_jntadr[m.body("deposition_chamber__specimen").id]],
            mujoco.mjtJoint.mjJNT_FREE,
        )
        self.assertEqual(m.neq, 0)

    def test_direct_moving_fixed_pairs_clear_through_combined_travel(self):
        s = session()
        a = ChamberAccess(s)
        m = s.model
        # Static pose copies cannot represent a rotating free coupon. Exclude
        # only that carrier in this geometric sweep; live retention is tested below.
        pairs = [
            (int(x), int(y)) for x, y in a.pairs if x != a.specimen and y != a.specimen
        ]
        pairs = np.array(pairs)
        minimum = 0.002
        for q in itertools.product(
            np.linspace(-1.7, 0, 9),
            np.linspace(-np.pi, np.pi, 5),
            np.linspace(0, np.pi / 2, 5),
        ):
            s.data.qpos[a.qadr] = q
            mujoco.mj_forward(m, s.data)
            rot = s.data.geom_xmat.reshape(-1, 3, 3)
            center = s.data.geom_xpos + np.einsum("nij,nj->ni", rot, m.geom_aabb[:, :3])
            half = np.einsum("nij,nj->ni", np.abs(rot), m.geom_aabb[:, 3:])
            x, y = pairs.T
            sep = np.maximum(np.abs(center[x] - center[y]) - half[x] - half[y], 0)
            for first, second in pairs[np.linalg.norm(sep, axis=1) < 0.002]:
                value = mujoco.mj_geomDistance(
                    m, s.data, int(first), int(second), 0.002, None
                )
                minimum = min(minimum, value)
                self.assertGreaterEqual(
                    value, 0.0005, msg=(m.geom(first).name, m.geom(second).name, q)
                )
        self.assertAlmostEqual(minimum, 0.0006, places=7)

    def test_top_opening_is_real_and_closed_lid_blocks_it(self):
        s = session()
        m = s.model
        hit = np.array([-1], dtype=np.int32)
        origin = np.array([-1.6, 0.15, 1.8])
        direction = np.array([0.0, 0.0, -1.0])
        mujoco.mj_ray(m, s.data, origin, direction, None, 1, -1, hit)
        self.assertEqual(hit[0], geom(s, "lid_disc"))
        c = s.controls["deposition_chamber", "chamber_lid"]
        s.data.qpos[m.jnt_qposadr[c.joint]] = -1.5
        mujoco.mj_forward(m, s.data)
        mujoco.mj_ray(m, s.data, origin, direction, None, 1, -1, hit)
        self.assertEqual(m.geom(hit[0]).name, "deposition_chamber__specimen__collision")

    def test_original_axes_track_actual_commands(self):
        s = session()
        for key in ("chamber_lid", "platen", "vacuum_valve"):
            s.reset()
            c = s.controls["deposition_chamber", key]
            for fraction in (0.15, 0.85, 0.25):
                result = s.command(
                    "deposition_chamber", key, c.lower + (c.upper - c.lower) * fraction
                )
                self.assertLess(abs(result["error"]), 0.025 * (c.upper - c.lower))
                self.assertTrue(np.isfinite(s.data.qpos).all())


class UvaAccessTests(unittest.TestCase):
    def test_portable_asset_paths_keep_the_authored_binding(self):
        root, manifest = build_scene("uva_deposition")
        resources = _portable_resources(root)
        assets = {row["file"]: Path(row["source"]).read_bytes() for row in resources}
        model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"), assets
        )
        s = InstrumentSession(model, manifest)
        self.assertTrue(ChamberAccess(s).binding["scene_sha256"])
        self.assertEqual(s.data.time, 0)

    def test_free_coupon_loss_is_a_failure_not_a_hidden_attachment(self):
        s = session()
        a = ChamberAccess(s)
        s.data.xfrc_applied[a.sample, 2] = s.model.body_mass[a.sample] * 9.81 * 3
        report = a.open_inspect_return()
        self.assertEqual(report["status"], "failed")
        self.assertIn("free_carrier_left_support", report["failure"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(report["physics_steps"], report["fully_checked_steps"])
        self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
        self.assertEqual(s.model.neq, 0)

    def test_disabled_geometry_global_flags_and_wrong_digest_rejected(self):
        for key in ("lid_disc", "platen_spindle", "hinge_pin", "valve_handle"):
            s = session()
            i = geom(s, key)
            s.model.geom_contype[i] = s.model.geom_conaffinity[i] = 0
            with self.assertRaises(ValueError):
                ChamberAccess(s)
        for key in (
            "deposition_chamber__specimen__collision",
            "deposition_chamber__carrier_bearing_pad",
        ):
            s = session()
            i = s.model.geom(key).id
            s.model.geom_contype[i] = s.model.geom_conaffinity[i] = 0
            with self.assertRaises(ValueError):
                ChamberAccess(s)
        for bit in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            s = session()
            s.model.opt.disableflags |= int(bit)
            with self.assertRaises(ValueError):
                ChamberAccess(s)
        s = session()
        s.manifest["scene_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            ChamberAccess(s)

    def test_foreign_data_and_replaced_state_rejected_before_command(self):
        for after in (False, True):
            s = session()
            a = ChamberAccess(s) if after else None
            foreign = copy(s.model)
            foreign.geom_pos[geom(s, "lid_disc"), 2] += 0.5
            s.data = mujoco.MjData(foreign)
            before = s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time
            with self.assertRaises(ValueError):
                a.open_inspect_return() if after else ChamberAccess(s)
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, before[2])

    def test_post_constructor_collider_change_rejected_without_hold(self):
        s = session()
        a = ChamberAccess(s)
        i = geom(s, "hinge_pin")
        s.model.geom_contype[i] = s.model.geom_conaffinity[i] = 0
        before = s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time
        with self.assertRaises(ValueError):
            a.open_inspect_return()
        np.testing.assert_array_equal(s.data.qpos, before[0])
        np.testing.assert_array_equal(s.data.ctrl, before[1])
        self.assertEqual(s.data.time, before[2])

    def test_unexpected_obstacle_is_rejected_as_changed_source_geometry(self):
        root, manifest = build_scene("uva_deposition")
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="unexpected_obstacle",
            type="box",
            pos="-1.6 .15 1.4",
            size=".05 .05 .02",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            manifest,
        )
        with self.assertRaises(ValueError):
            ChamberAccess(s)
        self.assertEqual(s.data.time, 0)

    def test_applied_load_failure_has_partial_coverage_and_no_reset(self):
        s = session()
        a = ChamberAccess(s)
        s.data.qfrc_applied[a.dadr[0]] = 100
        signature = _model_signature(s.model)
        report = a.open_inspect_return()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(report["physics_steps"], report["fully_checked_steps"])
        self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
        self.assertEqual(signature, _model_signature(s.model))
        self.assertGreater(s.data.time, 0)
        np.testing.assert_allclose(
            s.data.ctrl[a.actuators], np.clip(s.data.qpos[a.qadr], a.lower, a.upper)
        )

    def test_nonselected_loaded_control_is_not_silently_driven(self):
        s = session()
        a = ChamberAccess(s)
        c = s.controls["spin_coater", "lid_joint"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 50
        report = a.open_inspect_return()
        self.assertEqual(report["status"], "failed")
        self.assertIn("held_control", report["failure"])
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(s.data.ctrl[c.actuator], 0)

    def test_measured_loaded_round_trip_and_visible_geometric_access(self):
        s = session()
        a = ChamberAccess(s)
        signature = _model_signature(s.model)
        report = a.open_inspect_return()
        self.assertEqual(report["status"], "succeeded", report.get("failure"))
        self.assertEqual(report["physics_steps"], report["fully_checked_steps"])
        self.assertGreater(report["physics_steps"], 10000)
        self.assertFalse(report["reset_performed"])
        self.assertFalse(report["hold_requested"])
        self.assertTrue(report["observation"]["visible"])
        self.assertTrue(report["observation"]["upward_300mm_rays_clear"])
        self.assertLess(max(report["return_error_rad"]), 0.003)
        self.assertLess(report["maximum_support_penetration_m"], 0.0001)
        self.assertGreaterEqual(report["minimum_motion_support_contacts"], 1)
        self.assertGreater(report["observation"]["measured_platen_rad"], 0.497)
        rotation = np.array(report["observation"]["measured_coupon_rotation"]).reshape(
            3, 3
        )
        self.assertAlmostEqual(
            np.arctan2(rotation[1, 0], rotation[0, 0]), 0.5, delta=0.01
        )
        self.assertEqual(signature, _model_signature(s.model))
        self.assertEqual(s.model.neq, 0)


if __name__ == "__main__":
    unittest.main()
