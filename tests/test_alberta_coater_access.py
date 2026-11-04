"""Supported open XY guides, a physical door and measured dry registration."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.alberta_coater_motion import (
    CoaterAccess,
    CoaterObserver,
    inspect_alberta_coater,
)
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class AlbertaCoaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("alberta_nanofab_exactacoat")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_xy_frames_ranges_drives_and_original_views(self):
        s = self.session()
        m = s.model
        np.testing.assert_array_equal(m.jnt_range[:2], [[-0.21, 0.21], [-0.13, 0.13]])
        np.testing.assert_array_equal(m.jnt_axis[:2], [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(
            m.body("coater_alignment__body_dry_head_x").pos, [0, 0.215, 1.57]
        )
        np.testing.assert_array_equal(
            m.body("coater_alignment__body_dry_head_y").pos, [0, -0.19, -0.04]
        )
        np.testing.assert_array_equal(m.actuator_gainprm[:2, 0], [450, 420])
        np.testing.assert_array_equal(
            m.actuator_forcerange[:2], [[-120, 120], [-120, 120]]
        )
        self.assertEqual(
            s.manifest["camera"]["workstation"],
            dict(position=[0.05, -1, 1.47], target=[0, 0.58, 1.4], fovy=46),
        )
        self.assertEqual(m.njnt, 3)

    def test_every_visible_actuated_group_has_real_collision_geometry(self):
        report = screen_contact_coverage(self.model)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["visible_actuated_weld_group_count"], 3)
        for row in report["groups"]:
            self.assertGreater(row["contact_enabled_count"], 0)
        self.assertEqual(
            self.model.geom("coater_alignment__coater_door_contact_glass").contype, 1
        )

    def test_full_combined_travel_checks_parent_filtered_pairs(self):
        s = self.session()
        observer = CoaterObserver(s)
        for q in product(
            np.linspace(-0.21, 0.21, 5),
            np.linspace(-0.13, 0.13, 5),
            np.linspace(0, 1.25, 9),
        ):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            observer.check(s.model, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-7])
        self.assertEqual(observer.checked, 225)
        self.assertGreater(observer.minimum, 0.00059)

    def test_guides_and_layered_head_have_continuous_fixed_attachments(self):
        s = self.session()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(
                m,
                d,
                m.geom("coater_alignment__coater_" + a).id,
                m.geom("coater_alignment__coater_" + b).id,
                0.1,
                None,
            )

        for x in (-0.105, 0.105):
            self.assertLess(gap(f"y_rail_{x}", "x_bridge"), 0)
            for y in (-0.035, 0.035):
                self.assertLessEqual(abs(gap(f"y_shoe_lug_{x}_{y}", "y_plate")), 1e-7)
        self.assertLess(gap("head_backplate", "y_plate"), 0)
        self.assertAlmostEqual(gap("upper_head", "head_coupling"), -0.005, places=6)
        self.assertAlmostEqual(gap("lower_head", "lower_stem"), 0, places=7)
        self.assertAlmostEqual(gap("lower_stem", "coater_dry_nozzle"), 0, places=7)
        for sign in (-1, 1):
            self.assertLess(
                gap(f"capped_connector_{sign}", f"connector_stand_{sign}"), 0
            )
        for x in (-0.425, 0.425):
            self.assertLess(gap(f"parked_loop_stay_{x}", "parked_loop_beam"), 0)
            self.assertLess(gap(f"parked_loop_stay_{x}", f"rear_post_{x}"), 0)

    def test_every_static_controller_lead_part_has_a_physical_mount_path(self):
        s = self.session()
        m, d = s.model, s.data
        prefix = "coater_alignment__coater_"
        anchors = {
            m.geom(prefix + key).id for key in ("control_fascia", "hand_controller")
        }
        components = {
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(prefix + "lead_")
            or m.geom(g).name.startswith(prefix + "coiled_lead_")
        }
        connected = set(anchors)
        pending = set(components)
        while pending:
            attached = {
                a
                for a in pending
                if any(
                    mujoco.mj_geomDistance(m, d, a, b, 0.0001, None) <= 1e-7
                    for b in connected
                )
            }
            if not attached:
                break
            connected.update(attached)
            pending.difference_update(attached)
        self.assertFalse(pending, [m.geom(g).name for g in sorted(pending)])
        self.assertEqual(len(components), 95)
        connector = m.geom(prefix + "lead_connector").id
        fascia = m.geom(prefix + "control_fascia").id
        upper = m.geom(prefix + "lead_top_1").id
        self.assertLess(mujoco.mj_geomDistance(m, d, connector, fascia, 0.1, None), 0)
        self.assertLessEqual(
            mujoco.mj_geomDistance(m, d, connector, upper, 0.1, None), 1e-7
        )

    def test_hinge_supports_and_real_annular_clearance_through_travel(self):
        s = self.session()
        m, d = s.model, s.data
        prefix = "coater_alignment__coater_"
        for q in (0, 0.65, 1.25):
            d.qpos[2] = q
            mujoco.mj_forward(m, d)
            for z, local in ((1.105, -0.395), (1.895, 0.395)):
                pin = m.geom(prefix + f"hinge_pin_{z}").id
                for i in range(32):
                    sleeve = m.geom(prefix + f"hinge_sleeve_{local}_contact_{i}").id
                    self.assertGreater(
                        mujoco.mj_geomDistance(m, d, pin, sleeve, 0.01, None), 0.0009
                    )
                moving = m.geom(prefix + f"hinge_moving_leaf_{local}").id
                frame = m.geom(prefix + "door_vertical_0.05").id
                self.assertLess(
                    mujoco.mj_geomDistance(m, d, moving, frame, 0.1, None), 0
                )
                for sign in (-1, 1):
                    leaf = m.geom(prefix + f"hinge_fixed_leaf_{z}_{sign}").id
                    post = m.geom(prefix + "fixed_front_side_-0.49").id
                    self.assertLess(
                        mujoco.mj_geomDistance(m, d, leaf, post, 0.1, None), 0
                    )

    def test_physical_surface_and_declared_dry_standoff(self):
        s = self.session()
        m, d = s.model, s.data
        wafer = m.geom("coater_alignment__coater_supported_300mm_wafer")
        marker = m.geom("coater_alignment__coater_physical_marker")
        site = m.site("coater_alignment__site_coater_wafer")
        self.assertAlmostEqual(2 * wafer.size[0], 0.3)
        self.assertAlmostEqual(site.pos[2], marker.pos[2] + marker.size[2])
        d.qpos[:2] = [0.07, -0.085]
        mujoco.mj_forward(m, d)
        result = CoaterAccess(s).registration()
        np.testing.assert_allclose(result["xy_error_m"], [0, 0], atol=1e-12)
        self.assertAlmostEqual(result["marker_standoff_m"], 0.048)
        self.assertAlmostEqual(
            d.site_xpos[m.site("coater_alignment__site_dry_nozzle_tip").id, 2]
            - d.geom_xpos[wafer.id, 2]
            - wafer.size[1],
            0.05,
        )

    def test_closed_glass_blocks_front_access_and_open_door_exposes_marker(self):
        s = self.session()
        task = CoaterAccess(s)
        before = _model_signature(s.model)
        closed = task.access_observation()
        self.assertEqual(closed["clear_front_access_rays"], 0)
        s.data.qpos[:] = [-0.18, 0.10, 1.15]
        mujoco.mj_forward(s.model, s.data)
        opened = task.access_observation()
        self.assertEqual(opened["clear_front_access_rays"], 25)
        self.assertTrue(opened["visible"])
        self.assertEqual(_model_signature(s.model), before)

    def test_actual_registration_access_return_and_complete_checks(self):
        s = self.session()
        before = _model_signature(s.model)
        report = inspect_alberta_coater(s)
        self.assertEqual(report["status"], "succeeded", report)
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 10)
        self.assertEqual(
            report["open_access_observation"]["clear_front_access_rays"], 25
        )
        self.assertEqual(
            report["closed_access_observation"]["clear_front_access_rays"], 0
        )
        np.testing.assert_allclose(
            report["registration"]["xy_error_m"], [0, 0], atol=0.0002
        )
        np.testing.assert_allclose(
            report["measured_return_error"], [0, 0, 0], atol=0.0002
        )
        for phase in report["phases"]:
            observed = phase["direct_clearance"]
            self.assertTrue(observed["coverage_complete"])
            self.assertEqual(observed["checked_steps"], phase["physics_steps"])
            self.assertGreater(observed["whole_leg_clearance_lower_bound_m"], 0.00025)
        self.assertEqual(_model_signature(s.model), before)

    def test_disabled_essential_geometry_physics_and_metadata_reject_before_mutation(
        self,
    ):
        mutations = [
            lambda s: setattr(
                s.model.geom("coater_alignment__coater_x_bridge"), "contype", 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom("coater_alignment__coater_x_rail_1.51").id, 2), 1.0
            ),
            lambda s: s.model.opt.gravity.__setitem__(2, 0),
            lambda s: s.equipment["coater_alignment"]["sites"].__setitem__(
                "coater_wafer", "wrong"
            ),
        ]
        for flag in (
            "mjDSBL_CONTACT",
            "mjDSBL_CONSTRAINT",
            "mjDSBL_LIMIT",
            "mjDSBL_ACTUATION",
        ):
            mutations.append(
                lambda s, flag=flag: setattr(
                    s.model.opt,
                    "disableflags",
                    int(getattr(mujoco.mjtDisableBit, flag)),
                )
            )
        for change in mutations:
            s = self.session()
            change(s)
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                CoaterAccess(s)
            for a, b in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(a, b)

    def test_foreign_replaced_state_and_later_mask_changes_reject_without_hold_mutation(
        self,
    ):
        for mode in ("foreign", "replace", "mask"):
            s = self.session()
            task = CoaterAccess(s)
            if mode == "foreign":
                s.data = mujoco.MjData(copy(s.model))
            elif mode == "replace":
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_conaffinity[
                    s.model.geom("coater_alignment__coater_door_contact_glass").id
                ] = 0
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                task.move([0.05, 0, 0])
            for a, b in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(a, b)

    def test_external_door_load_fails_without_reset_and_reports_partial_coverage(self):
        s = self.session()
        s.data.qfrc_applied[2] = 100
        result = CoaterAccess(s).move([0.05, 0, 0])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 0)
        self.assertGreater(s.data.qpos[2], 0)
        self.assertFalse(result["direct_clearance"]["coverage_complete"])
        self.assertIsNone(
            result["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        np.testing.assert_allclose(
            s.data.ctrl, np.clip(s.data.qpos, [-0.21, -0.13, 0], [0.21, 0.13, 1.25])
        )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, [0, 0, 0])

    def test_observer_failure_preserves_measured_state_and_complete_checked_subset(
        self,
    ):
        s = self.session()
        count = 0

        def stop(m, d):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("external_readonly_failure")

        result = CoaterAccess(s).move([0.02, 0, 0], step_observer=stop)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["physics_steps"], 3)
        self.assertEqual(result["direct_clearance"]["checked_steps"], 3)
        self.assertTrue(result["direct_clearance"]["coverage_complete"])
        self.assertTrue(result["hold_requested"])


if __name__ == "__main__":
    unittest.main()
