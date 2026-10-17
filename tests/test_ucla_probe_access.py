"""Actual hollow port, supported open guides and measured dry insertion."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.ucla_probe_motion import ProbeAccess, ProbeObserver, inspect_ucla_probe


class UclaProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("ucla_bapsf_lapd")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_frames_ranges_drives_and_views_remain(self):
        m = self.model
        np.testing.assert_array_equal(m.jnt_range, [[-0.04, 0.04], [-0.20, 0]])
        np.testing.assert_array_equal(m.jnt_axis, [[0, 0, 1], [1, 0, 0]])
        np.testing.assert_array_equal(
            m.body("lapd_segment__body_probe_vertical_alignment").pos, [2.2, -2.4, 1.25]
        )
        np.testing.assert_array_equal(
            m.body("lapd_segment__body_probe_approach").pos, [-0.3, 0, -0.017]
        )
        np.testing.assert_array_equal(m.actuator_gainprm[:, 0], [380, 340])
        np.testing.assert_array_equal(m.actuator_forcerange, [[-120, 120], [-120, 120]])
        self.assertEqual(
            self.manifest["camera"]["workstation"],
            dict(position=[2.2, -3.8, 1.9], target=[0.3, -2.4, 1.3], fovy=51),
        )
        self.assertEqual(m.njnt, 2)

    def test_both_visible_moving_weld_groups_have_enabled_shapes(self):
        report = screen_contact_coverage(self.model)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["visible_actuated_weld_group_count"], 2)
        for group in report["groups"]:
            self.assertGreater(group["contact_enabled_count"], 0)

    def test_complete_combined_travel_uses_direct_parent_filtered_queries(self):
        s = self.session()
        observer = ProbeObserver(s)
        for q in product(np.linspace(-0.04, 0.04, 9), np.linspace(-0.2, 0, 17)):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            observer.check(s.model, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-7])
        self.assertEqual(observer.checked, 153)
        self.assertGreater(observer.minimum, 0.00059)

    def test_enabled_wall_and_neck_leave_a_real_open_passage(self):
        s = self.session()
        m = copy(s.model)
        d = mujoco.MjData(m)
        # Private ray groups isolate real fixed wall/neck sectors. No live mask
        # or collision configuration is altered by this inspection test.
        m.geom_group[:] = 1
        for g in range(m.ngeom):
            name = m.geom(g).name
            if any(
                name.startswith("lapd_segment__lapd_" + k)
                for k in (
                    "vessel_contact",
                    "access_neck_contact",
                    "access_flange_contact",
                )
            ):
                m.geom_group[g] = 0
        mujoco.mj_forward(m, d)
        for y, z in product((-0.04, 0, 0.04), (-0.025, 0, 0.025)):
            origin = np.array([-0.25, -2.4 + y, 1.298 + z])
            hit = np.array([-1], np.int32)
            distance = mujoco.mj_ray(
                m,
                d,
                origin,
                np.array([-1.0, 0, 0]),
                np.array([1, 0, 0, 0, 0, 0], np.uint8),
                1,
                -1,
                hit,
            )
            self.assertGreater(distance, 1.4)
            self.assertTrue(
                m.geom(int(hit[0])).name.startswith("lapd_segment__lapd_vessel_contact")
            )
        hit = np.array([-1], np.int32)
        blocked = mujoco.mj_ray(
            m,
            d,
            np.array([-0.25, -2.29, 1.298]),
            np.array([-1.0, 0, 0]),
            np.array([1, 0, 0, 0, 0, 0], np.uint8),
            1,
            -1,
            hit,
        )
        self.assertLess(blocked, 0.3)
        self.assertIn("access_flange_contact", m.geom(int(hit[0])).name)
        self.assertEqual(_model_signature(s.model), _model_signature(self.model))

    def test_each_moving_welded_assembly_has_a_continuous_attachment_path(self):
        s = self.session()
        m, d = s.model, s.data
        for body_name in ("probe_vertical_alignment", "probe_approach"):
            body = m.body("lapd_segment__body_" + body_name).id
            geoms = {
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == body
                and (m.geom_contype[g] or m.geom_conaffinity[g])
            }
            connected = {next(iter(geoms))}
            pending = geoms - connected
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
                connected |= attached
                pending -= attached
            self.assertFalse(pending, [m.geom(g).name for g in sorted(pending)])

    def test_main_vessel_coils_and_services_have_continuous_floor_supports(self):
        s = self.session()
        m, d = s.model, s.data
        p = "lapd_segment__lapd_"

        def gap(a, b):
            return mujoco.mj_geomDistance(
                m, d, m.geom(p + a).id, m.geom(p + b).id, 0.01, None
            )

        walls = [
            i for i in range(m.ngeom) if m.geom(i).name.startswith(p + "vessel_contact")
        ]
        for y in (-3.5, -1.75, 0, 1.75, 3.5):
            support = m.geom(p + f"vessel_support_{y}").id
            self.assertLessEqual(
                min(
                    mujoco.mj_geomDistance(m, d, support, g, 0.001, None) for g in walls
                ),
                1e-7,
            )
            for x in (-0.6, 0.6):
                self.assertLessEqual(
                    gap(f"vessel_support_{y}", f"vessel_leg_{x}_{y}"), 1e-7
                )
        for i in range(26):
            stand = m.geom(p + f"coil_pedestal_{i}").id
            sectors = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name.startswith(p + f"coil_proxy_{i}_contact")
            ]
            self.assertLessEqual(
                min(
                    mujoco.mj_geomDistance(m, d, stand, g, 0.001, None) for g in sectors
                ),
                1e-7,
            )
        for i in range(3):
            self.assertLess(gap(f"junction_hanger_{i}", "back_manifold"), 0)
            self.assertLess(gap(f"junction_hanger_{i}", f"junction_box_{i}"), 0)
        for i in range(11):
            for a, b in (
                (f"fixed_stand_post_{i}", f"fixed_stand_foot_{i}"),
                (f"fixed_stand_top_{i}", f"fixed_stand_post_{i}"),
                (f"fixed_drive_rod_{i}", f"fixed_stand_top_{i}"),
                (f"fixed_drive_end_{i}", f"fixed_drive_clamp_{i}"),
            ):
                self.assertLessEqual(gap(a, b), 1e-7, (a, b))

    def test_physical_head_site_and_actual_inside_cavity_observation(self):
        s = self.session()
        s.data.qpos[:] = [0, -0.19]
        mujoco.mj_forward(s.model, s.data)
        report = ProbeAccess(s).observe()
        self.assertTrue(report["head_inside_empty_chamber"])
        self.assertGreater(report["head_radial_margin_inside_chamber_m"], 0.003)
        self.assertGreater(
            report["tip_depth_inside_inner_cylindrical_envelope_m"], 0.025
        )
        self.assertTrue(report["visible"])
        np.testing.assert_allclose(report["axis_alignment_error_m"], [0, 0], atol=1e-12)
        head = s.model.geom("lapd_segment__lapd_inert_probe_head")
        site = s.model.site("lapd_segment__site_probe_target")
        self.assertAlmostEqual(site.pos[0], head.pos[0] + head.size[0])
        self.assertLess(abs(site.pos[1] - head.pos[1]), head.size[1])
        self.assertLess(abs(site.pos[2] - head.pos[2]), head.size[2])

    def test_measured_alignment_insertion_observation_return_without_reset(self):
        s = self.session()
        signature = _model_signature(s.model)
        report = inspect_ucla_probe(s)
        self.assertEqual(report["status"], "succeeded", report)
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 10)
        self.assertFalse(report["initial_observation"]["head_inside_empty_chamber"])
        self.assertAlmostEqual(
            report["checkpoints"]["offset_check"]["axis_alignment_error_m"][1],
            0.025,
            delta=0.0002,
        )
        self.assertTrue(
            report["checkpoints"]["insert_probe"]["head_inside_empty_chamber"]
        )
        self.assertTrue(report["checkpoints"]["insert_probe"]["visible"])
        self.assertFalse(
            report["checkpoints"]["retract_probe"]["head_inside_empty_chamber"]
        )
        for phase in report["phases"]:
            observed = phase["direct_clearance"]
            self.assertTrue(observed["coverage_complete"])
            self.assertEqual(observed["checked_steps"], phase["physics_steps"])
        np.testing.assert_allclose(
            report["measured_return_error_m"], [0, 0], atol=0.0002
        )
        self.assertEqual(_model_signature(s.model), signature)

    def test_altered_essential_geometry_flags_and_metadata_reject_before_mutation(self):
        mutations = [
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom("lapd_segment__lapd_vertical_bridge").id, 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom("lapd_segment__lapd_diagnostic_x_guide_-0.09").id, 1), 0.5
            ),
            lambda s: s.model.opt.gravity.__setitem__(2, 0),
            lambda s: s.equipment["lapd_segment"]["sites"].__setitem__(
                "probe_tip", "wrong"
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
                ProbeAccess(s)
            for a, b in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(a, b)

    def test_foreign_or_replaced_state_and_later_contact_change_reject_unchanged(self):
        for mode in ("foreign", "replacement", "mask"):
            s = self.session()
            task = ProbeAccess(s)
            if mode == "foreign":
                s.data = mujoco.MjData(copy(s.model))
            elif mode == "replacement":
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_conaffinity[
                    s.model.geom("lapd_segment__lapd_probe_shaft").id
                ] = 0
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                task.move([0, -0.05])
            for a, b in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(a, b)

    def test_vertical_load_on_held_axis_fails_with_honest_partial_coverage(self):
        s = self.session()
        s.data.qfrc_applied[0] = 100
        report = ProbeAccess(s).move([0, -0.05])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 0)
        self.assertLess(
            report["direct_clearance"]["checked_steps"], report["physics_steps"]
        )
        self.assertIsNone(
            report["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        np.testing.assert_allclose(
            s.data.ctrl, np.clip(s.data.qpos, [-0.04, -0.2], [0.04, 0])
        )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, [0, 0])

    def test_readonly_callback_failure_does_not_relabel_failure_as_success(self):
        s = self.session()
        states = []

        def abort(m, d):
            states.append((d.qpos.copy(), d.qvel.copy()))
            raise RuntimeError("requested_readonly_abort")

        report = ProbeAccess(s).move([0.02, 0], step_observer=abort)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["physics_steps"], 1)
        self.assertTrue(report["direct_clearance"]["coverage_complete"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        np.testing.assert_array_equal(s.data.qpos, states[-1][0])
        np.testing.assert_array_equal(s.data.qvel, states[-1][1])


if __name__ == "__main__":
    unittest.main()
