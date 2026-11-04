"""Open gimbal supports, actual dry alignment and source-bound failure checks."""

from copy import copy
import itertools
import json
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.heidelberg_alignment_geometry import REGISTERED_ANGLES
from real_labs.heidelberg_alignment_motion import (
    AlignmentInspection,
    inspect_heidelberg_alignment,
    SCENE,
)
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def full_state(s):
    d = s.data
    return (
        float(d.time),
        *(
            getattr(d, k).tobytes()
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "eq_active",
            )
        ),
    )


class HeidelbergAlignmentTests(unittest.TestCase):
    def test_original_joint_and_true_target_contract(self):
        s = session()
        m = s.model
        for key, axis, limits, kp, mass, pos in (
            ("alignment_tip", [1, 0, 0], [-0.16, 0.16], 100, 0.06, [0.37, -0.57, 1.13]),
            ("alignment_tilt", [0, 0, 1], [-0.20, 0.20], 85, 0.035, [0, 0, 0]),
        ):
            c = s.controls["hqa_alignment", key]
            body = int(m.jnt_bodyid[c.joint])
            np.testing.assert_allclose(m.jnt_axis[c.joint], axis)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
            np.testing.assert_allclose(m.body_pos[body], pos)
            self.assertAlmostEqual(m.body_mass[body], mass)
            self.assertAlmostEqual(m.actuator_gainprm[c.actuator, 0], kp)
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
            self.assertAlmostEqual(m.dof_damping[m.jnt_dofadr[c.joint]], 0.8)
        disc = m.geom("hqa_alignment__hqa_inert_alignment_disc").id
        np.testing.assert_allclose(m.geom_size[disc, :2], [0.046, 0.008])
        site = m.site("hqa_alignment__site_hqa_target").id
        np.testing.assert_allclose(m.site_pos[site], [0, -0.011, 0])
        np.testing.assert_allclose(s.data.site_xpos[site], [0.37, -0.281, 1.13])
        button = m.geom("hqa_alignment__hqa_witness_button").id
        normal = s.data.geom_xmat[button].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(
            s.data.geom_xpos[button] + normal * 0.0015, s.data.site_xpos[site]
        )

    def test_both_actuated_groups_have_contact_and_no_child_coverage_shortcut(self):
        from real_labs.contact_coverage import screen_contact_coverage

        s = session()
        report = screen_contact_coverage(s.model)
        self.assertFalse(report["issues"])
        for key in ("alignment_tip", "alignment_tilt"):
            c = s.controls["hqa_alignment", key]
            weld = int(s.model.body_weldid[s.model.jnt_bodyid[c.joint]])
            gids = [
                g
                for g in range(s.model.ngeom)
                if int(s.model.body_weldid[s.model.geom_bodyid[g]]) == weld
            ]
            self.assertGreater(
                sum(
                    bool(s.model.geom_contype[g] or s.model.geom_conaffinity[g])
                    for g in gids
                ),
                3,
            )

    def test_full_coupled_travel_including_parent_filtered_pairs(self):
        s = session()
        task = AlignmentInspection(s)
        m, d = s.model, s.data
        pairs = {
            frozenset((m.geom(int(a)).name, m.geom(int(b)).name)) for a, b in task.pairs
        }
        self.assertIn(
            frozenset(
                (
                    "hqa_alignment__hqa_x_bearing_1_proxy_0",
                    "hqa_alignment__hqa_x_spindle_1",
                )
            ),
            pairs,
        )
        self.assertIn(
            frozenset(
                (
                    "hqa_alignment__hqa_z_bearing_1_proxy_0",
                    "hqa_alignment__hqa_z_spindle_1",
                )
            ),
            pairs,
        )
        for a, b in itertools.product(
            np.linspace(-0.16, 0.16, 13), np.linspace(-0.20, 0.20, 17)
        ):
            d.qpos[:] = [a, b]
            mujoco.mj_forward(m, d)
            task.check(m, d)
            self.assertFalse(any(c.dist < -1e-7 for c in d.contact), (a, b))
        self.assertGreaterEqual(task.minimum, 0.0005999)

    def test_static_mounts_and_retained_target_are_connected(self):
        s = session()
        m, d = s.model, s.data
        pairs = (
            ("mount_foot", "mount_post"),
            ("mount_post", "fork_bridge"),
            ("fork_bridge", "fork_cheek_1"),
            ("fork_bridge", "fork_cheek_-1"),
            ("inert_alignment_disc", "z_spindle_1"),
            ("inert_alignment_disc", "z_spindle_-1"),
            ("inert_alignment_disc", "witness_button"),
            ("reference_foot", "reference_post"),
        )
        for a, b in pairs:
            gap = mujoco.mj_geomDistance(
                m,
                d,
                m.geom("hqa_alignment__hqa_" + a).id,
                m.geom("hqa_alignment__hqa_" + b).id,
                0.2,
                None,
            )
            self.assertLessEqual(gap, 2e-8, (a, b, gap))
        tabletop = next(
            g for g in range(m.ngeom) if "optical_tabletop" in m.geom(g).name
        )
        for name in ("mount_foot", "reference_foot"):
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(
                    m, d, m.geom("hqa_alignment__hqa_" + name).id, tabletop, 0.2, None
                ),
                0,
                delta=1e-8,
            )
        reference = m.geom("hqa_alignment__hqa_reference_post").id
        ring = [
            g
            for g in range(m.ngeom)
            if "hqa_reference_aperture_proxy_" in m.geom(g).name
        ]
        self.assertLessEqual(
            min(mujoco.mj_geomDistance(m, d, reference, g, 0.02, None) for g in ring),
            1e-8,
        )
        for sign in (-1, 1):
            cheek = m.geom(f"hqa_alignment__hqa_fork_cheek_{sign}").id
            bearing = [
                g
                for g in range(m.ngeom)
                if f"hqa_x_bearing_{sign}_proxy_" in m.geom(g).name
            ]
            self.assertLessEqual(
                min(
                    mujoco.mj_geomDistance(m, d, cheek, g, 0.03, None) for g in bearing
                ),
                1e-8,
            )

    def test_overhead_grid_rows_have_continuous_support(self):
        s = session()
        m, d = s.model, s.data
        low = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("hqa_alignment__")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and abs(d.geom_xpos[g, 2] - 2.272) < 1e-8
        ]
        high = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("hqa_alignment__")
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            and abs(d.geom_xpos[g, 2] - 2.282) < 1e-8
        ]
        self.assertEqual(len(low), 23)
        self.assertEqual(len(high), 14)
        for row in high:
            self.assertAlmostEqual(
                min(
                    mujoco.mj_geomDistance(m, d, row, support, 0.01, None)
                    for support in low
                ),
                0,
                delta=1e-8,
            )

    def test_registration_measures_real_normal_and_true_open_reference(self):
        s = session()
        task = AlignmentInspection(s)
        initial = task.observe()
        self.assertGreater(initial["surface_normal_error_rad"], 0.1)
        self.assertFalse(initial["axis_segment_clear_through_reference"])
        s.data.qpos[:] = REGISTERED_ANGLES
        mujoco.mj_forward(s.model, s.data)
        aligned = task.observe()
        self.assertLess(aligned["surface_normal_error_rad"], 1e-6)
        self.assertLess(aligned["reference_center_miss_m"], 1e-7)
        self.assertAlmostEqual(aligned["axis_intersection_distance_m"], 0.169, places=8)
        self.assertTrue(aligned["axis_segment_clear_through_reference"])
        self.assertTrue(aligned["visible"])
        self.assertGreater(aligned["estimated_target_diameter_px"], 15)
        # This checks physical geometry, not a claim about final pixel contrast.

    def test_actual_route_has_complete_checks_and_returns_without_reset(self):
        s = session()
        start = s.data.qpos.copy()
        report = inspect_heidelberg_alignment(s)
        self.assertEqual(report["status"], "succeeded", report["phases"][-1])
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["coverage_complete"])
        self.assertGreater(report["executed_steps"], 0)
        self.assertEqual(report["checked_steps"], report["executed_steps"])
        self.assertGreaterEqual(report["whole_run_clearance_lower_bound_m"], 0.000599)
        registration = next(
            x for x in report["observations"] if x["phase"] == "register_surface_normal"
        )
        self.assertLess(registration["surface_normal_error_rad"], 0.001)
        self.assertLess(registration["reference_center_miss_m"], 0.0002)
        self.assertTrue(registration["axis_segment_clear_through_reference"])
        self.assertTrue(all(o["visible"] for o in report["observations"]))
        np.testing.assert_allclose(s.data.qpos, start, atol=1e-4)
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 1)

    def test_disabled_or_changed_model_rejects_before_state_mutation(self):
        mutations = (
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom("hqa_alignment__hqa_x_spindle_1").id, 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom("hqa_alignment__hqa_reference_post").id, 0), 1.0
            ),
            lambda s: s.model.opt.__setattr__(
                "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda s: s.model.opt.__setattr__(
                "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            ),
            lambda s: s.model.opt.__setattr__(
                "disableflags", int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            ),
            lambda s: s.model.opt.__setattr__(
                "disableflags", int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
            ),
            lambda s: s.equipment["hqa_alignment"]["alignment_interface"].__setitem__(
                "reference_aperture_radius_m", 0.1
            ),
        )
        for mutate in mutations:
            s = session()
            mutate(s)
            before = full_state(s)
            with self.assertRaises(ValueError):
                AlignmentInspection(s)
            self.assertEqual(full_state(s), before)

    def test_foreign_replaced_or_postconstruction_modified_state_rejects(self):
        s = session()
        s.data = mujoco.MjData(copy(s.model))
        before = full_state(s)
        with self.assertRaises(ValueError):
            AlignmentInspection(s)
        self.assertEqual(full_state(s), before)
        for mutation in ("replace", "geometry"):
            s = session()
            task = AlignmentInspection(s)
            if mutation == "replace":
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_size[
                    s.model.geom("hqa_alignment__hqa_witness_button").id, 0
                ] *= 2
            before = full_state(s)
            with self.assertRaises(ValueError):
                task.move([0.01, 0])
            self.assertEqual(full_state(s), before)

    def test_force_failure_reports_partial_coverage_and_holds_actual_state(self):
        s = session()
        task = AlignmentInspection(s)
        control = s.controls["hqa_alignment", "alignment_tip"]
        q = int(s.model.jnt_qposadr[control.joint])
        v = int(s.model.jnt_dofadr[control.joint])
        s.data.qfrc_applied[v] = 100
        initial = s.data.qpos.copy()
        report = task.move([0, 0.05])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(
            report["physics_steps"], report["direct_clearance"]["checked_steps"]
        )
        self.assertIsNone(
            report["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
        )
        self.assertGreater(s.data.time, 0)
        self.assertFalse(np.array_equal(initial, s.data.qpos))
        self.assertAlmostEqual(
            s.data.ctrl[control.actuator],
            np.clip(s.data.qpos[q], control.lower, control.upper),
        )

    def test_existing_command_contracts_reach_full_travel_targets(self):
        s = session()
        for key, c in s.controls.items():
            s.reset()
            for fraction in (0.15, 0.85, 0.25):
                result = s.command(
                    *key, c.lower + fraction * (c.upper - c.lower), duration=2
                )
                self.assertLessEqual(abs(result["error"]), 0.025 * (c.upper - c.lower))
