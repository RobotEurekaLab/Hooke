"""Supported open guides and measured free-carrier inspection, CPU only."""

from copy import copy, deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.eth_air_motion import FilterAccess, inspect_eth_filter
from real_labs.export import export_scene
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class EthFilterAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("eth_air_quality")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_retained_frames_limits_free_carrier_and_force_cap(self):
        s = self.session()
        for key, position, axis, limits in (
            ("filter_drawer", [-0.14, -0.025, 0.156], [0, -1, 0], [0, 0.14]),
            ("flow_valve", [0.005, -0.083, 0.075], [0, 0, 1], [0, np.pi]),
        ):
            c = s.controls["aerosol_sampler", key]
            np.testing.assert_allclose(
                s.model.body_pos[s.model.jnt_bodyid[c.joint]], position
            )
            np.testing.assert_allclose(s.model.jnt_axis[c.joint], axis)
            np.testing.assert_allclose(s.model.jnt_range[c.joint], limits)
        carrier = s.model.body("aerosol_sampler__specimen").id
        self.assertEqual(
            s.model.jnt_type[s.model.body_jntadr[carrier]], mujoco.mjtJoint.mjJNT_FREE
        )
        np.testing.assert_allclose(
            s.model.geom("aerosol_sampler__specimen__collision").size,
            [0.0185, 0.0185, 0.001],
        )
        c = s.controls["aerosol_sampler", "filter_drawer"]
        np.testing.assert_allclose(
            s.model.actuator_forcerange[c.actuator], [-0.25, 0.25]
        )
        self.assertFalse(
            any(
                s.model.eq_type[i]
                in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT)
                and carrier in (s.model.eq_obj1id[i], s.model.eq_obj2id[i])
                for i in range(s.model.neq)
            )
        )

    def test_full_combined_travel_queries_parent_filtered_pairs(self):
        s = self.session()
        task = FilterAccess(s)
        m, d = s.model, s.data
        pairs = [
            (int(a), int(b))
            for a, b in task.pairs
            if m.geom(a).name.startswith("aerosol_sampler__")
            and m.geom(b).name.startswith("aerosol_sampler__")
            and task.specimen not in (a, b)
        ]
        valve = s.controls["aerosol_sampler", "flow_valve"]
        closest = 0.002
        for q, angle in product(np.linspace(0, 0.14, 15), np.linspace(0, np.pi, 9)):
            d.qpos[task.qadr] = q
            d.qpos[m.jnt_qposadr[valve.joint]] = angle
            mujoco.mj_forward(m, d)
            for a, b in pairs:
                gap = min(float(mujoco.mj_geomDistance(m, d, a, b, 0.002, None)), 0.002)
                closest = min(closest, gap)
                self.assertGreater(
                    gap, 0.0001, (q, angle, m.geom(a).name, m.geom(b).name)
                )
        self.assertGreater(len(pairs), 1000)
        self.assertGreater(closest, 0.0004)

    def test_rails_and_stems_have_continuous_supported_paths(self):
        s = self.session()
        m, d = s.model, s.data
        for sign in (-1, 1):
            rail = m.geom(f"aerosol_sampler__rail_{sign}").id
            front = m.geom(f"aerosol_sampler__rail_post_{sign}_front").id
            rear = m.geom(f"aerosol_sampler__rail_post_{sign}_rear").id
            foot = m.geom(f"aerosol_sampler__rail_foot_{sign}").id
            for a, b in (
                (rail, front),
                (rail, rear),
                (front, foot),
                (rear, m.geom("aerosol_sampler__monitor_lower").id),
            ):
                self.assertLessEqual(
                    mujoco.mj_geomDistance(m, d, a, b, 0.1, None), 1e-8
                )
        for a, b in (
            ("stack_body", "stack_neck"),
            ("stack_neck", "stack_cap"),
            ("stack_cap", "stack_gasket"),
            ("stack_gasket", "stack_top"),
            ("valve_stem", "valve_knob"),
        ):
            self.assertLessEqual(
                mujoco.mj_geomDistance(
                    m,
                    d,
                    m.geom("aerosol_sampler__" + a).id,
                    m.geom("aerosol_sampler__" + b).id,
                    0.1,
                    None,
                ),
                1e-8,
            )
        task = FilterAccess(s)
        for q in (0, 0.07, 0.14):
            d.qpos[task.qadr] = q
            mujoco.mj_forward(m, d)
            for sign in (-1, 1):
                rail = m.geom(f"aerosol_sampler__rail_{sign}").id
                # The full moving bearing length stays between support posts;
                # actual hollow top/bottom walls remain 0.5 mm from the rail.
                for wall in ("top", "bottom"):
                    shoe = m.geom(f"aerosol_sampler__bearing_{wall}_{sign}").id
                    self.assertAlmostEqual(
                        mujoco.mj_geomDistance(m, d, rail, shoe, 0.01, None),
                        0.0005,
                        places=7,
                    )
                    self.assertGreater(
                        d.geom_xpos[shoe, 1] - m.geom_size[shoe, 1],
                        d.geom_xpos[rail, 1] - m.geom_size[rail, 1],
                    )
                    self.assertLess(
                        d.geom_xpos[shoe, 1] + m.geom_size[shoe, 1],
                        d.geom_xpos[rail, 1] + m.geom_size[rail, 1],
                    )

    def test_loaded_round_trip_checks_actual_access_and_returns_without_reset(self):
        s = self.session()
        initial = _model_signature(s.model)
        report = inspect_eth_filter(s)
        self.assertTrue(report["success"], report)
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["physics_steps"], report["checked_steps"])
        self.assertGreater(report["physics_steps"], 4000)
        self.assertGreater(report["minimum_noncontact_clearance_lower_bound_m"], 0.0004)
        self.assertLess(report["maximum_carrier_tracking_error_m"], 0.0002)
        self.assertLess(report["maximum_carrier_penetration_m"], 0.00015)
        self.assertLess(abs(report["measured_return_error_m"]), 0.0002)
        self.assertFalse(report["reset_performed"])
        obs = report["observation"]
        self.assertTrue(obs["visible"])
        self.assertTrue(obs["upward_100mm_rays_clear"])
        self.assertEqual(len(obs["upward_rays"]), 25)
        self.assertGreater(obs["time_s"], 1)
        self.assertGreater(s.data.time, obs["time_s"])
        for phase in report["phases"]:
            self.assertEqual(phase["executed_steps"], phase["checked_steps"])
        self.assertEqual(_model_signature(s.model), initial)

    def test_exposed_carrier_can_lift_out_by_force_without_a_hidden_attachment(self):
        s = self.session()
        task = FilterAccess(s)
        task.phase("expose_for_removal_probe", 0.14, 3.5, 1.0)
        before = float(s.data.xpos[task.carrier, 2])
        s.data.xfrc_applied[task.carrier, 2] = (
            3 * s.model.body_mass[task.carrier] * 9.81
        )
        mujoco.mj_step(s.model, s.data, nstep=round(0.2 / s.model.opt.timestep))
        mujoco.mj_forward(s.model, s.data)
        self.assertGreater(s.data.xpos[task.carrier, 2] - before, 0.05)
        self.assertAlmostEqual(s.data.qpos[task.qadr], 0.14, delta=0.0002)

    def test_portable_package_has_same_bound_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            export_scene("eth_air_quality", directory, validate=False)
            p = Path(directory)
            s = InstrumentSession.from_files(p / "scene.xml", p / "manifest.json")
            task = FilterAccess(s)
            task.observe()
            self.assertEqual(task.report["checked_steps"], 1)

    def test_all_essential_colliders_and_physics_mutations_reject_before_state_change(
        self,
    ):
        names = [
            "rail_-1",
            "bearing_top_1",
            "filter_plate",
            "filter_puck",
            "valve_bearing_0",
            "valve_stem",
            "valve_knob",
            "stand_post",
            "carrier_bearing_pad",
            "carrier_guide_0_-1",
            "specimen__collision",
        ]
        for name in names:
            for mask in ("geom_contype", "geom_conaffinity"):
                s = self.session()
                getattr(s.model, mask)[s.model.geom("aerosol_sampler__" + name).id] = 0
                self.assert_rejected_unchanged(s)
        for flag in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            s = self.session()
            s.model.opt.disableflags |= int(flag)
            self.assert_rejected_unchanged(s)
        s = self.session()
        s.model.geom_size[s.model.geom("aerosol_sampler__valve_stem").id, 0] *= 2
        self.assert_rejected_unchanged(s)
        s = self.session()
        s.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
        self.assert_rejected_unchanged(s)

    def assert_rejected_unchanged(self, s, operation=None):
        before = (
            s.data.qpos.copy(),
            s.data.qvel.copy(),
            s.data.ctrl.copy(),
            s.data.time,
        )
        with self.assertRaises(ValueError):
            (operation or (lambda: FilterAccess(s)))()
        for actual, expected in zip(
            (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time), before
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_foreign_state_metadata_and_postconstruction_changes_reject(self):
        s = self.session()
        s.data = mujoco.MjData(copy(s.model))
        self.assert_rejected_unchanged(s)
        s = self.session()
        task = FilterAccess(s)
        s.data = mujoco.MjData(s.model)
        self.assert_rejected_unchanged(s, task.run)
        s = self.session()
        task = FilterAccess(s)
        s.model.geom_contype[s.model.geom("aerosol_sampler__rail_1").id] = 0
        self.assert_rejected_unchanged(s, task.run)
        s = self.session()
        s.manifest["camera"]["filter_inspection"]["fovy"] = 70
        self.assert_rejected_unchanged(s)
        s = self.session()
        carrier = s.model.body("aerosol_sampler__specimen").id
        s.data.qpos[s.model.jnt_qposadr[s.model.body_jntadr[carrier]]] += 0.01
        mujoco.mj_forward(s.model, s.data)
        self.assert_rejected_unchanged(s)

    def test_nonfinite_initial_state_rejects_without_a_failure_hold(self):
        for field in ("qpos", "qvel", "ctrl", "qfrc_applied"):
            s = self.session()
            getattr(s.data, field)[0] = float("nan")
            self.assert_rejected_unchanged(s)

    def test_external_load_failure_keeps_state_holds_and_reports_incomplete_coverage(
        self,
    ):
        s = self.session()
        task = FilterAccess(s)
        s.data.qfrc_applied[task.dadr] = 2
        report = task.run()
        self.assertFalse(report["success"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(report["physics_steps"], 0)
        self.assertLess(report["checked_steps"], report["physics_steps"])
        self.assertFalse(report["coverage_complete"])
        self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
        self.assertGreater(s.data.time, 0)
        self.assertAlmostEqual(
            s.data.ctrl[task.actuator], np.clip(s.data.qpos[task.qadr], 0, 0.14)
        )
        s.reset()
        self.assertEqual(s.data.time, 0)
        self.assertEqual(s.data.qpos[task.qadr], 0)

    def test_held_axes_keep_nonzero_setpoints_and_force_failure_is_not_hidden(self):
        s = self.session()
        valve = s.controls["aerosol_sampler", "flow_valve"]
        s.data.qpos[s.model.jnt_qposadr[valve.joint]] = 0.4
        s.data.ctrl[valve.actuator] = 0.4
        mujoco.mj_forward(s.model, s.data)
        task = FilterAccess(s)
        task.phase("small_loaded_move", 0.02, 1.0, 0.5)
        self.assertEqual(s.data.ctrl[valve.actuator], 0.4)
        self.assertAlmostEqual(s._measured(valve), 0.4, places=8)
        t = self.session()
        task = FilterAccess(t)
        t.data.qfrc_applied[t.model.jnt_dofadr[valve.joint]] = 2
        report = task.run()
        self.assertFalse(report["success"])
        self.assertEqual(report["failure"], "held_control_changed_or_drifted")
        self.assertLess(report["checked_steps"], report["physics_steps"])

    def test_readonly_observer_abort_has_honest_checked_coverage(self):
        s = self.session()
        calls = 0

        def stop(model, data):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("independent_review_stop")

        report = inspect_eth_filter(s, step_observer=stop)
        self.assertFalse(report["success"])
        self.assertEqual(report["physics_steps"], 4)
        self.assertEqual(report["checked_steps"], 4)
        self.assertTrue(report["coverage_complete"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])


if __name__ == "__main__":
    unittest.main()
