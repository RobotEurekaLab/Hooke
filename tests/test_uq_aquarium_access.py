"""True open guides and dry probe access, without a water-sampling claim."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.runtime import InstrumentSession
from real_labs.uq_aquarium_motion import AquariumAccess, SCENE, inspect_uq_aquarium


def session():
    root, metadata = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), metadata
    )


def state(s):
    return (
        s.data.time,
        *(
            getattr(s.data, key).tobytes()
            for key in (
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


def gid(s, key):
    return s.model.geom("water_sampling_bridge__uq_" + key).id


def gap(s, a, b):
    return float(
        mujoco.mj_geomDistance(s.model, s.data, gid(s, a), gid(s, b), 0.1, None)
    )


class AquariumAccessTests(unittest.TestCase):
    def test_optical_water_contains_lowered_cap_and_meets_actual_basin(self):
        s = session()
        s.data.qpos[:] = [0.3, 0.18]
        mujoco.mj_forward(s.model, s.data)
        m, d = s.model, s.data
        water, floor, cap = (
            gid(s, key) for key in ("display_water", "tank_bottom", "probe_cap")
        )
        bottom = d.geom_xpos[water, 2] - m.geom_size[water, 1]
        top = d.geom_xpos[water, 2] + m.geom_size[water, 1]
        floor_top = d.geom_xpos[floor, 2] + m.geom_size[floor, 1]
        self.assertAlmostEqual(top, 1.064)
        self.assertAlmostEqual(floor_top - bottom, 0.0002)
        self.assertAlmostEqual(m.geom_size[water, 0] - 1.115, 0.0002)
        self.assertLess(bottom, d.geom_xpos[cap, 2] - m.geom_size[cap, 1])
        self.assertGreater(top, d.geom_xpos[cap, 2] + m.geom_size[cap, 1])
        self.assertEqual(m.geom_contype[water], 0)
        self.assertEqual(m.geom_conaffinity[water], 0)
        self.assertEqual(m.geom_group[water], 2)
        self.assertTrue(m.mat(m.geom_matid[water]).name.endswith("__mat_clear_water"))

    def test_original_axes_scale_and_true_surface(self):
        s = session()
        for key, pos, axis, limits, mass, gain in (
            ("probe_position", [0, 0, 1.744], [1, 0, 0], [-0.55, 0.55], 1.0, 250),
            ("probe_depth", [0, 0, -0.03], [0, 0, -1], [0, 0.28], 0.30, 180),
        ):
            c = s.controls["water_sampling_bridge", key]
            body = s.model.jnt_bodyid[c.joint]
            np.testing.assert_allclose(s.model.body_pos[body], pos)
            np.testing.assert_allclose(s.model.jnt_axis[c.joint], axis)
            np.testing.assert_allclose(s.model.jnt_range[c.joint], limits)
            self.assertEqual(s.model.body_mass[body], mass)
            self.assertEqual(s.model.actuator_gainprm[c.actuator, 0], gain)
            np.testing.assert_allclose(
                s.model.actuator_forcerange[c.actuator], [-120, 120]
            )
        cap = gid(s, "probe_cap")
        np.testing.assert_allclose(s.model.geom_size[cap, :2], [0.02, 0.004])
        center = s.model.site("water_sampling_bridge__site_probe_tip").id
        surface = s.model.site("water_sampling_bridge__site_probe_surface").id
        np.testing.assert_allclose(s.model.site_pos[center], [0, 0, -0.532])
        np.testing.assert_allclose(
            s.data.site_xpos[surface], s.data.geom_xpos[cap] + [0, -0.02, 0]
        )
        self.assertEqual(
            screen_contact_coverage(s.model)["zero_contact_visible_group_count"], 0
        )

    def test_full_coupled_travel_and_parent_filtered_bores(self):
        s = session()
        task = AquariumAccess(s)
        pairs = {frozenset((int(a), int(b))) for a, b in task.pairs}
        self.assertIn(
            frozenset((gid(s, "x_rail_0"), gid(s, "x_bearing_0_0_proxy_0"))), pairs
        )
        self.assertIn(
            frozenset((gid(s, "probe_shaft"), gid(s, "z_bearing_0_proxy_0"))), pairs
        )
        for x, z in itertools.product(
            np.linspace(-0.55, 0.55, 13), np.linspace(0, 0.28, 15)
        ):
            s.data.qpos[:] = [x, z]
            mujoco.mj_forward(s.model, s.data)
            task.check(s.model, s.data)
            self.assertFalse(any(c.dist < -1e-7 for c in s.data.contact))
        self.assertGreaterEqual(task.minimum, 0.0005999)

    def test_fixed_and_moving_load_paths(self):
        s = session()
        for a, b in (
            ("foot_0_0", "post_0_0"),
            ("post_0_0", "end_crossbar_0"),
            ("end_crossbar_0", "beam_0"),
            ("beam_0", "rail_anchor_0_0"),
            ("rail_anchor_0_0", "x_rail_0"),
            ("post_0_0", "brace_0_0"),
            ("brace_0_0", "beam_0"),
            ("beam_1", "datum_top_bracket"),
            ("datum_top_bracket", "datum_strip"),
            ("datum_strip", "datum_witness"),
            ("post_1_0", "controller_bracket"),
            ("controller_bracket", "controller"),
            ("controller", "controller_screen"),
            ("controller", "controller_button"),
            ("controller", "parked_controller_lead"),
            ("parked_controller_lead", "parked_connector"),
            ("tank_foot_0", "tank_bottom"),
            ("tank_bottom", "drain_boss"),
            ("drain_boss", "drain_cap"),
            ("carriage_side_0", "carriage_end_0"),
            ("carriage_side_0", "guide_upright_0"),
            ("guide_upright_0", "guide_arm_0_0"),
            ("carriage_side_0", "actuator_foot"),
            ("actuator_foot", "actuator_housing"),
            ("actuator_housing", "actuator_cap"),
            ("probe_shaft", "probe_head"),
            ("probe_head", "probe_cap"),
        ):
            self.assertLessEqual(gap(s, a, b), 2e-7, (a, b, gap(s, a, b)))
        for key, attachment in (
            ("x_bearing_0_0", "carriage_side_0"),
            ("z_bearing_0", "guide_arm_0_0"),
        ):
            self.assertLessEqual(
                min(gap(s, f"{key}_proxy_{i}", attachment) for i in range(48)), 2e-7
            )
        m, d = s.model, s.data
        for depth in (0, 0.14, 0.28):
            d.qpos[1] = depth
            mujoco.mj_forward(m, d)
            shaft = gid(s, "probe_shaft")
            bottom = d.geom_xpos[shaft, 2] - m.geom_size[shaft, 1]
            top = d.geom_xpos[shaft, 2] + m.geom_size[shaft, 1]
            for k in (0, 1):
                bearing = gid(s, f"z_bearing_{k}_proxy_0")
                self.assertLess(bottom, d.geom_xpos[bearing, 2] - 0.019)
                self.assertGreater(top, d.geom_xpos[bearing, 2] + 0.019)

    def test_ceiling_hangers_and_service_graph_reach_actual_roof(self):
        s = session()
        m, d = s.model, s.data
        roof = m.geom("arch_roof_slab").id
        hangers = [
            g for g in range(m.ngeom) if "facility_uq_utility_hanger" in m.geom(g).name
        ]
        self.assertEqual(len(hangers), 9)
        for g in hangers:
            self.assertLessEqual(mujoco.mj_geomDistance(m, d, g, roof, 0.2, None), 0)
            self.assertGreaterEqual(
                mujoco.mj_geomDistance(m, d, g, roof, 0.2, None), -0.00901
            )
        # Check actual touching paths for every ceiling-dependent service,
        # including main/header/branch/return pipes, not only endpoint numbers.
        remaining = {
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith("arch_roof_facility_uq")
        }
        reached = {roof}
        while remaining:
            connected = {
                g
                for g in remaining
                if any(
                    mujoco.mj_geomDistance(m, d, g, h, 0.000001, None) <= 1e-7
                    for h in reached
                )
            }
            if not connected:
                break
            reached |= connected
            remaining -= connected
        self.assertFalse(remaining, [m.geom(g).name for g in remaining])

    def test_actual_position_observe_and_return(self):
        s = session()
        report = inspect_uq_aquarium(s)
        self.assertEqual(report["status"], "succeeded", report["phases"][-1])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["executed_steps"], report["checked_steps"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 1)
        np.testing.assert_allclose(s.data.qpos, [0, 0], atol=1e-4)
        for observation in report["observations"]:
            self.assertTrue(observation["cap_solid_line_of_sight"])
            self.assertGreater(observation["estimated_cap_diameter_px"], 20)
            self.assertAlmostEqual(observation["intentional_datum_offset_y_m"], -0.344)
        registered = report["observations"][1]
        self.assertTrue(registered["camera_path_crosses_display_water"])
        self.assertFalse(registered["rendered_visibility_qualified"])
        self.assertLess(abs(registered["vertical_registration_error_m"]), 0.0002)
        self.assertLess(abs(registered["lateral_registration_error_m"]), 0.0002)
        self.assertGreater(report["whole_run_clearance_lower_bound_m"], 0.00059)

    def test_source_masks_physics_and_metadata_reject_without_mutation(self):
        for case in (
            "cap",
            "rail",
            "bearing",
            "geometry",
            "mass",
            "contact",
            "constraint",
            "limit",
            "actuation",
            "metadata",
        ):
            s = session()
            if case in ("cap", "rail", "bearing"):
                key = {
                    "cap": "probe_cap",
                    "rail": "x_rail_0",
                    "bearing": "z_bearing_0_proxy_0",
                }[case]
                s.model.geom_contype[gid(s, key)] = 0
                s.model.geom_conaffinity[gid(s, key)] = 0
            elif case == "geometry":
                s.model.geom_pos[gid(s, "probe_head"), 0] += 0.1
            elif case == "mass":
                s.model.body_mass[s.model.geom_bodyid[gid(s, "probe_cap")]] *= 2
            elif case == "metadata":
                s.equipment["water_sampling_bridge"] = dict(
                    s.equipment["water_sampling_bridge"], probe_interface={}
                )
            else:
                s.model.opt.disableflags |= int(
                    getattr(mujoco.mjtDisableBit, "mjDSBL_" + case.upper())
                )
            before = state(s)
            with self.assertRaises(ValueError, msg=case):
                AquariumAccess(s)
            self.assertEqual(state(s), before, case)

    def test_foreign_and_replaced_state_rejected_before_mutation(self):
        s, other = session(), session()
        bad = copy(s)
        bad.data = other.data
        before = state(bad)
        with self.assertRaises(ValueError):
            AquariumAccess(bad)
        self.assertEqual(state(bad), before)
        task = AquariumAccess(s)
        s.data = other.data
        before = state(s)
        with self.assertRaises(ValueError):
            task.move((0.1, 0))
        self.assertEqual(state(s), before)

    def test_held_depth_load_reports_partial_coverage_and_no_reset(self):
        s = session()
        task = AquariumAccess(s)
        c = s.controls["water_sampling_bridge", "probe_depth"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 500
        result = task.move((0.1, 0))
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["physics_steps"], 0)
        coverage = result["direct_clearance"]
        self.assertLess(coverage["checked_steps"], coverage["executed_steps"])
        self.assertFalse(coverage["coverage_complete"])
        self.assertIsNone(coverage["whole_leg_clearance_lower_bound_m"])
        self.assertTrue(result["hold_requested"])
        self.assertGreater(s.data.time, 0)

    def test_live_mask_change_aborts_before_next_complete_check(self):
        s = session()
        task = AquariumAccess(s)
        s.model.geom_contype[gid(s, "probe_cap")] = 0
        s.model.geom_conaffinity[gid(s, "probe_cap")] = 0
        with self.assertRaisesRegex(RuntimeError, "contact_configuration_changed"):
            task.check(s.model, s.data)
        self.assertEqual(task.checked, 0)


if __name__ == "__main__":
    unittest.main()
