"""168-flap geometry, complete-array controls and honest failure observations."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.flowave_geometry import (
    COUNT,
    HINGE_Z,
    LIMIT,
    PREFIX,
    TEST_FLOOR_Z,
    WATER_Z,
    key,
)
from real_labs.flowave_motion import FlapInspection, full_angle_bounds
from real_labs.runtime import InstrumentSession


class FloWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("edinburgh_flowave_basin")
        cls.xml = ET.tostring(root, encoding="unicode")
        cls.model = mujoco.MjModel.from_xml_string(cls.xml)
        cls.session = InstrumentSession(cls.model, cls.manifest)
        cls.inspection = FlapInspection(cls.session)

    def setUp(self):
        self.session.reset()

    def test_published_datums_versioned_angles_and_real_water(self):
        m, d = self.model, self.session.data
        self.assertEqual(m.njnt, 168)
        self.assertTrue(np.all(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
        np.testing.assert_array_equal(m.jnt_range, np.tile([-LIMIT, LIMIT], (COUNT, 1)))
        np.testing.assert_allclose(d.xanchor[:, 2], HINGE_Z, atol=1e-12)
        self.assertAlmostEqual(WATER_Z - TEST_FLOOR_Z, 2)
        self.assertAlmostEqual(WATER_Z - HINGE_Z, 1.7)
        water = m.geom(PREFIX + "water_display_volume").id
        self.assertFalse(m.geom_contype[water] or m.geom_conaffinity[water])
        self.assertEqual(m.geom_size[water, 1], 1.0)
        self.assertEqual(m.mat(m.geom_matid[water]).name, "wave_basin__mat_clear_water")
        self.assertEqual(
            len(self.manifest["equipment"][0]["coordinated_inspection_targets"]), 168
        )
        with self.assertRaisesRegex(ValueError, "complete version-2"):
            self.inspection.move({"paddle_west": 0.05, "paddle_east": 0.05})

    def test_open_grid_and_continuous_structural_bearings(self):
        m, d = self.model, self.session.data

        def inside(name, point, tol=1e-9):
            g = m.geom(name).id
            self.assertEqual(m.geom_type[g], mujoco.mjtGeom.mjGEOM_BOX)
            local = d.geom_xmat[g].reshape(3, 3).T @ (
                np.asarray(point) - d.geom_xpos[g]
            )
            return np.all(abs(local) <= m.geom_size[g] + tol)

        # Witness points lie in actual metal on both sides of each connection;
        # this avoids interpreting a capped distant query or tangent-box bug as support.
        for i in range(COUNT):
            a = i * 2 * np.pi / COUNT
            c, s = np.cos(a), np.sin(a)
            for side in (-1, 1):
                local = np.array([0.034, side * 0.213, 0])
                world = (
                    np.array([12.53 * c, 12.53 * s, HINGE_Z])
                    + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ local
                )
                self.assertTrue(inside(PREFIX + f"bearing_{i}_{side}_pedestal", world))
                self.assertTrue(
                    any(
                        inside(PREFIX + f"bearing_{i}_{side}_sector_{j}", world)
                        for j in range(12)
                    )
                )
            point = ((12.53 + 0.062) * c, (12.53 + 0.062) * s, HINGE_Z - 0.345 + 0.01)
            # Mount bottom overlaps the real hinge support ring.
            point = ((12.53 + 0.062) * c, (12.53 + 0.062) * s, HINGE_Z - 0.30)
            self.assertTrue(inside(PREFIX + f"mount_base_{i}", point))
            self.assertTrue(
                any(
                    inside(PREFIX + f"hinge_support_ring_{j}", point)
                    for j in (i, (i - 1) % 168)
                )
            )
        for i in range(1344):
            a = i * 2 * np.pi / 1344
            point = (13.70 * np.cos(a), 13.70 * np.sin(a), -0.028)
            self.assertTrue(inside(f"flowave_context_grate_radial_{i}", point))
            self.assertTrue(
                any(
                    inside(f"flowave_context_grate_ring_beam_1_{j%168}", point)
                    for j in (i // 8, i // 8 - 1)
                )
            )
        # One genuine opening is empty at deck level; no collision slab is hidden below bars.
        a = 2 * np.pi * (24.5) / 1344
        r = 13.12
        group = np.ones(6, np.uint8)
        hit = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            m,
            d,
            np.array([r * np.cos(a), r * np.sin(a), 0.01]),
            np.array([0.0, 0.0, -1.0]),
            group,
            True,
            -1,
            hit,
        )
        self.assertGreater(distance, 0.04)
        self.assertGreaterEqual(hit[0], 0)

    def test_full_angular_domain_bounds_and_neighbor_extrema(self):
        m, op = self.model, self.inspection
        low, high = full_angle_bounds(m)
        scratch = mujoco.MjData(m)
        rng = np.random.default_rng(724)
        patterns = [
            np.full(COUNT, -LIMIT),
            np.full(COUNT, LIMIT),
            np.where(np.arange(COUNT) % 2, LIMIT, -LIMIT),
            np.where(np.arange(COUNT) % 2, -LIMIT, LIMIT),
            np.zeros(COUNT),
        ]
        patterns.extend(rng.uniform(-LIMIT, LIMIT, COUNT) for _ in range(8))
        for pattern in patterns:
            scratch.qpos[op.addresses] = pattern
            mujoco.mj_forward(m, scratch)
            mat = scratch.geom_xmat.reshape(-1, 3, 3)
            center = scratch.geom_xpos + np.einsum(
                "nij,nj->ni", mat, m.geom_aabb[:, :3]
            )
            # Every actual oriented box corner, not its looser world AABB, lies
            # in the analytic full-angle enclosure.
            for signs in itertools.product((-1, 1), repeat=3):
                corners = center + np.einsum(
                    "nij,nj->ni", mat, m.geom_aabb[:, 3:] * signs
                )
                self.assertTrue(np.all(corners >= low - 1e-9))
                self.assertTrue(np.all(corners <= high + 1e-9))
            gap, _ = op.direct_clearance(scratch)
            self.assertGreater(gap, 0.00059)
            self.assertEqual(scratch.ncon, 0)

    def test_real_obstacle_and_parent_filtered_contacts_are_rejected(self):
        # The original enabled fixed bearing is moved into the real axle on a
        # private model; direct checks must fail even if engine parent filters it.
        m = mujoco.MjModel.from_xml_string(self.xml)
        s = InstrumentSession(m, self.manifest)
        op = FlapInspection(s)
        g = m.geom(PREFIX + "bearing_0_1_sector_0").id
        m.geom_pos[g] = [0, 0.213, 0]
        mujoco.mj_forward(m, s.data)
        with self.assertRaisesRegex(RuntimeError, "clearance lost"):
            op.direct_clearance(s.data)
        with self.assertRaises(ValueError):
            op.move({key(i): 0 for i in range(COUNT)})

    def test_model_data_flags_and_incomplete_vectors_reject_unchanged(self):
        for mutation in ("geometry", "flags", "foreign_data", "incomplete", "nan_time"):
            m = mujoco.MjModel.from_xml_string(self.xml)
            s = InstrumentSession(m, self.manifest)
            op = FlapInspection(s)
            target = {key(i): 0.01 for i in range(COUNT)}
            if mutation == "geometry":
                m.geom_size[m.geom(PREFIX + "panel_0").id, 0] *= 0.5
            if mutation == "flags":
                m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            if mutation == "foreign_data":
                s.data = mujoco.MjData(mujoco.MjModel.from_xml_string(self.xml))
            if mutation == "incomplete":
                target.pop(key(167))
            if mutation == "nan_time":
                s.data.time = float("nan")
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                op.move(target)
            for field, value in zip(("qpos", "qvel", "ctrl"), before[:3]):
                np.testing.assert_array_equal(getattr(s.data, field), value)
            self.assertTrue(
                np.isnan(s.data.time)
                if np.isnan(before[3])
                else s.data.time == before[3]
            )

    def test_noncanonical_instances_compile_without_name_collisions(self):
        from real_labs.equipment import _Builder
        from real_labs.flowave_geometry import add_basin

        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(root, "actuator")
        for name, x in (("basin_a", -20), ("basin_b", 20)):
            builder = _Builder(root, name)
            base = ET.SubElement(world, "body", name=name, pos=f"{x} 0 0")
            add_basin(builder, base, {})
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual(model.njnt, 336)
        self.assertGreaterEqual(model.geom("basin_a__flowave_v2_panel_0").id, 0)
        self.assertGreaterEqual(model.geom("basin_b__flowave_v2_panel_0").id, 0)

    def test_position_only_settling_counterexample_is_rejected(self):
        # Independently discovered actual load/velocity combination crosses the
        # target at 0.5 s with ~33 mrad/s, although position error is near zero.
        self.session.data.qvel[0] = 0.4
        self.session.data.qfrc_applied[0] = -9.520508090342076
        mujoco.mj_forward(self.model, self.session.data)
        report = self.inspection.move(
            {key(i): 0 for i in range(COUNT)}, ramp_s=0.5, settle_s=0
        )
        self.assertLess(report["maximum_target_error_rad"], 0.0002)
        self.assertGreater(report["maximum_terminal_speed_rad_s"], 0.03)
        self.assertEqual(report["status"], "failed")
        self.assertIn("residual speed", report["reason"])
        self.assertTrue(report["hold_requested"])
        self.assertEqual(report["physics_steps"], report["checked_steps"])
        self.assertFalse(report["reset_used"])

    def test_native_reset_is_reported_and_existing_warning_rejected(self):
        m = mujoco.MjModel.from_xml_string(self.xml)
        session = InstrumentSession(m, self.manifest)
        op = FlapInspection(session)
        session.data.time = 0.7
        session.data.qfrc_applied[op.velocities[0]] = 1e25
        warnings = []
        previous = mujoco.get_mju_user_warning()
        try:
            mujoco.set_mju_user_warning(warnings.append)
            report = op.move({key(i): 0 for i in range(COUNT)})
        finally:
            mujoco.set_mju_user_warning(previous)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(warnings)
        self.assertTrue(report["automatic_reset_detected"])
        self.assertTrue(report["reset_used"])
        self.assertEqual(report["physics_steps"], 1)
        self.assertEqual(report["checked_steps"], 0)
        self.assertIsNone(report["whole_leg_clearance_lower_bound_m"])
        state = op.state()
        with self.assertRaises(ValueError):
            op.move({key(i): 0 for i in range(COUNT)})
        self.assertEqual(state, op.state())

    def test_actual_full_array_move_return_and_failure_coverage(self):
        op = self.inspection
        target = {key(i): 0.035 if i % 2 else -0.035 for i in range(COUNT)}
        first = op.move(target, ramp_s=1.0, settle_s=1.5)
        self.assertEqual(first["status"], "succeeded", first["reason"])
        self.assertEqual(first["physics_steps"], first["checked_steps"])
        self.assertFalse(first["reset_used"])
        second = op.move({key(i): 0 for i in range(COUNT)}, ramp_s=1.0, settle_s=1.5)
        self.assertEqual(second["status"], "succeeded", second["reason"])
        self.assertLess(np.max(abs(op.d.qpos)), 0.0002)
        start = float(op.d.time)
        op.d.qfrc_applied[op.velocities[7]] = 100000
        failure = op.move(target, ramp_s=1.0, settle_s=1.0)
        self.assertEqual(failure["status"], "failed")
        self.assertTrue(failure["hold_requested"])
        self.assertLess(failure["checked_steps"], failure["physics_steps"])
        self.assertIsNone(failure["whole_leg_clearance_lower_bound_m"])
        self.assertGreater(op.d.time, start)
        self.assertFalse(failure["reset_used"])


if __name__ == "__main__":
    unittest.main()
