"""Support, real apertures and measured wire/slot motion without state teleport."""

import itertools
import json
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.birmingham_fixture import RUNNING_CLEARANCE_M, WIRE_RADIUS_M
from real_labs.motion import PlanningError
from real_labs.runtime import InstrumentSession
from real_labs.wire_alignment import WireAlignment


PREFIX = "primary_apparatus__wire_fixture_"


def session():
    root, manifest = build_scene("birmingham_advanced_micro_edm")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def gid(s, key):
    return s.model.geom(PREFIX + key).id


def distance(s, first, second):
    return float(mujoco.mj_geomDistance(s.model, s.data, first, second, 10, None))


def proxies(s, prefix):
    return [
        i
        for i in range(s.model.ngeom)
        if s.model.geom(i).name.startswith(PREFIX + prefix)
    ]


def base_box(s, position, half_size):
    base = s.model.body("primary_apparatus").id
    return next(
        i
        for i in range(s.model.ngeom)
        if s.model.geom_bodyid[i] == base
        and s.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        and np.allclose(s.model.geom_pos[i], position)
        and np.allclose(s.model.geom_size[i], half_size)
    )


class BirminghamFixtureGeometryTests(unittest.TestCase):
    def test_fixed_supports_connect_to_grounded_machine(self):
        s = session()
        cabinet = base_box(s, (0, 0, 0.39), (0.66, 0.52, 0.39))
        back = base_box(s, (0, 0.49, 1.30), (0.66, 0.06, 0.52))
        self.assertAlmostEqual(
            s.data.geom_xpos[cabinet, 2] - s.model.geom_size[cabinet, 2], 0
        )
        self.assertLessEqual(distance(s, back, cabinet), 1e-6)
        for row in range(2):
            runner = gid(s, f"bed_runner_{row}")
            self.assertAlmostEqual(distance(s, cabinet, runner), 0, places=7)
            self.assertAlmostEqual(distance(s, gid(s, "bed"), runner), 0, places=7)
        self.assertLessEqual(distance(s, back, gid(s, "lower_arm")), 0)
        for side in (-1, 1):
            self.assertLessEqual(distance(s, back, gid(s, f"upper_fork_{side}")), 0)
            spacer, plate = gid(s, f"wheel_back_spacer_{side}"), gid(
                s, f"wheel_plate_{side}"
            )
            self.assertLessEqual(distance(s, back, spacer), 0)
            self.assertAlmostEqual(distance(s, spacer, plate), 0, places=7)
            for level in range(2):
                axle, wheel = gid(s, f"wheel_axle_{side}_{level}"), gid(
                    s, f"wheel_{side}_{level}"
                )
                self.assertLessEqual(distance(s, plate, axle), 0)
                self.assertLessEqual(distance(s, axle, wheel), 0)
                standoff = gid(s, f"panel_standoff_{side}_{level}")
                self.assertLessEqual(distance(s, back, standoff), 0)
                self.assertLessEqual(distance(s, gid(s, f"panel_{side}"), standoff), 0)
        console = base_box(s, (0.92, -0.24, 0.95), (0.32, 0.27, 0.045))
        display = base_box(s, (0.92, -0.04, 1.20), (0.32, 0.04, 0.23))
        bracket = gid(s, "display_bracket")
        self.assertLessEqual(distance(s, console, bracket), 1e-6)
        self.assertLessEqual(distance(s, bracket, display), 1e-6)
        for key in [f"key_pad_{i}" for i in range(6)] + ["button_pad"]:
            self.assertAlmostEqual(distance(s, console, gid(s, key)), 0, places=7)

    def test_original_axes_datums_and_true_wire_scale(self):
        s = session()
        m = s.model
        self.assertEqual((m.njnt, m.nu), (2, 2))
        np.testing.assert_allclose(m.jnt_range, [[-0.12, 0.12], [-0.10, 0.10]])
        np.testing.assert_allclose(m.qpos0, [0, 0])
        legacy = m.site("primary_apparatus__site_inert_coupon").id
        slot = m.site("primary_apparatus__site_calibration_slot").id
        surface = m.site("primary_apparatus__site_coupon_surface").id
        np.testing.assert_allclose(s.data.site_xpos[legacy], [-0.3, 0.7, 1.04])
        np.testing.assert_allclose(s.data.site_xpos[slot], s.data.site_xpos[legacy])
        np.testing.assert_allclose(s.data.site_xpos[surface], [-0.26, 0.7, 1.065])
        self.assertAlmostEqual(m.geom_size[gid(s, "wire"), 0], 0.0001)
        self.assertEqual(WIRE_RADIUS_M, 0.0001)
        self.assertTrue(
            {"overview", "interior", "workstation", "wire_interface"}.issubset(
                {m.camera(i).name for i in range(m.ncam)}
            )
        )
        self.assertEqual(
            s.equipment["primary_apparatus"]["wire_alignment"][
                "rear_open_slot_width_m"
            ],
            0.008,
        )

    def test_bearing_bores_and_load_path_through_full_travel(self):
        s = session()
        m = s.model
        bed, xplate = gid(s, "bed"), gid(s, "x_plate")
        for row, end in itertools.product(range(2), repeat=2):
            pedestal, rail = gid(s, f"x_pedestal_{row}_{end}"), gid(s, f"x_rail_{row}")
            self.assertAlmostEqual(distance(s, bed, pedestal), 0, places=7)
            self.assertAlmostEqual(distance(s, rail, pedestal), 0, places=7)
            pedestal, rail = gid(s, f"y_pedestal_{row}_{end}"), gid(s, f"y_rail_{row}")
            self.assertAlmostEqual(distance(s, xplate, pedestal), 0, places=7)
            self.assertAlmostEqual(distance(s, rail, pedestal), 0, places=7)
        for axis, limits in enumerate(((-0.12, 0.12), (-0.1, 0.1))):
            letter = "xy"[axis]
            for q in np.linspace(*limits, 49):
                s.data.qpos[:] = 0
                s.data.qpos[axis] = q
                mujoco.mj_forward(m, s.data)
                for row, shoe in itertools.product(range(2), repeat=2):
                    rail = gid(s, f"{letter}_rail_{row}")
                    pieces = proxies(s, f"{letter}_bearing_{row}_{shoe}_proxy_")
                    self.assertEqual(len(pieces), 24)
                    self.assertEqual(m.geom_contype[rail], 1)
                    self.assertAlmostEqual(
                        min(distance(s, rail, g) for g in pieces),
                        RUNNING_CLEARANCE_M,
                        places=7,
                    )
                    mount = gid(s, f"{letter}_shoe_mount_{row}_{shoe}")
                    self.assertLessEqual(
                        min(distance(s, mount, g) for g in pieces), 1e-6
                    )
                    deck = xplate if axis == 0 else gid(s, f"y_side_{row}")
                    self.assertLessEqual(distance(s, mount, deck), 1e-6)
                    self.assertGreater(distance(s, rail, deck), 0.0005)

    def test_coupon_has_real_open_slot_retention_and_supported_guides(self):
        s = session()
        m = s.model
        wire = gid(s, "wire")
        for side, row in ((-1, 0), (1, 1)):
            pad, coupon = gid(s, f"coupon_pad_{side}"), gid(s, f"coupon_side_{side}")
            self.assertAlmostEqual(distance(s, pad, coupon), 0, places=6)
            self.assertLessEqual(distance(s, pad, gid(s, f"y_side_{row}")), 0)
            # Exact aligned-face support interval avoids coplanar GJK degeneracy.
            tab = gid(s, f"coupon_tab_{side}")
            self.assertAlmostEqual(
                s.data.geom_xpos[tab, 2] - m.geom_size[tab, 2],
                s.data.geom_xpos[coupon, 2] + m.geom_size[coupon, 2],
            )
        for key in ("lower_guide", "upper_guide"):
            pieces = proxies(s, key + "_proxy_")
            self.assertEqual(len(pieces), 48)
            self.assertAlmostEqual(
                min(distance(s, wire, g) for g in pieces), 0.0002, places=7
            )
        lower = proxies(s, "lower_guide_proxy_")
        self.assertLessEqual(
            min(distance(s, gid(s, "lower_arm"), g) for g in lower), 1e-6
        )
        upper = proxies(s, "upper_guide_proxy_")
        for side in (-1, 1):
            self.assertLessEqual(
                min(distance(s, gid(s, f"upper_fork_{side}"), g) for g in upper), 1e-6
            )
        s.data.qpos[:] = [0, 0.08]
        mujoco.mj_forward(m, s.data)
        # The old solid coupon was 64 mm above its wire. Now the wire passes
        # between two actual metal faces with 3.9 mm lateral clearance.
        self.assertAlmostEqual(
            distance(s, wire, gid(s, "coupon_side_-1")), 0.0039, places=6
        )
        self.assertAlmostEqual(
            distance(s, wire, gid(s, "coupon_side_1")), 0.0039, places=6
        )
        self.assertLess(s.data.geom_xpos[wire, 2] - m.geom_size[wire, 1], 1.015)
        self.assertGreater(s.data.geom_xpos[wire, 2] + m.geom_size[wire, 1], 1.065)
        # The fixed calibration wire terminates on actual supported surfaces;
        # it no longer floats axially inside both open eyelets.
        self.assertAlmostEqual(distance(s, wire, gid(s, "lower_arm")), 0, places=7)
        cap = gid(s, "upper_wire_termination")
        self.assertAlmostEqual(distance(s, wire, cap), 0, places=7)
        self.assertAlmostEqual(min(distance(s, cap, g) for g in upper), 0, places=7)

    def test_filtered_pairs_full_travel_only_expose_real_wire_obstacles(self):
        s = session()
        m, d = s.model, s.data
        ids = np.array(
            [
                g
                for g in range(m.ngeom)
                if m.geom_contype[g] and m.geom(g).name.startswith("primary_apparatus")
            ]
        )
        welds = m.body_weldid[m.geom_bodyid[ids]]
        first, second = np.where(
            (welds[:, None] != welds)
            & (np.arange(len(ids))[:, None] < np.arange(len(ids)))
        )
        aa, bb = ids[first], ids[second]
        poses = list(
            itertools.product(np.linspace(-0.12, 0.12, 9), np.linspace(-0.1, 0.1, 9))
        )
        poses += [(q, 0) for q in np.linspace(-0.12, 0.12, 49)]
        poses += [(0, q) for q in np.linspace(-0.1, 0.1, 49)]
        blocked = 0
        for pose in poses:
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            near = (
                np.linalg.norm(d.geom_xpos[aa] - d.geom_xpos[bb], axis=1)
                <= m.geom_rbound[aa] + m.geom_rbound[bb] + 0.0001
            )
            for a, b in zip(aa[near], bb[near]):
                gap = distance(s, int(a), int(b))
                if gap >= -1e-5:
                    continue
                names = (m.geom(a).name, m.geom(b).name)
                self.assertIn(PREFIX + "wire", names, (pose, names, gap))
                other = next(n for n in names if n != PREFIX + "wire")
                self.assertTrue(
                    other.startswith(PREFIX + "coupon_"), (pose, names, gap)
                )
                blocked += 1
        self.assertGreater(
            blocked,
            0,
            "The preserved full workspace must expose real blocked configurations",
        )


class BirminghamWireTaskTests(unittest.TestCase):
    def test_reported_clearance_is_a_capped_lower_bound(self):
        s = session()
        task = WireAlignment(s)
        # A real reset query can return a large sentinel at a 10 mm cutoff,
        # while another primitive pair returns an uncapped analytic distance.
        task._observe(s.model, s.data)
        actual = min(distance(s, task.wire, g) for g in task.motion.moving_geoms)
        self.assertGreater(actual, 0.01)
        self.assertAlmostEqual(task.minimum_wire_gap, 0.01)
        self.assertLessEqual(task.minimum_wire_gap, actual)
        s.data.qpos[:] = [0, 0.08]
        mujoco.mj_forward(s.model, s.data)
        task._observe(s.model, s.data)
        self.assertAlmostEqual(task.minimum_wire_gap, 0.0039, places=6)

    def test_disabled_essential_colliders_are_rejected_before_motion(self):
        for key in (
            "wire",
            "lower_guide_proxy_0",
            "upper_guide_proxy_47",
            "coupon_side_-1",
            "coupon_front",
            "lower_arm",
            "upper_wire_termination",
        ):
            with self.subTest(key=key):
                s = session()
                s.model.geom_contype[gid(s, key)] = 0
                s.model.geom_conaffinity[gid(s, key)] = 0
                with self.assertRaisesRegex(ValueError, "collider changed"):
                    WireAlignment(s)
                self.assertEqual(s.data.time, 0)
                np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)

    def test_global_disable_and_postconstruction_changes_are_rejected(self):
        for flag in (
            mujoco.mjtDisableBit.mjDSBL_CONTACT,
            mujoco.mjtDisableBit.mjDSBL_CONSTRAINT,
            mujoco.mjtDisableBit.mjDSBL_LIMIT,
            mujoco.mjtDisableBit.mjDSBL_ACTUATION,
        ):
            s = session()
            s.model.opt.disableflags |= int(flag)
            with self.assertRaisesRegex(PlanningError, "must remain enabled"):
                WireAlignment(s)
            self.assertEqual(s.data.time, 0)
        s = session()
        task = WireAlignment(s)
        s.model.geom_contype[task.wire] = 0
        before = s.data.qpos.copy()
        report = task.move([0, 0.08])
        self.assertEqual(report["status"], "blocked")
        self.assertIn("Model changed", report["failure"])
        self.assertEqual(report["physics_steps"], 0)
        self.assertTrue(report["hold_requested"])
        np.testing.assert_array_equal(s.data.qpos, before)
        self.assertEqual(s.data.time, 0)

    def test_interface_shape_and_semantic_binding_are_verified(self):
        for change in (
            "wire_radius",
            "guide_position",
            "wire_body",
            "slot_site",
            "controls",
        ):
            with self.subTest(change=change):
                s = session()
                if change == "wire_radius":
                    s.model.geom_size[gid(s, "wire"), 0] = 0.00005
                elif change == "guide_position":
                    s.model.geom_pos[gid(s, "lower_guide_proxy_0"), 0] += 0.002
                elif change == "wire_body":
                    s.model.geom_bodyid[gid(s, "wire")] = s.model.body(
                        "primary_apparatus__body_fixture_y"
                    ).id
                elif change == "slot_site":
                    s.equipment["primary_apparatus"]["sites"][
                        "calibration_slot"
                    ] = "primary_apparatus__site_coupon_surface"
                else:
                    xkey, ykey = ("primary_apparatus", "fixture_x"), (
                        "primary_apparatus",
                        "fixture_y",
                    )
                    s.controls[xkey], s.controls[ykey] = (
                        s.controls[ykey],
                        s.controls[xkey],
                    )
                with self.assertRaisesRegex(ValueError, "changed|datum"):
                    WireAlignment(s)
                self.assertEqual(s.data.time, 0)

    def test_measured_round_trip_and_complete_observer_coverage(self):
        s = session()
        before_masks = s.model.geom_contype.copy()
        result = WireAlignment(s).run()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["status"], "succeeded")
        self.assertLess(result["alignment"]["lateral_alignment_error_m"], 0.0002)
        self.assertLess(result["measured_return_error_m"], 0.0002)
        self.assertFalse(result["reset_performed"])
        self.assertGreater(result["alignment"]["simulation_time_s"], 0)
        self.assertEqual(len(result["alignment"]["qpos"]), s.model.nq)
        for leg in result["phases"]:
            self.assertTrue(leg["whole_leg_clearance_observed"])
            self.assertEqual(
                leg["physics_steps"], leg["wire_clearance_fully_checked_steps"]
            )
            self.assertGreaterEqual(leg["minimum_measured_wire_gap_m"], 0.0001)
            self.assertGreaterEqual(leg["minimum_observed_guide_gap_m"], 0.0001)
        np.testing.assert_array_equal(before_masks, s.model.geom_contype)

    def test_blocked_goal_and_raw_contact_retain_real_collision(self):
        s = session()
        task = WireAlignment(s)
        self.assertEqual(task.move([0, 0.08])["status"], "succeeded")
        before, time = s.data.qpos.copy(), s.data.time
        blocked = task.move([0.01, 0.08])
        self.assertEqual(blocked["status"], "blocked")
        self.assertTrue(blocked["hold_requested"])
        self.assertEqual(s.data.time, time)
        np.testing.assert_array_equal(s.data.qpos, before)
        direct = s.command("primary_apparatus", "fixture_x", 0.01, duration=2)
        self.assertGreater(abs(direct["error"]), 0.005)
        self.assertLess(direct["measured"], 0.0041)
        self.assertTrue(
            any(
                gid(s, "wire") in (c.geom1, c.geom2) and c.dist < 0
                for c in s.data.contact
            )
        )
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)
        np.testing.assert_array_equal(s.data.qvel, np.zeros(s.model.nv))

    def test_observer_failure_holds_measured_state_without_reset(self):
        s = session()
        task = WireAlignment(s)
        count = 0

        def stop(model, data):
            nonlocal count
            count += 1
            if count == 23:
                raise RuntimeError("test_observer_stop")

        result = task.move([-0.06, -0.04], step_observer=stop)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"], "test_observer_stop")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["whole_leg_clearance_observed"])
        self.assertEqual(result["physics_steps"], 23)
        self.assertEqual(result["wire_clearance_fully_checked_steps"], 23)
        np.testing.assert_allclose(s.data.ctrl, s.data.qpos)
        self.assertGreater(np.linalg.norm(s.data.qpos), 0)
        self.assertGreater(np.linalg.norm(s.data.qvel), 0)
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, s.model.qpos0)
        np.testing.assert_array_equal(s.data.qvel, np.zeros(s.model.nv))

    def test_core_failure_has_no_unobserved_clearance_claim(self):
        s = session()
        task = WireAlignment(s)
        # A selected-axis force fault trips core tracking on the first step,
        # before the fixture observer is called. No failed leg is certified.
        s.data.qfrc_applied[0] = 10000
        result = task.move([-0.06, -0.04])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertGreater(result["physics_steps"], 0)
        self.assertEqual(result["wire_clearance_observed_steps"], 0)
        self.assertEqual(result["wire_clearance_fully_checked_steps"], 0)
        self.assertIsNone(result["minimum_measured_wire_gap_m"])
        self.assertIsNone(result["minimum_observed_guide_gap_m"])
        self.assertFalse(result["whole_leg_clearance_observed"])

    def test_round_trip_rejects_nonhome_without_resetting(self):
        s = session()
        s.command("primary_apparatus", "fixture_x", -0.06, duration=2)
        before, time = s.data.qpos.copy(), s.data.time
        with self.assertRaises(PlanningError):
            WireAlignment(s).run()
        np.testing.assert_array_equal(s.data.qpos, before)
        self.assertEqual(s.data.time, time)


if __name__ == "__main__":
    unittest.main()
