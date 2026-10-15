"""Source-bound PolyU free-coupon geometry, contact and interface migration."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.polyu_access_geometry import OLD, NEW, EQUIPMENT
from real_labs.polyu_access_motion import (
    CouponAccess,
    FreeCouponReciprocation,
    SCENE,
    FIELDS,
)
from real_labs.polyu_contact_motion import LoadedReciprocation as RevisionTwoTask
from real_labs.polyu_fixture_contacts import finite_face_gap
from real_labs.runtime import InstrumentSession


def load():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(session):
    return (
        session.data.time,
        *(
            getattr(session.data, n).tobytes()
            for n in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
        ),
    )


class PolyUContactAccessTests(unittest.TestCase):
    def test_original_frames_drives_inertias_and_actual_free_surface(self):
        s = load()
        m = s.model
        for name, frame, axis, limits, mass, kp in (
            ("coupon_x", (0, -0.025, 0.229), (1, 0, 0), (-0.065, 0.065), 0.11, 240),
            ("probe_height", (0, -0.025, 0.62), (0, 0, 1), (-0.10, 0.08), 0.16, 280),
        ):
            c = s.controls[EQUIPMENT, name]
            body = m.jnt_bodyid[c.joint]
            for actual, expected in (
                (m.body_pos[body], frame),
                (m.jnt_axis[c.joint], axis),
                (m.jnt_range[c.joint], limits),
                (m.actuator_ctrlrange[c.actuator], limits),
            ):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(m.body_mass[body], mass)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], kp)
            np.testing.assert_array_equal(
                m.actuator_forcerange[c.actuator], [-120, 120]
            )
        coupon = m.geom(OLD + "coupon").id
        body = int(m.geom_bodyid[coupon])
        surface = m.site(EQUIPMENT + "__site_tribology_coupon").id
        self.assertEqual(m.body_jntnum[body], 1)
        self.assertEqual(m.jnt_type[m.body_jntadr[body]], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.site_bodyid[surface], body)
        np.testing.assert_array_equal(m.site_pos[surface], [0, 0, 0.002])
        self.assertEqual(m.body_mass[body], 0.01728)
        np.testing.assert_array_equal(m.geom_size[coupon], [0.02, 0.02, 0.002])
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.opt.timestep, 0.002)
        for key in ("coupon", "counterface"):
            np.testing.assert_array_equal(m.geom(OLD + key).solref, [0.008, 1])
        self.assertEqual(s.equipment[EQUIPMENT]["contact_interface"]["revision"], 3)
        self.assertEqual(len(s.manifest["specimens"]), 1)
        self.assertEqual(
            s.equipment[EQUIPMENT]["sample_site"], EQUIPMENT + "__site_loading_center"
        )

    def test_supported_lips_and_open_hinge_interfaces(self):
        t = FreeCouponReciprocation(load())
        m, d = t.m, t.d
        for stop, sign in zip(t.stops, (-1, 1)):
            self.assertTrue(m.geom_contype[stop] and m.geom_conaffinity[stop])
            self.assertAlmostEqual(
                finite_face_gap(m, d, t.coupon, stop, sign, axis=0), 0.00015, places=12
            )
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, stop, t.puck, 0.01, None),
                -0.001,
                places=10,
            )
        for left, right in itertools.product((0, 0.87, 1.74532925), repeat=2):
            d.qpos[t.qa[t.ids["guard_left"]]] = left
            d.qpos[t.qa[t.ids["guard_right"]]] = right
            mujoco.mj_forward(m, d)
            for side in ("left", "right"):
                for z in (0.25, 0.77):
                    pin = m.geom(NEW + f"hinge_pin_{side}_{z}").id
                    sectors = [
                        g
                        for g in range(m.ngeom)
                        if m.geom(g).name.startswith(
                            NEW + f"hinge_hinge_{side}_{z}_proxy_"
                        )
                    ]
                    self.assertEqual(len(sectors), 48)
                    self.assertGreater(
                        min(
                            mujoco.mj_geomDistance(m, d, pin, g, 0.002, None)
                            for g in sectors
                        ),
                        0.00059,
                    )

    def test_exact_finite_lip_depth_rejects_native_half_depth(self):
        t = FreeCouponReciprocation(load())
        m, d = t.m, t.d
        q = int(m.joint(NEW + "free_coupon_joint").qposadr[0])
        initial = d.qpos.copy()
        for sign, stop in zip((-1, 1), t.stops):
            d.qpos[:] = initial
            d.qpos[q] += sign * 0.00027
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(
                finite_face_gap(m, d, t.coupon, stop, sign, axis=0), -0.00012, places=12
            )
            with self.assertRaisesRegex(RuntimeError, "support_or_clamp_penetration"):
                t.observe()

    def test_revision_two_masks_metadata_and_foreign_data_reject_unchanged(self):
        s = load()
        before = state(s)
        with self.assertRaisesRegex(ValueError, "revision2"):
            RevisionTwoTask(s)
        self.assertEqual(state(s), before)
        for case in ("mask", "geometry", "metadata", "foreign", "replacement", "clock"):
            s = load()
            t = FreeCouponReciprocation(s)
            if case == "mask":
                s.model.geom_contype[t.stops[0]] = 0
            elif case == "geometry":
                s.model.geom_pos[t.tip, 0] += 0.1
            elif case == "metadata":
                s.equipment[EQUIPMENT]["contact_interface"]["revision"] = 2
            elif case == "foreign":
                s.data = mujoco.MjData(copy(s.model))
            elif case == "replacement":
                s.data = mujoco.MjData(s.model)
            else:
                s.data.time = float("inf")
            before = state(s)
            with self.assertRaises(ValueError):
                t.move_contact({}, 0.01, mode="access", settle=0)
            self.assertEqual(state(s), before, case)

    def test_nonzero_constructor_bounds_keep_every_close_native_pair(self):
        # These are private geometric probes, not executed sample transfers.
        for x, z in ((-0.01, -0.04006), (0.01, -0.04006), (0.003, 0.06)):
            s = load()
            m, d = s.model, s.data
            for key, value in (("coupon_x", x), ("probe_height", z)):
                c = s.controls[EQUIPMENT, key]
                d.qpos[m.jnt_qposadr[c.joint]] = value
                d.ctrl[c.actuator] = value
            d.qpos[m.joint(NEW + "free_coupon_joint").qposadr[0]] += x
            mujoco.mj_forward(m, d)
            task = FreeCouponReciprocation(s)
            retained = set(map(tuple, task.pairs))
            excluded = [(a, b) for a, b in task.full_pairs if (a, b) not in retained]
            free = set(np.flatnonzero(m.geom_bodyid == task.free))
            self.assertFalse(any(a in free or b in free for a, b in excluded))
            for a, b in excluded:
                self.assertGreaterEqual(
                    mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None),
                    0.002,
                    (x, z, m.geom(int(a)).name, m.geom(int(b)).name),
                )

    def test_real_contact_interlocks_disturbance_and_retraction(self):
        s = load()
        task = FreeCouponReciprocation(s)
        for changes, duration, mode in (
            ({"clamp_front": -0.0001, "clamp_rear": -0.0001}, 0.5, "access"),
            ({"coupon_x": 0, "probe_height": -0.0405}, 3, "load"),
            ({"coupon_x": -0.002, "probe_height": -0.0405}, 2, "slide"),
            ({"coupon_x": 0, "probe_height": -0.0405}, 2, "slide"),
        ):
            report = task.move_contact(changes, duration, mode=mode)
            self.assertEqual(report["status"], "succeeded", report["failure"])
            self.assertEqual(report["physics_steps"], report["checked_steps"])
        loaded = task.state()
        self.assertGreater(task.observe()["normal_force_N"], 0.05)
        report = task.move_contact({"probe_height": 0}, 3, mode="retract")
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["final_observation"]["normal_force_N"], 0)
        self.assertGreaterEqual(report["final_observation"]["ball_gap_capped_m"], 0.001)
        # Replay this test's actual loaded state in the private test session.
        for key, value in loaded.items():
            if key == "time_s":
                s.data.time = value
            else:
                getattr(s.data, key)[:] = np.asarray(value).reshape(
                    getattr(s.data, key).shape
                )
        s.data.xfrc_applied[task.free, 0] = 1.3
        mujoco.mj_forward(s.model, s.data)
        failed = task.move_contact({}, 0.1, mode="slide", settle=0)
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["hold_requested"])
        self.assertFalse(failed["reset_performed"])
        self.assertGreater(failed["physics_steps"], failed["checked_steps"])
        self.assertIsNone(failed["whole_leg_capped_clearance_lower_bound_m"])

    def test_constructing_next_workflow_preserves_actual_complete_state(self):
        session = load()
        session.command(EQUIPMENT, "coupon_x", 0.001, duration=0.2)
        before = {key: getattr(session.data, key).copy() for key in FIELDS}
        time = session.data.time
        for operation in (CouponAccess, FreeCouponReciprocation):
            operation(session)
            self.assertEqual(session.data.time, time)
            for key, value in before.items():
                np.testing.assert_array_equal(
                    getattr(session.data, key), value, err_msg=key
                )

    def test_native_reset_and_ambiguous_clock_remain_explicit(self):
        for kind in ("native", "unknown"):
            s = load()
            task = FreeCouponReciprocation(s)
            if kind == "native":
                s.data.time = 1
                s.data.qfrc_applied[task.da[task.ids["probe_height"]]] = 1e25
            else:
                s.data.time = 1e30
            report = task.move_contact({}, 0.01, mode="access", settle=0)
            self.assertEqual((report["physics_steps"], report["checked_steps"]), (1, 0))
            self.assertIs(report["reset_performed"], True if kind == "native" else None)
            self.assertFalse(report["explicit_reset_performed"])
            self.assertIsNone(report["whole_leg_capped_clearance_lower_bound_m"])


if __name__ == "__main__":
    unittest.main()
