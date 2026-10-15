"""Mechanical support and clearance of the explicitly authored dry fixture."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.imperial_fixture import RUNNING_CLEARANCE_M
from real_labs.runtime import InstrumentSession


PREFIX = "thinfilm_cluster__dry_fixture_"


def session():
    root, manifest = build_scene("imperial_thinfilm_bessemer")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def named(model, key):
    return model.geom(PREFIX + key).id


def distance(s, a, b):
    return mujoco.mj_geomDistance(s.model, s.data, a, b, 10, None)


def fixed_at(model, pos, half):
    body = model.body("thinfilm_cluster").id
    return next(
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and np.allclose(model.geom_pos[g], pos)
        and np.allclose(model.geom_size[g], half)
    )


class ImperialFixtureGeometryTests(unittest.TestCase):
    def test_original_axes_reference_pose_sample_and_cameras(self):
        s = session()
        m = s.model
        self.assertEqual((m.njnt, m.nu), (2, 2))
        door = s.controls["thinfilm_cluster", "access_door"]
        carrier = s.controls["thinfilm_cluster", "coupon_x"]
        np.testing.assert_allclose([door.lower, door.upper], [0, 1.2])
        np.testing.assert_allclose([carrier.lower, carrier.upper], [-0.12, 0.12])
        np.testing.assert_allclose(m.qpos0, [0.72, 0])
        np.testing.assert_allclose(
            m.body_pos[m.jnt_bodyid[carrier.joint]], [0.94, -0.73, 1.015]
        )
        site = m.site("thinfilm_cluster__site_inert_thinfilm_coupon").id
        np.testing.assert_allclose(s.data.site_xpos[site], [0.94, -0.73, 1.057])
        np.testing.assert_allclose(
            m.geom_size[named(m, "coupon")], [0.049, 0.045, 0.007]
        )
        np.testing.assert_allclose(m.geom_pos[named(m, "coupon")], [0, 0, 0.032])
        self.assertTrue(
            {"overview", "interior", "workstation", "chambers"}.issubset(
                {m.camera(i).name for i in range(m.ncam)}
            )
        )
        self.assertIn(
            "not copied",
            s.equipment["thinfilm_cluster"]["dry_fixture_adaptations"]["basis"],
        )

    def test_table_chamber_rail_bearing_load_path_through_travel(self):
        s = session()
        m = s.model
        table = fixed_at(m, (0.945, -0.735, 0.865), (0.41, 0.29, 0.025))
        floor = fixed_at(m, (0.945, -0.73, 0.921), (0.355, 0.245, 0.02))
        plate = named(m, "carrier_plate")
        for side, x in enumerate((0.615, 1.275)):
            runner = named(m, f"chamber_runner_{side}")
            wall = fixed_at(m, (x, -0.73, 1.08), (0.025, 0.225, 0.18))
            self.assertAlmostEqual(distance(s, runner, table), 0, places=6)
            self.assertAlmostEqual(distance(s, runner, wall), 0, places=6)
            self.assertLess(distance(s, wall, floor), 0)
        for row, end in itertools.product(range(2), repeat=2):
            pedestal = named(m, f"rail_{row}_pedestal_{end}")
            guide = named(m, f"guide_{row}")
            self.assertAlmostEqual(distance(s, pedestal, floor), 0, places=6)
            self.assertAlmostEqual(distance(s, pedestal, guide), 0, places=6)
            self.assertEqual(m.geom_contype[guide], 1)
        control = s.controls["thinfilm_cluster", "coupon_x"]
        for q in np.linspace(-0.12, 0.12, 49):
            s.data.qpos[m.jnt_qposadr[control.joint]] = q
            mujoco.mj_forward(m, s.data)
            for row, shoe in itertools.product(range(2), repeat=2):
                proxies = [
                    g
                    for g in range(m.ngeom)
                    if m.geom(g).name.startswith(
                        PREFIX + f"bearing_{row}_{shoe}_proxy_"
                    )
                ]
                self.assertEqual(len(proxies), 24)
                self.assertLessEqual(min(distance(s, plate, g) for g in proxies), 1e-6)
                self.assertAlmostEqual(
                    min(distance(s, named(m, f"guide_{row}"), g) for g in proxies),
                    RUNNING_CLEARANCE_M,
                    places=6,
                )
                self.assertGreater(min(distance(s, floor, g) for g in proxies), 0.02)

    def test_coupon_shim_and_clamp_pads_retain_unchanged_geometry(self):
        s = session()
        m = s.model
        plate, coupon, shim = [
            named(m, key) for key in ("carrier_plate", "coupon", "coupon_shim")
        ]
        self.assertAlmostEqual(distance(s, plate, shim), 0, places=7)
        self.assertAlmostEqual(distance(s, shim, coupon), 0, places=7)
        for side in (-1, 1):
            clamp = named(m, f"clamp_{side}")
            foot = named(m, f"clamp_foot_{side}")
            pad = named(m, f"retaining_pad_{side}")
            tab = named(m, f"hold_down_{side}")
            for first, second in (
                (plate, foot),
                (foot, clamp),
                (pad, clamp),
                (tab, coupon),
            ):
                self.assertAlmostEqual(distance(s, first, second), 0, places=7)
            # MuJoCo's distance query can choose the wrong EPA face for these
            # coplanar, equally tall boxes (returning their full 14 mm height).
            # Their exact aligned-box separating-axis intervals prove contact
            # without relaxing a penetration tolerance or shifting the coupon.
            for geom in (pad, coupon):
                np.testing.assert_allclose(
                    s.data.geom_xmat[geom].reshape(3, 3), np.eye(3)
                )
            interval_gaps = (
                np.abs(s.data.geom_xpos[pad] - s.data.geom_xpos[coupon])
                - m.geom_size[pad]
                - m.geom_size[coupon]
            )
            self.assertAlmostEqual(interval_gaps[0], 0, places=12)
            self.assertTrue(np.all(interval_gaps[1:] < 0))
            self.assertLess(distance(s, tab, clamp), 0)
            self.assertEqual(int(m.geom_bodyid[pad]), int(m.geom_bodyid[coupon]))

    def test_hinge_and_handle_connect_without_a_solid_pin_bore(self):
        s = session()
        m = s.model
        bracket, pin, plate, handle = [
            named(m, k)
            for k in ("hinge_bracket", "hinge_pin", "door_plate", "door_handle")
        ]
        wall = fixed_at(m, (0.615, -0.73, 1.08), (0.025, 0.225, 0.18))
        self.assertLess(distance(s, bracket, wall), 0)
        for side in (-1, 1):
            lug = named(m, f"hinge_lug_{side}")
            self.assertLess(distance(s, lug, bracket), 0)
            self.assertLess(distance(s, lug, pin), 0)
            foot, post = [
                named(m, f"{key}_{side}") for key in ("handle_base", "handle_standoff")
            ]
            self.assertLess(distance(s, plate, foot), 0)
            self.assertLess(distance(s, foot, post), 0)
            self.assertLess(distance(s, post, handle), 0)
        control = s.controls["thinfilm_cluster", "access_door"]
        for q in np.linspace(0, 1.2, 65):
            s.data.qpos[m.jnt_qposadr[control.joint]] = q
            mujoco.mj_forward(m, s.data)
            for side in (-1, 1):
                proxies = [
                    g
                    for g in range(m.ngeom)
                    if m.geom(g).name.startswith(PREFIX + f"hinge_hub_{side}_proxy_")
                ]
                self.assertAlmostEqual(
                    min(distance(s, pin, g) for g in proxies),
                    RUNNING_CLEARANCE_M,
                    places=6,
                )
                self.assertLess(min(distance(s, plate, g) for g in proxies), 0)

    def test_direct_full_travel_and_combined_states_preserve_clearance(self):
        s = session()
        m = s.model
        controls = list(s.controls.values())
        bodies = {m.jnt_bodyid[c.joint] for c in controls}
        moving = np.array(
            [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] in bodies and m.geom_contype[g]
            ]
        )
        fixed = np.array(
            [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] not in bodies and m.geom_contype[g]
            ]
        )
        states = [(q, 0) for q in np.linspace(0, 1.2, 65)]
        states += [(0.72, q) for q in np.linspace(-0.12, 0.12, 65)]
        states += list(
            itertools.product(np.linspace(0, 1.2, 9), np.linspace(-0.12, 0.12, 9))
        )
        for values in states:
            s.reset()
            for control, value in zip(controls, values):
                s.data.qpos[m.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(m, s.data)
            self.assertFalse([c for c in s.data.contact if c.dist < -1e-5], values)
            for first in moving:
                candidates = np.concatenate(
                    (fixed, moving[m.geom_bodyid[moving] != m.geom_bodyid[first]])
                )
                centers = np.linalg.norm(
                    s.data.geom_xpos[candidates] - s.data.geom_xpos[first], axis=1
                )
                near = candidates[
                    centers < m.geom_rbound[candidates] + m.geom_rbound[first] + 0.001
                ]
                for second in near:
                    gap = distance(s, int(first), int(second))
                    self.assertGreaterEqual(
                        gap,
                        -1e-5,
                        (values, m.geom(first).name, m.geom(second).name, gap),
                    )

    def test_open_bearing_meshes_and_door_access_are_not_filled_surfaces(self):
        s = session()
        m = s.model
        # Check the actual visual triangle meshes, not only hollow collision
        # proxies. A convex-hull presentation would otherwise hide the bore.
        for row, shoe in itertools.product(range(2), repeat=2):
            g = named(m, f"bearing_{row}_{shoe}_shell")
            start = np.array([0.60, (-0.80, -0.65)[row], 0.970])
            self.assertEqual(
                mujoco.mj_rayMesh(m, s.data, g, start, np.array([1.0, 0, 0])), -1
            )
        for side in (-1, 1):
            g = named(m, f"hinge_hub_{side}_shell")
            self.assertEqual(
                mujoco.mj_rayMesh(
                    m,
                    s.data,
                    g,
                    np.array([0.583, -0.993, 0.80]),
                    np.array([0.0, 0, 1.0]),
                ),
                -1,
            )
        m.geom_group[:] = ((m.geom_contype != 0) | (m.geom_conaffinity != 0)).astype(
            np.int32
        )
        group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        hit = np.array([-1], dtype=np.int32)
        control = s.controls["thinfilm_cluster", "access_door"]
        for value, expected in ((0, "door_plate"), (1.2, "coupon")):
            s.data.qpos[m.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(m, s.data)
            gap = mujoco.mj_ray(
                m,
                s.data,
                np.array([0.94, -1.60, 1.045]),
                np.array([0.0, 1.0, 0]),
                group,
                True,
                -1,
                hit,
            )
            self.assertGreater(gap, 0)
            self.assertEqual(hit[0], named(m, expected))

    def test_actual_reversals_hold_other_axis_and_reset_reference_pose(self):
        s = session()
        m = s.model
        for key, control in s.controls.items():
            s.reset()
            held = {k: s._measured(c) for k, c in s.controls.items() if k != key}
            for fraction in (0.15, 0.85, 0.25, 0, 1, 0.5):
                target = control.lower + fraction * (control.upper - control.lower)
                s.data.ctrl[control.actuator] = target
                for _ in range(round(2 / m.opt.timestep)):
                    mujoco.mj_step(m, s.data)
                    self.assertFalse([c for c in s.data.contact if c.dist < -1e-5], key)
                    self.assertTrue(np.isfinite(s.data.qpos).all())
                self.assertLess(
                    abs(s._measured(control) - target),
                    0.025 * (control.upper - control.lower),
                )
                for other, value in held.items():
                    self.assertLess(abs(s._measured(s.controls[other]) - value), 1e-5)
            self.assertFalse(any(w.number for w in s.data.warning))
        s.reset()
        np.testing.assert_array_equal(s.data.qpos, m.qpos0)
        np.testing.assert_allclose(s.data.ctrl, [0.72, 0])
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
