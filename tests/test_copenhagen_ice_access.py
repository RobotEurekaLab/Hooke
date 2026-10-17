"""Legacy revision1 dry guide regression; revision2 has separate contact tests."""

import itertools
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.layout_audit import (
    audit_layout,
    doorway_obstacle_warnings,
    model_obstacles,
)
from real_labs.runtime import InstrumentSession

SCENE = "copenhagen_pice_ice_workroom"
PREFIX = "ice_station__pice_"


def session():
    definition = scene(SCENE)
    next(e for e in definition["equipment"] if e["id"] == "ice_station")["params"][
        "interface_revision"
    ] = 1
    with patch("real_labs.builder.scene", return_value=definition):
        root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def moving_pairs(model):
    groups = {
        model.body_weldid[model.jnt_bodyid[model.joint("ice_station__joint_" + key).id]]
        for key in ("wheel_guard", "inert_core_carriage")
    }
    ids = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    weld = model.body_weldid[model.geom_bodyid]
    return np.array(
        [
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if weld[a] != weld[b] and (weld[a] in groups or weld[b] in groups)
        ],
        dtype=int,
    )


def checked_clearance(model, data, pairs):
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", abs(rotation), model.geom_aabb[:, 3:])
    a, b = pairs.T
    lower = np.linalg.norm(
        np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
    )
    return min(
        [0.002]
        + [
            float(mujoco.mj_geomDistance(model, data, int(a), int(b), 0.002, None))
            for a, b in pairs[lower < 0.002]
        ]
    )


class CopenhagenIceAccessTests(unittest.TestCase):
    def test_original_control_and_inert_core_contract(self):
        s = session()
        m = s.model
        for key, position, axis, limits, mass, inertia in (
            (
                "wheel_guard",
                (-0.285, -0.023, 1.48),
                (0, 0, -1),
                (0, 1.1),
                0.28,
                0.00168,
            ),
            (
                "inert_core_carriage",
                (-0.115, -0.22, 0.92),
                (0, 1, 0),
                (-0.075, 0.075),
                0.18,
                0.00108,
            ),
        ):
            c = s.controls["ice_station", key]
            body = m.jnt_bodyid[c.joint]
            np.testing.assert_allclose(m.body_pos[body], position)
            np.testing.assert_allclose(m.jnt_axis[c.joint], axis)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
            self.assertEqual(m.body_mass[body], mass)
            np.testing.assert_allclose(m.body_inertia[body], [inertia] * 3)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], 180)
            self.assertAlmostEqual(
                m.actuator_biasprm[c.actuator, 2], -2 * np.sqrt(180 * mass), places=6
            )
            np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
        body = m.body("ice_station__body_inert_core_carriage").id
        core = [
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] == body
            and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and m.geom_contype[i]
        ]
        self.assertEqual(len(core), 1)
        np.testing.assert_allclose(m.geom_size[core[0], :2], [0.034, 0.066])
        np.testing.assert_allclose(m.geom_pos[core[0]], [0, 0, 0.048])
        site = m.site(s.equipment["ice_station"]["sample_site"]).id
        np.testing.assert_allclose(m.site_pos[site], [0, 0, 0.048])
        self.assertEqual(s.equipment["ice_station"]["sample_attachment"], "clamped")

    def test_open_guides_have_continuous_support(self):
        s = session()
        m, d = s.model, s.data
        body = m.body("ice_station__body_inert_core_carriage").id
        ids = [
            i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_contype[i]
        ]
        adjacency = {i: set() for i in ids}
        for a, b in itertools.combinations(ids, 2):
            if mujoco.mj_geomDistance(m, d, a, b, 1e-7, None) <= 1e-8:
                adjacency[a].add(b)
                adjacency[b].add(a)
        seen, todo = {ids[0]}, [ids[0]]
        while todo:
            for item in adjacency[todo.pop()] - seen:
                seen.add(item)
                todo.append(item)
        self.assertEqual(seen, set(ids))
        for rail in range(2):
            a = m.geom(PREFIX + f"rail_{rail}").id
            self.assertEqual((m.geom_contype[a], m.geom_conaffinity[a]), (1, 1))
            for end in range(2):
                b = m.geom(PREFIX + f"rail_{rail}_support_{end}").id
                self.assertLessEqual(mujoco.mj_geomDistance(m, d, a, b, 0.1, None), 0)
                self.assertAlmostEqual(m.geom_pos[b, 2] - m.geom_size[b, 2], 0.876)
            for i in range(48):
                geom = m.geom(PREFIX + f"bearing_{rail}_contact_{i}").id
                self.assertEqual(
                    (m.geom_contype[geom], m.geom_conaffinity[geom]), (1, 1)
                )
                # The radial inner face is exactly 7.6 mm from the 7 mm rail.
                center = m.geom_pos[geom] - [(-0.105, 0.095)[rail], 0, -0.029]
                self.assertAlmostEqual(
                    np.linalg.norm(center[[0, 2]]) - m.geom_size[geom, 0],
                    0.0076,
                    delta=2e-8,
                )
        for y in (-0.075, 0.075):
            self.assertGreater(-0.22 + y - 0.024, -0.36 + 0.012)
            self.assertLess(-0.22 + y + 0.024, -0.055 - 0.012)

    def test_full_coupled_travel_checks_parent_filtered_pairs(self):
        s = session()
        m, d = s.model, s.data
        pairs = moving_pairs(m)
        guard, carriage = [
            s.controls["ice_station", k] for k in ("wheel_guard", "inert_core_carriage")
        ]
        for angle in np.linspace(0, 1.1, 9):
            for travel in np.linspace(-0.075, 0.075, 13):
                d.qpos[m.jnt_qposadr[guard.joint]] = angle
                d.qpos[m.jnt_qposadr[carriage.joint]] = travel
                mujoco.mj_forward(m, d)
                self.assertGreaterEqual(checked_clearance(m, d, pairs), 0.0005999)
        self.assertGreater(len(pairs), 10000)

    def test_actual_loaded_transit_and_return_without_reset(self):
        s = session()
        m, d = s.model, s.data
        controls = [
            s.controls["ice_station", k] for k in ("wheel_guard", "inert_core_carriage")
        ]
        home = [s._measured(c) for c in controls]
        pairs = moving_pairs(m)
        checked = 0
        for load in (-0.18, 0.18):
            d.qfrc_applied[m.jnt_dofadr[controls[1].joint]] = load
            for fraction in (0.15, 0.85, 0.25):
                for c in controls:
                    d.ctrl[c.actuator] = c.lower + fraction * (c.upper - c.lower)
                for _ in range(round(2 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    self.assertGreaterEqual(checked_clearance(m, d, pairs), 0.0005998)
                    checked += 1
                for c in controls:
                    self.assertLess(
                        abs(s._measured(c) - d.ctrl[c.actuator]),
                        0.025 * (c.upper - c.lower),
                    )
        d.qfrc_applied[:] = 0
        for c, value in zip(controls, home):
            d.ctrl[c.actuator] = value
        for _ in range(round(2 / m.opt.timestep)):
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            self.assertGreaterEqual(checked_clearance(m, d, pairs), 0.0005998)
            checked += 1
        for c, value in zip(controls, home):
            self.assertLess(abs(s._measured(c) - value), 1e-6)
        self.assertEqual(checked, round(14 / m.opt.timestep))
        self.assertAlmostEqual(d.time, 14, places=8)

    def test_end_door_approach_and_furniture_support(self):
        s = session()
        m, d = s.model, s.data
        self.assertTrue(audit_layout(s.manifest)["passed"])
        obstacles = model_obstacles(m, d)
        self.assertEqual(doorway_obstacle_warnings(s.manifest, obstacles), [])
        low, high = np.array([1.25, -0.75]), np.array([2.36, 0.35])
        blockers = [
            o["name"]
            for o in obstacles
            if np.all(
                np.minimum(high, o["maximum_xy"]) - np.maximum(low, o["minimum_xy"])
                > 1e-8
            )
        ]
        self.assertEqual(blockers, [])
        for key in ("left_bench", "right_bench", "near_table", "end_bench"):
            top = m.geom("pice_room_" + key + "_wood_top").id
            for side in (-1, 1):
                rail = m.geom("pice_room_" + key + f"_rail_{side}").id
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(m, d, top, rail, 0.1, None), 0, delta=1e-8
                )
        wall = m.geom("arch_cutaway_wall_right").id
        for suffix in (
            "freezer_closed_panel",
            "freezer_jamb_0",
            "freezer_jamb_1",
            "freezer_header",
            "freezer_exit_backing",
        ):
            geom = m.geom("arch_cutaway_wall_right_pice_" + suffix).id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, geom, wall, 0.1, None), 1e-8
            )
        self.assertEqual(s.manifest["room"]["size"], [5, 4.5, 2.9])

    def test_context_camera_observes_the_closed_end_door(self):
        s = session()
        m, d = s.model, s.data
        view = s.manifest["camera"]["reference"]
        origin = np.asarray(view["position"], dtype=float)
        direction = np.asarray(view["target"]) - origin
        direction /= np.linalg.norm(direction)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(m, d, origin, direction, None, 1, -1, hit)
        self.assertGreater(distance, 0)
        self.assertEqual(
            m.geom(int(hit[0])).name,
            "arch_cutaway_wall_right_pice_freezer_closed_panel",
        )
        self.assertEqual(
            s.manifest["camera"]["interior"]["position"], [0.65, -1.82, 1.65]
        )
        self.assertEqual(
            s.manifest["camera"]["workstation"]["position"], [0.76, -0.1, 1.69]
        )
        # This geometric ray is not an image-contrast or calibrated camera claim.


if __name__ == "__main__":
    unittest.main()
