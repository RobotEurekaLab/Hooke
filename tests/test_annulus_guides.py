"""Open profiler bearings: source contract, exact clearance and loaded motion."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession

SCENE = "anu_climate_fluid_physics"
EQUIPMENT = "large_rotating_annulus"
PREFIX = EQUIPMENT + "__annulus_guide_"


def session():
    root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def profiler_pairs(model):
    body = model.body(EQUIPMENT + "__body_probe_height").id
    group = model.body_weldid[body]
    ids = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    moving = [int(i) for i in ids if model.body_weldid[model.geom_bodyid[i]] == group]
    other = [int(i) for i in ids if model.body_weldid[model.geom_bodyid[i]] != group]
    return np.array([(a, b) for a in moving for b in other], dtype=int)


def minimum_distance(model, data, pairs):
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


def analytic_rail_box_clearance(model, data):
    """Exact parallel rail-segment/box distance, independent of GJK queries."""
    body = model.body(EQUIPMENT + "__body_probe_height").id
    boxes = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == body
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    gaps = []
    for index in range(2):
        rail = model.geom(PREFIX + f"rail_{index}").id
        axis = data.geom_xmat[rail].reshape(3, 3)[:, 2]
        ends = data.geom_xpos[rail] + np.outer((-1, 1), axis * model.geom_size[rail, 1])
        for box in boxes:
            local = (ends - data.geom_xpos[box]) @ data.geom_xmat[box].reshape(3, 3)
            np.testing.assert_allclose(local[0, :2], local[1, :2], atol=1e-12)
            delta = np.maximum(abs(local[0, :2]) - model.geom_size[box, :2], 0)
            dz = max(
                min(local[:, 2]) - model.geom_size[box, 2],
                -model.geom_size[box, 2] - max(local[:, 2]),
                0,
            )
            gaps.append(np.linalg.norm([*delta, dz]) - model.geom_size[rail, 0])
    return min(gaps)


class AnnulusGuideTests(unittest.TestCase):
    def test_preserved_axis_drive_mass_and_tip(self):
        s = session()
        m, d = s.model, s.data
        c = s.controls[EQUIPMENT, "probe_height"]
        body = m.jnt_bodyid[c.joint]
        np.testing.assert_allclose(m.body_pos[body], [0.625, 0.32, 0.77])
        np.testing.assert_allclose(m.jnt_axis[c.joint], [0, 0, 1])
        np.testing.assert_allclose(m.jnt_range[c.joint], [-0.07, 0.17])
        self.assertEqual(m.body_mass[body], 0.08)
        np.testing.assert_allclose(m.body_inertia[body], [0.00048] * 3)
        self.assertEqual(m.actuator_gainprm[c.actuator, 0], 220)
        self.assertAlmostEqual(
            m.actuator_biasprm[c.actuator, 2], -2 * np.sqrt(220 * 0.08), places=7
        )
        np.testing.assert_allclose(m.actuator_forcerange[c.actuator], [-120, 120])
        stylus = m.geom(PREFIX + "stylus").id
        np.testing.assert_allclose(m.geom_pos[stylus], [-0.2, 0, -0.245])
        np.testing.assert_allclose(m.geom_size[stylus, :2], [0.002, 0.245])
        self.assertEqual((m.geom_contype[stylus], m.geom_conaffinity[stylus]), (1, 1))
        tip = m.site(EQUIPMENT + "__site_probe_tip").id
        target = m.site(EQUIPMENT + "__site_calibration_target").id
        np.testing.assert_allclose(d.site_xpos[tip, :2], d.site_xpos[target, :2])
        np.testing.assert_allclose(m.site_pos[tip], [-0.2, 0, -0.492])
        self.assertEqual(s.equipment[EQUIPMENT]["profiling_guide"]["revision"], 2)

    def test_connected_moving_support_and_fixed_rail_mounts(self):
        s = session()
        m, d = s.model, s.data
        body = m.body(EQUIPMENT + "__body_probe_height").id
        ids = [
            i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_contype[i]
        ]
        adjacency = {i: set() for i in ids}
        for index, a in enumerate(ids):
            for b in ids[index + 1 :]:
                if mujoco.mj_geomDistance(m, d, a, b, 1e-7, None) <= 1e-8:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        seen, todo = {ids[0]}, [ids[0]]
        while todo:
            for item in adjacency[todo.pop()] - seen:
                seen.add(item)
                todo.append(item)
        self.assertEqual(seen, set(ids))
        for rail in ("rail_0", "rail_1"):
            a = m.geom(PREFIX + rail).id
            self.assertEqual((m.geom_contype[a], m.geom_conaffinity[a]), (1, 1))
            for support in ("rim_bracket", "top_bridge"):
                b = m.geom(PREFIX + support).id
                self.assertLessEqual(
                    mujoco.mj_geomDistance(m, d, a, b, 0.1, None), 1e-8
                )
        bracket = m.geom(PREFIX + "rim_bracket").id
        tank = m.body(EQUIPMENT + "__body_tank_rotation").id
        walls = [
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] == tank
            and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            and np.isclose(m.geom_pos[i, 2], 0.25)
            and np.isclose(m.geom_size[i, 2], 0.20)
        ]
        self.assertTrue(walls)
        self.assertLessEqual(
            min(mujoco.mj_geomDistance(m, d, bracket, i, 0.1, None) for i in walls),
            1e-8,
        )

    def test_full_coupled_travel_has_real_bores_and_contact_clearance(self):
        s = session()
        m, d = s.model, s.data
        pairs = profiler_pairs(m)
        rotation = s.controls[EQUIPMENT, "tank_rotation"]
        height = s.controls[EQUIPMENT, "probe_height"]
        for angle in np.linspace(-0.5, 0.5, 9):
            for z in np.linspace(-0.07, 0.17, 17):
                d.qpos[m.jnt_qposadr[rotation.joint]] = angle
                d.qpos[m.jnt_qposadr[height.joint]] = z
                mujoco.mj_forward(m, d)
                self.assertGreaterEqual(minimum_distance(m, d, pairs), 0.00059999)
                self.assertAlmostEqual(
                    analytic_rail_box_clearance(m, d), 0.0006, delta=2e-8
                )
                # Full bearing length stays inside the supported rail span.
                self.assertGreaterEqual(0.77 + z - 0.032, 0.45)
                self.assertLessEqual(0.77 + z + 0.032, 1.0)

    def test_actual_loaded_bidirectional_transit_preserves_clearance(self):
        s = session()
        m, d = s.model, s.data
        pairs = profiler_pairs(m)
        height = s.controls[EQUIPMENT, "probe_height"]
        rotation = s.controls[EQUIPMENT, "tank_rotation"]
        checked = 0
        # Explicit small opposing/assisting force on an ideal authored axis;
        # not a real profiler load rating, fluid drag, or measurement model.
        for load in (-0.15, 0.15):
            d.qfrc_applied[m.jnt_dofadr[height.joint]] = load
            for fraction in (0.15, 0.85, 0.25):
                for c in (height, rotation):
                    d.ctrl[c.actuator] = c.lower + fraction * (c.upper - c.lower)
                for _ in range(round(2 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    self.assertGreaterEqual(minimum_distance(m, d, pairs), 0.0005999)
                    checked += 1
                for c in (height, rotation):
                    self.assertLessEqual(
                        abs(s._measured(c) - d.ctrl[c.actuator]),
                        0.025 * (c.upper - c.lower),
                    )
        self.assertEqual(checked, round(12 / m.opt.timestep))
        self.assertAlmostEqual(d.time, 12, places=8)


if __name__ == "__main__":
    unittest.main()
