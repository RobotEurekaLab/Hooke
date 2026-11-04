"""Dry traverse, mounted observation and independent complete-pair checks."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.sydney_tank_motion import TankInspection, SCENE, EQUIPMENT


class SydneyTankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_axes_drives_plate_scale_and_world_views(self):
        m = self.model
        np.testing.assert_array_equal(m.jnt_range, [[-0.55, 0.55], [-0.09, 0.09]])
        np.testing.assert_array_equal(m.jnt_axis, [[1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(m.actuator_gainprm[:, 0], [450, 180])
        np.testing.assert_array_equal(
            m.body(EQUIPMENT + "__body_probe_x").pos, [0, 0.28, 1.88]
        )
        np.testing.assert_array_equal(
            m.body(EQUIPMENT + "__body_probe_height").pos, [0, -0.28, 0]
        )
        np.testing.assert_allclose(
            m.geom(EQUIPMENT + "__sydney_calibration_plate").size * 2,
            [0.030, 0.002, 0.020],
        )
        self.assertTrue(
            {"overview", "workstation", "interior"} <= self.manifest["camera"].keys()
        )

    def test_independent_unculled_geometry_across_full_combined_travel(self):
        s = self.session()
        m, d = s.model, s.data
        ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        pairs = [
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
        ]
        task = TankInspection(s)
        # Independent list covers all enabled cross-weld geometry without AABB culling.
        self.assertEqual(set(pairs), {tuple(p) for p in task.pairs})
        for q in product(np.linspace(-0.55, 0.55, 5), np.linspace(-0.09, 0.09, 5)):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            direct = min(
                mujoco.mj_geomDistance(m, d, a, b, 0.002, None) for a, b in pairs
            )
            self.assertGreater(direct, 0.00059)
            task.reset_counts()
            task.check(m, d)
            self.assertAlmostEqual(direct, task.minimum, places=8)

    def test_guides_remain_engaged_and_target_has_a_mount(self):
        s = self.session()
        m, d = s.model, s.data
        prefix = EQUIPMENT + "__sydney_"
        for height in (-0.09, 0.09):
            d.qpos[1] = height
            mujoco.mj_forward(m, d)
            shaft = m.geom(prefix + "probe_shaft").id
            z = d.geom_xpos[shaft, 2]
            half = m.geom_size[shaft, 1]
            # Shaft extends beyond both ends of the 140 mm guide at both limits.
            guide_center = 1.95
            self.assertLess(z - half, guide_center - 0.07)
            self.assertGreater(z + half, guide_center + 0.07)
            mount = m.geom(prefix + "plate_mount").id
            plate = m.geom(prefix + "calibration_plate").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, shaft, mount, 0.01, None), 1e-7
            )
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, plate, mount, 0.01, None), 1e-7
            )

    def test_actual_camera_sees_real_marker_without_mutating_live_model(self):
        s = self.session()
        task = TankInspection(s)
        before = _model_signature(s.model)
        for q in ((0, 0), (-0.20, 0.035), (0.20, -0.035)):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            result = task.observe()
            self.assertTrue(result["visible"], result)
            self.assertEqual(
                result["first_opaque_hit"], EQUIPMENT + "__sydney_plate_marker"
            )
        self.assertEqual(_model_signature(s.model), before)

    def test_tilted_camera_has_a_continuous_physical_support_to_the_feet(self):
        s = self.session()
        m, d = s.model, s.data
        ids = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(EQUIPMENT + "__sydney_camera_")
        ]
        self.assertGreaterEqual(len(ids), 12)
        self.assertTrue(all(m.geom_contype[g] for g in ids))
        connected = {m.geom(EQUIPMENT + "__sydney_camera_foot_-0.12").id}
        pending = set(ids) - connected
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
            connected.update(attached)
            pending.difference_update(attached)
        self.assertFalse(pending, [m.geom(g).name for g in pending])
        camera = m.camera(EQUIPMENT + "__inspection_camera").id
        np.testing.assert_allclose(d.cam_xpos[camera], [0, -0.8129, 1.6], atol=1e-9)
        axis = -d.cam_xmat[camera].reshape(3, 3)[:, 2]
        direction = np.array([0, 0.1986, 1.23]) - d.cam_xpos[camera]
        np.testing.assert_allclose(
            axis, direction / np.linalg.norm(direction), atol=1e-9
        )

    def test_tilted_view_retains_actual_full_travel_visibility(self):
        s = self.session()
        task = TankInspection(s)
        for q in product(np.linspace(-0.55, 0.55, 5), np.linspace(-0.09, 0.09, 5)):
            s.data.qpos[:] = q
            mujoco.mj_forward(s.model, s.data)
            result = task.observe()
            self.assertTrue(result["visible"], (q, result))

    def test_measured_round_trip_and_every_step_direct_clearance(self):
        s = self.session()
        before = _model_signature(s.model)
        result = TankInspection(s).run()
        self.assertEqual(result["status"], "succeeded", result)
        self.assertFalse(result["reset_performed"])
        self.assertEqual(len(result["observations"]), 3)
        self.assertGreater(s.data.time, 40)
        for phase in result["phases"]:
            coverage = phase["direct_clearance"]
            self.assertTrue(coverage["coverage_complete"])
            self.assertEqual(coverage["checked_steps"], phase["physics_steps"])
            self.assertGreater(coverage["whole_leg_clearance_lower_bound_m"], 0.00059)
        np.testing.assert_allclose(
            result["measured_return_error_m"], [0, 0], atol=0.0002
        )
        self.assertEqual(_model_signature(s.model), before)

    def test_foreign_or_modified_model_metadata_and_state_rejected(self):
        changes = (
            lambda s: s.model.geom_contype.fill(0),
            lambda s: s.model.actuator_gainprm.__setitem__((0, 0), 999),
            lambda s: s.manifest.__setitem__("scene_sha256", "forged"),
            lambda s: s.equipment[EQUIPMENT]["inspection_interface"].__setitem__(
                "guide_clearance_m", 0.1
            ),
            lambda s: setattr(s, "data", mujoco.MjData(copy(s.model))),
        )
        for change in changes:
            s = self.session()
            change(s)
            before = s.data.qpos.copy()
            with self.assertRaises(ValueError):
                TankInspection(s)
            np.testing.assert_array_equal(s.data.qpos, before)
        s = self.session()
        task = TankInspection(s)
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            task.move((0.01, 0))

    def test_forced_failure_requests_hold_without_teleport_or_false_coverage(self):
        s = self.session()
        task = TankInspection(s)
        s.data.qfrc_applied[0] = 500
        result = task.move((0.02, 0))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertGreater(s.data.time, 0)
        self.assertNotEqual(s.data.qpos[0], 0)
        coverage = result["direct_clearance"]
        self.assertEqual(
            coverage["unchecked_steps"],
            result["physics_steps"] - coverage["checked_steps"],
        )
        np.testing.assert_allclose(s.data.ctrl, s.data.qpos, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
