"""Open supports, a truly mounted camera, and actual dry inspection failures."""

from copy import copy, deepcopy
import itertools
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.export import _portable_resources
from real_labs.mit_rotating_geometry import CAMERA
from real_labs.mit_rotating_motion import RotatingInspection, inspect_mit, _binding
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class MitRotatingInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("mit_paoc_rotating_fluids")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_frames_ranges_sites_and_real_camera_attachment(self):
        s = self.session()
        m, d = s.model, s.data
        for key, pos, axis, limits in (
            ("table_rotation", (0, 0, 0.855), (0, 0, 1), (-np.pi, np.pi)),
            ("camera_traverse", (0, 0, 1.85), (1, 0, 0), (-0.34, 0.34)),
        ):
            c = s.controls["rotating_table", key]
            np.testing.assert_allclose(
                m.body_pos[m.jnt_bodyid[c.joint]], pos, atol=1e-10
            )
            np.testing.assert_allclose(m.jnt_axis[c.joint], axis, atol=1e-10)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits, atol=5e-8)
        for key, pos in (
            ("calibration_target", (0.22, 0, 0.051)),
            ("camera_center", (0, 0, -0.184)),
        ):
            np.testing.assert_allclose(
                m.site_pos[m.site("rotating_table__site_" + key).id], pos
            )
        camera = m.camera(CAMERA).id
        c = s.controls["rotating_table", "camera_traverse"]
        self.assertEqual(m.cam_bodyid[camera], m.jnt_bodyid[c.joint])
        self.assertIn(CAMERA, s.manifest["camera_names"])
        self.assertNotIn(CAMERA, s.manifest["camera"])
        start = d.cam_xpos[camera].copy()
        d.qpos[m.jnt_qposadr[c.joint]] = 0.23
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.cam_xpos[camera] - start, (0.23, 0, 0), atol=1e-12)
        np.testing.assert_allclose(
            d.cam_xmat[camera].reshape(3, 3)[:, 2], (0, 0, 1), atol=1e-12
        )
        self.assertEqual(m.cam_fovy[camera], 45)

    def test_enabled_open_bearings_have_real_clearance_and_continuous_support(self):
        s = self.session()
        m, d = s.model, s.data

        def g(key):
            return m.geom("rotating_table__mit_" + key).id

        spindle = g("spindle")
        rail = g("camera_rail")
        platter = g("rotating_platter")
        for i in range(48):
            proxy = g("spindle_bearing_proxy_" + str(i))
            self.assertEqual((m.geom_contype[proxy], m.geom_conaffinity[proxy]), (1, 1))
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, spindle, proxy, 0.01, None),
                0.0006,
                places=7,
            )
        for name in ("crossbar_-1", "crossbar_1", "side_-1", "side_1"):
            proxy = g("camera_bearing_" + name)
            self.assertEqual((m.geom_contype[proxy], m.geom_conaffinity[proxy]), (1, 1))
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, rail, proxy, 0.01, None), 0.0006, places=7
            )
        self.assertLess(mujoco.mj_geomDistance(m, d, spindle, platter, 0.01, None), 0)
        self.assertAlmostEqual(d.geom_xpos[spindle, 2] - m.geom_size[spindle, 1], 0.781)
        case = g("camera_case")
        self.assertAlmostEqual(d.geom_xpos[case, 2] + m.geom_size[case, 2], 1.80)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m,
            d,
            np.array((-0.53, 0.25, 1.85)),
            np.array((1.0, 0, 0)),
            None,
            1,
            m.body("rotating_table").id,
            hit,
        )
        self.assertTrue(distance < 0 or distance > 0.30)

    def test_static_full_range_pruning_keeps_every_near_pair(self):
        s = self.session()
        m, d = s.model, s.data
        run = RotatingInspection(s)
        roots = {int(m.jnt_bodyid[c.joint]) for c in run.controls}
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        pairs = [
            (int(a), int(b))
            for i, a in enumerate(enabled)
            for b in enabled[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in roots
                or m.body_weldid[m.geom_bodyid[b]] in roots
            )
        ]
        kept = {tuple(x) for x in run.pairs.tolist()}
        minimum = 0.01
        for qr, qc in itertools.product(
            np.linspace(-np.pi, np.pi, 9), np.linspace(-0.34, 0.34, 5)
        ):
            d.qpos[run.q] = (qr, qc)
            mujoco.mj_forward(m, d)
            for a, b in pairs:
                value = mujoco.mj_geomDistance(m, d, a, b, 0.003, None)
                if value < 0.003:
                    self.assertIn((a, b), kept)
                    minimum = min(minimum, value)
                    self.assertGreaterEqual(
                        value, 0.00059, (m.geom(a).name, m.geom(b).name, qr, qc)
                    )
        self.assertAlmostEqual(minimum, 0.0006, places=7)

    def test_source_context_has_grounded_computer_cart_and_supported_projector(self):
        s = self.session()
        m, d = s.model, s.data

        def gap(first, second):
            return mujoco.mj_geomDistance(
                m, d, m.geom(first).id, m.geom(second).id, 1, None
            )

        for a, b in (
            ("mit_cart_round_shelf_0.205", "mit_cart_side_-1"),
            ("mit_cart_round_shelf_0.86", "mit_cart_side_-1"),
            ("mit_cart_round_shelf_0.205", "mit_cart_computer"),
            ("mit_cart_round_shelf_0.86", "mit_monitor_base"),
            ("mit_monitor_base", "mit_monitor_stem"),
            ("mit_overhead_bridge", "mit_projector_mount"),
            ("mit_projector_mount", "mit_projector"),
            ("mit_overhead_rail_-0.85", "mit_overhead_bridge"),
        ):
            self.assertAlmostEqual(gap(a, b), 0, places=7, msg=(a, b))
        self.assertFalse(
            any(m.geom(i).name == "mit_cart_camera_boom" for i in range(m.ngeom))
        )
        green = m.material("rotating_table__mat_paoc_green").id
        self.assertGreater(m.mat_rgba[green, 1], m.mat_rgba[green, 0])

    def test_actual_index_inspect_return_observes_every_step_and_two_distinct_sensor_poses(
        self,
    ):
        s = self.session()
        signature = _model_signature(s.model)
        r = inspect_mit(s)
        self.assertTrue(r["success"], r["failure"])
        self.assertEqual(r["physics_steps"], r["completed_checks"])
        self.assertEqual(r["geometry_checks"], r["physics_steps"])
        self.assertTrue(r["coverage_complete"])
        self.assertGreater(r["physics_steps"], 15000)
        self.assertGreater(r["whole_run_clearance_lower_bound_m"], 0.00059)
        self.assertEqual(len(r["captures"]), 2)
        for capture in r["captures"]:
            self.assertTrue(capture["visible"])
            self.assertLess(capture["axis_error_m"], 0.0005)
            self.assertGreater(capture["target_diameter_px"], 90)
            np.testing.assert_allclose(capture["target_pixel"], (560, 350), atol=0.5)
            self.assertEqual(len(capture["state"]["qpos"]), s.model.nq)
        first, second = r["captures"]
        self.assertAlmostEqual(
            first["camera_position_m"][0] - second["camera_position_m"][0],
            0.44,
            delta=0.0002,
        )
        self.assertGreater(second["state"]["qpos"][0], 3.14)
        self.assertTrue(all(p["success"] for p in r["phases"]))
        self.assertFalse(r["reset_performed"])
        self.assertFalse(r["hold_requested"])
        self.assertEqual(signature, _model_signature(s.model))
        self.assertLess(r["return_error"][0], 0.002)
        self.assertLess(r["return_error"][1], 0.0001)

    def test_canonical_hardware_physics_camera_and_foreign_state_reject_before_commands(
        self,
    ):
        for change in ("guide", "camera", "physics", "mask", "foreign", "metadata"):
            s = self.session()
            m, d = s.model, s.data
            if change == "guide":
                m.geom_pos[m.geom("rotating_table__mit_camera_rail").id, 1] += 0.25
            elif change == "camera":
                m.cam_pos[m.camera(CAMERA).id, 0] += 0.1
            elif change == "physics":
                m.opt.gravity[2] = 0
            elif change == "mask":
                m.geom_contype[m.geom("rotating_table__mit_spindle").id] = 0
            elif change == "foreign":
                s.data = mujoco.MjData(copy(m))
            else:
                s.manifest["mounted_cameras"] = []
            d = s.data
            flags = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(d.model, flags))
            mujoco.mj_getState(d.model, d, before, flags)
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspect_mit(s)
            after = np.empty_like(before)
            mujoco.mj_getState(d.model, d, after, flags)
            np.testing.assert_array_equal(after, before)
        s = self.session()
        run = RotatingInspection(s)
        old = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            run.execute()
        self.assertEqual(old.time, 0)
        self.assertEqual(s.data.time, 0)

    def test_actual_external_load_abort_has_partial_coverage_and_hold_without_reset(
        self,
    ):
        for axis, force in ((0, 1000), (1, 250)):
            s = self.session()
            run = RotatingInspection(s)
            s.data.qfrc_applied[run.v[axis]] = force
            signature = _model_signature(s.model)
            r = run.execute()
            self.assertFalse(r["success"])
            self.assertTrue(r["hold_requested"])
            self.assertFalse(r["reset_performed"])
            self.assertFalse(r["coverage_complete"])
            self.assertEqual(r["physics_steps"], r["completed_checks"] + 1)
            self.assertIsNone(r["whole_run_clearance_lower_bound_m"])
            self.assertEqual(signature, _model_signature(s.model))
            np.testing.assert_allclose(
                s.data.ctrl[run.a], np.clip(s.data.qpos[run.q], run.lower, run.upper)
            )
            self.assertGreater(s.data.time, 0)
            self.assertFalse(r["phases"][-1]["success"])

    def test_portable_binding_accepts_actual_nested_camera(self):
        root = deepcopy(self.root)
        resources = _portable_resources(root)
        m = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"),
            assets={r["file"]: Path(r["source"]).read_bytes() for r in resources},
        )
        manifest = deepcopy(self.manifest)
        manifest["scene_sha256"] = _binding()["scene_sha256"]
        run = RotatingInspection(InstrumentSession(m, manifest))
        self.assertEqual(
            run.camera_body, m.body("rotating_table__body_camera_traverse").id
        )

    def test_obstacle_and_unsafe_start_are_rejected_without_hold_or_reset(self):
        root = deepcopy(self.root)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="inspection_obstacle",
            type="box",
            pos="-.18 .25 1.45",
            size=".03 .03 .03",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            deepcopy(self.manifest),
        )
        before = s.data.ctrl.copy(), s.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "binding changed"):
            inspect_mit(s)
        self.assertEqual(s.data.time, 0)
        np.testing.assert_array_equal(s.data.ctrl, before[0])
        np.testing.assert_array_equal(s.data.qpos, before[1])
        s = self.session()
        s.data.ctrl[0] = 1.0
        with self.assertRaisesRegex(ValueError, "resting"):
            inspect_mit(s)
        self.assertEqual(s.data.ctrl[0], 1.0)
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
