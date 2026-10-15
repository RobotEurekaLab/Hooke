"""Open instrument guides, retained inert slide and measured inspection failures."""

from copy import copy, deepcopy
import itertools
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.export import _portable_resources
from real_labs.kyoto_instrument_geometry import CAMERA
from real_labs.kyoto_instrument_motion import SlideInspection, inspect_kyoto
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class KyotoInstrumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("kyoto_yokokawa_nanometrics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_frames_limits_target_and_actual_moving_camera(self):
        s = self.session()
        m, d = s.model, s.data
        for key, origin, axis, limits in (
            ("slide_x", (0, -0.09, 0.205), (1, 0, 0), (-0.018, 0.018)),
            ("slide_y", (0, 0, 0.019), (0, 1, 0), (-0.014, 0.014)),
            ("objective_focus", (0, -0.09, 0.132), (0, 0, 1), (-0.003, 0.003)),
        ):
            control = s.controls["tirf_workspace", key]
            np.testing.assert_allclose(m.body_pos[m.jnt_bodyid[control.joint]], origin)
            np.testing.assert_allclose(m.jnt_axis[control.joint], axis)
            np.testing.assert_allclose(m.jnt_range[control.joint], limits)
        target = m.site("tirf_workspace__site_inert_target").id
        np.testing.assert_allclose(m.site_pos[target], (0, 0, 0.00816))
        np.testing.assert_allclose(d.site_xpos[target], (0.91, -0.59, 1.08816))
        item = s.equipment["tirf_workspace"]
        self.assertEqual(item["sample_attachment"], "clamped")
        self.assertEqual(item["sample_size_m"], [0.002, 0.002, 0.00016])
        camera = m.camera(CAMERA).id
        focus = s.controls["tirf_workspace", "objective_focus"]
        self.assertEqual(m.cam_bodyid[camera], m.jnt_bodyid[focus.joint])
        self.assertIn(CAMERA, s.manifest["camera_names"])
        self.assertNotIn(CAMERA, s.manifest["camera"])
        np.testing.assert_allclose(
            d.cam_xmat[camera].reshape(3, 3)[:, 2], (0, 0, -1), atol=1e-12
        )
        before = d.cam_xpos[camera].copy()
        d.qpos[m.jnt_qposadr[focus.joint]] = 0.002
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(
            d.cam_xpos[camera] - before, (0, 0, 0.002), atol=1e-12
        )

    def test_open_focus_bearing_and_xy_supports(self):
        s = self.session()
        m, d = s.model, s.data
        g = lambda key: m.geom("tirf_workspace__kyoto_" + key).id
        barrel = g("focus_carriage")
        for i in range(48):
            proxy = g(f"focus_bearing_proxy_{i}")
            self.assertEqual((m.geom_contype[proxy], m.geom_conaffinity[proxy]), (1, 1))
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, barrel, proxy, 0.01, None),
                0.0006,
                places=7,
            )
        for side in (-1, 1):
            rail = g(f"x_rail_{side}")
            for key in ("crossbar_-1", "crossbar_1", "side_-1", "side_1"):
                sleeve = g(f"x_bearing_{side}_{key}")
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(m, d, rail, sleeve, 0.01, None),
                    0.0005,
                    places=7,
                )
            for end in (-1, 1):
                leg = g(f"x_pedestal_{side}_{end}")
                self.assertAlmostEqual(
                    d.geom_xpos[leg, 2] + m.geom_size[leg, 2],
                    d.geom_xpos[rail, 2] - m.geom_size[rail, 2],
                )
                self.assertAlmostEqual(d.geom_xpos[leg, 2] - m.geom_size[leg, 2], 0.926)
                mount = g(f"y_rail_mount_{side}_{end}")
                yrail = g(f"y_rail_{side}")
                self.assertAlmostEqual(
                    d.geom_xpos[mount, 2] + m.geom_size[mount, 2],
                    d.geom_xpos[yrail, 2] - m.geom_size[yrail, 2],
                )
            for key in ("crossbar_-1", "crossbar_1", "side_-1", "side_1"):
                sleeve = g(f"y_bearing_{side}_{key}")
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(m, d, yrail, sleeve, 0.01, None),
                    0.0005,
                    places=7,
                )
        # Lens face is a physical surface, with 4 mm authored home clearance.
        run = SlideInspection(s)
        self.assertAlmostEqual(run.working_gap(), 0.004)
        self.assertAlmostEqual(
            d.site_xpos[run.face_site, 2],
            d.geom_xpos[run.face, 2] + m.geom_size[run.face, 1],
        )

    def test_retaining_hardware_contacts_the_slide_and_stage(self):
        s = self.session()
        m, d = s.model, s.data
        g = lambda key: m.geom("tirf_workspace__kyoto_" + key).id
        glass = g("glass_slide")
        for key in (
            "glass_slide",
            "inert_target",
            "registration_target",
            "objective_face",
        ):
            geom = g(key)
            self.assertEqual((m.geom_contype[geom], m.geom_conaffinity[geom]), (1, 1))
        np.testing.assert_allclose(m.geom_size[glass], (0.0375, 0.0125, 0.0005))
        for side in (-1, 1):
            post, pad, bridge = [
                g(f"keeper_{key}_{side}") for key in ("post", "pad", "bridge")
            ]
            for geom in (post, pad, bridge):
                self.assertEqual(
                    (m.geom_contype[geom], m.geom_conaffinity[geom]), (1, 1)
                )
            self.assertAlmostEqual(
                d.geom_xpos[pad, 2] - m.geom_size[pad, 2],
                d.geom_xpos[glass, 2] + m.geom_size[glass, 2],
            )
            self.assertAlmostEqual(
                d.geom_xpos[pad, 2] + m.geom_size[pad, 2],
                d.geom_xpos[bridge, 2] - m.geom_size[bridge, 2],
            )
            self.assertAlmostEqual(
                d.geom_xpos[post, 2] + m.geom_size[post, 2],
                d.geom_xpos[bridge, 2] - m.geom_size[bridge, 2],
            )
            self.assertAlmostEqual(d.geom_xpos[post, 2] - m.geom_size[post, 2], 1.087)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m,
            d,
            np.array((0.91, -0.59, 1.084)),
            np.array((0.0, 0.0, 1.0)),
            None,
            1,
            -1,
            hit,
        )
        self.assertEqual(hit[0], glass)
        self.assertAlmostEqual(distance, 0.003, places=7)

    def test_all_parent_and_other_enabled_pairs_clear_full_travel(self):
        s = self.session()
        m, d = s.model, s.data
        run = SlideInspection(s)
        self.assertEqual(run.report["pairs_pruned"], 0)
        roots = {m.jnt_bodyid[c.joint] for c in run.controls}
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        expected = {
            (int(a), int(b))
            for i, a in enumerate(enabled)
            for b in enabled[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
            and (
                m.body_weldid[m.geom_bodyid[a]] in roots
                or m.body_weldid[m.geom_bodyid[b]] in roots
            )
        }
        self.assertEqual(set(map(tuple, run.pairs)), expected)
        poses = list(
            itertools.product(
                (-0.018, 0, 0.018), (-0.014, 0, 0.014), (-0.003, 0, 0.003)
            )
        )
        poses.extend(np.random.default_rng(1743).uniform(run.lower, run.upper, (18, 3)))
        for pose in poses:
            d.qpos[run.q] = pose
            mujoco.mj_forward(m, d)
            for a, b in run.pairs:
                gap = mujoco.mj_geomDistance(m, d, int(a), int(b), 0.001, None)
                self.assertGreaterEqual(
                    gap,
                    0.0001999,
                    (pose, m.geom(int(a)).name, m.geom(int(b)).name, gap),
                )
        self.assertFalse(any(c.dist < 0 for c in d.contact))

    def test_photo_informed_mounts_have_supported_faces_and_connected_services(self):
        s = self.session()
        m, d = s.model, s.data
        g = lambda key: m.geom("tirf_workspace__kyoto_" + key).id
        for i in range(4):
            post, base, bracket = [
                g(f"{key}_{i}")
                for key in ("post", "optical_base", "micrometer_bracket")
            ]
            self.assertAlmostEqual(
                d.geom_xpos[bracket, 2] - m.geom_size[bracket, 2],
                d.geom_xpos[base, 2] + m.geom_size[base, 2],
            )
            if i < 2:
                lens = g(f"optic_face_{i}")
                self.assertGreater(
                    mujoco.mj_geomDistance(m, d, post, lens, 0.1, None), 0.005
                )
            else:
                self.assertAlmostEqual(
                    mujoco.mj_geomDistance(
                        m, d, g(f"mirror_face_{i}"), g(f"mirror_back_{i}"), 0.01, None
                    ),
                    0.0,
                    places=7,
                )
        for i, record in enumerate(
            s.equipment["tirf_workspace"]["inspection_interface"]["parked_services"]
        ):
            geoms = [
                j
                for j in range(m.ngeom)
                if m.geom(j).name.startswith(
                    f"tirf_workspace__kyoto_parked_service_{i}_segment_"
                )
            ]
            self.assertGreater(len(geoms), 3)
            origin = d.xpos[m.body("tirf_workspace").id]
            for geom, point in (
                (geoms[0], record["start"]),
                (geoms[-1], record["end"]),
            ):
                axis = d.geom_xmat[geom].reshape(3, 3)[:, 2]
                endpoints = [
                    d.geom_xpos[geom] + direction * m.geom_size[geom, 1] * axis
                    for direction in (-1, 1)
                ]
                self.assertLess(
                    min(np.linalg.norm(p - origin - point) for p in endpoints), 1e-9
                )

    def test_portable_model_and_metadata_accept_the_same_contract(self):
        root = deepcopy(self.root)
        resources = _portable_resources(root)
        model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"),
            assets={r["file"]: Path(r["source"]).read_bytes() for r in resources},
        )
        run = SlideInspection(InstrumentSession(model, deepcopy(self.manifest)))
        self.assertEqual(len(run.controls), 3)
        self.assertGreater(len(run.pairs), 1000)

    def test_changed_geometry_metadata_masks_and_obstacles_reject_without_mutation(
        self,
    ):
        for change in ("rail", "mask", "gain", "camera", "target"):
            s = self.session()
            m, d = s.model, s.data
            if change == "rail":
                m.geom_pos[m.geom("tirf_workspace__kyoto_x_rail_1").id, 0] += 0.25
            elif change == "mask":
                g = m.geom("tirf_workspace__kyoto_objective_face").id
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            elif change == "gain":
                m.actuator_gainprm[0, 0] *= 2
            elif change == "camera":
                s.manifest["mounted_cameras"][0]["fovy_deg"] = 20
            else:
                s.equipment["tirf_workspace"]["inspection_interface"][
                    "inspection_working_clearance_m"
                ] = 0.06
            before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time))
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspect_kyoto(s)
            for actual, original in zip((d.qpos, d.qvel, d.ctrl), before[:3]):
                np.testing.assert_array_equal(actual, original)
            self.assertEqual(d.time, before[3])
        root = deepcopy(self.root)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="inspection_obstacle",
            type="sphere",
            size=".008",
            pos=".91 -.59 1.085",
            contype="1",
            conaffinity="1",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            deepcopy(self.manifest),
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            inspect_kyoto(s)
        self.assertEqual(s.data.time, 0)
        self.assertEqual(
            s.model.geom_contype[s.model.geom("inspection_obstacle").id], 1
        )

    def test_foreign_and_replaced_data_reject_before_mutation(self):
        s = self.session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaisesRegex(ValueError, "exact model"):
            inspect_kyoto(s)
        s = self.session()
        run = SlideInspection(s)
        old = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaisesRegex(ValueError, "replaced"):
            run.phase("invalid", run.home, 0.002, 0.01)
        self.assertEqual(old.time, 0)
        self.assertEqual(s.data.time, 0)
        np.testing.assert_array_equal(old.ctrl, 0)
        np.testing.assert_array_equal(s.data.ctrl, 0)

    def test_actual_inspection_has_complete_checks_captures_and_return(self):
        s = self.session()
        before = _model_signature(s.model)
        result = inspect_kyoto(s)
        self.assertTrue(result["success"], result["failure"])
        self.assertEqual(result["physics_steps"], 4626)
        self.assertEqual(result["physics_steps"], result["observer_entries"])
        self.assertEqual(result["physics_steps"], result["geometry_checks"])
        self.assertEqual(result["physics_steps"], result["completed_checks"])
        self.assertTrue(result["coverage_complete"])
        self.assertFalse(result["reset_performed"])
        self.assertGreaterEqual(result["whole_run_clearance_lower_bound_m"], 0.000499)
        self.assertLess(max(result["return_error_m"]), 0.00002)
        self.assertEqual(len(result["captures"]), 2)
        for capture in result["captures"]:
            self.assertTrue(capture["visible"])
            self.assertLess(capture["axis_error_m"], 0.00002)
            self.assertAlmostEqual(
                capture["physical_working_clearance_m"], 0.002, delta=0.00002
            )
            self.assertEqual(len(capture["state"]["qpos"]), s.model.nq)
        self.assertEqual(_model_signature(s.model), before)
        for phase in result["phases"]:
            self.assertTrue(phase["success"])
            self.assertTrue(phase["coverage_complete"])
            self.assertLess(phase["maximum_held_axis_error_m"], 0.0001)

    def test_actual_force_failures_do_not_certify_unchecked_steps(self):
        for force in (0.2, 100.0):
            s = self.session()
            c = s.controls["tirf_workspace", "objective_focus"]
            s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = force
            signature = _model_signature(s.model)
            report = inspect_kyoto(s)
            self.assertFalse(report["success"])
            self.assertTrue(report["hold_requested"])
            self.assertFalse(report["reset_performed"])
            self.assertFalse(report["coverage_complete"])
            self.assertIsNone(report["whole_run_clearance_lower_bound_m"])
            self.assertGreater(report["physics_steps"], report["completed_checks"])
            self.assertFalse(report["phases"][-1]["success"])
            self.assertFalse(report["phases"][-1]["coverage_complete"])
            self.assertGreater(s.data.time, 0)
            self.assertGreater(np.linalg.norm(s.data.qpos), 0)
            self.assertEqual(_model_signature(s.model), signature)
            if report["completed_checks"] == 0:
                self.assertIsNone(report["minimum_checked_clearance_lower_bound_m"])
            else:
                self.assertIsNotNone(report["minimum_checked_clearance_lower_bound_m"])


if __name__ == "__main__":
    unittest.main()
