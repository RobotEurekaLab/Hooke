"""Open pan support, actual cameras, cache access and measured failure semantics."""

from copy import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.nasa_mast_geometry import CAMERAS
from real_labs.nasa_rover_geometry import GROUND_DATUM_OFFSET_M, START
from real_labs.nasa_rover_task import NasaRoverTask
from real_labs.nasa_mast_motion import MastInspection, inspect_nasa_mast
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene("nasa_planetary")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class NasaMastTests(unittest.TestCase):
    def test_representative_rover_api_keeps_default_non_nasa_behavior(self):
        from real_labs.equipment import add_equipment

        root = ET.Element("mujoco", model="standalone_rover")
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
        world = ET.SubElement(root, "worldbody")
        item = add_equipment(
            root,
            world,
            "rover",
            "other_rover",
            (1, 2, 0),
            yaw=0.4,
            params={"free_base": True},
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.assertEqual(model.ncam, 0)
        self.assertNotIn("mast_inspection", item)
        self.assertEqual(item["freejoint"], "other_rover__free_base")
        self.assertIn("mast_pan", item["joints"])

    def test_original_pan_and_cache_local_sites_preserved_on_grounded_base(self):
        s = session()
        c = s.controls["rover", "mast_pan"]
        np.testing.assert_allclose(
            s.model.body_pos[s.model.jnt_bodyid[c.joint]],
            [-0.22, 0, 1.026 - GROUND_DATUM_OFFSET_M],
        )
        np.testing.assert_allclose(
            s.model.jnt_range[c.joint], [-np.pi, np.pi], atol=1e-7
        )
        np.testing.assert_allclose(s.model.jnt_axis[c.joint], [0, 0, 1])
        self.assertEqual(len(s.controls), 24)
        self.assertEqual(
            s.model.jnt_type[s.model.joint("rover__free_base").id],
            mujoco.mjtJoint.mjJNT_FREE,
        )
        for i in range(3):
            site = s.model.site(f"rover__site_sample_cache_{i+1}").id
            np.testing.assert_allclose(
                s.model.site_pos[site],
                [-0.26 + i * 0.06, 0, 0.497 - GROUND_DATUM_OFFSET_M],
                atol=1e-9,
            )
            np.testing.assert_allclose(
                s.data.site_xpos[site],
                np.array(START) + [-0.26 + i * 0.06, 0, 0.497 - GROUND_DATUM_OFFSET_M],
                atol=1e-9,
            )
        self.assertEqual(s.model.ncam, len(s.manifest["camera_names"]))
        self.assertTrue(all(name in s.manifest["camera_names"] for name in CAMERAS))

    def test_enabled_surfaces_and_full_pan_parent_pair_clearance(self):
        s = session()
        m, d = s.model, s.data
        task = MastInspection(s)
        for name in (
            "column",
            "fixed_spindle",
            "head",
            "left_lens",
            "right_lens",
            "cache_holder",
            "cache_cap_1",
            "cache_cap_2",
            "cache_cap_3",
        ):
            gid = m.geom("rover__mast_" + name).id
            self.assertEqual(
                (int(m.geom_contype[gid]), int(m.geom_conaffinity[gid])), (1, 1)
            )
        sleeve = m.geom("rover__mast_pan_sleeve_proxy_0").id
        spindle = m.geom("rover__mast_fixed_spindle").id
        self.assertTrue(any(set(p) == {sleeve, spindle} for p in task.pairs))
        for angle in np.linspace(-np.pi, np.pi, 73):
            d.qpos[task.q] = angle
            mujoco.mj_forward(m, d)
            gaps, _ = task.distances()
            self.assertGreaterEqual(float(np.min(gaps)), 0.0005999)

    def test_cache_cylindrical_approach_and_real_support(self):
        root, manifest = build_scene("nasa_planetary")
        node = ET.SubElement(root.find("worldbody"), "body", name="approach_probe")
        ET.SubElement(node, "freejoint", name="probe_free")
        ET.SubElement(
            node, "geom", name="probe", type="cylinder", size=".008 .075", mass=".01"
        )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(m)
        gid = m.geom("probe").id
        q = m.jnt_qposadr[m.joint("probe_free").id]
        for index in range(3):
            d.qpos[q : q + 3] = np.array(START) + [
                -0.26 + index * 0.06,
                0,
                0.497 + 0.075 - GROUND_DATUM_OFFSET_M,
            ]
            d.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(m, d)
            for other in range(m.ngeom):
                if other == gid or not (
                    m.geom_contype[other] or m.geom_conaffinity[other]
                ):
                    continue
                gap = mujoco.mj_geomDistance(m, d, gid, other, 0.003, None)
                self.assertGreaterEqual(gap, 0.003 - 1e-8, msg=m.geom(other).name)
        # Connected welded support does not mean a floating retained cap.
        pairs = [
            ("deck_skin", "cache_plinth"),
            ("cache_plinth", "cache_holder"),
            ("cache_holder", "cache_cap_1"),
            ("foot", "column"),
            ("column", "upper_bridge"),
            ("upper_bridge", "bearing_seat"),
            ("bearing_seat", "fixed_spindle"),
        ]
        for a, b in pairs:
            gap = mujoco.mj_geomDistance(
                m,
                d,
                m.geom("rover__mast_" + a).id,
                m.geom("rover__mast_" + b).id,
                1,
                None,
            )
            self.assertLessEqual(gap, 1e-8, (a, b, gap))
        foot = m.geom("rover__mast_target_foot").id
        soil = m.geom("facility_grc1_lane_soil").id
        self.assertAlmostEqual(
            mujoco.mj_geomDistance(m, d, foot, soil, 1, None), 0, delta=1e-8
        )

    def test_mast_foot_clears_existing_solar_panels(self):
        s = session()
        m, d = s.model, s.data
        # These are the two original static blue solar plates, not arm paint.
        solar = [
            g
            for g in range(m.ngeom)
            if m.geom_matid[g] == m.material("rover__mat_blue").id
            and int(m.geom_bodyid[g]) == m.body("rover").id
        ]
        self.assertEqual(len(solar), 2)
        for name in ("foot", "column"):
            for panel in solar:
                gap = mujoco.mj_geomDistance(
                    m, d, m.geom("rover__mast_" + name).id, panel, 0.1, None
                )
                self.assertGreaterEqual(gap, 0.003 - 1e-9)

    def test_added_collision_obstacle_is_rejected_not_discarded(self):
        root, manifest = build_scene("nasa_planetary")
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="retained_obstacle",
            type="sphere",
            size=".04",
            pos="-5.15 2.54 1.018",
        )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        s = InstrumentSession(m, manifest)
        state, controls = s.data.qpos.copy(), s.data.ctrl.copy()
        with self.assertRaises(ValueError):
            inspect_nasa_mast(s)
        self.assertEqual(m.geom("retained_obstacle").contype, 1)
        np.testing.assert_array_equal(s.data.qpos, state)
        np.testing.assert_array_equal(s.data.ctrl, controls)
        self.assertEqual(s.data.time, 0)

    def test_cameras_are_mounted_parallel_and_move_with_pan(self):
        s = session()
        m, d = s.model, s.data
        task = MastInspection(s)
        initial = d.cam_xpos[task.cameras].copy()
        d.qpos[task.q] = 0.5
        mujoco.mj_forward(m, d)
        self.assertGreater(
            float(np.linalg.norm(d.cam_xpos[task.cameras] - initial)), 0.05
        )
        np.testing.assert_allclose(
            d.cam_xmat[task.cameras[0]], d.cam_xmat[task.cameras[1]]
        )
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    d.cam_xpos[task.cameras[0]] - d.cam_xpos[task.cameras[1]]
                )
            ),
            0.13,
        )
        self.assertTrue(
            all(
                int(m.cam_bodyid[c]) == int(m.jnt_bodyid[task.control.joint])
                for c in task.cameras
            )
        )
        self.assertTrue(all(name in s.manifest["camera_names"] for name in CAMERAS))

    def test_success_reports_actual_capture_steps_and_return(self):
        s = session()
        initial = s.data.qpos.copy()
        r = inspect_nasa_mast(s)
        self.assertTrue(r["success"], r["failure"])
        json.dumps(r, allow_nan=False)
        self.assertEqual(r["operation"], "parked_mast_inspection")
        self.assertEqual(r["physics_steps"], 20000)
        self.assertEqual(r["completed_checks"], r["physics_steps"])
        self.assertTrue(r["coverage_complete"])
        self.assertEqual(r["unchecked_steps"], 0)
        self.assertEqual(r["physics_events"], [])
        self.assertFalse(r["automatic_reset_detected"])
        self.assertFalse(r["time_discontinuity_detected"])
        self.assertTrue(r["no_resets"])
        self.assertAlmostEqual(r["simulated_seconds"], 10.0, places=8)
        # Native contact checks cover every executed step; direct geometry is
        # sampled every 50 ms plus phase boundaries, including parent pairs.
        self.assertGreaterEqual(r["geometry_checks"], 201)
        self.assertLess(r["geometry_checks"], r["physics_steps"])
        self.assertGreater(r["direct_queries"], 0)
        self.assertGreaterEqual(r["minimum_direct_m"], -0.00005)
        self.assertLess(r["max_parked_mast_drift_m"], 0.003)
        self.assertFalse(r["reset_performed"])
        self.assertNotIn("native_isaac_tested", r)
        self.assertEqual(r["execution_provenance"]["backend"], "mujoco")
        self.assertEqual(r["execution_provenance"]["physics_engine"], "MuJoCo")
        self.assertIn("named operation only", r["qualification_scope"])
        self.assertEqual(
            [p["name"] for p in r["phases"]],
            ["mast_align_complete", "mast_return_complete"],
        )
        cap = r["captures"][0]
        self.assertEqual(len(cap["views"]), 2)
        self.assertTrue(all(v["visible"] for v in cap["views"]))
        self.assertAlmostEqual(cap["measured_baseline_m"], 0.13)
        self.assertGreater(abs(cap["horizontal_disparity_px"]), 20)
        q = int(s.model.jnt_qposadr[s.controls["rover", "mast_pan"].joint])
        self.assertLess(abs(s.data.qpos[q] - initial[q]), 0.0001)
        np.testing.assert_allclose(
            s.data.xpos[s.model.body("rover").id][:2], START[:2], atol=0.003
        )
        # The command-line path must print the grounded report successfully.
        from real_labs import nasa_mast_motion

        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(
                nasa_mast_motion, "inspect_nasa_mast", return_value=r
            ), patch("sys.argv", ["nasa_mast_motion", "--output", str(output)]), patch(
                "builtins.print"
            ), self.assertRaises(
                SystemExit
            ) as stopped:
                nasa_mast_motion.main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(json.loads(output.read_text())["physics_steps"], 20000)

    def test_external_torque_is_rejected_before_any_step(self):
        s = session()
        c = s.controls["rover", "mast_pan"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 100
        before = s.data.qpos.copy()
        r = inspect_nasa_mast(s)
        self.assertFalse(r["success"])
        self.assertIn("External applied forces", r["failure"])
        self.assertEqual(r["physics_steps"], 0)
        self.assertEqual(s.data.time, 0)
        self.assertFalse(r["reset_performed"])
        np.testing.assert_array_equal(s.data.qpos, before)

    def test_grounded_scene_rejects_historical_anchored_protocol(self):
        s = session()
        task = MastInspection(s)
        before = s.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "anchored mast protocol is historical"):
            task.execute()
        self.assertEqual(s.data.time, 0)
        np.testing.assert_array_equal(s.data.qpos, before)

    def test_modified_geometry_masks_and_optics_reject_before_mutation(self):
        for mutate in (
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom("rover__mast_column").id, 0), -0.8
            ),
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom("rover__mast_head").id, 0
            ),
            lambda s: s.model.cam_fovy.__setitem__(s.model.camera(CAMERAS[0]).id, 45),
            lambda s: s.equipment["rover"]["mast_inspection"].__setitem__(
                "baseline_m", 0.2
            ),
        ):
            s = session()
            mutate(s)
            before = (s.data.qpos.copy(), s.data.ctrl.copy(), s.data.time)
            with self.assertRaises(ValueError):
                inspect_nasa_mast(s)
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, before[2])

    def test_foreign_or_replaced_state_and_nonfinite_input_reject(self):
        s = session()
        other = copy(s.model)
        s.data = mujoco.MjData(other)
        with self.assertRaises(ValueError):
            NasaRoverTask(s, record=False)
        s = session()
        task = NasaRoverTask(s, record=False)
        s.data = mujoco.MjData(s.model)
        with self.assertRaisesRegex(RuntimeError, "model/state replaced"):
            task.step()
        self.assertEqual(s.data.time, 0)
        s = session()
        s.data.qvel[0] = float("nan")
        with self.assertRaises(ValueError):
            NasaRoverTask(s, record=False)
        self.assertEqual(s.data.time, 0)
        self.assertTrue(np.isnan(s.data.qvel[0]))


if __name__ == "__main__":
    unittest.main()
