"""Connected guides, a free carrier, and measured dry registration failures."""

from copy import copy, deepcopy
import itertools
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.indiana_motion import _Inspection, _authored_binding, inspect_indiana
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class IndianaInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("indiana_microfluidics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_actual_registration_inspection_and_return(self):
        s = self.session()
        signature = _model_signature(s.model)
        r = inspect_indiana(s)
        self.assertTrue(r["success"], r)
        self.assertEqual(r["physics_steps"], 6752)
        self.assertEqual(r["completed_checks"], r["physics_steps"])
        self.assertTrue(r["coverage_complete"])
        self.assertGreater(r["parent_filtered_distance_queries"], 800000)
        self.assertGreaterEqual(r["minimum_parent_filtered_distance_m"], -1e-9)
        inspect = next(
            p for p in r["phases"] if p["name"] == "inspect_registered_carrier"
        )
        self.assertLess(inspect["registration_error_m"], 1e-5)
        self.assertAlmostEqual(inspect["objective_slide_gap_m"], 0.007, delta=1e-5)
        offset = next(p for p in r["phases"] if p["name"] == "offset_loaded_stage")
        np.testing.assert_allclose(
            offset["measured_m"][:2], (0.020, -0.015), atol=0.0001
        )
        for p in r["phases"]:
            self.assertTrue(p["success"])
            self.assertEqual(p["executed_steps"], p["completed_checks"])
            self.assertEqual(len(p["qpos"]), s.model.nq)
            self.assertEqual(len(p["qvel"]), s.model.nv)
            self.assertEqual(len(p["ctrl"]), s.model.nu)
        self.assertLess(r["maximum_carrier_tracking_error_m"], 0.0002)
        self.assertLess(r["maximum_carrier_penetration_m"], 0.0001)
        self.assertTrue(r["readiness"]["ready"])
        self.assertEqual(r["forbidden_contacts"], [])
        self.assertFalse(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        np.testing.assert_allclose(r["phases"][-1]["measured_m"], 0, atol=0.0001)
        self.assertEqual(signature, _model_signature(s.model))

    def test_full_axis_guide_and_objective_sweep_has_no_solid_intersections(self):
        s = self.session()
        m, d = s.model, s.data
        run = _Inspection(s)
        address = m.jnt_qposadr[m.body_jntadr[run.carrier]]
        carrier_home = d.qpos[address : address + 2].copy()
        for x, y, z in itertools.product(
            np.linspace(-0.057, 0.057, 5),
            np.linspace(-0.0365, 0.0365, 5),
            np.linspace(-0.003, 0.004, 5),
        ):
            d.qpos[run.q] = x, y, z
            # Private geometry sweep only; actual task tests never assign qpos.
            d.qpos[address : address + 2] = carrier_home + (x, y)
            mujoco.mj_forward(m, d)
            self.assertFalse([c for c in d.contact if c.dist < -1e-5], (x, y, z))
            for a, b in run.filtered:
                self.assertGreaterEqual(
                    mujoco.mj_geomDistance(m, d, a, b, 0.01, None),
                    -1e-9,
                    (x, y, z, m.geom(a).name, m.geom(b).name),
                )
        self.assertEqual(
            m.jnt_type[m.body_jntadr[run.carrier]], mujoco.mjtJoint.mjJNT_FREE
        )

    def test_real_slide_supports_controller_feet_and_optical_aperture(self):
        s = self.session()
        m, d = s.model, s.data

        def geom(key):
            return m.geom("microfluidic_station__indiana_" + key).id

        glass = geom("glass_slide")
        for side in (-1, 1):
            support = geom(f"slide_end_support_{side}")
            pad = geom(f"slide_keeper_pad_{side}")
            post = geom(f"slide_keeper_post_{side}")
            bridge = geom(f"slide_keeper_{side}")
            self.assertAlmostEqual(
                d.geom_xpos[support, 2] + m.geom_size[support, 2],
                d.geom_xpos[glass, 2] - m.geom_size[glass, 2],
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
        shell = geom("controller_shell")
        feet = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(
                "microfluidic_station__indiana_controller_foot_"
            )
        ]
        self.assertEqual(len(feet), 4)
        for foot in feet:
            self.assertAlmostEqual(d.geom_xpos[foot, 2] - m.geom_size[foot, 1], 0.8)
            self.assertAlmostEqual(
                d.geom_xpos[foot, 2] + m.geom_size[foot, 1],
                d.geom_xpos[shell, 2] - m.geom_size[shell, 2],
            )
        # Below the transparent slide, a ray passes the real stage aperture.
        origin = np.array((-1.55, 0.65, 1.152))
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m, d, origin, np.array((0.0, 0.0, 1.0)), None, 1, -1, hit
        )
        self.assertEqual(hit[0], glass)
        self.assertAlmostEqual(distance, 0.0043, delta=1e-7)
        guide = geom("focus_column_guide")
        arm = geom("focus_arm")
        self.assertAlmostEqual(
            mujoco.mj_geomDistance(m, d, guide, arm, 0.1, None), 0.0002, delta=1e-9
        )

    def test_trace_scale_and_parked_service_endpoints_are_explicit(self):
        s = self.session()
        m, d = s.model, s.data
        interface = s.equipment["microfluidic_station"]["inspection_interface"]
        traces = [
            g for g in range(m.ngeom) if "__indiana_channel_trace_" in m.geom(g).name
        ]
        self.assertEqual(len(traces), 6)
        for g in traces:
            self.assertAlmostEqual(2 * m.geom_size[g, 0], interface["trace_width_m"])
        self.assertEqual(interface["trace_width_m"], 0.0001)
        self.assertFalse(interface["services_connected_to_chip"])
        for i, lead in enumerate(interface["parked_leads"]):
            first = m.geom(f"microfluidic_station__indiana_parked_hose_{i}_0").id
            last = m.geom(f"microfluidic_station__indiana_parked_hose_{i}_15").id
            for g, site, sign in (
                (first, lead["source"], -1),
                (last, lead["destination"], 1),
            ):
                axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
                endpoints = [
                    d.geom_xpos[g] + direction * m.geom_size[g, 1] * axis
                    for direction in (-1, 1)
                ]
                target = d.site_xpos[m.site(site).id]
                self.assertLess(
                    min(np.linalg.norm(end - target) for end in endpoints), 1e-9
                )
        for key in ("chip_inlet", "chip_outlet"):
            cap = m.geom("microfluidic_station__indiana_" + key + "_dry_cap").id
            port = m.geom("microfluidic_station__indiana_" + key + "_fitting").id
            self.assertAlmostEqual(
                d.geom_xpos[cap, 2] - m.geom_size[cap, 1],
                d.geom_xpos[port, 2] + m.geom_size[port, 1],
            )

    def test_disabled_guides_and_foreign_or_replaced_state_rejected(self):
        names = (
            "objective_0",
            "objective_face_0",
            "focus_arm",
            "x_column_rail",
            "y_bearing_rail_-1",
            "xy_plate_side_-1",
            "glass_slide",
            "controller_foot_0.245_-0.032",
        )
        for name in names:
            s = self.session()
            g = s.model.geom("microfluidic_station__indiana_" + name).id
            s.model.geom_contype[g] = s.model.geom_conaffinity[g] = 0
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "collision"
            ):
                _Inspection(s)
            self.assertEqual(s.data.time, 0)
        s = self.session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaisesRegex(ValueError, "exact model"):
            inspect_indiana(s)
        s = self.session()
        run = _Inspection(s)
        old = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            run.phase("bad", run.home, 0.1, 0.1)
        self.assertEqual(old.time, 0)
        self.assertEqual(s.data.time, 0)
        s = self.session()
        run = _Inspection(s)
        s.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with self.assertRaisesRegex(RuntimeError, "compiled_model_changed"):
            run.phase("bad", run.home, 0.1, 0.1)
        self.assertEqual(s.data.time, 0)

    def test_canonical_binding_rejects_disconnected_guides_and_changed_physics(self):
        for change in ("guide", "mass", "gravity", "timestep", "gain"):
            s = self.session()
            m, d = s.model, s.data
            if change == "guide":
                m.geom_pos[
                    m.geom("microfluidic_station__indiana_focus_column_guide").id, 0
                ] += 0.25
                mujoco.mj_forward(m, d)
            elif change == "mass":
                m.body_mass[m.body("microfluidic_station__specimen").id] *= 2
            elif change == "gravity":
                m.opt.gravity[2] = 0
            elif change == "timestep":
                m.opt.timestep *= 0.5
            else:
                m.actuator_gainprm[
                    s.controls["microfluidic_station", "stage_x"].actuator, 0
                ] *= 2
            flags = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(m, flags))
            mujoco.mj_getState(m, d, before, flags)
            signature = _model_signature(m)
            with self.subTest(change=change), self.assertRaisesRegex(
                ValueError, "binding changed"
            ):
                inspect_indiana(s)
            after = np.empty_like(before)
            mujoco.mj_getState(m, d, after, flags)
            np.testing.assert_array_equal(after, before)
            self.assertEqual(_model_signature(m), signature)

    def test_canonical_binding_rejects_changed_metadata_and_accepts_portable_model(
        self,
    ):
        from pathlib import Path
        from real_labs.export import _portable_resources

        for change in ("equipment", "camera", "specimens", "scene_sha256"):
            s = self.session()
            if change == "equipment":
                s.equipment["microfluidic_station"]["inspection_interface"][
                    "trace_width_m"
                ] = 0.001
            elif change == "scene_sha256":
                s.manifest[change] = "0" * 64
            else:
                s.manifest[change] = None
            before = s.data.ctrl.copy()
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspect_indiana(s)
            self.assertEqual(s.data.time, 0)
            np.testing.assert_array_equal(s.data.ctrl, before)
        root = deepcopy(self.root)
        resources = _portable_resources(root)
        m = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"),
            assets={
                item["file"]: Path(item["source"]).read_bytes() for item in resources
            },
        )
        manifest = deepcopy(self.manifest)
        manifest["scene_sha256"] = _authored_binding()["scene_sha256"]
        run = _Inspection(InstrumentSession(m, manifest))
        self.assertEqual(run.report["authored_scene_sha256"], manifest["scene_sha256"])

    def test_retained_real_path_obstacle_fails_without_reset_or_false_coverage(self):
        root = deepcopy(self.root)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="registration_obstacle",
            type="box",
            pos="-1.535 .638 1.1605",
            size=".003 .003 .001",
            contype="1",
            conaffinity="1",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            self.manifest,
        )
        signature = _model_signature(s.model)
        # The public operation rejects a changed environment before mutation.
        # Test-only binding substitution exercises the downstream collision
        # guard independently; it is not a supported production bypass.
        with self.assertRaisesRegex(ValueError, "binding changed"):
            inspect_indiana(s)
        self.assertEqual(s.data.time, 0)
        with patch(
            "real_labs.indiana_motion._require_binding",
            return_value=_authored_binding(),
        ):
            r = inspect_indiana(s)
        self.assertFalse(r["success"])
        self.assertEqual(r["failure"], "unexpected_instrument_collision")
        self.assertGreater(r["physics_steps"], 0)
        self.assertEqual(r["completed_checks"], r["physics_steps"] - 1)
        self.assertEqual(r["geometry_checks"], r["physics_steps"])
        self.assertFalse(r["coverage_complete"])
        self.assertIn("registration_obstacle", r["forbidden_contacts"][0]["geoms"])
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertFalse(r["phases"][-1]["success"])
        self.assertEqual(signature, _model_signature(s.model))

    def test_parent_filtered_penetration_and_initial_obstacle_have_no_completed_coverage(
        self,
    ):
        for parent_overlap in (True, False):
            root = deepcopy(self.root)
            if parent_overlap:
                root.find(
                    ".//geom[@name='microfluidic_station__indiana_y_bearing_rail_-1']"
                ).set("pos", "-.104 0 .015")
            else:
                ET.SubElement(
                    root.find("worldbody"),
                    "geom",
                    name="initial_carrier_obstacle",
                    type="sphere",
                    pos="-1.55 .65 1.1595",
                    size=".001",
                    contype="1",
                    conaffinity="1",
                )
            s = InstrumentSession(
                mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
                self.manifest,
            )
            with self.assertRaisesRegex(ValueError, "binding changed"):
                inspect_indiana(s)
            with patch(
                "real_labs.indiana_motion._require_binding",
                return_value=_authored_binding(),
            ):
                r = inspect_indiana(s)
            self.assertFalse(r["success"])
            self.assertEqual(r["physics_steps"], 0)
            self.assertFalse(r["coverage_complete"])
            self.assertEqual(r["completed_checks"], 0)
            self.assertEqual(r["preflight_completed_checks"], 0)
            self.assertTrue(r["hold_requested"])
            if parent_overlap:
                self.assertTrue(
                    any(
                        c["detection"].startswith("direct")
                        for c in r["forbidden_contacts"]
                    )
                )

    def test_disturbed_held_axis_and_free_carrier_cannot_claim_success(self):
        for carrier in (False, True):
            s = self.session()
            if carrier:
                s.data.xfrc_applied[
                    s.model.body("microfluidic_station__specimen").id, 0
                ] = 2
            else:
                c = s.controls["microfluidic_station", "flow_valve"]
                s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 100
            r = inspect_indiana(s)
            self.assertFalse(r["success"])
            self.assertIn(
                r["failure"],
                (
                    "carrier_displaced_or_not_retained",
                    "inactive_control_changed_or_drifted",
                ),
            )
            self.assertEqual(r["completed_checks"], r["physics_steps"] - 1)
            self.assertLess(r["geometry_checks"], r["physics_steps"])
            self.assertFalse(r["coverage_complete"])
            self.assertTrue(r["hold_requested"])
            self.assertFalse(r["reset_performed"])


if __name__ == "__main__":
    unittest.main()
