"""Actual contact, assembly and failure checks for the authored Leeds fixture."""

from copy import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.leeds_motion import run_loaded_slide
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class LeedsContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("leeds_environmental_tribology")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), self.manifest)

    def test_actual_loaded_slide_and_retraction(self):
        session = self.session()
        before = _model_signature(session.model)
        held = session.data.ctrl[session.controls["tribology", "access_door"].actuator]
        report = run_loaded_slide(session)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["physics_steps"], 5125)
        self.assertEqual(report["completed_checks"], report["physics_steps"])
        self.assertTrue(report["coverage_complete"])
        self.assertGreater(report["contact_steps"], 3000)
        self.assertAlmostEqual(report["loaded_relative_path_m"], 0.040, delta=0.0002)
        self.assertGreater(report["phases"][0]["normal_force_n"], 3)
        self.assertLess(report["peak_normal_force_n"], 5)
        self.assertGreater(report["peak_tangential_force_n"], 0.1)
        self.assertLess(report["peak_tangential_force_n"], 2)
        self.assertGreater(report["tangential_impulse_magnitude_ns"], 0)
        for phase in report["phases"]:
            self.assertEqual(len(phase["qpos"]), session.model.nq)
            self.assertEqual(len(phase["qvel"]), session.model.nv)
            self.assertEqual(len(phase["ctrl"]), session.model.nu)
        # Force is consistently expressed on the coupon, in world coordinates.
        for start, end, expected_sign in ((3.2, 3.8, 1), (4.6, 5.8, -1)):
            samples = [
                row for row in report["telemetry"] if start < row["time_s"] < end
            ]
            self.assertTrue(samples)
            self.assertGreater(
                expected_sign
                * np.mean([row["force_on_coupon_world_n"][0] for row in samples]),
                0.1,
            )
        self.assertGreater(report["maximum_contact_penetration_m"], 0)
        self.assertLess(report["maximum_contact_penetration_m"], 0.0005)
        self.assertEqual(report["forbidden_contacts"], [])
        self.assertEqual(report["held_maximum_drift"], 0)
        self.assertFalse(report["reset_performed"])
        self.assertEqual(before, _model_signature(session.model))
        self.assertEqual(
            session.data.ctrl[session.controls["tribology", "access_door"].actuator],
            held,
        )
        np.testing.assert_allclose(report["final_position_m"], (0, 0), atol=1e-7)
        tip = session.model.site("tribology__site_counterface_tip").id
        coupon = session.model.site("tribology__site_dry_tribology_coupon").id
        self.assertAlmostEqual(
            session.data.site_xpos[tip, 2] - session.data.site_xpos[coupon, 2],
            0.020,
            places=8,
        )

    def test_retained_obstacle_aborts_motion_without_teleport(self):
        root, manifest = build_scene("leeds_environmental_tribology")
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="real_fixture_obstacle",
            type="box",
            pos=".15 .66 1.101",
            size=".015 .015 .003",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        session = InstrumentSession(model, manifest)
        signature = _model_signature(model)
        report = run_loaded_slide(session)
        self.assertFalse(report["success"])
        self.assertEqual(report["failure"], "unexpected_fixture_collision")
        self.assertGreater(report["physics_steps"], 0)
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertIn("real_fixture_obstacle", report["forbidden_contacts"][0]["geoms"])
        self.assertEqual(signature, _model_signature(model))
        self.assertGreater(session.data.time, 0)
        self.assertEqual(report["geometry_checks"], report["physics_steps"])
        self.assertEqual(report["completed_checks"], report["physics_steps"] - 1)
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(report["geometry_coverage_complete"])

    def test_a_phase_with_insufficient_measured_load_is_not_successful(self):
        session = self.session()
        coupon = session.model.geom("tribology__leeds_coupon").id
        # A misplaced coupon changes attainable compression without changing
        # the requested load. Actual response must fail the endpoint criterion.
        session.model.geom_pos[coupon, 2] -= 0.002
        report = run_loaded_slide(session)
        self.assertEqual(report["failure"], "normal_load_not_established")
        self.assertFalse(report["success"])
        self.assertFalse(report["phases"][0]["success"])
        self.assertLess(report["phases"][0]["normal_force_n"], 2.5)
        self.assertTrue(report["coverage_complete"])

    def test_preflight_collision_has_zero_executed_step_coverage(self):
        root, manifest = build_scene("leeds_environmental_tribology")
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="start_obstacle",
            type="sphere",
            pos=".15 .66 1.111",
            size=".004",
            contype="1",
            conaffinity="1",
        )
        session = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            manifest,
        )
        report = run_loaded_slide(session)
        self.assertFalse(report["success"])
        self.assertEqual(report["physics_steps"], 0)
        self.assertEqual(report["completed_checks"], 0)
        self.assertEqual(report["unchecked_steps"], 0)
        self.assertEqual(report["preflight_geometry_checks"], 1)
        self.assertEqual(report["preflight_completed_checks"], 0)
        self.assertFalse(report["coverage_complete"])

    def test_external_load_limit_and_held_drift_are_real_failures(self):
        session = self.session()
        control = session.controls["tribology", "contact_approach"]
        session.data.qfrc_applied[session.model.jnt_dofadr[control.joint]] = 4
        report = run_loaded_slide(session)
        self.assertFalse(report["success"])
        self.assertEqual(report["failure"], "normal_load_limit")
        self.assertGreater(report["peak_normal_force_n"], 7.5)
        self.assertEqual(report["completed_checks"], report["physics_steps"] - 1)
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        session = self.session()
        control = session.controls["tribology", "access_door"]
        session.data.qfrc_applied[session.model.jnt_dofadr[control.joint]] = 100
        report = run_loaded_slide(session)
        self.assertFalse(report["success"])
        self.assertIn(
            report["failure"],
            ("inactive_motion", "inactive_control_changed_or_drifted"),
        )
        self.assertEqual(report["observer_entries"], report["physics_steps"])
        self.assertLess(report["completed_checks"], report["physics_steps"])
        self.assertFalse(report["coverage_complete"])
        self.assertGreater(report["unchecked_steps"], 0)

    def test_disabled_hand_contacts_and_altered_force_contract_rejected(self):
        for name in (
            "counterface",
            "coupon",
            "flat_lever",
            "clip_pad_0",
            "approach_guide_-0.28",
            "slide_rail_-0.126",
            "carriage_contact_0",
        ):
            session = self.session()
            geom = session.model.geom("tribology__leeds_" + name).id
            session.model.geom_contype[geom] = session.model.geom_conaffinity[geom] = 0
            with self.assertRaisesRegex(ValueError, "contact|collision"):
                run_loaded_slide(session)
            self.assertEqual(session.data.time, 0)
        session = self.session()
        control = session.controls["tribology", "contact_approach"]
        session.model.actuator_forcelimited[control.actuator] = 0
        with self.assertRaisesRegex(ValueError, "force limits"):
            run_loaded_slide(session)

    def test_full_door_sweep_and_unloaded_slide_range(self):
        session = self.session()
        model, data = session.model, session.data
        door = model.jnt_qposadr[session.controls["tribology", "access_door"].joint]
        slide = model.jnt_qposadr[session.controls["tribology", "specimen_slide"].joint]
        for angle, x in itertools.product(np.linspace(0, 1.2, 61), (-0.012, 0, 0.012)):
            data.qpos[door], data.qpos[slide] = angle, x
            mujoco.mj_forward(model, data)
            self.assertEqual(
                [(c.geom1, c.geom2, c.dist) for c in data.contact if c.dist < -1e-5], []
            )

    def test_aperture_rays_and_continuous_supports(self):
        session = self.session()
        model, data = session.model, session.data
        body = model.body("tribology__body_access_door").id
        rotation = data.xmat[body].reshape(3, 3)
        glass = model.geom("tribology__leeds_viewport_glass").id
        model.geom_group[glass] = 4
        mask = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
        for x, z in ((0.49, 0), (0.45, 0), (0.53, 0), (0.49, 0.04), (0.49, -0.04)):
            point = data.xpos[body] + rotation @ np.array((x, -0.08, z))
            direction = rotation @ np.array((0.0, 1.0, 0.0))
            hit = np.zeros(1, dtype=np.int32)
            distance = mujoco.mj_ray(model, data, point, direction, mask, 1, -1, hit)
            self.assertTrue(distance < 0 or distance > 0.13, (x, z, distance, hit))

        # These planar support elevations are measured from the compiled model.
        def bounds(name):
            g = model.geom("tribology__leeds_" + name).id
            return (
                data.geom_xpos[g, 2] - model.geom_size[g, 1],
                data.geom_xpos[g, 2] + model.geom_size[g, 1],
            )

        foundation, core = bounds("stage_foundation"), bounds("stage_core")
        self.assertAlmostEqual(foundation[0], 0.885)
        self.assertAlmostEqual(foundation[1], core[0])
        self.assertAlmostEqual(core[1], 1.025)
        coupon = model.geom("tribology__leeds_coupon").id
        for i in range(3):
            pad = model.geom(f"tribology__leeds_clip_pad_{i}").id
            bridge = model.geom(f"tribology__leeds_clip_bridge_{i}").id
            post = model.geom(f"tribology__leeds_clip_post_{i}").id
            self.assertAlmostEqual(
                data.geom_xpos[pad, 2] - model.geom_size[pad, 2],
                data.geom_xpos[coupon, 2] + model.geom_size[coupon, 1],
            )
            self.assertAlmostEqual(
                data.geom_xpos[pad, 2] + model.geom_size[pad, 2],
                data.geom_xpos[bridge, 2] - model.geom_size[bridge, 2],
            )
            self.assertAlmostEqual(
                data.geom_xpos[post, 2] + model.geom_size[post, 2],
                data.geom_xpos[bridge, 2] - model.geom_size[bridge, 2],
            )
        self.assertEqual(model.ncam, 4)

    def test_coils_have_continuous_support_and_clear_specimen_rails(self):
        session = self.session()
        model, data = session.model, session.data
        core = model.geom("tribology__leeds_stage_core").id
        rings = [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and abs(model.geom_size[g, 0] - 0.009) < 1e-12
            and any(
                abs(data.geom_xpos[g, 2] - z) < 1e-12 for z in (0.981, 1.003, 1.025)
            )
        ]
        self.assertEqual(len(rings), 144)
        for geom in rings:
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(model, data, geom, core, 0.1, None),
                0,
                delta=1e-7,
            )
        for suffix in ("-0.126", "0.006"):
            rail = model.geom("tribology__leeds_slide_rail_" + suffix).id
            self.assertTrue(model.geom_contype[rail] and model.geom_conaffinity[rail])
            for geom in rings:
                self.assertGreater(
                    mujoco.mj_geomDistance(model, data, rail, geom, 1, None), 0.004
                )
        header = model.geom("tribology__leeds_coil_support_header").id
        foundation = model.geom("tribology__leeds_stage_foundation").id
        self.assertAlmostEqual(
            data.geom_xpos[header, 2] - model.geom_size[header, 1],
            data.geom_xpos[foundation, 2] + model.geom_size[foundation, 1],
        )

    def test_open_bearing_shell_and_enabled_columns_clear_full_travel(self):
        session = self.session()
        model, data = session.model, session.data
        contacts = [
            model.geom(name).id
            for name in session.equipment["tribology"]["bearing_contact_geoms"]
        ]
        surface = model.geom("tribology__leeds_load_carriage").id
        lever = model.geom("tribology__leeds_flat_lever").id
        guides = [
            model.geom("tribology__leeds_approach_guide_" + suffix).id
            for suffix in ("-0.28", "-0.22")
        ]
        address = model.jnt_qposadr[
            session.controls["tribology", "contact_approach"].joint
        ]
        for position in np.linspace(0, 0.025, 11):
            data.qpos[address] = position
            mujoco.mj_forward(model, data)
            for guide in guides:
                self.assertTrue(
                    model.geom_contype[guide] and model.geom_conaffinity[guide]
                )
                for contact in contacts:
                    self.assertTrue(
                        model.geom_contype[contact] and model.geom_conaffinity[contact]
                    )
                    self.assertGreater(
                        mujoco.mj_geomDistance(model, data, guide, contact, 1, None),
                        0.0004,
                    )
                self.assertGreater(
                    mujoco.mj_geomDistance(model, data, guide, lever, 1, None), 0.01
                )
                start = data.geom_xpos[guide].copy()
                start[2] = 1.30
                self.assertEqual(
                    mujoco.mj_rayMesh(
                        model, data, surface, start, np.array((0.0, 0.0, -1.0))
                    ),
                    -1,
                )
            # A ray between the holes hits the real shell, not an omitted body.
            between = (data.geom_xpos[guides[0]] + data.geom_xpos[guides[1]]) / 2
            between[2] = 1.30
            self.assertGreater(
                mujoco.mj_rayMesh(
                    model,
                    data,
                    surface,
                    between,
                    np.array((0.0, 0.0, -1.0)),
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
