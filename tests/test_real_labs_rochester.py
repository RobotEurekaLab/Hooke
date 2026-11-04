"""Rochester dry-contact and free-carrier access: geometry and measured failures."""

from copy import copy, deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.motion import _model_signature
from real_labs.rochester_motion import _Run, run_rochester_access
from real_labs.runtime import InstrumentSession


class RochesterAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("rochester_genomics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_actual_dry_contact_and_loaded_access_return(self):
        session = self.session()
        signature = _model_signature(session.model)
        report = run_rochester_access(session)
        self.assertTrue(report["success"], report)
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["physics_steps"], report["completed_checks"])
        self.assertEqual(report["physics_steps"], report["geometry_checks"])
        self.assertGreater(report["physics_steps"], 14000)
        self.assertGreater(report["tip_contact_steps"], 200)
        touch = next(p for p in report["phases"] if p["name"] == "dry_touch")
        self.assertEqual(touch["contacting_tips"], 8)
        self.assertGreater(touch["tip_normal_force_n"], 0.02)
        self.assertLess(report["peak_tip_normal_force_n"], 0.1)
        self.assertLess(report["maximum_tip_penetration_m"], 0.00005)
        self.assertLess(
            max(report["carrier_maximum_tracking_error_m"].values()), 0.0005
        )
        for phase in report["phases"]:
            self.assertTrue(phase["success"])
            self.assertEqual(len(phase["qpos"]), session.model.nq)
            self.assertEqual(len(phase["qvel"]), session.model.nv)
            self.assertEqual(len(phase["ctrl"]), session.model.nu)
            self.assertEqual(phase["completed_checks"], phase["executed_steps"])
        opened = next(
            p for p in report["phases"] if p["name"] == "inspect_loading_access"
        )
        self.assertAlmostEqual(opened["measured_m"][3], 0.25, delta=0.0002)
        self.assertIn(
            "sequencer_loading",
            report["open_loading_camera_screening"]["candidate_views"],
        )
        self.assertTrue(report["closed_readiness"]["ready"])
        self.assertEqual(report["forbidden_contacts"], [])
        self.assertFalse(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertEqual(signature, _model_signature(session.model))
        np.testing.assert_allclose(report["phases"][-1]["measured_m"], 0, atol=0.0003)

    def test_open_front_and_retained_free_carrier_clear_full_tray_sweep(self):
        session = self.session()
        m, d = session.model, session.data
        tray = session.controls["sequencer", "flowcell_tray"]
        carrier = m.body("sequencer__specimen").id
        cj = m.body_jntadr[carrier]
        self.assertEqual(m.jnt_type[cj], mujoco.mjtJoint.mjJNT_FREE)
        carrier_y = m.jnt_qposadr[cj] + 1
        initial_y = float(d.qpos[carrier_y])
        frames = [
            m.geom("sequencer__" + n).id
            for n in (
                "front_left",
                "front_right",
                "separator",
                "upper_lintel",
                "drawer_divider",
                "lower_sill",
            )
        ]
        sample = m.geom("sequencer__specimen__collision").id
        for q in np.linspace(0, 0.25, 51):
            d.qpos[m.jnt_qposadr[tray.joint]] = q
            d.qpos[carrier_y] = initial_y - q
            mujoco.mj_forward(m, d)
            for frame in frames:
                self.assertTrue(m.geom_contype[frame] and m.geom_conaffinity[frame])
                self.assertGreater(
                    mujoco.mj_geomDistance(m, d, sample, frame, 1, None), 0.005
                )
            for side in (-1, 1):
                guide = m.geom(f"sequencer__flowcell_guide_{side}").id
                runner = m.geom(f"sequencer__flowcell_runner_{side}").id
                front = m.geom("sequencer__flowcell_front").id
                self.assertAlmostEqual(
                    d.geom_xpos[runner, 2] - m.geom_size[runner, 2],
                    d.geom_xpos[guide, 2] + m.geom_size[guide, 2],
                )
                overlap = min(
                    d.geom_xpos[runner, 1] + m.geom_size[runner, 1],
                    d.geom_xpos[guide, 1] + m.geom_size[guide, 1],
                ) - max(
                    d.geom_xpos[runner, 1] - m.geom_size[runner, 1],
                    d.geom_xpos[guide, 1] - m.geom_size[guide, 1],
                )
                self.assertGreater(overlap, 0.15)
                self.assertGreater(
                    mujoco.mj_geomDistance(m, d, guide, front, 1, None), 0.004
                )
            self.assertFalse([c for c in d.contact if c.dist < -1e-5])

    def test_tip_and_carrier_collision_contract_cannot_be_disabled(self):
        names = [f"liquid_handler__tip_contact_{i}" for i in range(1, 9)]
        names += [f"liquid_handler__tip_mount_contact_{i}" for i in range(1, 9)]
        names += [
            "liquid_handler__head_contact",
            "liquid_handler__gantry_contact",
            "liquid_handler__carriage_contact",
            "sequencer__flowcell_front",
            "sequencer__specimen__collision",
        ]
        for name in names:
            s = self.session()
            g = s.model.geom(name).id
            s.model.geom_contype[g] = s.model.geom_conaffinity[g] = 0
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "collision"
            ):
                run_rochester_access(s)
            self.assertEqual(s.data.time, 0)
        s = self.session()
        s.model.actuator_forcelimited[
            s.controls["liquid_handler", "pipette_z"].actuator
        ] = 0
        with self.assertRaisesRegex(ValueError, "force limits"):
            run_rochester_access(s)

    def test_binding_rejects_wrong_tool_frame_and_model_mutation(self):
        s = self.session()
        s.equipment["liquid_handler"]["dry_contact_interface"][
            "tool_site"
        ] = "sequencer__site_flowcell_1"
        with self.assertRaisesRegex(ValueError, "moving head"):
            run_rochester_access(s)
        s = self.session()
        run = _Run(s)
        s.model.geom_contype[s.model.geom("liquid_handler__tip_contact_1").id] = 0
        with self.assertRaisesRegex(RuntimeError, "compiled_model_changed"):
            run.phase("unsafe", run.home, 0.1, 0.1)
        self.assertEqual(s.data.time, 0)

    def test_real_obstacle_aborts_without_reset_or_false_coverage(self):
        root = deepcopy(self.root)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="dry_path_obstacle",
            type="box",
            pos="-1.554 1.1365 1.20",
            size=".055 .018 .006",
            contype="1",
            conaffinity="1",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            self.manifest,
        )
        signature = _model_signature(s.model)
        r = run_rochester_access(s)
        self.assertFalse(r["success"])
        self.assertEqual(r["failure"], "unexpected_instrument_collision")
        self.assertIn("dry_path_obstacle", r["forbidden_contacts"][0]["geoms"])
        self.assertGreater(r["physics_steps"], 0)
        self.assertEqual(r["geometry_checks"], r["physics_steps"])
        self.assertEqual(r["completed_checks"], r["physics_steps"] - 1)
        self.assertFalse(r["coverage_complete"])
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertFalse(r["phases"][-1]["success"])
        self.assertEqual(signature, _model_signature(s.model))
        self.assertGreater(s.data.time, 0)

    def test_preflight_obstacle_does_not_establish_executed_coverage(self):
        root = deepcopy(self.root)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="initial_tool_obstacle",
            type="box",
            pos="-1.7 1.269 1.335",
            size=".04 .01 .005",
            contype="1",
            conaffinity="1",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            self.manifest,
        )
        r = run_rochester_access(s)
        self.assertFalse(r["success"])
        self.assertEqual(r["physics_steps"], 0)
        self.assertEqual(r["completed_checks"], 0)
        self.assertFalse(r["coverage_complete"])
        self.assertEqual(r["preflight_geometry_checks"], 1)
        self.assertEqual(r["preflight_completed_checks"], 0)
        self.assertTrue(r["hold_requested"])

    def test_external_force_on_held_axis_fails_before_completed_check(self):
        s = self.session()
        held = s.controls["liquid_handler", "front_door_hinge"]
        s.data.qfrc_applied[s.model.jnt_dofadr[held.joint]] = 1000
        r = run_rochester_access(s)
        self.assertFalse(r["success"])
        self.assertEqual(r["failure"], "inactive_control_changed_or_drifted")
        self.assertGreater(r["physics_steps"], 0)
        self.assertEqual(r["completed_checks"], r["physics_steps"] - 1)
        self.assertLess(r["geometry_checks"], r["physics_steps"])
        self.assertFalse(r["coverage_complete"])
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])

    def test_lost_free_carrier_fails_without_hidden_repositioning(self):
        s = self.session()
        body = s.model.body("liquid_handler__specimen").id
        s.data.xfrc_applied[body, 0] = 2
        r = run_rochester_access(s)
        self.assertEqual(r["failure"], "carrier_displaced_or_not_retained")
        self.assertFalse(r["success"])
        self.assertEqual(r["physics_steps"], 1)
        self.assertEqual(r["geometry_checks"], 0)
        self.assertEqual(r["completed_checks"], 0)
        self.assertFalse(r["coverage_complete"])
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])

    def test_carrier_weld_cannot_replace_retention(self):
        root = deepcopy(self.root)
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            body1="sequencer__specimen",
            body2="sequencer__body_flowcell_tray",
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            self.manifest,
        )
        with self.assertRaisesRegex(ValueError, "equality constraints"):
            run_rochester_access(s)
        self.assertEqual(s.data.time, 0)

    def test_foreign_or_replaced_data_rejected_before_execution(self):
        s = self.session()
        foreign = InstrumentSession(copy(s.model), self.manifest)
        s.data = foreign.data
        before = s.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "exact model"):
            run_rochester_access(s)
        np.testing.assert_array_equal(s.data.qpos, before)
        self.assertEqual(s.data.time, 0)
        s = self.session()
        run = _Run(s)
        old_data = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            run.phase("replaced_state", run.home, 0.1, 0.1)
        self.assertEqual(old_data.time, 0)
        self.assertEqual(s.data.time, 0)

    def test_parent_filtered_guides_clear_the_full_retained_axis_ranges(self):
        s = self.session()
        run = _Run(s)
        m, d = s.model, s.data
        self.assertGreater(len(run.filtered_pairs), 50)
        for x in (-0.28, 0, 0.28):
            for y in (-0.22, 0, 0.22):
                for z in np.linspace(0, 0.24, 13):
                    d.qpos[run.qadr[:3]] = x, y, z
                    mujoco.mj_forward(m, d)
                    for a, b in run.filtered_pairs:
                        self.assertGreaterEqual(
                            mujoco.mj_geomDistance(m, d, a, b, 0.001, None),
                            -1e-9,
                            (x, y, z, m.geom(a).name, m.geom(b).name),
                        )
                    for side in (-1, 1):
                        rail = m.geom(f"liquid_handler__z_rail_{side}").id
                        pad = m.geom(f"liquid_handler__z_bearing_{side}").id
                        overlap = min(
                            d.geom_xpos[rail, 2] + m.geom_size[rail, 2],
                            d.geom_xpos[pad, 2] + m.geom_size[pad, 2],
                        ) - max(
                            d.geom_xpos[rail, 2] - m.geom_size[rail, 2],
                            d.geom_xpos[pad, 2] - m.geom_size[pad, 2],
                        )
                        self.assertGreater(overlap, 0.04)
                        self.assertAlmostEqual(
                            mujoco.mj_geomDistance(m, d, rail, pad, 0.1, None),
                            0,
                            delta=1e-9,
                        )
                        yr = m.geom(f"liquid_handler__y_rail_{side}").id
                        for kind in ("cap", "side"):
                            for edge in (-1, 1):
                                sleeve = m.geom(
                                    f"liquid_handler__y_bearing_{side}_{kind}_{edge}"
                                ).id
                                self.assertAlmostEqual(
                                    mujoco.mj_geomDistance(m, d, yr, sleeve, 0.1, None),
                                    0.0005,
                                    delta=1e-9,
                                )

    def test_parent_filtered_penetration_is_a_real_preflight_failure(self):
        root = deepcopy(self.root)
        root.find(".//geom[@name='liquid_handler__carriage_contact']").set(
            "pos", "0 0 .002"
        )
        s = InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            self.manifest,
        )
        r = run_rochester_access(s)
        self.assertFalse(r["success"])
        self.assertEqual(r["failure"], "unexpected_instrument_collision")
        self.assertEqual(r["physics_steps"], 0)
        self.assertFalse(r["coverage_complete"])
        self.assertEqual(r["preflight_completed_checks"], 0)
        self.assertTrue(
            any(
                c.get("detection", "").startswith("direct distance")
                for c in r["forbidden_contacts"]
            )
        )


if __name__ == "__main__":
    unittest.main()
