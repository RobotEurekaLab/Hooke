"""Loaded free carriers need real support, retention and resolved contact dynamics."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.specimens import add_specimens
from backends.capabilities import preflight


CASES = {
    "eth_air_quality": ("aerosol_sampler", "filter_drawer"),
    "indiana_microfluidics": ("microfluidic_station", "stage_y"),
    "rochester_genomics": ("sequencer", "flowcell_tray"),
    "uva_deposition": ("deposition_chamber", "platen"),
    # The supported rack is fixed; exercise the replacement Cartesian head drive.
    "uw_isolab": ("isotope_analyzer", "v3_gantry_x"),
    "waterloo_robohub": ("mobile_robot", "wheel_left"),
}


def session_for(scene):
    root, manifest = build_scene(scene)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def exercise(session, key):
    """Observe every integration step, including other carriers on the machine."""
    model, data = session.model, session.data
    carriers = {
        item["id"]: model.geom(
            item.get("existing_free_specimen", {}).get(
                "collision_geom", item["process"]["sample_body"] + "__collision"
            )
        ).id
        for item in session.equipment.values()
        if item.get("sample_attachment") == "supported"
    }
    control = session.controls[key]
    worst = 0.0
    rows = []
    for fraction in (0.15, 0.85, 0.25):
        target = control.lower + fraction * (control.upper - control.lower)
        data.ctrl[control.actuator] = target
        start = float(data.time)
        for _ in range(round(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if (
                    contact.geom1 in carriers.values()
                    or contact.geom2 in carriers.values()
                ):
                    worst = min(worst, float(contact.dist))
        rows.append(
            dict(
                error=session._measured(control) - target,
                elapsed=float(data.time) - start,
                carriers={name: session.process_readiness(name) for name in carriers},
            )
        )
    return worst, rows


class CarrierSupportTests(unittest.TestCase):
    def test_carrier_drives_keep_the_native_supported_actuator_form(self):
        for scene in (
            "uw_isolab",
            "uva_deposition",
            "indiana_microfluidics",
            "eth_air_quality",
        ):
            with self.subTest(scene=scene):
                root, manifest = build_scene(scene)
                xml = ET.tostring(root, encoding="unicode")
                model = mujoco.MjModel.from_xml_string(xml)
                self.assertTrue(preflight(model, xml)["can_attempt_native"])
                self.assertEqual(model.na, 0)

    def test_loaded_reset_and_reversals_preserve_contact_and_free_motion(self):
        for scene, key in CASES.items():
            with self.subTest(scene=scene):
                session = session_for(scene)
                model, data = session.model, session.data
                for item in session.equipment.values():
                    if item.get("sample_attachment") != "supported":
                        continue
                    body = model.body(item["process"]["sample_body"]).id
                    joint = model.body_jntadr[body]
                    self.assertEqual(model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
                    geom = model.geom(
                        item.get("existing_free_specimen", {}).get(
                            "collision_geom",
                            item["process"]["sample_body"] + "__collision",
                        )
                    ).id
                    self.assertEqual(model.geom_contype[geom], 1)
                    self.assertEqual(model.geom_conaffinity[geom], 1)
                    self.assertNotIn(
                        "sample_not_at_target",
                        session.process_readiness(item["id"])["reasons"],
                    )
                self.assertFalse(any(contact.dist < -1e-6 for contact in data.contact))
                worst, rows = exercise(session, key)
                self.assertGreaterEqual(worst, -1e-4)
                span = session.controls[key].upper - session.controls[key].lower
                for row in rows:
                    self.assertAlmostEqual(row["elapsed"], 2.0, places=8)
                    self.assertLessEqual(abs(row["error"]), span * 0.025)
                    for observed in row["carriers"].values():
                        self.assertNotIn("sample_not_at_target", observed["reasons"])
                        self.assertNotIn("sample_moving", observed["reasons"])
                self.assertFalse(any(warning.number for warning in data.warning))

    def test_mobile_payload_rests_on_the_visible_deck(self):
        session = session_for("waterloo_robohub")
        model, data = session.model, session.data
        item = session.equipment["mobile_robot"]
        body = model.body(item["process"]["sample_body"]).id
        mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
        mujoco.mj_forward(model, data)
        deck = model.geom("mobile_robot__box_2").id
        self.assertEqual(model.geom_contype[deck], 1)
        payload_bottom = data.xpos[body, 2] - item["sample_size_m"][2] / 2
        deck_top = data.geom_xpos[deck, 2] + model.geom_size[deck, 2]
        self.assertAlmostEqual(payload_bottom, deck_top, delta=1e-4)

    def test_open_vial_holder_retains_but_does_not_attach_the_carrier(self):
        session = session_for("uw_isolab")
        exercise(session, CASES["uw_isolab"])
        model, data = session.model, session.data
        body = model.body(
            session.equipment["isotope_analyzer"]["process"]["sample_body"]
        ).id
        height_before = float(data.xpos[body, 2])
        data.xfrc_applied[body, 2] = model.body_mass[body] * 9.81 * 3
        mujoco.mj_step(model, data, nstep=round(0.20 / model.opt.timestep))
        mujoco.mj_forward(model, data)
        self.assertGreater(data.xpos[body, 2] - height_before, 0.06)
        # A physical removal test must leave the loading gate, unlike a weld.
        self.assertIn(
            "sample_not_at_target",
            session.process_readiness("isotope_analyzer")["reasons"],
        )

    def test_half_step_convergence_keeps_contact_parameters_fixed(self):
        first = session_for("uw_isolab")
        second = session_for("uw_isolab")
        second.model.opt.timestep = first.model.opt.timestep / 2
        np.testing.assert_array_equal(first.model.geom_solref, second.model.geom_solref)
        np.testing.assert_array_equal(first.model.geom_solimp, second.model.geom_solimp)
        worst_a, rows_a = exercise(first, CASES["uw_isolab"])
        worst_b, rows_b = exercise(second, CASES["uw_isolab"])
        self.assertGreaterEqual(worst_a, -1e-4)
        self.assertGreaterEqual(worst_b, -1e-4)
        for a, b in zip(rows_a, rows_b):
            self.assertAlmostEqual(a["error"], b["error"], delta=1e-5)
            self.assertAlmostEqual(
                a["carriers"]["isotope_analyzer"]["sample_distance_m"],
                b["carriers"]["isotope_analyzer"]["sample_distance_m"],
                delta=1e-4,
            )

    def test_missing_physical_support_is_not_silently_replaced_by_the_floor(self):
        root = ET.fromstring(
            """<mujoco><worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body name="instrument" pos="0 0 1"><site name="sample"/></body>
        </worldbody></mujoco>"""
        )
        item = dict(
            id="instrument",
            kind="test",
            sample_attachment="supported",
            sample_size_m=[0.02, 0.02, 0.01],
            sample_site="sample",
        )
        with self.assertRaisesRegex(ValueError, "No nearby contact support"):
            add_specimens(root, [item])


if __name__ == "__main__":
    unittest.main()
