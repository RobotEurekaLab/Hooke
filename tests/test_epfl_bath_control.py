"""Dual-hood controls must move the bath while retaining fixed glassware."""

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from flask import Flask
import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from webui import real_labs_api


SCENE = "epfl_lspn_chemistry"


class EpflBathControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.xml = ET.tostring(root, encoding="unicode")
        cls.model = mujoco.MjModel.from_xml_string(cls.xml)

    def setUp(self):
        self.session = InstrumentSession(self.model, deepcopy(self.manifest))

    def heights(self):
        return {
            owner: drive.measured(self.session.data)
            for (owner, _), drive in self.session.mechanisms.items()
        }

    def test_namespaces_and_bath_reference_are_unambiguous(self):
        self.assertEqual(len(self.session.controls), 2)
        self.assertEqual(len(self.session.mechanisms), 2)
        self.assertEqual(self.model.nu, 4)
        left, right = self.session.mechanisms.values()
        self.assertTrue(set(left.joints.values()).isdisjoint(right.joints.values()))
        self.assertNotEqual(left.actuator, right.actuator)
        for item in self.session.inspect()["equipment"]:
            control = item["controls"]["bath_height"]
            self.assertEqual(control["range"], [0.245, 0.375])
            self.assertAlmostEqual(control["measured"], 0.245, places=8)
            self.assertEqual(control["units"], "m")
            self.assertNotIn("flask_lift", item["controls"])
        with self.assertRaisesRegex(KeyError, "flask is fixed"):
            self.session.command("left_hood", "flask_lift", 0.09)
        self.assertEqual(self.session.data.time, 0)

    def test_independent_baths_hold_during_sash_motion_and_reset(self):
        sites = [
            self.model.site(item["sample_site"]).id
            for item in self.manifest["equipment"]
        ]
        fixed = self.session.data.site_xpos[sites].copy()
        result = self.session.command("left_hood", "bath_height", 0.375)
        self.assertGreaterEqual(result["trajectory_duration_s"], 15 - 1e-8)
        self.assertTrue(result["target_reached"])
        for owner in ("left_hood", "right_hood"):
            self.session.command(owner, "sash_height", 0.48)
            self.assertAlmostEqual(self.heights()["left_hood"], 0.375, delta=0.0005)
            self.assertAlmostEqual(self.heights()["right_hood"], 0.245, delta=0.0005)
        result = self.session.command("right_hood", "bath_height", 0.31)
        self.assertTrue(result["target_reached"])
        self.assertAlmostEqual(self.heights()["left_hood"], 0.375, delta=0.0005)
        result = self.session.command("left_hood", "bath_height", 0.245)
        self.assertTrue(result["target_reached"])
        self.assertAlmostEqual(self.heights()["right_hood"], 0.31, delta=0.0005)
        np.testing.assert_allclose(
            self.session.data.site_xpos[sites], fixed, atol=1e-12
        )
        self.assertFalse(np.any(self.session.data.warning.number))
        self.session.reset()
        np.testing.assert_allclose(self.session.data.qpos, self.model.qpos0, atol=1e-12)
        self.assertEqual(self.session.data.time, 0)
        self.session.advance(500)
        for measured in self.heights().values():
            self.assertAlmostEqual(measured, 0.245, delta=0.0005)

    def test_shared_drive_ownership_and_invalid_commands_fail_before_motion(self):
        metadata = deepcopy(self.manifest)
        metadata["equipment"][1]["mechanisms"] = deepcopy(
            metadata["equipment"][0]["mechanisms"]
        )
        with self.assertRaisesRegex(ValueError, "already owned"):
            InstrumentSession(self.model, metadata)
        initial = self.session.data.qpos.copy()
        for value in (0.244, 0.376, float("nan")):
            with self.assertRaises(ValueError):
                self.session.command("left_hood", "bath_height", value)
        for steps in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                self.session.advance(steps)
        np.testing.assert_array_equal(self.session.data.qpos, initial)
        self.assertEqual(self.session.data.time, 0)

    def test_missing_physics_step_cannot_report_success(self):
        with patch("real_labs.runtime.mujoco.mj_step"):
            with self.assertRaisesRegex(RuntimeError, "time discontinuity"):
                self.session.advance(1)

    def test_final_step_error_in_the_other_hood_is_detected_immediately(self):
        original = mujoco.mj_step
        right = self.session.mechanisms["right_hood", "bath_height"]

        def disturbed(model, data):
            original(model, data)
            data.qpos[right.qpos["a1_angle"]] += 0.05

        with patch("real_labs.runtime.mujoco.mj_step", side_effect=disturbed):
            with self.assertRaisesRegex(RuntimeError, "tracking_error|closure_error"):
                self.session.advance(1)

    def test_web_exposes_observed_height_and_rejects_retired_interface(self):
        app = Flask(__name__)
        app.register_blueprint(real_labs_api.bp)
        with patch.object(real_labs_api, "_build_session", return_value=self.session):
            client = app.test_client()
            path = f"/api/real-labs/{SCENE}"
            state = client.get(path + "/state").get_json()
            self.assertIn("bath_height", state["equipment"][0]["controls"])
            response = client.post(
                path + "/command",
                json={
                    "equipment": "left_hood",
                    "control": "flask_lift",
                    "value": 0.09,
                },
            )
            self.assertEqual(response.status_code, 400)
            response = client.post(
                path + "/command",
                json={
                    "equipment": "left_hood",
                    "control": "bath_height",
                    "value": 0.255,
                },
            )
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertTrue(response.get_json()["result"]["target_reached"])
            state = client.post(path + "/reset", json={}).get_json()
            self.assertEqual(state["time_s"], 0)
            self.assertAlmostEqual(
                state["equipment"][0]["controls"]["bath_height"]["measured"],
                0.245,
                places=8,
            )

    def test_export_settling_keeps_both_baths_at_their_reset_height(self):
        from real_labs import export

        observed = {}
        original = export.export_snapshot

        def capture(model, data, destination):
            for item in self.manifest["equipment"]:
                info = item["mechanisms"]["bath_height"]
                rim = model.site(info["observation_site"]).id
                datum = model.site(info["datum_site"]).id
                observed[item["id"]] = float(
                    data.site_xpos[rim, 2] - data.site_xpos[datum, 2]
                )
            observed["time"] = float(data.time)
            return original(model, data, destination)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(export, "export_snapshot", side_effect=capture):
                result = export.export_scene(SCENE, directory, validate=False)
        self.assertTrue(result["compiled"])
        self.assertAlmostEqual(observed["time"], 1.0, places=8)
        for owner in ("left_hood", "right_hood"):
            self.assertAlmostEqual(observed[owner], 0.245, delta=0.0005)


if __name__ == "__main__":
    unittest.main()
