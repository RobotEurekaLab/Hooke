"""Simulation-preparation reports must not promote visual exports to qualification."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.export import export_scene
from real_labs.preflight import inspect_package, scene_inventory


class ScenePreflightTests(unittest.TestCase):
    def test_accessory_without_sample_interface_remains_in_inventory(self):
        xml = '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        accessory = dict(
            id="accessory",
            kind="reuse",
            role="Passive bench accessory",
            joints={},
            actuators={},
            reference_urls=["https://example.edu/laboratory"],
        )
        report = scene_inventory(
            model, data, {"id": "legacy", "equipment": [accessory]}, xml
        )
        item = report["equipment"][0]
        self.assertIsNone(item["target"])
        self.assertEqual(item["target_interface_status"], "not_defined")
        self.assertEqual(item["sources"], accessory["reference_urls"])
        self.assertEqual(report["counts"]["equipment"], 1)

        # An explicitly declared but missing site is still a corrupt contract.
        accessory["sample_site"] = "missing_target"
        with self.assertRaises(KeyError):
            scene_inventory(
                model, data, {"id": "invalid", "equipment": [accessory]}, xml
            )

    def test_archive_preserves_mechanics_without_qualifying_isaac(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            export_scene("uwa_ngcf_c72", path)
            report = inspect_package(path, native_archive=True)
            self.assertTrue(report["acceptance"]["structural_native_attempt"])
            for key in (
                "isaac_runtime_validated",
                "simready_validated",
                "robot_task_validated",
                "scientific_process_validated",
            ):
                self.assertFalse(report["acceptance"][key])
            target = report["equipment"][0]["target"]
            self.assertEqual(target["attachment"], "clamped")
            self.assertFalse(target["freely_graspable"])
            self.assertIsNone(report["cameras"][0]["polling_rate_hz"])
            self.assertIsNone(report["cameras"][0]["runtime_resolution"])
            original = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
            restored = mujoco.MjModel.from_xml_path(
                str(path / "native_source/scene.xml")
            )
            for field in ("body_mass", "body_inertia", "jnt_range", "actuator_gainprm"):
                np.testing.assert_allclose(
                    getattr(original, field),
                    getattr(restored, field),
                    rtol=1e-4,
                    atol=1e-7,
                )
            with np.load(
                path / "native_source/model.npz", allow_pickle=False
            ) as archive:
                self.assertIn("geom_friction", archive.files)
                self.assertIn("reset_qpos", archive.files)
            metadata = json.loads((path / "native_source/scene.json").read_text())
            self.assertEqual(metadata["isaac_qualification"], "unqualified")

    def test_modified_xml_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            export_scene("uwa_ngcf_c72", path)
            xml = path / "scene.xml"
            xml.write_text(xml.read_text() + "\n<!-- changed -->")
            with self.assertRaisesRegex(ValueError, "manifest"):
                inspect_package(path, native_archive=True)
            self.assertFalse((path / "native_source").exists())

    def test_native_structural_blocker_survives_inventory(self):
        xml = '<mujoco><worldbody><body name="ball"><joint type="ball"/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        report = scene_inventory(
            model, data, {"id": "unsupported", "equipment": []}, xml
        )
        self.assertFalse(report["acceptance"]["structural_native_attempt"])
        self.assertIn("ball_joints", report["native_structural_preflight"]["blockers"])
        self.assertFalse(report["bodies"][1]["fixed_to_world"])


if __name__ == "__main__":
    unittest.main()
