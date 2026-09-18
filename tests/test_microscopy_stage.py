"""Compiled vendor geometry, independent slide motion and unchanged SI cells."""

import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from microscopy.cad_assets import DEFAULT_ROOT, asset_evidence
from microscopy.tasks import make_task
from microscopy.control import apply_request

ASSETS = Path(os.environ.get("HOOKE_TEST_STAGE_ASSET_ROOT", DEFAULT_ROOT))


class StageSelection(unittest.TestCase):
    def test_vendor_stage_requires_its_explicit_instrument_and_assets(self):
        with patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="reference",
                                        HOOKE_MICROSCOPY_STAND="te2000-s-reference",
                                        HOOKE_MICROSCOPY_OPTICS="estimated",
                                        HOOKE_MICROSCOPY_STAGE="x-asr100")):
            with self.assertRaisesRegex(ValueError, "requires CAD assets"):
                make_task("cell_injection")


@unittest.skipUnless(all((ASSETS / name / "manifest.json").is_file()
                        for name in ("x-asr100", "ap114", "zaber-rhf")),
                     "Vendor STEP assets are optional private fixtures")
class VendorStage(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, dict(HOOKE_MICROSCOPY_ASSETS="cad",
                                        HOOKE_MICROSCOPY_ASSET_ROOT=str(ASSETS),
                                        HOOKE_MICROSCOPY_STAND="te2000-s-reference",
                                        HOOKE_MICROSCOPY_OPTICS="estimated",
                                        HOOKE_MICROSCOPY_STAGE="x-asr100"))
        env.start()
        self.addCleanup(env.stop)
        self.task = make_task("cell_injection")
        self.task.reset(0)

    def test_all_solids_are_unscaled_and_the_sample_keeps_micrometre_dimensions(self):
        t = self.task
        points = []
        for i in range(38):
            geom = t.model.geom(f"stage_cad_x_asr100_{i}")
            mesh = t.model.mesh(int(geom.dataid[0]))
            start, count = int(mesh.vertadr[0]), int(mesh.vertnum[0])
            vertices = t.model.mesh_vert[start:start+count]
            points.append(vertices @ t.data.geom_xmat[geom.id].reshape(3, 3).T + t.data.geom_xpos[geom.id])
            self.assertEqual(t.model.geom_contype[geom.id], 0)
        np.testing.assert_allclose(np.ptp(np.concatenate(points), axis=0),
                                   [.3329, .2518, .042525], rtol=0., atol=1e-7)
        for i in range(2):
            self.assertEqual(t.model.geom(f"stage_adapter_cad_ap114_{i}").bodyid,
                             t.model.body("stage_base").id)
        np.testing.assert_allclose(t.model.geom("cell_0_shell").size * 2, [36e-6, 28e-6, 8e-6])
        self.assertAlmostEqual(t.public_state()["calibration"]["field_width_m"], 160e-6)
        np.testing.assert_allclose(t.model.joint("stage_x").range, [-.06, .06])
        np.testing.assert_allclose(t.model.joint("stage_y").range, [-.05, .05])
        evidence = asset_evidence(t.model)["stage"]
        self.assertFalse(evidence["one_to_one_verified"])
        self.assertIn("license not established", evidence["rights"])

    def test_operating_envelope_rejects_unsafe_travel_before_pressure_or_motion(self):
        t = self.task
        state = t.public_state()
        self.assertEqual(state["actuator_limits_m"]["stage_y"], [-.05, .05])
        self.assertEqual(state["command_limits_m"]["stage_y"], [-.004, .004])
        before = (t.data.qpos.copy(), t.data.ctrl.copy(), t.data.userdata.copy(), t.data.time)
        with self.assertRaisesRegex(ValueError, "objective retraction"):
            apply_request(t, dict(command="step", actuators={"stage_y": -.05},
                                  pressure_pa=5000, target_pl=.5))
        for actual, expected in zip((t.data.qpos, t.data.ctrl, t.data.userdata), before[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(t.data.time, before[3])
        t.command(dict(stage_x=.004, stage_y=-.004), .3)
        np.testing.assert_allclose([t.public_state()["encoders_m"][name] for name in ("stage_x", "stage_y")],
                                   [.004, -.004], atol=2e-7)

    @unittest.skipUnless((ASSETS / "x-asr100/native_components.json").is_file(),
                         "Factory component metadata is an optional private fixture")
    def test_feedback_moves_only_the_matching_carriages_and_sample(self):
        t = self.task
        # Select critical parts independently from the vendor's component
        # feature labels, rather than reusing the builder's partition table.
        native = json.loads((ASSETS / "x-asr100/native_components.json").read_text())
        self.assertEqual(native["assembly_sha256"], asset_evidence(t.model)["stage"]["assembly_sha256"])
        vendor_features = native["components"]
        manifest = json.loads((ASSETS / "x-asr100/manifest.json").read_text())
        expected_features = {0: ("ASR_base", "CBORE for M3 SHCS1"),
                             9: ("ASR_mid", "Cut-Extrude38"),
                             19: ("ASR_mid", "Boss-Extrude37"),
                             20: ("ASR_mid", "Boss-Extrude38"),
                             25: ("ASR_top", "M3x0.5 Tapped Hole5")}
        for index, (component, feature) in expected_features.items():
            # STEP adds occurrence suffixes to repeated feature solids.
            label = re.sub(r"\[\d+\]$", "", manifest["parts"][index]["source_name"])
            self.assertEqual(label, feature)
            self.assertIn(feature, vendor_features[component]["feature_labels"])
        parts = (0, 9, 19, 20, 25, 26, 32)
        indices = [t.model.geom(f"stage_cad_x_asr100_{i}").id for i in parts]
        original = t.data.geom_xpos[indices].copy()
        cell = t.model.geom("cell_0_shell").id
        initial_cell = t.data.geom_xpos[cell].copy()
        t.command(dict(stage_x=.003, stage_y=-.004), .3)
        feedback = t.public_state()["encoders_m"]
        dx, dy = feedback["stage_x"], feedback["stage_y"]
        expected = [[0., 0., 0.], [0., dy, 0.], [0., dy, 0.], [0., dy, 0.],
                    [dx, dy, 0.], [0., 0., 0.], [0., dy, 0.]]
        np.testing.assert_allclose(t.data.geom_xpos[indices]-original, expected, rtol=0., atol=1e-12)
        np.testing.assert_allclose(t.data.geom_xpos[cell]-initial_cell,
                                   [feedback["stage_x"], feedback["stage_y"], 0.], atol=1e-12)
        np.testing.assert_allclose([feedback["stage_x"], feedback["stage_y"]], [.003, -.004], atol=2e-7)


if __name__ == "__main__":
    unittest.main()
