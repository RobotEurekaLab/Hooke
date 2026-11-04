"""Offline reconstruction rejects missing evidence and preserves contact timing."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))
from microscopy.refresh_recordings import (ArchivedSlides, digest, exact_rows,
                                          handling_contacts, recovered_episode)
from microscopy.scene import ORIGIN


class ArchivedMicroscopyTests(unittest.TestCase):
    def test_slide_forward_kinematics_uses_rotated_body_and_joint_reference(self):
        with TemporaryDirectory() as folder:
            source = Path(folder)
            (source/"scene.json").write_text(json.dumps({"names": {
                "joint": ["slide"], "geom": ["probe"], "site": ["tip"]}}))
            np.savez(source/"model.npz",
                body_parentid=[0, 0], body_pos=[[0, 0, 0], [1, 2, 3]],
                body_quat=[[1, 0, 0, 0], [2**-.5, 0, 0, 2**-.5]],
                jnt_type=[2], jnt_bodyid=[1], jnt_axis=[[1, 0, 0]], jnt_qposadr=[0], qpos0=[.2],
                geom_bodyid=[1], geom_pos=[[0, 1, 0]], geom_quat=[[1, 0, 0, 0]], geom_size=[[1, 1, 1]],
                site_bodyid=[1], site_pos=[[0, 0, 1]])
            model = ArchivedSlides(source)
            pose = model.pose({"slide": .7})
            np.testing.assert_allclose(model.element(pose, "site", "tip")[0], [1, 2.5, 4])
            np.testing.assert_allclose(model.geom(pose, "probe")[0], [0, 2.5, 3], atol=1e-15)
            with self.assertRaisesRegex(ValueError, "Missing actual joint feedback"):
                model.pose({})

    def test_observation_matching_does_not_interpolate_or_accept_invalid_trace(self):
        times = np.array([.001, .002, .003])
        np.testing.assert_array_equal(exact_rows(times, [.001, .003]), [0, 2])
        for observed in ([.0015], [.004]):
            with self.assertRaisesRegex(ValueError, "exact interaction sample"):
                exact_rows(times, observed)
        for invalid in (np.array([]), np.array([1., 1.]), np.array([np.nan])):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                exact_rows(invalid, [.001])

    def test_probe_compression_uses_contact_tick_and_resulting_cell_pose(self):
        radius, probe_radius = 18e-6, 2e-6
        # Contact was eligible at the pre-solve cell position. Compression is
        # evaluated after the overdamped solve, as in the mechanics update.
        trace = {"contact_cell_world_m": np.array([[19e-6, 0, 0]]),
                 "cell_world_m": np.array([[19.8e-6, 0, 0]])}
        model = SimpleNamespace(geom=lambda pose, name: (np.zeros(3), np.eye(3), np.full(3, probe_radius)))
        state = {"cell_radii_m": [radius]*3}
        handling_contacts(model, None, trace, 0, state, "push")
        self.assertAlmostEqual(state["indentation_m"], .2e-6, places=15)
        np.testing.assert_allclose(state["membrane_contacts"][0]["offset_m"], [-radius, 0, 0])

    def recovered_fixture(self, root):
        episode, recovered = root/"original", root/"recovered"
        for path in (episode, recovered):
            (path/"observations/00000").mkdir(parents=True)
        source_state = {"time_s": .001, "encoders_m": {"focus": 0.}}
        (episode/"observations/00000/state.json").write_text(json.dumps(source_state))
        np.savez(episode/"interaction.npz", time_s=[.001], tip_world_m=[ORIGIN], cell_world_m=[ORIGIN])
        checksum = digest(episode/"interaction.npz")
        (episode/"result.json").write_text(json.dumps({"interaction_trace": {"sha256": checksum}}))
        image = {"time_s": .001, "focus_m": 0., "tip_m": [0., 0., 0.], "centres_m": [[0., 0., 0.]]}
        image_path = recovered/"observations/00000/image_state.json"
        image_path.write_text(json.dumps(image))
        evidence = {"source_episode": str(episode), "source_interaction_sha256": checksum,
                    "exact_trace_match": True, "exact_encoder_match": True,
                    "image_state_sha256": {"00000": digest(image_path)}}
        (recovered/"evidence.json").write_text(json.dumps(evidence))
        return episode, recovered, image_path, image, evidence

    def test_recovered_states_require_verified_content_and_matching_time(self):
        with TemporaryDirectory() as folder:
            episode, recovered, image_path, image, evidence = self.recovered_fixture(Path(folder))
            observations, _ = recovered_episode(episode, recovered)
            self.assertEqual(len(observations), 1)
            image["time_s"] = .002
            image_path.write_text(json.dumps(image))
            with self.assertRaisesRegex(ValueError, "differs from its replay evidence"):
                recovered_episode(episode, recovered)
            evidence["image_state_sha256"]["00000"] = digest(image_path)
            (recovered/"evidence.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "state time differs"):
                recovered_episode(episode, recovered)

    def test_recovered_states_reject_missing_files_or_unverified_replay(self):
        with TemporaryDirectory() as folder:
            episode, recovered, image_path, _, evidence = self.recovered_fixture(Path(folder))
            image_path.unlink()
            with self.assertRaises(FileNotFoundError):
                recovered_episode(episode, recovered)
            evidence["exact_encoder_match"] = False
            (recovered/"evidence.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "verified exact trace and encoder equality"):
                recovered_episode(episode, recovered)


if __name__ == "__main__":
    unittest.main()
