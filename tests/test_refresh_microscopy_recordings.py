"""Offline reconstruction rejects missing evidence and preserves contact timing."""

import json
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))
from microscopy.refresh_recordings import (ArchivedSlides, digest, exact_rows,
                                          handling_contacts, recovered_episode, refresh_episode)
from microscopy.pipettes import INJECTION_PROFILE
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

    def test_recovered_archive_keeps_its_own_pipette_geometry(self):
        with TemporaryDirectory() as folder:
            episode, recovered, image_path, _, _ = self.recovered_fixture(Path(folder))
            state_path = episode/"observations/00000/state.json"
            state = json.loads(state_path.read_text())
            profile = [[0., .6e-6, .25e-6], [100e-6, 1e-6, .6e-6]]
            state["sample_scale"] = {"injection_profile_m": profile}
            state_path.write_text(json.dumps(state))
            original_hash = digest(image_path)
            observations, _ = recovered_episode(episode, recovered)
            appearance = observations[0][2]
            self.assertEqual(appearance["injection_profile_m"], profile)
            self.assertEqual(appearance["pipette_mouth_outline"], "closed_polygon")
            self.assertEqual(digest(image_path), original_hash)

    def presentation_fixture(self, root, operation="suction_injection"):
        episode = root/"original"
        observations, hashes = [], []

        def render(state, *, channel="phase_contrast"):
            value = 40 if channel == "phase_contrast" else 80
            if state["pipette_mouth_outline"] == "longitudinal_walls":
                value += 20
            return Image.new("RGB", (4, 4), (value, value, value))

        for index, indentation in enumerate((0., 1e-6)):
            folder = episode/"observations"/f"{index:05d}"
            folder.mkdir(parents=True)
            state = dict(operation=operation, time_s=index*.25, force_nn=index*25., delivered_pl=index*.2,
                         indentation_m=indentation, punctured=bool(index),
                         injection_profile_m=[[0., .6e-6, .25e-6], [100e-6, 1e-6, .6e-6]],
                         pipette_mouth_outline="closed_polygon", pipette_depth_mode="overlay")
            for name in ("state.json", "image_state.json"):
                (folder/name).write_text(json.dumps(state))
            for name in ("overview.png", "closeup.png"):
                Image.new("RGB", (4, 4), "blue").save(folder/name)
            render(state).save(folder/"microscope.png")
            render(state, channel="fluorescence").save(folder/"microscope_fluorescence.png")
            observations.append((folder, deepcopy(state), state))
            hashes.append({path.name: digest(path) for path in folder.iterdir()})
        recording = root/"demo.mp4.json"
        recording.write_text(json.dumps(dict(episode=str(episode), observation_hashes=hashes,
                                             sha256="original-video-hash", fps=4)))
        scope = SimpleNamespace(render=Mock(side_effect=render),
                                public_calibration=lambda: {"pipette_depth_mode": "cell_volume"})
        return recording, observations, hashes, scope

    def test_current_presentation_validates_archive_before_regenerating_every_frame(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            recording, observations, hashes, scope = self.presentation_fixture(root)
            original_states = deepcopy([item[2] for item in observations])
            with patch("microscopy.refresh_recordings.reconstruct_episode", return_value=(observations, {})), \
                    patch("microscopy.refresh_recordings.CellMicroscope", return_value=scope):
                evidence = refresh_episode(recording, root/"source", root/"revised",
                                           encode=False, current_presentation=True)
            self.assertEqual(evidence["microscope_observations_regenerated"], 2)
            self.assertEqual(evidence["microscope_observations_reused"], 0)
            self.assertTrue(evidence["baseline_pixel_identical"])
            self.assertNotIn("cell_projection_mode", evidence["presentation_overrides"])
            self.assertNotIn("capillary_blur_model", evidence["presentation_overrides"])
            self.assertEqual([item[2] for item in observations], original_states)
            calls = scope.render.call_args_list
            self.assertEqual(len(calls), 6)  # Two original baseline channels, then two per frame.
            for call, channel in zip(calls[:2], ("phase_contrast", "fluorescence")):
                self.assertEqual(call.args[0], original_states[0])
                self.assertEqual(call.kwargs["channel"], channel)
            for (original, _, _), expected_hashes, old_state in zip(observations, hashes, original_states):
                destination = root/"revised/observations"/original.name
                revised = json.loads((destination/"image_state.json").read_text())
                overrides = evidence["presentation_overrides"]
                self.assertEqual(revised["injection_profile_m"], [list(row) for row in INJECTION_PROFILE.sections])
                self.assertEqual(revised["pipette_mouth_outline"], "longitudinal_walls")
                self.assertEqual(revised["pipette_depth_mode"], "cell_volume")
                self.assertEqual({key: value for key, value in revised.items() if key not in overrides},
                                 {key: value for key, value in old_state.items() if key not in overrides})
                for name, expected in expected_hashes.items():
                    self.assertEqual(digest(original/name), expected)
                for name in ("state.json", "overview.png", "closeup.png"):
                    self.assertTrue((destination/name).is_symlink())
                    self.assertEqual((destination/name).resolve(), (original/name).resolve())
                for name, value in (("microscope.png", 60), ("microscope_fluorescence.png", 100)):
                    self.assertFalse((destination/name).is_symlink())
                    with Image.open(destination/name) as image:
                        self.assertTrue(np.all(np.asarray(image) == value))

    def test_current_adherent_presentation_records_optical_modes_after_baseline_check(self):
        for operation in ("cell_injection", "injection"):
            with self.subTest(operation=operation), TemporaryDirectory() as folder:
                root = Path(folder)
                recording, observations, _, scope = self.presentation_fixture(root, operation)
                with patch("microscopy.refresh_recordings.reconstruct_episode", return_value=(observations, {})), \
                        patch("microscopy.refresh_recordings.CellMicroscope", return_value=scope):
                    evidence = refresh_episode(recording, root/"source", root/"revised",
                                               encode=False, current_presentation=True)
                for key, value in (("cell_projection_mode", "phase_defocus"),
                                   ("capillary_blur_model", "aperture")):
                    self.assertEqual(evidence["presentation_calibration"][key], value)
                    for call in scope.render.call_args_list[:2]:
                        self.assertNotIn(key, call.args[0])
                    for path in (root/"revised/observations").glob("*/image_state.json"):
                        self.assertEqual(json.loads(path.read_text())[key], value)
                self.assertIn("phase_contrast.py", evidence["renderer_sha256"])

    def test_current_presentation_rejects_bad_baseline_before_creating_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            recording, observations, _, scope = self.presentation_fixture(root)
            scope.render.side_effect = lambda state, **kwargs: Image.new("RGB", (4, 4), "red")
            with patch("microscopy.refresh_recordings.reconstruct_episode", return_value=(observations, {})), \
                    patch("microscopy.refresh_recordings.CellMicroscope", return_value=scope):
                with self.assertRaisesRegex(ValueError, "Undeformed baseline differs"):
                    refresh_episode(recording, root/"source", root/"revised",
                                    encode=False, current_presentation=True)
            self.assertFalse((root/"revised").exists())
            scope.render.assert_called_once()
            self.assertEqual(scope.render.call_args.args[0]["pipette_depth_mode"], "overlay")


if __name__ == "__main__":
    unittest.main()
