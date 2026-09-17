"""Invalid mechanics, calibration, unknown spectra and public-data boundaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from archetypes.task_catalog import CATALOG
from experiments.analysis import estimate_mass, oscillation
from experiments.mechanics import MechanicalError
from experiments.private_state import persistent_host_key, sample_seed
from experiments.records import ScienceRecords
from experiments.qualify import qualify
from experiments.spectrometer import WAVELENGTH_UM, analyze_spectrum, test_profile
from webui.space_experiment_api import bp


def oscillation_record(identifier, sample, mass, dt=0.02):
    t = np.arange(dt, 6 + dt / 2, dt)
    decay = 0.02 / (2 * mass)
    omega = np.sqrt(20 / mass - decay**2)
    x = 0.005 * np.exp(-decay * t) * np.sin(omega * t)
    return dict(
        measurement_id=identifier,
        sample_id=sample,
        method="inertial",
        time_position=np.column_stack([t, x]).tolist(),
        valid=True,
        reference_mass_kg=0.1 if sample == "reference" else None,
        encoder_noise_std_m=1e-6,
    )


def spectrum_record(identifier, sample, profile=0):
    r = (
        np.zeros(192)
        if sample is None
        else (np.full(192, 0.95) if sample == "reference" else test_profile(profile))
    )
    return dict(
        measurement_id=identifier,
        sample_id=sample,
        raw_counts=(1.2 * r + 0.02).tolist(),
        known_reflectance=r.tolist() if sample == "reference" else None,
        wavelength_um=WAVELENGTH_UM.tolist(),
        calibration_version="test-v1",
        valid=True,
        saturated=False,
    )


class SpaceExperiments(unittest.TestCase):
    def test_qualification_cannot_overwrite_existing_episode_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "isaac/space_lunar_sample_transfer/seed_000").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                qualify(
                    root, ["isaac"], [0], False, True, ["lunar"], ["sample_transfer"], 6
                )

    def test_science_http_analysis_reads_only_registered_job_measurements(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "measurements"
            folder.mkdir()
            for identifier, sample, mass in (
                ("m0001", None, 0.2),
                ("m0002", "reference", 0.3),
                ("m0003", "candidate", 0.36),
            ):
                record = dict(
                    oscillation_record(identifier, sample, mass), private_mass_kg=mass
                )
                (folder / (identifier + ".json")).write_text(json.dumps(record))
            with patch(
                "webui.space_experiment_api.get_job", return_value={"output": directory}
            ):
                base = "/api/science/jobs/" + "a" * 32
                public = client.get(base + "/measurements/m0003").get_json()
                self.assertNotIn("private_mass_kg", public)
                payload = dict(
                    method="mass", measurement_ids=["m0001", "m0002", "m0003"]
                )
                result = client.post(base + "/analyze", json=payload)
                self.assertEqual(result.status_code, 200)
                self.assertAlmostEqual(
                    result.get_json()["total_mass_kg"], 0.16, places=5
                )
                for invalid in (
                    dict(payload, path="/etc/passwd"),
                    dict(payload, method="read_stage"),
                    dict(payload, measurement_ids=["m0001"] * 3),
                    dict(payload, measurement_ids=["m0003", "m0002", "m0001"]),
                ):
                    self.assertEqual(
                        client.post(base + "/analyze", json=invalid).status_code, 400
                    )
        self.assertEqual(
            client.get("/api/science/jobs/" + "b" * 32 + "/measurements").status_code,
            404,
        )

    def test_persistent_host_campaign_pairs_backends_without_publishing_the_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            first = persistent_host_key(path)
            second = persistent_host_key(path)
            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(sample_seed(first, 3), sample_seed(second, 3))
            self.assertNotEqual(sample_seed(first, 3), sample_seed(first, 4))

    def test_invalid_operations_do_not_release_or_attach_samples(self):
        for world in ("orbital", "lunar", "martian"):
            cls, _ = CATALOG[f"space_{world}_sample_transfer"].load_classes()
            task = cls(cls.load())
            task.reset(0)
            active = task.data.eq_active.copy()
            for operation in (
                lambda: task.mechanics.release_storage(task.data, "candidate"),
                lambda: task.mechanics.lock(task.data, "candidate"),
                lambda: task.mechanics.unlock(task.data, "candidate"),
                lambda: task.mechanics.start_measurement(
                    task.data, "candidate", "inertial"
                ),
                lambda: task.spectrometer.measure(task.data, "candidate"),
            ):
                with self.assertRaises(MechanicalError):
                    operation()
            np.testing.assert_array_equal(task.data.eq_active, active)
            self.assertFalse(task.check())

    def test_microgravity_static_weighing_is_rejected_not_reported_as_zero_mass(self):
        cls, _ = CATALOG["space_orbital_mass_measurement"].load_classes()
        task = cls(cls.load())
        task.reset(0)
        with self.assertRaisesRegex(MechanicalError, "unidentifiable"):
            task.mechanics.start_measurement(task.data, None, "static_force")

    def test_reset_restores_fixture_transforms_and_seeded_mass(self):
        cls, _ = CATALOG["space_lunar_sample_transfer"].load_classes()
        task = cls(cls.load())
        task.reset(5)
        original = task.model.eq_data.copy()
        mass = float(task.model.body("sample_candidate").mass[0])
        task.model.eq_data[-1, 3] += 0.1
        task.mechanics.events.append({"operation": "invalid"})
        task.reset(5)
        np.testing.assert_array_equal(task.model.eq_data, original)
        self.assertEqual(float(task.model.body("sample_candidate").mass[0]), mass)
        self.assertEqual(task.mechanics.events, [])
        self.assertFalse(task.check())

    def test_damping_corrected_mass_estimate_survives_sample_rate_changes(self):
        for dt in (0.01, 0.02, 0.04):
            records = [
                oscillation_record("m0001", None, 0.2, dt),
                oscillation_record("m0002", "reference", 0.3, dt),
                oscillation_record("m0003", "candidate", 0.36, dt),
            ]
            self.assertAlmostEqual(
                estimate_mass(*records)["total_mass_kg"], 0.16, places=5
            )

    def test_short_invalid_or_motionless_encoder_data_cannot_estimate_mass(self):
        record = oscillation_record("m0001", None, 0.2)
        for replacement in (
            {"valid": False},
            {"time_position": record["time_position"][:20]},
            {"time_position": [[t, 0] for t, _ in record["time_position"]]},
        ):
            with self.assertRaises(ValueError):
                oscillation(dict(record, **replacement))

    def test_spectral_calibration_and_unknown_rejection(self):
        dark = spectrum_record("s0001", None)
        reference = spectrum_record("s0002", "reference")
        for index in range(4):
            sample = spectrum_record("s0003", "candidate", index)
            result = analyze_spectrum(dark, reference, [sample])
            self.assertEqual(
                result["match"], None if index == 3 else f"analytic_profile_{index}"
            )
        with self.assertRaisesRegex(ValueError, "versions"):
            analyze_spectrum(
                dark, dict(reference, calibration_version="other"), [sample]
            )
        with self.assertRaises(ValueError):
            analyze_spectrum(dark, reference, [dict(sample, saturated=True)])

    def test_public_records_reject_paths_truth_fields_and_unregistered_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "measurements"
            folder.mkdir()
            record = dict(
                oscillation_record("m0001", None, 0.2),
                private_mass_kg=123,
                hidden_material="secret",
                stage_path="secret",
            )
            (folder / "m0001.json").write_text(json.dumps(record))
            store = ScienceRecords(directory)
            self.assertEqual(store.list_measurements(), ["m0001"])
            public = store.read_measurement("m0001")
            self.assertNotIn("private_mass_kg", public)
            self.assertNotIn("hidden_material", public)
            self.assertNotIn("stage_path", public)
            for identifier in (
                "../source/model",
                "/etc/passwd",
                "m0001.json",
                "m00001",
            ):
                with self.assertRaises(ValueError):
                    store.read_measurement(identifier)
            with self.assertRaises(ValueError):
                store.analyze("read_stage", ["m0001"])

    def test_gallery_and_capability_routes_reject_unknown_media(self):
        app = Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()
        data = client.get("/api/space-experiments").get_json()
        self.assertEqual(len(data["worlds"]), 3)
        self.assertEqual(len(data["operations"]), 3)
        self.assertIn("read_mass_truth", data["capabilities"]["unavailable"])
        for url in (
            "/api/space-experiments/earth/sample_transfer/isaac/image",
            "/api/space-experiments/orbital/read_stage/isaac/image",
            "/api/space-experiments/orbital/sample_transfer/isaac/private",
        ):
            self.assertEqual(client.get(url).status_code, 404)


if __name__ == "__main__":
    unittest.main()
