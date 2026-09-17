"""A narrow scientific interface over a completed episode's public evidence.

The host selects the episode directory. Clients select registered measurement
IDs, never filesystem paths, simulation attributes or private evaluator data.
This API boundary is not an operating-system sandbox for arbitrary agent code.
"""

import json
from pathlib import Path
import re

from experiments.analysis import estimate_mass
from experiments.spectrometer import analyze_spectrum, public_templates

MEASUREMENT_ID = re.compile(r"[ms][0-9]{4}\Z")
PUBLIC_FIELDS = frozenset(
    (
        "measurement_id",
        "sample_id",
        "method",
        "started_s",
        "simulation_s",
        "time_position",
        "encoder_units",
        "reference_mass_kg",
        "fidelity",
        "encoder_noise_std_m",
        "spring_stiffness_n_m",
        "gravity_m_s2",
        "valid",
        "wavelength_um",
        "raw_counts",
        "known_reflectance",
        "calibration_version",
        "noise_std_counts",
        "saturated",
        "data_source",
    )
)


class ScienceRecords:
    def __init__(self, episode_directory):
        self.directory = Path(episode_directory)

    def list_measurements(self):
        folder = self.directory / "measurements"
        return sorted(
            p.stem for p in folder.glob("*.json") if MEASUREMENT_ID.fullmatch(p.stem)
        )

    def read_measurement(self, identifier):
        if not isinstance(identifier, str) or not MEASUREMENT_ID.fullmatch(identifier):
            raise ValueError("Unknown measurement ID")
        path = self.directory / "measurements" / (identifier + ".json")
        if not path.is_file():
            raise KeyError("Measurement is not available")
        value = json.loads(path.read_text())
        if value.get("measurement_id") != identifier:
            raise ValueError("Measurement identity mismatch")
        return {key: value[key] for key in PUBLIC_FIELDS if key in value}

    def analyze(self, method, measurement_ids):
        if (
            not isinstance(measurement_ids, list)
            or any(not isinstance(i, str) for i in measurement_ids)
            or len(set(measurement_ids)) != len(measurement_ids)
        ):
            raise ValueError("Unique measurement IDs are required")
        if method not in ("mass", "spectrum"):
            raise ValueError("Analysis method is not registered")
        if (method == "mass" and len(measurement_ids) != 3) or (
            method == "spectrum" and not 3 <= len(measurement_ids) <= 12
        ):
            raise ValueError("Incorrect number of measurements")
        records = [self.read_measurement(identifier) for identifier in measurement_ids]
        if method == "mass":
            return estimate_mass(*records)
        return analyze_spectrum(records[0], records[1], records[2:])

    @staticmethod
    def capabilities():
        return dict(
            read_measurement="registered_public_evidence_only",
            analyze=["mass", "spectrum"],
            public_spectral_templates=public_templates(),
            spectral_data_source="analytic_test_profiles_not_usgs",
            unavailable=[
                "read_stage",
                "read_mass_truth",
                "read_hidden_material",
                "arbitrary_path",
                "arbitrary_python",
                "reset_physics",
            ],
            isolation="http_tool_boundary_not_os_sandbox",
        )
