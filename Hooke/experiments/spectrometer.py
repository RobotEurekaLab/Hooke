"""A mechanically gated synthetic reflectance instrument.

The bundled profiles are analytic test curves, not USGS spectra or planetary
geology. Observations contain instrument counts; calibrated references and
public templates are provided separately for reproducible analysis.
"""

import numpy as np

from simulation import System
from experiments.mechanics import MechanicalError

WAVELENGTH_UM = np.linspace(0.5, 2.4, 192)


def test_profile(index):
    centers = ((0.9, 1.9), (1.2, 2.2), (0.7, 1.6), (1.05, 2.05))
    w = WAVELENGTH_UM
    first, second = centers[index]
    return (
        0.65
        + 0.04 * (w - 1.5)
        - 0.20 * np.exp(-(((w - first) / 0.08) ** 2))
        - 0.28 * np.exp(-(((w - second) / 0.11) ** 2))
    )


def public_templates():
    return {f"analytic_profile_{i}": test_profile(i).tolist() for i in range(3)}


def analyze_spectrum(dark, reference, samples):
    records = [dark, reference, *samples]
    if not samples or any(not r.get("valid") for r in records):
        raise ValueError("Valid dark, reference and sample observations are required")
    if (
        dark["sample_id"] is not None
        or reference["sample_id"] != "reference"
        or any(r["sample_id"] != "candidate" for r in samples)
    ):
        raise ValueError("Measurement identities do not match the calibration sequence")
    if len({r["calibration_version"] for r in records}) != 1:
        raise ValueError("Calibration versions do not match")
    wavelengths = np.asarray(dark["wavelength_um"])
    for r in records:
        if (
            r.get("saturated")
            or not np.array_equal(r["wavelength_um"], wavelengths)
            or not np.isfinite(r["raw_counts"]).all()
        ):
            raise ValueError("Spectral axes, saturation or finite values are invalid")
    zero = np.asarray(dark["raw_counts"])
    gain = (np.asarray(reference["raw_counts"]) - zero) / np.asarray(
        reference["known_reflectance"]
    )
    if np.any(gain < 0.1):
        raise ValueError("Reference response is unidentifiable")
    curves = np.asarray([(np.asarray(r["raw_counts"]) - zero) / gain for r in samples])
    reflectance = curves.mean(axis=0)
    scores = {
        name: float(np.sqrt(np.mean((reflectance - np.asarray(template)) ** 2)))
        for name, template in public_templates().items()
    }
    ranked = sorted(scores, key=scores.get)
    supported = (
        scores[ranked[0]] < 0.03 and scores[ranked[1]] - scores[ranked[0]] > 0.01
    )
    return dict(
        match=ranked[0] if supported else None,
        abstained=not supported,
        scores_rms=scores,
        reflectance=reflectance.tolist(),
        wavelength_um=wavelengths.tolist(),
        repeat_rms=float(np.sqrt(np.mean((curves - curves.mean(axis=0)) ** 2))),
        measurement_ids=[r["measurement_id"] for r in records],
        fidelity="synthetic_analytic_reflectance_instrument_not_planetary_composition",
        data_source="hooke_analytic_test_profiles_not_usgs",
    )


class Spectrometer(System):
    def _configure(self, mechanics):
        self.mechanics = mechanics

    def _reload(self, model):
        self.shutter_qa = int(model.joint("spectrometer_shutter_slide").qposadr[0])
        self.shutter_drive = model.actuator("spectrometer_shutter_drive").id

    def _reset(self, data):
        self.rng = np.random.default_rng(0)
        self.profile_index = 0
        self.records = {}
        self.calibration_version = "analytic-v1"

    def close(self, data):
        sample = self.mechanics.loaded(data)
        if sample and self.mechanics.contacts(data, sample, self.mechanics.robot_geoms):
            raise MechanicalError(
                "Robot must clear the sample before closing the shutter"
            )
        data.ctrl[self.shutter_drive] = 0

    def open(self, data):
        data.ctrl[self.shutter_drive] = 0.12

    def measure(self, data, sample_id):
        if abs(float(data.qpos[self.shutter_qa])) > 0.001:
            raise MechanicalError("Optical shutter is not closed")
        if self.mechanics.loaded(data) != sample_id:
            raise MechanicalError("Requested sample is not locked in the optical slot")
        if sample_id and self.mechanics.contacts(
            data, sample_id, self.mechanics.robot_geoms
        ):
            raise MechanicalError("Robot contact invalidates the optical observation")
        w = WAVELENGTH_UM
        # Explicit reduced instrument response, independent of visible texture.
        response = 1.1 + 0.07 * (w - w.mean())
        baseline = 0.02 + 0.003 * np.sin(3 * w)
        reflectance = (
            np.zeros_like(w)
            if sample_id is None
            else (
                np.full_like(w, 0.95)
                if sample_id == "reference"
                else test_profile(self.profile_index)
            )
        )
        raw = response * reflectance + baseline + self.rng.normal(0, 0.001, len(w))
        record = dict(
            measurement_id=f"s{len(self.records)+1:04d}",
            sample_id=sample_id,
            simulation_s=float(data.time),
            wavelength_um=w.tolist(),
            raw_counts=raw.tolist(),
            known_reflectance=(
                reflectance.tolist() if sample_id == "reference" else None
            ),
            calibration_version=self.calibration_version,
            noise_std_counts=0.001,
            saturated=bool(np.any(raw > 1.5)),
            valid=True,
            fidelity="synthetic_analytic_reflectance",
            data_source="hooke_analytic_test_profiles_not_usgs",
        )
        self.records[record["measurement_id"]] = record
        self.mechanics.event(
            data,
            "spectrum_measured",
            sample_id=sample_id,
            measurement_id=record["measurement_id"],
            shutter_closed=True,
        )
        return record
