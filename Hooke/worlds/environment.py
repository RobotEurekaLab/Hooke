"""A reduced radiative witness driven by the actual episode clock."""

from dataclasses import dataclass
import math

from simulation import System

SIGMA = 5.670374419e-8


@dataclass
class ThermalWitness:
    temperature_k: float = 293.15
    heat_capacity_j_k: float = 90.0
    area_m2: float = 0.01
    projected_area_m2: float = 0.001
    absorptivity: float = 0.3
    emissivity: float = 0.8
    elapsed_s: float = 0.0
    net_energy_j: float = 0.0

    def __post_init__(self):
        for value in (self.temperature_k, self.heat_capacity_j_k, self.area_m2):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Thermal state and dimensions must be positive and finite"
                )
        if (
            not math.isfinite(self.projected_area_m2)
            or not 0 <= self.projected_area_m2 <= self.area_m2
        ):
            raise ValueError("Projected area must lie within the radiating area")
        if not 0 <= self.absorptivity <= 1 or not 0 <= self.emissivity <= 1:
            raise ValueError("Optical coefficients must lie within [0, 1]")
        self.initial_temperature_k = self.temperature_k

    def step(self, dt, environment, solar_flux):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError(
                "Radiative witness requires a timestep in (0, 0.1] seconds"
            )
        if not math.isfinite(solar_flux) or solar_flux < 0:
            raise ValueError("Solar irradiance must be finite and nonnegative")
        solar = self.absorptivity * self.projected_area_m2 * solar_flux
        radiation = (
            self.emissivity
            * SIGMA
            * self.area_m2
            * (environment.boundary_temperature_k**4 - self.temperature_k**4)
        )
        energy = (solar + radiation) * dt
        temperature = self.temperature_k + energy / self.heat_capacity_j_k
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Thermal witness left its declared valid state")
        self.temperature_k = temperature
        self.net_energy_j += energy
        self.elapsed_s += dt

    def report(self):
        stored = self.heat_capacity_j_k * (
            self.temperature_k - self.initial_temperature_k
        )
        return {
            "temperature_k": self.temperature_k,
            "elapsed_s": self.elapsed_s,
            "energy_residual_j": stored - self.net_energy_j,
            "fidelity": "reduced_uncalibrated_radiative_witness",
            "coupling": "episode_time_only_no_material_or_rigid_body_feedback",
            "parameters": {
                "heat_capacity_j_k": self.heat_capacity_j_k,
                "area_m2": self.area_m2,
                "projected_area_m2": self.projected_area_m2,
                "absorptivity": self.absorptivity,
                "emissivity": self.emissivity,
            },
            "convection": "disabled_in_all_profiles_not_equivalent_to_absent_martian_or_cabin_air",
            "occlusion_and_view_factors": "prescribed_boundary_without_ray_casting",
        }


class EnvironmentSystem(System):
    def _configure(self, profile):
        self.profile = profile

    def _reset(self, data):
        self.witness = ThermalWitness()
        self.previous_time = float(data.time)

    def _update(self, data):
        elapsed = float(data.time) - self.previous_time
        if elapsed > 0:
            self.witness.step(
                elapsed, self.profile.workspace, self.profile.solar_flux_w_m2
            )
            self.previous_time = float(data.time)

    def report(self):
        return {
            "profile": self.profile.report(),
            "thermal_witness": self.witness.report(),
        }
