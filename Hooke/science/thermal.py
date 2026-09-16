"""Two well-mixed thermal nodes with measured energy conservation."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ThermalParameters:
    block_capacity_j_k: float = 20.0
    sample_capacity_j_k: float = 0.84
    block_sample_w_k: float = 0.2
    ambient_conductance_w_k: float = 0.1
    max_power_w: float = 20.0
    gain_w_k: float = 5.0
    biot_number: float = 0.05

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Thermal parameters must be finite and positive")
        if self.biot_number >= 0.1:
            raise ValueError("Lumped thermal nodes require Biot number below 0.1")


class ThermalBath:
    def __init__(self, parameters=None, ambient_c=25.0):
        if not math.isfinite(ambient_c):
            raise ValueError("Ambient temperature must be finite")
        self.parameters = parameters or ThermalParameters()
        self.ambient_c = self.block_c = self.sample_c = ambient_c
        self.supplied_j = self.ambient_loss_j = self.elapsed_s = 0.0

    def step(self, dt, target_c):
        p = self.parameters
        if (
            not math.isfinite(dt)
            or dt <= 0
            or not math.isfinite(target_c)
            or not 10 <= target_c <= 90
        ):
            raise ValueError(
                "Thermal model requires positive dt and target in 10–90 °C"
            )
        rate = (
            p.block_sample_w_k + p.ambient_conductance_w_k
        ) / p.block_capacity_j_k + p.block_sample_w_k / p.sample_capacity_j_k
        if dt * rate > 0.5:
            raise ValueError(
                "Thermal timestep exceeds the explicit integration validity bound"
            )
        power = max(
            -p.max_power_w, min(p.max_power_w, p.gain_w_k * (target_c - self.block_c))
        )
        exchange = p.block_sample_w_k * (self.block_c - self.sample_c)
        loss = p.ambient_conductance_w_k * (self.block_c - self.ambient_c)
        self.block_c += dt * (power - exchange - loss) / p.block_capacity_j_k
        self.sample_c += dt * exchange / p.sample_capacity_j_k
        self.supplied_j += power * dt
        self.ambient_loss_j += loss * dt
        self.elapsed_s += dt

    def report(self):
        p = self.parameters
        stored = p.block_capacity_j_k * (
            self.block_c - self.ambient_c
        ) + p.sample_capacity_j_k * (self.sample_c - self.ambient_c)
        return dict(
            model="two_node_thermal_v1",
            fidelity="reduced_uncalibrated",
            valid=True,
            parameters=asdict(p),
            block_c=self.block_c,
            sample_c=self.sample_c,
            elapsed_s=self.elapsed_s,
            supplied_j=self.supplied_j,
            ambient_loss_j=self.ambient_loss_j,
            stored_energy_j=stored,
            energy_residual_j=stored - self.supplied_j + self.ambient_loss_j,
            limitations=[
                "Well-mixed nodes; no spatial heat field, evaporation or phase changes.",
                "Parameters are declared examples, not calibrated Eppendorf measurements.",
            ],
        )
