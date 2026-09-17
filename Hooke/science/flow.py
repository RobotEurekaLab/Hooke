"""Declared steady laminar capillary connection with conservative volumes."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CapillaryParameters:
    radius_m: float = 0.0001
    length_m: float = 0.02
    viscosity_pa_s: float = 0.001
    density_kg_m3: float = 1000.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Capillary parameters must be finite and positive")


class CapillaryFlow:
    def __init__(self, ledger, source, destination, parameters=None):
        if source == destination:
            raise ValueError("Capillary requires distinct reservoirs")
        ledger.state(source)
        ledger.state(destination)
        self.ledger, self.source, self.destination = ledger, source, destination
        self.parameters = parameters or CapillaryParameters()
        self.transferred_m3 = self.elapsed_s = self.max_reynolds = 0.0
        self.rate_m3_s = 0.0

    def step(self, dt, pressure_pa):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(pressure_pa):
            raise ValueError(
                "Capillary requires a positive timestep and finite pressure"
            )
        p = self.parameters
        rate = (
            math.pi * p.radius_m**4 * pressure_pa / (8 * p.viscosity_pa_s * p.length_m)
        )
        reynolds = (
            2 * p.density_kg_m3 * abs(rate) / (math.pi * p.radius_m * p.viscosity_pa_s)
        )
        if reynolds >= 1000 or p.length_m < 0.05 * reynolds * 2 * p.radius_m:
            raise ValueError(
                "Capillary is outside laminar fully developed flow validity"
            )
        origin, target = (
            (self.source, self.destination)
            if rate >= 0
            else (self.destination, self.source)
        )
        actual = self.ledger.transfer(origin, target, abs(rate) * dt)
        self.rate_m3_s = math.copysign(actual / dt, rate)
        self.transferred_m3 += math.copysign(actual, rate)
        self.elapsed_s += dt
        self.max_reynolds = max(self.max_reynolds, reynolds)
        return self.rate_m3_s

    def report(self):
        return dict(
            model="laminar_capillary_v1",
            fidelity="reduced_uncalibrated",
            valid=True,
            parameters=asdict(self.parameters),
            elapsed_s=self.elapsed_s,
            transferred_m3=self.transferred_m3,
            flow_m3_s=self.rate_m3_s,
            max_reynolds=self.max_reynolds,
            reservoirs=self.ledger.snapshot(),
            conservation_error_m3=self.ledger.total_m3 - self.ledger.initial_total_m3,
            limitations=[
                "A declared pressure boundary and a fully filled circular capillary.",
                "No air compliance, free-surface CFD, droplets or physical pipette calibration.",
            ],
        )
