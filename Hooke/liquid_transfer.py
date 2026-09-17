"""Ideal volume accounting, independent of geometry and simulation engines.

All volumes are in cubic metres. This model does not simulate pressure, flow,
air compressibility or droplets; geometry adapters decide where transfers go.
"""
from dataclasses import asdict, dataclass, replace
from math import fsum, isfinite


@dataclass(frozen=True)
class VolumeState:
    volume_m3: float
    capacity_m3: float | None

    def __post_init__(self):
        if not isfinite(self.volume_m3) or self.volume_m3 < 0:
            raise ValueError('Volume must be finite and nonnegative')
        if self.capacity_m3 is not None:
            if not isfinite(self.capacity_m3) or self.capacity_m3 <= 0:
                raise ValueError('Capacity must be finite and positive')
            if self.volume_m3 > self.capacity_m3:
                raise ValueError('Volume exceeds capacity')


class VolumeLedger:
    def __init__(self, reservoirs: dict[str, VolumeState]):
        self._reservoirs = dict(reservoirs)
        self.initial_total_m3 = self.total_m3

    def state(self, name: str) -> VolumeState:
        return self._reservoirs[name]

    @property
    def total_m3(self) -> float:
        return fsum(state.volume_m3 for state in self._reservoirs.values())

    def transfer(self, source: str, destination: str, requested_m3: float) -> float:
        if not isfinite(requested_m3) or requested_m3 < 0:
            raise ValueError('Requested volume must be finite and nonnegative')
        if source == destination:
            raise ValueError('Transfer requires distinct reservoirs')
        origin, target = self.state(source), self.state(destination)
        available = requested_m3 if target.capacity_m3 is None else target.capacity_m3-target.volume_m3
        transferred = min(requested_m3, origin.volume_m3, available)
        new_origin = replace(origin, volume_m3=origin.volume_m3-transferred)
        new_target = replace(target, volume_m3=target.volume_m3+transferred)
        self._reservoirs.update({source: new_origin, destination: new_target})
        return transferred

    def snapshot(self) -> dict:
        return {name: asdict(state) for name, state in self._reservoirs.items()}


class PistonPipette:
    def __init__(self, ledger: VolumeLedger, tip: str, pressed_fraction: float = 0.):
        self.ledger, self.tip = ledger, tip
        self.capacity_m3 = ledger.state(tip).capacity_m3
        if self.capacity_m3 is None:
            raise ValueError('Pipette tip requires a finite capacity')
        self.previous_fraction = self._fraction(pressed_fraction)
        self.aspirated_m3 = self.dispensed_m3 = self.air_stroke_m3 = 0.

    @staticmethod
    def _fraction(value: float) -> float:
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Pressed fraction must be finite and within [0, 1]')
        return value

    def update(self, pressed_fraction: float, submerged_source: str | None, destination: str):
        fraction = self._fraction(pressed_fraction)
        delta = fraction-self.previous_fraction
        requested = abs(delta)*self.capacity_m3
        if delta < 0:
            transferred = 0. if submerged_source is None else self.ledger.transfer(submerged_source, self.tip, requested)
            self.aspirated_m3 += transferred
            self.air_stroke_m3 += requested-transferred
        elif delta > 0:
            self.dispensed_m3 += self.ledger.transfer(self.tip, destination, requested)
        self.previous_fraction = fraction

    def snapshot(self) -> dict:
        return dict(model='ideal_piston_volume_v1', reservoirs=self.ledger.snapshot(),
                    initial_total_m3=self.ledger.initial_total_m3, total_m3=self.ledger.total_m3,
                    conservation_error_m3=self.ledger.total_m3-self.ledger.initial_total_m3,
                    aspirated_m3=self.aspirated_m3, dispensed_m3=self.dispensed_m3,
                    air_stroke_m3=self.air_stroke_m3, pressed_fraction=self.previous_fraction)
