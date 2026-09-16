"""Connect declared reduced models to existing instrument and liquid systems."""

import math
from simulation import System
from liquid_transfer import VolumeLedger, VolumeState
from science.flow import CapillaryFlow
from science.thermal import ThermalBath


class PressureTransferSystem(System):
    def _configure(self, source, destination, pressure_pa=5000.0):
        if not math.isfinite(pressure_pa):
            raise ValueError("Pressure boundary must be finite")
        self.containers = {"source": source, "destination": destination}
        self.pressure_pa = pressure_pa

    def _reset(self, data):
        ledger = VolumeLedger(
            {
                name: VolumeState(
                    float(system.container.volume),
                    float(system.definition.interior.volume * 0.9),
                )
                for name, system in self.containers.items()
            }
        )
        self.flow = CapillaryFlow(ledger, "source", "destination")
        self.previous_time = float(data.time)

    def _update(self, data):
        dt = float(data.time) - self.previous_time
        if dt <= 0:
            return
        self.flow.step(dt, self.pressure_pa)
        for name, system in self.containers.items():
            volume = self.flow.ledger.state(name).volume_m3
            if volume != system.container.volume:
                system.set_volume(data, volume)
        self.previous_time = float(data.time)

    def report(self):
        return dict(
            self.flow.report(),
            pressure_boundary_pa=self.pressure_pa,
            geometry_coupling="original_container_liquid_surfaces",
            mechanical_mass_feedback=False,
        )


def configure(task, name):
    """Explicit programme boundaries for model qualification, not robot actions."""
    if name == "thermal":
        if not hasattr(task.instrument, "thermal_model"):
            raise ValueError("Thermal qualification requires the thermal_mixer task")
        bath = ThermalBath()
        task.instrument.thermal_model = bath
        task.instrument.thermal_time = float(task.data.time)
        task.instrument.ui_state.main_parameter.set_temperature = 33.0
        return bath
    if name == "capillary":
        transfer = getattr(task, "liquid_transfer", None)
        if transfer is None or "destination" not in transfer.containers:
            raise ValueError(
                "Capillary qualification requires the pipette_transfer task"
            )
        flow = PressureTransferSystem(
            source=transfer.containers["source"],
            destination=transfer.containers["destination"],
        )
        flow.reload(task.model)
        flow.reset(task.data)
        # This is a connected-reservoir experiment; piston accounting would
        # compete with this ledger for ownership of the same liquid surfaces.
        task.manager.systems = tuple(
            flow if system is transfer else system for system in task.manager.systems
        )
        return flow
    raise ValueError("Unknown scientific model")
