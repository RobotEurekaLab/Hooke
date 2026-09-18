"""Reduced, uncalibrated membrane interaction and conservative picolitre flow."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from liquid_transfer import VolumeLedger, VolumeState
from microscopy.cell_scene import CELL_RADII, NEEDLE_BACK, NUCLEAR_RADII
from microscopy.pipettes import INJECTION_PROFILE
from microscopy.scene import ORIGIN
from microscopy.kinematics import tool_axes
from simulation import System


@dataclass(frozen=True)
class CellParameters:
    stiffness_n_m: float = .03
    relaxation_s: float = .06
    puncture_indentation_m: float = 1e-6
    puncture_force_n: float = 24e-9
    maximum_depth_m: float = 5e-6
    pressure_response_s: float = .05
    viscosity_pa_s: float = .001
    resting_backpressure_pa: float = 300.

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Cell parameters must be finite and positive")


class CellMechanics(System):
    def _configure(self, parameters=None, cell_radii=CELL_RADII, nuclear_radii=NUCLEAR_RADII):
        self.parameters = parameters or CellParameters()
        self.cell_radii = np.asarray(cell_radii, dtype=float)
        self.nuclear_radii = np.asarray(nuclear_radii, dtype=float)
        if (self.cell_radii.shape != (3,) or self.nuclear_radii.shape != (3,)
                or not np.isfinite(self.cell_radii).all() or not np.isfinite(self.nuclear_radii).all()
                or np.any(self.cell_radii <= 0) or np.any(self.nuclear_radii <= 0)):
            raise ValueError("Cell and nuclear radii must be three finite positive lengths")

    def _reload(self, model):
        self.tip = model.site("injector_tcp").id
        self.centre = model.site("cell_center").id
        self.optical_reference_height_m = float(model.site_pos[self.centre, 2])
        self.shells = [model.geom(f"cell_{i}_shell").id for i in range(6)]
        from microscopy.kinematics import tool_dofs
        self.axis_dofs = tool_dofs(model, "injector")
        self.axis_directions = tool_axes(model, "injector")
        self.stage_dofs = [model.joint("stage_"+axis).dofadr[0] for axis in "xy"]
        self.ray_radius_m = 1 / np.linalg.norm(NEEDLE_BACK / self.cell_radii)
        self.resistance_pa_s_m3 = INJECTION_PROFILE.resistance(self.parameters.viscosity_pa_s)
        self.cell_volume_m3 = 4/3*np.pi*np.prod(self.cell_radii)

    def centre_position(self, data):
        return data.site_xpos[self.centre]

    def apply_reaction(self, data, reaction):
        from microscopy.kinematics import site_force
        data.qfrc_applied[self.axis_dofs] = site_force(self.model, data, self.tip, reaction)[self.axis_dofs]
        data.qfrc_applied[self.stage_dofs] = -reaction[:2]

    def _reset(self, data):
        self.ledger = VolumeLedger({"supply": VolumeState(100e-15, 100e-15),
                                   "cell": VolumeState(0., self.cell_volume_m3*.15),
                                   "environment": VolumeState(0., None)})
        self.indentation_m = self.force_n = self.actual_pressure_pa = self.compensation_pa = 0.
        self.maximum_indentation_m = self.maximum_force_n = self.maximum_depth_m = 0.
        self.contact_s = self.blocked_injection_s = self.intracellular_s = 0.
        self.punctured = self.withdrawn = self.overdepth = self.nuclear_contact = False
        self.puncture_time_s = self.withdrawal_time_s = None
        self.last_time_s = float(data.time)
        self.inside = False
        self.needle_clogged = False
        data.qfrc_applied[self.axis_dofs] = 0.
        data.qfrc_applied[self.stage_dofs] = 0.

    def _update(self, data):
        dt = float(data.time)-self.last_time_s
        if dt <= 0:
            return
        self.last_time_s = float(data.time)
        p = self.parameters
        offset = data.site_xpos[self.tip]-self.centre_position(data)
        normalized = float(np.linalg.norm(offset / self.cell_radii))
        depth = max(0., (1-normalized)*self.ray_radius_m)
        self.inside = bool(normalized < 1)
        self.indentation_m += (depth-self.indentation_m) * (-math.expm1(-dt/p.relaxation_s))
        self.maximum_indentation_m = max(self.maximum_indentation_m, self.indentation_m)
        if depth:
            self.contact_s += dt
        force = p.stiffness_n_m*self.indentation_m
        if not self.punctured and depth and self.contact_s >= .01:
            self.maximum_force_n = max(self.maximum_force_n, force)
            if self.indentation_m >= p.puncture_indentation_m and force >= p.puncture_force_n:
                self.punctured = True
                self.puncture_time_s = float(data.time)
        self.force_n = force * (.12 if self.punctured else 1.) if depth else 0.
        # This is a real generalized reaction force read by both engine adapters.
        normal = offset / self.cell_radii**2
        length = np.linalg.norm(normal)
        reaction = self.force_n*normal/length if length else np.zeros(3)
        self.apply_reaction(data, reaction)
        if self.punctured:
            self.maximum_depth_m = max(self.maximum_depth_m, depth)
            self.overdepth |= bool(depth > p.maximum_depth_m)
            self.nuclear_contact |= bool(np.linalg.norm(offset / self.nuclear_radii) < 1)
            if normalized > 1.4 and not self.withdrawn:
                self.withdrawn = True
                self.withdrawal_time_s = float(data.time)
        command = max(float(data.userdata[0]), self.compensation_pa)
        remaining = max(0., float(data.userdata[1])*1e-15-self.ledger.state("cell").volume_m3)
        quota_closed = self.punctured and remaining <= 1e-28
        if quota_closed:
            command = 0.
        old_pressure = self.actual_pressure_pa
        self.actual_pressure_pa += (command-old_pressure) * (-math.expm1(-dt/p.pressure_response_s))
        average_pressure = (old_pressure+self.actual_pressure_pa)/2
        intracellular = self.punctured and self.inside and not self.withdrawn
        if data.userdata[0] > 0 and (not intracellular or self.needle_clogged):
            self.blocked_injection_s += dt
        if self.needle_clogged:
            return
        if intracellular:
            self.intracellular_s += dt
            backpressure = p.resting_backpressure_pa + 500*self.ledger.state("cell").volume_m3/self.cell_volume_m3
            requested = max(0., average_pressure-backpressure)*dt/self.resistance_pa_s_m3
            self.ledger.transfer("supply", "cell", min(requested, remaining))
        elif not quota_closed:
            self.ledger.transfer("supply", "environment", average_pressure*dt/self.resistance_pa_s_m3)

    def image_state(self, data):
        return dict(time_s=float(data.time), focus_m=float(data.qpos[self.model.joint("focus").qposadr[0]]),
                    centres_m=[(data.geom_xpos[g]-ORIGIN).tolist() for g in self.shells],
                    tip_m=(data.site_xpos[self.tip]-ORIGIN).tolist(),
                    indentation_m=self.indentation_m, punctured=self.punctured,
                    withdrawn=self.withdrawn, delivered_pl=self.ledger.state("cell").volume_m3*1e15,
                    force_nn=self.force_n*1e9, actual_pressure_pa=self.actual_pressure_pa,
                    cell_radii_m=self.cell_radii.tolist(), nuclear_radii_m=self.nuclear_radii.tolist(),
                    optical_reference_height_m=self.optical_reference_height_m)

    def report(self, data):
        return dict(model="adherent_cell_phantom_v1", fidelity="reduced_uncalibrated",
                    parameters=asdict(self.parameters), cell_radii_m=self.cell_radii.tolist(),
                    nominal_cell_volume_pl=self.cell_volume_m3*1e15,
                    delivered_pl=self.ledger.state("cell").volume_m3*1e15,
                    environmental_pl=self.ledger.state("environment").volume_m3*1e15,
                    reservoirs=self.ledger.snapshot(),
                    conservation_error_m3=self.ledger.total_m3-self.ledger.initial_total_m3,
                    needle_resistance_pa_s_m3=self.resistance_pa_s_m3,
                    actual_pressure_pa=self.actual_pressure_pa, contact_s=self.contact_s,
                    needle_clogged=self.needle_clogged,
                    punctured=self.punctured, withdrawn=self.withdrawn,
                    puncture_time_s=self.puncture_time_s, withdrawal_time_s=self.withdrawal_time_s,
                    maximum_indentation_um=self.maximum_indentation_m*1e6,
                    maximum_force_nn=self.maximum_force_n*1e9,
                    maximum_depth_um=self.maximum_depth_m*1e6,
                    overdepth=self.overdepth, nuclear_contact=self.nuclear_contact,
                    blocked_intracellular_injection_s=self.blocked_injection_s,
                    intracellular_s=self.intracellular_s,
                    scope="One adherent cell phantom; background cells are visual context.")
