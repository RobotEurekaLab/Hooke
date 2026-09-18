"""Pressure-limited holding coupled to a reduced overdamped cell model.

Instrument motion is supplied by the chosen rigid-body backend. The suspended
cell is an analytical phantom, integrated here without a rigid-body pose drive
or weld. Pressure capacity, seal loss and needle load determine its translation.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from microscopy.cell_mechanics import CellMechanics
from microscopy.kinematics import tool_axes
from microscopy.scene import ORIGIN
from microscopy.suction_scene import CELL_RADII, HOLDER_BACK, HOLDER_INNER_M


@dataclass(frozen=True)
class HoldingParameters:
    pressure_response_s: float = .04
    seal_gap_m: float = 1.2e-6
    seal_tangent_m: float = 3e-6
    seal_establishment_s: float = .03
    minimum_vacuum_pa: float = 300.
    stiffness_n_m: float = .06
    medium_viscosity_pa_s: float = .001
    medium_density_kg_m3: float = 1000.
    cell_density_kg_m3: float = 1000.

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Holding parameters must be finite and positive")


class SuctionMechanics(CellMechanics):
    def _configure(self, parameters=None, holding_parameters=None):
        super()._configure(parameters, cell_radii=CELL_RADII, nuclear_radii=(6e-6,)*3)
        self.holding_parameters = holding_parameters or HoldingParameters()

    def _reload(self, model):
        super()._reload(model)
        self.holder_tip = model.site("holder_tcp").id
        from microscopy.kinematics import tool_dofs
        self.holder_dofs = tool_dofs(model, "holder")
        self.holder_axes = tool_axes(model, "holder")
        self.holding_area_m2 = math.pi*HOLDER_INNER_M**2
        self.drag_n_s_m = 6*math.pi*self.holding_parameters.medium_viscosity_pa_s*CELL_RADII[0]
        self.buoyant_weight_n = np.array([0., 0., -9.81*self.cell_volume_m3*(
            self.holding_parameters.cell_density_kg_m3-self.holding_parameters.medium_density_kg_m3)])

    def _reset(self, data):
        super()._reset(data)
        self.position_m = data.site_xpos[self.centre].copy()
        self.initial_position_m = self.position_m.copy()
        self.holding_pressure_pa = self.actual_holding_pressure_pa = 0.
        self.sealed = self.established = self.released_after_withdrawal = False
        self.holding_occluded = self.held_during_delivery = False
        self.delivery_without_hold = self.premature_release = False
        self.seal_contact_s = self.hold_s = self.maximum_holding_force_n = 0.
        self.maximum_displacement_m = 0.
        self.seal_losses = 0
        self.seal_time_s = self.release_time_s = None
        self.anchor_offset_m = np.zeros(3)
        self.needle_load_n = self.holding_force_n = np.zeros(3)
        data.qfrc_applied[self.holder_dofs] = 0.

    def centre_position(self, data):
        return self.position_m

    @property
    def holding_capacity_n(self):
        return 0. if self.holding_occluded else max(0., -self.actual_holding_pressure_pa)*self.holding_area_m2

    def apply_reaction(self, data, reaction):
        from microscopy.kinematics import site_force
        data.qfrc_applied[self.axis_dofs] = site_force(self.model, data, self.tip, reaction)[self.axis_dofs]
        data.qfrc_applied[self.stage_dofs] = 0.
        self.needle_load_n = -reaction

    def _lose_seal(self, data):
        self.sealed = False
        self.seal_losses += 1
        self.release_time_s = float(data.time)
        safe = self.withdrawn and data.userdata[0] == 0 and self.compensation_pa == 0
        self.released_after_withdrawal |= safe
        self.premature_release |= not safe

    def _update(self, data):
        dt = float(data.time)-self.last_time_s
        if dt <= 0:
            return
        p = self.holding_parameters
        self.actual_holding_pressure_pa += (self.holding_pressure_pa-self.actual_holding_pressure_pa)*(
            -math.expm1(-dt/p.pressure_response_s))
        holder = data.site_xpos[self.holder_tip]
        offset = self.position_m-holder
        axial = float(offset @ -HOLDER_BACK)
        tangent = np.linalg.norm(offset+HOLDER_BACK*axial)
        touching = (abs(axial-CELL_RADII[0]) <= p.seal_gap_m and tangent <= p.seal_tangent_m)
        vacuum = self.actual_holding_pressure_pa <= -p.minimum_vacuum_pa and not self.holding_occluded
        if not self.sealed:
            self.seal_contact_s = self.seal_contact_s+dt if touching and vacuum else 0.
            if self.seal_contact_s >= p.seal_establishment_s:
                self.sealed = self.established = True
                self.seal_time_s = float(data.time)
                self.anchor_offset_m = offset.copy()
        elif not vacuum:
            self._lose_seal(data)
        previous_volume = self.ledger.state("cell").volume_m3
        super()._update(data)
        load = self.needle_load_n+self.buoyant_weight_n
        self.holding_force_n = np.zeros(3)
        if self.sealed:
            anchor = holder+self.anchor_offset_m
            decay = math.exp(-p.stiffness_n_m*dt/self.drag_n_s_m)
            trial = anchor+(self.position_m-anchor)*decay+load/p.stiffness_n_m*(1-decay)
            force = p.stiffness_n_m*(anchor-trial)
            if np.linalg.norm(force) > self.holding_capacity_n:
                self._lose_seal(data)
            else:
                self.position_m = trial
                self.holding_force_n = force
                self.hold_s += dt
        if not self.sealed:
            self.position_m += load*dt/self.drag_n_s_m
        from microscopy.kinematics import site_force
        data.qfrc_applied[self.holder_dofs] = site_force(self.model, data, self.holder_tip, -self.holding_force_n)[self.holder_dofs]
        self.maximum_holding_force_n = max(self.maximum_holding_force_n, float(np.linalg.norm(self.holding_force_n)))
        self.maximum_displacement_m = max(self.maximum_displacement_m,
                                         float(np.linalg.norm(self.position_m-self.initial_position_m)))
        if self.ledger.state("cell").volume_m3 > previous_volume:
            self.held_during_delivery |= self.sealed
            self.delivery_without_hold |= not self.sealed

    def image_state(self, data):
        state = super().image_state(data)
        state["centres_m"][0] = (self.position_m-ORIGIN).tolist()
        state.update(holder_tip_m=(data.site_xpos[self.holder_tip]-ORIGIN).tolist(),
                     holder_back=HOLDER_BACK.tolist(), holding_sealed=self.sealed,
                     holding_pressure_pa=self.actual_holding_pressure_pa)
        return state

    def runtime_visuals(self):
        tracer = float(min(1., self.ledger.state("cell").volume_m3/.5e-15))
        return [dict(type=4, role=role, size=radii, pos=self.position_m.tolist(),
                     mat=np.eye(3).ravel().tolist(), rgba=rgba,
                     surface=dict(roughness=.8, specular_color=[.015]*3))
                for role, radii, rgba in (("cell_shell", self.cell_radii.tolist(), [.76-.25*tracer, .84, .79-.2*tracer, .55]),
                                         ("cell_nucleus", self.nuclear_radii.tolist(), [.32, .46, .60, .9]))]

    def report(self, data):
        report = super().report(data)
        report.update(model="suspended_suction_cell_phantom_v1", holding_parameters=asdict(self.holding_parameters),
                      scope="One density-matched suspended spherical phantom; background cells are visual context.",
                      cell_translation="Analytical overdamped force balance; exact held linear response per physics tick.",
                      cell_position_m=self.position_m.tolist(), drag_n_s_m=self.drag_n_s_m,
                      buoyant_weight_n=self.buoyant_weight_n.tolist(), holding_area_m2=self.holding_area_m2,
                      holding_pressure_pa=self.holding_pressure_pa,
                      actual_holding_pressure_pa=self.actual_holding_pressure_pa,
                      holding_capacity_nn=self.holding_capacity_n*1e9,
                      holding_force_nn=float(np.linalg.norm(self.holding_force_n))*1e9,
                      maximum_holding_force_nn=self.maximum_holding_force_n*1e9,
                      maximum_cell_displacement_um=self.maximum_displacement_m*1e6,
                      holding_occluded=self.holding_occluded, holding_sealed=self.sealed,
                      holding_established=self.established, hold_s=self.hold_s, seal_losses=self.seal_losses,
                      seal_time_s=self.seal_time_s, release_time_s=self.release_time_s,
                      released_after_withdrawal=self.released_after_withdrawal,
                      premature_release=self.premature_release, held_during_delivery=self.held_during_delivery,
                      delivery_without_hold=self.delivery_without_hold)
        return report
