"""Image-guided bilateral holding and cytoplasmic injection of a phantom."""

import mujoco
import numpy as np

from microscopy.cell_scene import NEEDLE_BACK
from microscopy.cell_tasks import CellInjection, CellInjectionExpert
from microscopy.suction_mechanics import SuctionMechanics
from microscopy.suction_scene import CELL_CENTER, CELL_RADII, HOLDER_BACK, build_xml


class SuctionInjection(CellInjection):
    operation = "suction_injection"
    default_task = "microscopy_suction_injection"
    mechanics_type = SuctionMechanics
    cell_center = CELL_CENTER
    task_prefix = "Suspended-cell phantom: electric holding and cytoplasmic injection"
    specimen_kind = "suspended_cell_phantom"

    @classmethod
    def load(cls, scene=None):
        return super().load(scene) if scene is not None else mujoco.MjSpec.from_string(build_xml())

    def reset(self, seed=None):
        info = super().reset(seed)
        info["cell_solver"] = "Reduced analytical overdamped force balance; not backend soft-body physics"
        return info

    def runtime_visuals(self):
        return self.mechanics.runtime_visuals()+super().runtime_visuals()

    def public_state(self):
        m = self.mechanics
        return dict(super().public_state(), holding_pressure_pa=m.holding_pressure_pa,
                    actual_holding_pressure_pa=m.actual_holding_pressure_pa,
                    holding_sealed=m.sealed, holding_occluded=m.holding_occluded,
                    holding_capacity_nn=m.holding_capacity_n*1e9,
                    holding_force_nn=float(np.linalg.norm(m.holding_force_n))*1e9)

    def microscopy_checks(self):
        m = self.mechanics
        return dict(super().microscopy_checks(), holding_established=m.established,
                    holding_100ms=m.hold_s >= .1,
                    held_during_injection=m.held_during_delivery and not m.delivery_without_hold,
                    no_premature_release=not m.premature_release,
                    holding_released=m.released_after_withdrawal and not m.sealed,
                    vacuum_off=m.holding_pressure_pa == 0 and abs(m.actual_holding_pressure_pa) < 1,
                    finite_cell_state=bool(np.isfinite(m.position_m).all()))

    def microscopy_report(self):
        report = super().microscopy_report()
        report["limitations"] = [line for line in report["limitations"]
                                 if not line.startswith("Nominal 3D cell geometry")]
        report["controller"] = "Separate synthetic nuclear-fluorescence image centroid, nominal height/radius and actual electric-stage encoders."
        report["limitations"] += [
            "Density-matched spherical phantom; translation is an analytical overdamped model, not a PhysX or MuJoCo soft body.",
            "Holding capacity is vacuum times pipette mouth area; seal stiffness, thresholds and contact geometry are uncalibrated.",
            "No resolved aspiration tongue, cell rotation, resealing, membrane adhesion or bath flow.",
            "3D cell translation and tracer colour follow the reduced model; membrane deformation is synthetic microscopy only.",
        ]
        return report


class SuctionInjectionExpert(CellInjectionExpert, SuctionInjection):
    def suction_injection(self):
        self.phase("locate_cell")
        centre = np.array([*self.locate("cell"), CELL_CENTER[2]])
        radius = CELL_RADII[0]
        self.phase("approach_holding_pipette")
        self.move("holder", centre+HOLDER_BACK*(radius+80e-6), .8)
        self.move("holder", centre+HOLDER_BACK*radius, .7)
        self.phase("establish_suction")
        self.mechanics.holding_pressure_pa = -1500.
        self.wait(.2)
        if not self.mechanics.sealed:
            raise RuntimeError("Holding seal was not established; injection aborted")
        centre[:2] = self.locate("cell")
        self.move("injector", centre+NEEDLE_BACK*(radius+100e-6), .8)
        self.mechanics.compensation_pa = 500.
        self.phase("approach_cell")
        self.move("injector", centre+NEEDLE_BACK*(radius+10e-6), .7)
        self.phase("membrane_contact")
        self.move("injector", centre+NEEDLE_BACK*(radius-.6e-6), .5)
        self.phase("puncture_cell")
        self.move("injector", centre+NEEDLE_BACK*(radius-3e-6), .7)
        if not self.mechanics.punctured or not self.mechanics.sealed:
            raise RuntimeError("A supported membrane puncture was not established; injection aborted")
        self.phase("cytoplasm_injection")
        self.data.userdata[0] = 5000.
        for _ in range(2500):
            self.step_and_log({})
            if not self.mechanics.sealed:
                self.data.userdata[0] = 0.
                self.mechanics.compensation_pa = 0.
                raise RuntimeError("Holding seal was lost during injection")
            if self.mechanics.ledger.state("cell").volume_m3 >= .5e-15-1e-28:
                break
        self.data.userdata[0] = 0.
        self.mechanics.compensation_pa = 0.
        self.phase("withdraw_cell_needle")
        self.move("injector", centre+NEEDLE_BACK*(radius+25e-6), .6)
        self.wait(.3)
        self.phase("release_suction")
        self.mechanics.holding_pressure_pa = 0.
        self.wait(.4)
        self.move("holder", centre+HOLDER_BACK*(radius+50e-6), .6)
        self.phase("tracer_review")


SuctionInjection.Expert = SuctionInjectionExpert
