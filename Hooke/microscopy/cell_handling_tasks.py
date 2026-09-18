"""Image-guided cell pushing and gentle electric-forceps transfer."""

import mujoco
import numpy as np

from microscopy.cell_tasks import CellInjection, CellInjectionExpert
from microscopy.cell_handling_mechanics import CellHandlingMechanics
from microscopy.cell_handling_scene import (CELL_CENTER, CELL_RADII, JAW_CLOSED_M,
    JAW_OPEN_M, LIFT_M, PAD_HALFSIZE_M, PICK_TARGET, PROBE_RADIUS_M, PUSH_TARGET, build_xml)
from microscopy.scene import ORIGIN


class CellHandling(CellInjection):
    mechanics_type = CellHandlingMechanics
    cell_center = CELL_CENTER
    specimen_kind = "suspended_cell_phantom"

    @classmethod
    def load(cls, scene=None):
        return super().load(scene) if scene is not None else mujoco.MjSpec.from_string(build_xml(cls.operation))

    def reset(self, seed=None):
        info = super().reset(seed)
        for side in ("a", "b"):
            if "jaw_"+side in self.drives:
                self.data.qpos[self.model.joint("jaw_"+side).qposadr[0]] = JAW_OPEN_M
                self.data.ctrl[self.drives["jaw_"+side]] = JAW_OPEN_M
        self.data.userdata[:] = [0., 0.]
        mujoco.mj_forward(self.model, self.data)
        self.mechanics.reset(self.data)
        info["cell_solver"] = "Reduced compliant-contact overdamped model, not backend soft-body physics"
        return info

    def runtime_visuals(self):
        return self.mechanics.runtime_visuals()+super().runtime_visuals()

    def public_state(self):
        m = self.mechanics
        return dict(super().public_state(), grasped=m.grasped,
                    force_nn=m.handling_force_n*1e9,
                    cell_compression_um=m.maximum_compression_m*1e6)

    def sample_scale(self):
        scale = super().sample_scale()
        if self.operation == "push":
            scale["probe_tip_outer_diameter_m"] = 2*PROBE_RADIUS_M
        else:
            scale["forceps_pad_dimensions_m"] = (2*PAD_HALFSIZE_M).tolist()
        return scale

    def microscopy_checks(self):
        m = self.mechanics
        target = PUSH_TARGET if self.operation == "push" else PICK_TARGET
        relative = m.position_m-ORIGIN
        checks = dict(episode_completed=self.completed, autofocus=self.focused,
            finite_state=bool(np.isfinite(self.data.qpos).all() and np.isfinite(m.position_m).all()),
            cell_at_target=bool(np.linalg.norm(relative[:2]-target) < 3e-6),
            safe_cell_compression=m.maximum_compression_m <= m.handling_parameters.maximum_compression_m,
            membrane_intact=not m.punctured, pressure_off=bool(self.data.userdata[0] == 0))
        if self.operation == "push":
            checks["cell_probe_contact_50ms"] = m.probe_contact_s >= .05
        else:
            checks.update(cell_bilateral_grasp_50ms=m.bilateral_contact_s >= .05,
                          cell_lift_20um=bool(m.maximum_lift_m >= 20e-6),
                          grasp_maintained_until_release=m.grip_losses == 1,
                          cell_released=m.released and not m.grasped)
        return checks

    def microscopy_report(self):
        report = super().microscopy_report()
        report["limitations"] = [
            "A 36 micrometre simulated cell with estimated texture; not a real biological specimen or measured response.",
            "Uncalibrated unilateral spring contacts, friction-limited grasp and overdamped cell translation; no resolved soft membrane or fluid solver.",
            "Rigid-body engines move instruments; cell positions follow shared force balance rather than a grasp weld or commanded pose.",
            "Background cells are visual context; cell-cell contacts, survival and adhesion are not resolved.",
            "Scalar phase contrast and empirical axial blur are uncalibrated; no measured hardware accuracy."]
        return report


class CellPush(CellHandling):
    operation = "push"
    default_task = "microscopy_push"
    task_prefix = "Image-guided gentle cell pushing"


class CellPickPlace(CellHandling):
    operation = "pick_place"
    default_task = "microscopy_pick_place"
    task_prefix = "Electric microforceps cell lift and transfer"


class CellHandlingExpert(CellInjectionExpert):
    def jaws(self, opening):
        self.command({"jaw_"+side: opening for side in ("a", "b") if "jaw_"+side in self.drives}, .3)

    def push(self):
        self.phase("locate_cell")
        radius = CELL_RADII[0]+PROBE_RADIUS_M
        for _ in range(5):
            xy = self.locate("cell")
            error = PUSH_TARGET-xy
            distance = float(np.linalg.norm(error))
            if distance < 1.5e-6:
                break
            direction = error/distance
            start = xy-direction*(radius+5e-6)
            self.phase("approach_cell_probe")
            self.move("probe", [*start, CELL_CENTER[2]+45e-6], .4)
            self.move("probe", [*start, CELL_CENTER[2]], .4)
            self.phase("push_cell")
            end = xy+direction*min(distance, 15e-6)-direction*(radius-.5e-6)
            self.move("probe", [*end, CELL_CENTER[2]], .7)
        self.phase("withdraw_cell_probe")
        xy = self.locate("cell")
        self.move("probe", [xy[0]-70e-6, xy[1], CELL_CENTER[2]+45e-6], .5)

    def pick_place(self):
        self.phase("locate_cell")
        xy = self.locate("cell")
        self.phase("approach_cell_forceps")
        self.move("gripper", [*xy, CELL_CENTER[2]+45e-6], .5)
        self.move("gripper", [*xy, CELL_CENTER[2]], .5)
        self.phase("grasp_cell")
        self.jaws(JAW_CLOSED_M)
        if not self.mechanics.grasped:
            raise RuntimeError("Bilateral cell grasp was not established")
        self.phase("lift_cell")
        self.move("gripper", [*xy, CELL_CENTER[2]+LIFT_M], .6, focus_m=LIFT_M)
        self.phase("transfer_cell")
        self.move("gripper", [*PICK_TARGET, CELL_CENTER[2]+LIFT_M], .7, focus_m=LIFT_M)
        self.phase("place_cell")
        self.move("gripper", [*PICK_TARGET, CELL_CENTER[2]], .6, focus_m=0.)
        self.phase("release_cell")
        self.jaws(JAW_OPEN_M)
        self.move("gripper", [*PICK_TARGET, CELL_CENTER[2]+60e-6], .5)


class CellPushExpert(CellHandlingExpert, CellPush):
    pass


class CellPickPlaceExpert(CellHandlingExpert, CellPickPlace):
    pass


CellPush.Expert = CellPushExpert
CellPickPlace.Expert = CellPickPlaceExpert
