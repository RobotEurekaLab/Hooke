"""Compliant cell contact and overdamped motion from observed instrument poses.

Both rigid-body backends drive the instruments. Cell translation uses an
implicit force balance, with unilateral elastic contacts and friction-limited
forceps gripping; it never assigns a cell pose from a commanded tool pose.
"""

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np

from microscopy.cell_mechanics import CellMechanics
from microscopy.cell_handling_scene import CELL_RADII, PROBE_RADIUS_M
from microscopy.scene import ORIGIN


@dataclass(frozen=True)
class HandlingParameters:
    stiffness_n_m: float = .03
    viscosity_pa_s: float = .001
    friction_coefficient: float = .6
    maximum_compression_m: float = 4.5e-6

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Cell handling parameters must be finite and positive")


class CellHandlingMechanics(CellMechanics):
    def _configure(self, parameters=None, handling_parameters=None):
        super()._configure(parameters, cell_radii=CELL_RADII, nuclear_radii=(6e-6,)*3)
        self.handling_parameters = handling_parameters or HandlingParameters()
        self.drag_n_s_m = 6*math.pi*self.handling_parameters.viscosity_pa_s*CELL_RADII[0]

    def _reload(self, model):
        super()._reload(model)
        self.probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        self.pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"jaw_{s}_pad") for s in ("a", "b")]
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
        from microscopy.kinematics import tool_dofs
        self.handling_dofs = sorted({dof for tool in ("probe", "gripper")
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tool+"_tcp") >= 0
            for dof in tool_dofs(model, tool)} | {
                int(model.joint("jaw_"+side).dofadr[0]) for side in ("a", "b")
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_"+side) >= 0})

    def _reset(self, data):
        super()._reset(data)
        self.position_m = data.site_xpos[self.centre].copy()
        self.initial_position_m = self.position_m.copy()
        self.grasp_offset_m = None
        self.probe_contact_s = self.bilateral_contact_s = 0.
        self.grasped = self.ever_grasped = self.released = False
        self.maximum_lift_m = self.maximum_compression_m = self.handling_force_n = 0.
        self.compression_m = 0.
        self.grip_losses = 0
        self.contact_history = []

    def centre_position(self, data):
        return self.position_m

    def apply_reaction(self, data, reaction):
        super().apply_reaction(data, reaction)
        data.qfrc_applied[self.stage_dofs] = 0.

    def _contacts(self, data):
        contacts = []
        radius = self.cell_radii[0]
        stiffness = self.handling_parameters.stiffness_n_m
        if self.probe >= 0:
            centre = data.geom_xpos[self.probe]
            delta = self.position_m-centre
            length = float(np.linalg.norm(delta))
            if 0 < length < radius+PROBE_RADIUS_M:
                normal = delta/length
                contacts.append(("probe", self.probe, normal, centre+normal*(radius+PROBE_RADIUS_M), stiffness))
        for side, (pad, sign) in enumerate(zip(self.pads, (1, -1))):
            if pad < 0:
                continue
            rotation = data.geom_xmat[pad].reshape(3, 3)
            normal = -sign*rotation[:, 1]
            halfsize = self.model.geom_size[pad]
            centre = data.geom_xpos[pad]
            local = rotation.T@(self.position_m-centre)
            face = centre+normal*halfsize[1]
            gap = float((self.position_m-face)@normal)
            if (0 <= gap < radius and abs(local[0]) < radius+halfsize[0]
                    and abs(local[2]) < radius+halfsize[2]):
                contacts.append((f"jaw_{side}", pad, normal, face+normal*radius, stiffness))
        return contacts

    def _update(self, data):
        dt = float(data.time)-self.last_time_s
        if dt <= 0:
            return
        data.qfrc_applied[self.handling_dofs] = 0.
        super()._update(data)
        contacts = self._contacts(data)
        coefficient = self.drag_n_s_m/dt
        matrix = np.eye(3)*coefficient
        rhs = self.position_m*coefficient
        for _, _, normal, equilibrium, stiffness in contacts:
            spring = stiffness*np.outer(normal, normal)
            matrix += spring
            rhs += spring@equilibrium
        floor = data.site_xpos[self.centre][2]
        if self.position_m[2] < floor:
            matrix[2, 2] += .3
            rhs[2] += .3*floor
        jaws = [contact for contact in contacts if contact[0].startswith("jaw_")]
        bilateral = len(jaws) == 2
        anchor = None
        tangent = np.diag([1., 0., 1.])
        stiffness = self.handling_parameters.stiffness_n_m
        if bilateral:
            if self.grasp_offset_m is None:
                self.grasp_offset_m = self.position_m-data.site_xpos[self.gripper]
            anchor = data.site_xpos[self.gripper]+self.grasp_offset_m
            candidate = np.linalg.solve(matrix+stiffness*tangent, rhs+stiffness*tangent@anchor)
            capacity = self.handling_parameters.friction_coefficient*min(
                max(0., stiffness*float((equilibrium-candidate)@normal)) for _, _, normal, equilibrium, _ in jaws)
            shear = stiffness*tangent@(anchor-candidate)
            magnitude = float(np.linalg.norm(shear))
            if magnitude <= capacity:
                matrix += stiffness*tangent
                rhs += stiffness*tangent@anchor
            else:
                rhs += shear*capacity/magnitude
                bilateral = False
        position = np.linalg.solve(matrix, rhs)
        self.probe_contact_s += dt*any(c[0] == "probe" for c in contacts)
        self.bilateral_contact_s += dt*bilateral
        if self.grasped and not bilateral:
            self.grip_losses += 1
            self.released = True
        self.grasped = bilateral
        self.ever_grasped |= bilateral
        if not bilateral:
            self.grasp_offset_m = None
        compression = 0.
        force_sum = 0.
        for _, geom, normal, equilibrium, k in contacts:
            overlap = max(0., float((equilibrium-position)@normal))
            force = overlap*k*normal
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacGeom(self.model, data, jacobian, None, geom)
            data.qfrc_applied -= jacobian.T@force
            compression = max(compression, overlap)
            force_sum += float(np.linalg.norm(force))
        if bilateral and anchor is not None:
            from microscopy.kinematics import site_force
            data.qfrc_applied -= site_force(self.model, data, self.gripper, stiffness*tangent@(anchor-position))
            self.maximum_lift_m = max(self.maximum_lift_m, position[2]-self.initial_position_m[2])
        self.position_m = position
        self.handling_force_n = force_sum
        self.compression_m = compression
        self.maximum_compression_m = max(self.maximum_compression_m, compression)
        # A small event log records actual contact transitions, not control phases.
        event = (any(c[0] == "probe" for c in contacts), bilateral)
        if not self.contact_history or tuple(self.contact_history[-1]["contacts"]) != event:
            self.contact_history.append(dict(time_s=float(data.time), contacts=list(event)))

    def image_state(self, data):
        state = super().image_state(data)
        state["centres_m"][0] = (self.position_m-ORIGIN).tolist()
        state["indentation_m"] = self.compression_m
        if self.probe >= 0:
            state.update(probe_tip_m=(data.geom_xpos[self.probe]-ORIGIN).tolist(), probe_radius_m=PROBE_RADIUS_M)
        if self.gripper >= 0:
            state["jaw_centres_m"] = [(data.geom_xpos[g]-ORIGIN).tolist() for g in self.pads]
            state["jaw_halfsize_m"] = self.model.geom_size[self.pads[0]].tolist()
        return state

    def runtime_visuals(self):
        return [dict(type=4, role=role, size=radii.tolist(), pos=self.position_m.tolist(),
                     mat=np.eye(3).ravel().tolist(), rgba=colour,
                     surface=dict(roughness=.8, specular_color=[.015]*3))
                for role, radii, colour in (("cell_shell", self.cell_radii, [.76, .84, .79, .55]),
                                           ("cell_nucleus", self.nuclear_radii, [.32, .46, .60, .9]))]

    def report(self, data):
        return dict(super().report(data), model="overdamped_compliant_cell_handling_v1",
                    scope="One density-matched suspended simulated cell; background cells are visual context",
                    handling_parameters=asdict(self.handling_parameters),
                    cell_position_m=self.position_m.tolist(), drag_n_s_m=self.drag_n_s_m,
                    probe_contact_s=self.probe_contact_s, bilateral_contact_s=self.bilateral_contact_s,
                    grasped=self.grasped, released=self.released, grip_losses=self.grip_losses,
                    maximum_lift_um=self.maximum_lift_m*1e6,
                    maximum_compression_um=self.maximum_compression_m*1e6,
                    contact_history=self.contact_history,
                    cell_translation="Implicit overdamped force balance from actual instrument feedback; no pose attachment or weld")
