"""One-reset centrifuge operation using the original 5430 geometry and robot."""

import numpy as np
import mujoco

from centrifuge_control import SpinController
from mani_centrifuge_5430_composite import (
    CentrifugeInsertCloseComposite,
    CentrifugeInsertCloseCompositeExpert,
)
from archetypes.lid_lock import lid_lock_passes
from simulation import System
from kinematics import mul_pose, neg_pose, Pose

MOTOR = "centrifuge_rotor_motor"
ROTOR = "/centrifuge_eppendorf_5430:rotor"
MOUNTS = ("centrifuge_sample_mount_1", "centrifuge_sample_mount_2")


class RotorProgramSystem(System):
    def _configure(self, task):
        self.task = task

    def _reload(self, model):
        self.motor_id = model.actuator(MOTOR).id
        self.joint_id = model.joint(ROTOR).id
        self.velocity_id = int(model.jnt_dofadr[self.joint_id])
        self.rotor_body = int(model.jnt_bodyid[self.joint_id])
        self.mount_ids = [model.equality(name).id for name in MOUNTS]
        self.loads = []
        for tube in (self.task.tube, self.task.tube2):
            owner = int(model.body_weldid[tube.body_id])
            members = np.flatnonzero(model.body_weldid == owner)
            self.loads.append((members, model.body_mass[members].copy()))

    def _reset(self, data):
        self.controller = SpinController()
        self.unlocked = False
        self.events = []
        self.previous_state = "IDLE"
        self.parking = False
        data.ctrl[self.motor_id] = 0.0
        data.eq_active[self.mount_ids] = 0

    def speed(self, data):
        return float(data.qvel[self.velocity_id])

    def balance_ratio(self, data):
        # A joint axis need not pass through the mesh body's origin. Use the
        # actual hinge anchor and the complete welded load's centre of mass.
        rotor = data.xanchor[self.joint_id]
        axis = data.xaxis[self.joint_id]
        offsets, masses = [], []
        for members, weights in self.loads:
            mass = float(weights.sum())
            offset = (
                np.sum(data.xipos[members] * weights[:, None], axis=0) / mass - rotor
            )
            offsets.append(offset - axis * np.dot(offset, axis))
            masses.append(mass)
        scale = sum(
            mass * np.linalg.norm(offset) for mass, offset in zip(masses, offsets)
        )
        return (
            float(
                np.linalg.norm(
                    np.sum(np.asarray(offsets) * np.asarray(masses)[:, None], axis=0)
                )
                / scale
            )
            if scale > 0
            else 1.0
        )

    def start(self, data):
        rotor = Pose(data.xpos[self.rotor_body], data.xquat[self.rotor_body])
        for tube, mount in zip((self.task.tube, self.task.tube2), self.mount_ids):
            slot = (
                self.task.slot_id
                if tube is self.task.tube
                else (self.task.slot_id + self.task.instrument.num_slots // 2)
                % self.task.instrument.num_slots
            )
            target = self.task.instrument.get_tube_pose(data, slot, "proximal")
            if np.linalg.norm(tube.get_body_pose(data).pos - target.pos) >= 0.005:
                raise RuntimeError("A sample is not seated in its rotor slot")
            relative = mul_pose(neg_pose(rotor), tube.get_body_pose(data))
            self.model.eq_data[mount, 3:6] = relative.pos
            self.model.eq_data[mount, 6:10] = relative.quat
        self.controller.start(
            self.speed(data),
            lid_lock_passes(data, self.task.instrument),
            bool(data.eq_active[self.task.instrument.lid_lock]),
            self.balance_ratio(data),
        )
        # A declared holder engages at the measured seating pose. Neither
        # engine copies a target pose into the load's physical state.
        data.eq_active[self.mount_ids] = 1
        self.events.append(
            dict(
                time_s=float(data.time),
                event="sample_holders_engaged",
                model="compliant_weld_uncalibrated",
                captured_eq_data=self.model.eq_data[self.mount_ids].tolist(),
            )
        )

    def unlock(self, data):
        if not self.controller.can_unlock(self.speed(data)):
            raise RuntimeError("Cannot unlock a moving or active centrifuge")
        self.unlocked = True
        data.eq_active[self.task.instrument.lid_lock] = 0
        data.eq_active[self.mount_ids] = 0
        self.events.append(dict(time_s=float(data.time), event="safe_unlock"))

    def _update(self, data):
        lock = self.task.instrument.lid_lock
        torque = self.controller.update(
            self.model.opt.timestep,
            self.speed(data),
            abs(
                float(data.qpos[self.task.instrument.lid_qposadr])
                - float(self.model.eq_data[lock, 0])
            )
            < 0.01,
            bool(data.eq_active[lock]),
            self.balance_ratio(data),
        )
        if self.parking and self.controller.state == "IDLE":
            limit = self.controller.program.torque_limit_nm
            torque = float(
                np.clip(
                    -self.controller.program.speed_gain * self.speed(data),
                    -limit,
                    limit,
                )
            )
        data.ctrl[self.motor_id] = torque
        if self.controller.state != self.previous_state:
            self.events.append(
                dict(
                    time_s=float(data.time),
                    event=self.controller.state,
                    speed_rad_s=self.speed(data),
                )
            )
            self.previous_state = self.controller.state
        if self.unlocked:
            # The legacy instrument auto-latches a closed lid. This programme
            # owns an explicit safe unlock after observed rotor standstill.
            data.eq_active[lock] = 0


class CentrifugeCycle(CentrifugeInsertCloseComposite):
    default_task = "centrifuge_5430_cycle"
    time_limit = 60.0

    @classmethod
    def prepare(cls, spec):
        spec = super().prepare(spec)
        motor = spec.add_actuator(
            name=MOTOR, target=ROTOR, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        motor.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        motor.biastype = mujoco.mjtBias.mjBIAS_NONE
        motor.dyntype = mujoco.mjtDyn.mjDYN_NONE
        motor.gainprm[:] = 0.0
        motor.biasprm[:] = 0.0
        motor.gainprm[0] = 1.0
        motor.gear[:] = 0.0
        motor.gear[0] = 1.0
        motor.ctrllimited = motor.forcelimited = True
        motor.ctrlrange = motor.forcerange = [-0.1, 0.1]
        for name, body in zip(
            MOUNTS, ("1/centrifuge_1-5ml_screw_body", "2/centrifuge_1-5ml_screw_body")
        ):
            mount = spec.add_equality(name=name)
            mount.type = mujoco.mjtEq.mjEQ_WELD
            mount.objtype = mujoco.mjtObj.mjOBJ_BODY
            mount.name1 = "/centrifuge_eppendorf_5430:rotor"
            mount.name2 = body
            mount.active = False
            mount.solref = [0.005, 1.0]
            mount.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
            # The scene's equality default is a joint polynomial [0,1,...].
            # Weld anchors must not inherit that one-metre coefficient.
            mount.data = np.r_[np.zeros(10), 1.0]
        for key in spec.keys:
            if len(key.ctrl):
                key.ctrl = np.r_[key.ctrl, 0.0]
        return spec

    def __init__(self, spec):
        super().__init__(spec)
        self.rotor_program = RotorProgramSystem(task=self)
        self.rotor_program.reload(self.model)
        self.manager.set_systems((*self.manager.systems, self.rotor_program))
        self.expert_phases = []

    def reset(self, seed=None):
        info = super().reset(seed)
        self.expert_phases = []
        return info

    def check(self):
        control = self.rotor_program.controller
        return bool(
            control.state == "COMPLETE"
            and control.fault is None
            and self.rotor_program.unlocked
            and control.rotation_rad > 2 * np.pi
            and control.hold_s >= control.program.hold_seconds
            and self.rotor_program.balance_ratio(self.data)
            <= control.program.balance_ratio_limit
            and abs(self.rotor_program.speed(self.data))
            <= control.program.safe_speed_rad_s
            and not self.data.eq_active[self.instrument.lid_lock]
        )


class CentrifugeCycleExpert(CentrifugeCycle, CentrifugeInsertCloseCompositeExpert):
    def _phase(self, name, operation):
        start = float(self.data.time)
        operation()
        self.expert_phases.append(
            dict(phase=name, start_s=start, end_s=float(self.data.time))
        )

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span].copy()
        self.planner = self._insert_planner
        self._phase("insert", self._execute_insert)
        self.planner = self._lever_planner
        self._phase("close_lock", self._execute_close_lid)
        # Inserting a load can impart real angular momentum to the free rotor.
        # Brake it through the motor before requesting a measured-speed start.
        self.rotor_program.parking = True
        start = float(self.data.time)
        stationary = 0.0
        program = self.rotor_program.controller.program
        while stationary < program.stopped_seconds:
            if self.data.time >= self.time_limit:
                raise TimeoutError("Rotor did not reach a safe start speed")
            self.step_and_log({})
            stationary = (
                stationary + self.dt
                if abs(self.rotor_program.speed(self.data)) <= program.safe_speed_rad_s
                else 0.0
            )
        self.expert_phases.append(
            dict(phase="park_rotor", start_s=start, end_s=float(self.data.time))
        )
        self.rotor_program.start(self.data)
        start = float(self.data.time)
        while self.rotor_program.controller.state not in ("COMPLETE", "FAULT"):
            if self.data.time >= self.time_limit:
                raise TimeoutError(
                    "Centrifuge programme exceeded its declared 60 seconds"
                )
            self.step_and_log({})
        self.expert_phases.append(
            dict(phase="spin_brake", start_s=start, end_s=float(self.data.time))
        )
        if self.rotor_program.controller.state == "FAULT":
            raise RuntimeError(self.rotor_program.controller.fault)
        self._phase("safe_unlock", lambda: self.rotor_program.unlock(self.data))
        self.step_and_log({})
        self.finish()


CentrifugeCycle.Expert = CentrifugeCycleExpert
