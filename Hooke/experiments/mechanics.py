"""Contact-gated retention and observed encoder measurements.

Only storage and the declared instrument create welds. The robot has no grasp
attachment. A fixture captures the current transform without moving a sample.
"""

import mujoco
import numpy as np

from simulation import System
from experiments.scene import SAMPLES, STORAGE, REFERENCE_MASS_KG


class MechanicalError(ValueError):
    """An operation cannot proceed from the observed mechanical state."""


class ExperimentMechanics(System):
    def _configure(self, profile):
        self.profile = profile

    def _reload(self, model):
        self.bodies = {name: model.body(f"sample_{name}").id for name in SAMPLES}
        self.shells = {name: model.geom(f"sample_{name}_shell").id for name in SAMPLES}
        self.storage_locks = {
            name: model.equality(f"storage_{name}_lock").id for name in SAMPLES
        }
        self.instrument_locks = {
            name: model.equality(f"instrument_{name}_lock").id for name in SAMPLES
        }
        self.tray = model.body("measurement_tray").id
        self.slot = model.site("instrument_sample_pose").id
        self.seat = model.geom("instrument_seat").id
        self.left = {model.geom(f"/ur:2f85:left_pad{i}").id for i in (1, 2)}
        self.right = {model.geom(f"/ur:2f85:right_pad{i}").id for i in (1, 2)}
        self.robot_geoms = {
            i
            for i in range(model.ngeom)
            if model.body(int(model.geom_bodyid[i])).name.startswith("/ur:")
        }
        joint = model.joint("measurement_slide")
        self.qa, self.va = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.latch_drive = model.actuator("instrument_latch_drive").id
        self.latch_qa = int(model.joint("instrument_latch_slide").qposadr[0])

    def _reset(self, data):
        self.events = []
        self.measurements = {}
        self.hold = True
        self.hold_position = 0.0
        self.excitation_until = -1.0
        self.active_recording = None
        self.encoder_rows = []
        self.next_sample_time = 0.0
        self.rng = np.random.default_rng(0)
        self.calibration_version = "unset"
        self.previous_time = float(data.time)
        self.handling = {
            name: dict(
                contact_grasp_s=0.0,
                maximum_lift_m=0.0,
                locked_robot_clear_s=0.0,
                returned_after_loading=False,
            )
            for name in SAMPLES
        }
        self.seen_loaded = set()

    def contacts(self, data, name, geoms):
        shell = self.shells[name]
        return any(
            shell in (int(c.geom1), int(c.geom2))
            and ({int(c.geom1), int(c.geom2)} - {shell}) & geoms
            for c in data.contact
        )

    def gripped(self, data, name):
        return self.contacts(data, name, self.left) and self.contacts(
            data, name, self.right
        )

    def event(self, data, operation, **details):
        self.events.append(
            dict(time_s=float(data.time), operation=operation, **details)
        )

    def capture(self, data, equality):
        if data.eq_active[equality]:
            raise MechanicalError("Fixture is already locked")
        a, b = map(
            int, (self.model.eq_obj1id[equality], self.model.eq_obj2id[equality])
        )
        inverse_pos, inverse_quat, position, orientation = (
            np.zeros(3),
            np.zeros(4),
            np.zeros(3),
            np.zeros(4),
        )
        mujoco.mju_negPose(inverse_pos, inverse_quat, data.xpos[a], data.xquat[a])
        mujoco.mju_mulPose(
            position,
            orientation,
            inverse_pos,
            inverse_quat,
            data.xpos[b],
            data.xquat[b],
        )
        self.model.eq_data[equality, :3] = 0
        self.model.eq_data[equality, 3:6] = position
        self.model.eq_data[equality, 6:10] = orientation
        data.eq_active[equality] = True

    def release_storage(self, data, name):
        if name not in SAMPLES or not data.eq_active[self.storage_locks[name]]:
            raise MechanicalError("Sample is not retained in storage")
        if not self.gripped(data, name):
            raise MechanicalError(
                "Both gripper pads must contact the sample before release"
            )
        data.eq_active[self.storage_locks[name]] = False
        self.event(data, "storage_released", sample_id=name, bilateral_contact=True)

    def loaded(self, data):
        names = [
            name for name in SAMPLES if data.eq_active[self.instrument_locks[name]]
        ]
        if len(names) > 1:
            raise MechanicalError("Multiple samples are locked in one slot")
        return names[0] if names else None

    def lock(self, data, name):
        if (
            name not in SAMPLES
            or self.loaded(data) is not None
            or data.eq_active[self.storage_locks[name]]
        ):
            raise MechanicalError(
                "Instrument is occupied or sample has not been released"
            )
        position_error = float(
            np.linalg.norm(data.xpos[self.bodies[name]] - data.site_xpos[self.slot])
        )
        angle_error = float(
            2
            * np.arccos(
                np.clip(
                    abs(np.dot(data.xquat[self.bodies[name]], data.xquat[self.tray])),
                    0,
                    1,
                )
            )
        )
        if (
            position_error > 0.006
            or angle_error > 0.12
            or not self.contacts(data, name, {self.seat})
        ):
            raise MechanicalError(
                f"Sample must be seated: position={position_error:.6f}, angle={angle_error:.6f}"
            )
        self.capture(data, self.instrument_locks[name])
        self.seen_loaded.add(name)
        data.ctrl[self.latch_drive] = 0.025
        self.event(
            data,
            "instrument_locked",
            sample_id=name,
            position_error_m=position_error,
            angle_error_rad=angle_error,
            seat_contact=True,
        )

    def unlock(self, data, name):
        if self.loaded(data) != name or not self.gripped(data, name):
            raise MechanicalError("Correct sample must be gripped before unlocking")
        if self.active_recording:
            raise MechanicalError("Cancel measurement before unlocking")
        data.eq_active[self.instrument_locks[name]] = False
        data.ctrl[self.latch_drive] = 0
        self.event(data, "instrument_unlocked", sample_id=name, bilateral_contact=True)

    def store(self, data, name):
        if not self.gripped(data, name) or self.loaded(data) == name:
            raise MechanicalError(
                "Sample must be gripped and released from the instrument"
            )
        if np.linalg.norm(data.xpos[self.bodies[name]] - STORAGE[name]) > 0.006:
            raise MechanicalError("Sample is not at its storage seat")
        seat = self.model.geom(f"storage_{name}_seat").id
        if not self.contacts(data, name, {seat}):
            raise MechanicalError("Sample must contact its storage seat")
        self.capture(data, self.storage_locks[name])
        self.handling[name]["returned_after_loading"] = name in self.seen_loaded
        self.event(data, "storage_locked", sample_id=name, seat_contact=True)

    def start_measurement(self, data, sample_id, method):
        if method not in ("inertial", "static_force"):
            raise MechanicalError("Unknown measurement method")
        if self.active_recording:
            raise MechanicalError("Measurement already running")
        if self.loaded(data) != sample_id:
            raise MechanicalError("Requested sample is not locked in the instrument")
        if sample_id and self.contacts(data, sample_id, self.robot_geoms):
            raise MechanicalError("Robot must clear the sample before measuring")
        if sample_id and float(data.qpos[self.latch_qa]) < 0.024:
            raise MechanicalError(
                "Instrument latch has not reached its closed position"
            )
        if method == "static_force" and self.profile.gravity_m_s2 < 0.05:
            raise MechanicalError("Static weighing is unidentifiable in microgravity")
        if (method == "inertial") != (self.profile.name == "orbital"):
            raise MechanicalError("Method does not match this instrument axis")
        self.hold = False
        self.excitation_until = float(data.time) + 0.08 if method == "inertial" else -1
        self.active_recording = dict(
            sample_id=sample_id, method=method, started_s=float(data.time), valid=True
        )
        self.encoder_rows = []
        self.next_sample_time = float(data.time)
        self.event(data, "measurement_started", **self.active_recording)

    def finish_measurement(self, data):
        if not self.active_recording:
            raise MechanicalError("No measurement is running")
        record = dict(
            self.active_recording,
            measurement_id=f"m{len(self.measurements)+1:04d}",
            time_position=self.encoder_rows.copy(),
            encoder_units="s,m",
            reference_mass_kg=(
                REFERENCE_MASS_KG
                if self.active_recording["sample_id"] == "reference"
                else None
            ),
            fidelity="rigid_body_spring_encoder_with_declared_noise",
            encoder_noise_std_m=1e-6,
            spring_stiffness_n_m=float(
                self.model.jnt_stiffness[self.model.joint("measurement_slide").id]
            ),
            gravity_m_s2=self.profile.gravity_m_s2,
        )
        record["valid"] &= self.loaded(data) == record["sample_id"]
        record["calibration_version"] = self.calibration_version
        self.measurements[record["measurement_id"]] = record
        self.active_recording = None
        self.hold = True
        self.hold_position = 0.0
        self.event(
            data,
            "measurement_finished",
            measurement_id=record["measurement_id"],
            sample_id=record["sample_id"],
        )
        return record

    def cancel(self, data):
        if self.active_recording:
            self.event(
                data,
                "measurement_cancelled",
                sample_id=self.active_recording["sample_id"],
                discarded_samples=len(self.encoder_rows),
            )
        self.active_recording = None
        self.hold = True
        self.hold_position = 0.0
        self.encoder_rows = []
        data.qfrc_applied[self.va] = 0

    def _update(self, data):
        position, velocity = float(data.qpos[self.qa]), float(data.qvel[self.va])
        data.qfrc_applied[self.va] = (
            -500 * (position - self.hold_position) - 20 * velocity
            if self.hold
            else (0.15 if data.time < self.excitation_until else 0.0)
        )
        elapsed = float(data.time) - self.previous_time
        self.previous_time = float(data.time)
        for name in SAMPLES:
            metrics = self.handling[name]
            retained = bool(data.eq_active[self.storage_locks[name]])
            if not retained and self.gripped(data, name):
                metrics["contact_grasp_s"] += elapsed
                metrics["maximum_lift_m"] = max(
                    metrics["maximum_lift_m"],
                    float(data.xpos[self.bodies[name], 2] - STORAGE[name][2]),
                )
            if data.eq_active[self.instrument_locks[name]] and not self.contacts(
                data, name, self.robot_geoms
            ):
                metrics["locked_robot_clear_s"] += elapsed
        if self.active_recording and data.time + 1e-10 >= self.next_sample_time:
            sample = self.active_recording["sample_id"]
            if (
                self.loaded(data) != sample
                or abs(position) >= 0.029
                or (sample and self.contacts(data, sample, self.robot_geoms))
            ):
                self.active_recording["valid"] = False
            self.encoder_rows.append(
                [
                    float(data.time) - self.active_recording["started_s"],
                    position + float(self.rng.normal(0, 1e-6)),
                ]
            )
            self.next_sample_time += 0.02

    def report(self):
        return {
            "events": self.events,
            "handling": self.handling,
            "measurement_ids": list(self.measurements),
            "robot_attachment": False,
            "fixture_lock": "contact_gated_captured_weld_with_driven_visual_indicator",
            "fixture_compliance": "not_calibrated",
            "seal": "closed_rigid_geometry_only",
        }

    def handling_checks(self, names):
        return {
            name: bool(
                self.handling[name]["contact_grasp_s"] >= 0.1
                and self.handling[name]["maximum_lift_m"] >= 0.08
                and self.handling[name]["locked_robot_clear_s"] >= 0.2
                and self.handling[name]["returned_after_loading"]
            )
            for name in names
        }
