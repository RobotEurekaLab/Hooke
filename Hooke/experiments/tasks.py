"""Continuous, contact-driven robot episodes shared by the three worlds."""

import mujoco
import numpy as np

from expert_common import UR5eArm, ExpertMotionMixin, make_topp_planner, set_gravcomp
from kinematics import Pose, mul_pose, neg_pose
from simulation import Manager
from task import Task, Expert, SCENE_ROOT
from experiments.analysis import estimate_mass
from experiments.mechanics import ExperimentMechanics, MechanicalError
from experiments.private_state import host_key, sample_seed
from experiments.scene import STORAGE, SAMPLES
from experiments.spectrometer import Spectrometer, analyze_spectrum
from worlds.environment import EnvironmentSystem
from worlds.profiles import WORLDS


def grasp_quaternion():
    # Site +Z points down; the finger opening axis is horizontal.
    return np.array([0.0, 0.0, 1.0, 0.0])


def make_experiment_task(world_name, operation):
    profile = WORLDS[world_name]

    class SpaceExperiment(Task):
        default_scene = SCENE_ROOT / f"space_{world_name}_experiment.gen.xml"
        default_task = f"space_{world_name}_{operation}"
        time_limit = 120.0
        early_stop = False

        @classmethod
        def prepare(cls, spec):
            set_gravcomp(spec.body("/ur:world"))
            return spec

        def __init__(self, spec):
            self.mechanics = ExperimentMechanics(profile=profile)
            self.environment = EnvironmentSystem(profile=profile)
            self.spectrometer = Spectrometer(mechanics=self.mechanics)
            super().__init__(
                Manager.from_spec(
                    spec, [self.environment, self.mechanics, self.spectrometer]
                )
            )
            self.arm = UR5eArm(self.model, "/ur:")
            self.original_eq_data = self.model.eq_data.copy()
            self.original_candidate_mass = float(
                self.model.body("sample_candidate").mass[0]
            )
            self.original_candidate_inertia = self.model.body(
                "sample_candidate"
            ).inertia.copy()
            self.original_candidate_spec = (
                float(spec.body("sample_candidate").mass),
                np.array(spec.body("sample_candidate").inertia),
            )
            self.private_campaign_key = host_key()

        def reset(self, seed=None):
            super().reset(seed)
            self.model.eq_data[:] = self.original_eq_data
            rng = np.random.default_rng(sample_seed(self.private_campaign_key, seed))
            mass = float(rng.uniform(0.12, 0.18))
            ratio = mass / self.original_candidate_mass
            self.model.body("sample_candidate").mass[0] = mass
            self.model.body("sample_candidate").inertia[:] = (
                self.original_candidate_inertia * ratio
            )
            self.spec.body("sample_candidate").mass = mass
            self.spec.body("sample_candidate").inertia = (
                self.original_candidate_spec[1] * ratio
            )
            self.spec.compile()  # Refresh the canonical XML serialization cache.
            mujoco.mj_setConst(self.model, self.data)
            self.manager.reset(keyframe=0)
            self.mechanics.rng = np.random.default_rng(
                np.random.SeedSequence([seed or 0, 518])
            )
            self.spectrometer.rng = np.random.default_rng(
                np.random.SeedSequence([seed or 0, 519])
            )
            self.spectrometer.profile_index = int(rng.integers(0, 4))
            version = f"episode-{sample_seed(self.private_campaign_key, f'calibration:{seed or 0}'):032x}"
            # The identifier is a one-way campaign digest, not a public RNG seed.
            self.mechanics.calibration_version = version + "-mass-v1"
            self.spectrometer.calibration_version = version + "-optical-v1"
            self.evaluator_truth = dict(
                candidate_total_mass_kg=mass,
                candidate_spectral_profile=self.spectrometer.profile_index,
            )
            self.spectrometer.open(self.data)
            self.completed = False
            self.conclusion = None
            self.phase_history = []
            self.task_info = dict(
                prefix=f"{profile.label}: {operation}",
                seed=seed,
                state_indices=self.arm.state_indices,
                action_indices=self.arm.action_indices,
                camera_mapping={
                    "image": "experiment_closeup",
                    "world": "world_overview",
                },
                display_only=False,
                space_environment=profile.report(),
                experiment=operation,
                fidelity="rigid_body_sample_handling_and_spring_encoder",
                sample_mass_target="cartridge_total_mass",
                grasp_attachment=False,
            )
            return self.task_info

        def check(self):
            return all(self.experiment_checks().values())

        def experiment_checks(self):
            operated = ("candidate",) if operation == "sample_transfer" else SAMPLES
            return dict(
                episode_completed=self.completed,
                samples_returned=all(
                    self.data.eq_active[self.mechanics.storage_locks[name]]
                    for name in SAMPLES
                ),
                instrument_empty=self.mechanics.loaded(self.data) is None,
                measurements_analyzed=operation == "sample_transfer"
                or self.conclusion is not None,
                **{
                    f"{name}_contact_lift_lock_return": value
                    for name, value in self.mechanics.handling_checks(operated).items()
                },
            )

        def experiment_ui(self):
            return dict(
                operation=operation,
                loaded_sample_id=self.mechanics.loaded(self.data),
                storage_locked={
                    name: bool(self.data.eq_active[self.mechanics.storage_locks[name]])
                    for name in SAMPLES
                },
                measurement_running=self.mechanics.active_recording is not None,
                recorded_measurements=len(self.mechanics.measurements)
                + len(self.spectrometer.records),
                phase=(
                    self.phase_history[-1]["phase"] if self.phase_history else "ready"
                ),
            )

        def experiment_report(self):
            return dict(
                self.mechanics.report(),
                spectral_measurement_ids=list(self.spectrometer.records),
                operation=operation,
                world=world_name,
                conclusion=self.conclusion,
                source_success=self.check(),
            )

    class SpaceExperimentExpert(SpaceExperiment, Expert, ExpertMotionMixin):
        def __init__(self, spec, freq=20):
            super().__init__(spec)
            self.freq = freq
            self.period = round(1 / (self.dt * freq))
            self.arm.register_ik(self.data)
            self.planner = make_topp_planner(
                self.arm.dof, self.arm.ik.solve, qc_vel=1.5, qc_acc=2.0
            )

        def phase(self, name, sample=None):
            self.phase_history.append(
                dict(phase=name, sample_id=sample, time_s=float(self.data.time))
            )

        def wait_seconds(self, seconds):
            for _ in range(round(seconds / self.dt)):
                self.step_and_log({})

        def body_pose(self, name):
            body = self.mechanics.bodies[name]
            return Pose(self.data.xpos[body].copy(), self.data.xquat[body].copy())

        def grasp(self, name):
            self.phase("grasp", name)
            position = self.data.site_xpos[
                self.model.site(f"sample_{name}_grasp").id
            ].copy()
            quat = grasp_quaternion()
            self.gripper_control(0, delay=100)
            self.move_to(Pose(position + np.array([0, 0, 0.14]), quat), 8)
            self.move_to(Pose(position, quat), 6)
            self.gripper_control(230, delay=200)
            if not self.mechanics.gripped(self.data, name):
                raise MechanicalError(f"Contact grasp failed for {name}")

        def lift(self, name):
            self.phase("lift", name)
            initial = self.body_pose(name).pos.copy()
            pose = self.arm.get_site_pose(self.data)
            self.move_to(Pose(pose.pos + np.array([0, 0, 0.12]), pose.quat), 8)
            self.wait_seconds(0.2)
            if self.body_pose(name).pos[2] - initial[
                2
            ] < 0.08 or not self.mechanics.gripped(self.data, name):
                raise MechanicalError("Sample did not lift while held by both pads")
            self.mechanics.event(
                self.data,
                "sample_lifted",
                sample_id=name,
                rise_m=float(self.body_pose(name).pos[2] - initial[2]),
                bilateral_contact=True,
            )

        def transfer_to(self, name, target_position):
            grasp = mul_pose(
                neg_pose(self.body_pose(name)), self.arm.get_site_pose(self.data)
            )
            above = Pose(
                np.asarray(target_position) + [0, 0, 0.12], np.array([1.0, 0, 0, 0])
            )
            self.move_to(mul_pose(above, grasp), 8)
            grasp = mul_pose(
                neg_pose(self.body_pose(name)), self.arm.get_site_pose(self.data)
            )
            target = Pose(
                np.asarray(target_position) + [0, 0, -0.001], np.array([1.0, 0, 0, 0])
            )
            self.move_to(mul_pose(target, grasp), 8)
            self.wait_seconds(0.2)

        def seat_with_feedback(self, name, target_position, seat_geom):
            """Bound small insertion corrections by actual contact, never attach."""
            for correction in range(3):
                if self.mechanics.contacts(self.data, name, {seat_geom}):
                    return
                if not self.mechanics.gripped(self.data, name):
                    raise MechanicalError("Contact grasp was lost during seating")
                self.phase("seat_contact_feedback", name)
                grasp = mul_pose(
                    neg_pose(self.body_pose(name)), self.arm.get_site_pose(self.data)
                )
                target = Pose(
                    np.asarray(target_position) + [0, 0, -0.002 - correction * 0.001],
                    np.array([1.0, 0, 0, 0]),
                )
                self.move_to(mul_pose(target, grasp), 4)
                self.wait_seconds(0.1)
            if not self.mechanics.contacts(self.data, name, {seat_geom}):
                raise MechanicalError("Bounded insertion did not reach the seat")

        def retreat(self):
            pose = self.arm.get_site_pose(self.data)
            self.move_to(Pose(pose.pos + [0, 0, 0.14], pose.quat), 8)

        def load_sample(self, name):
            self.spectrometer.open(self.data)
            self.wait_seconds(0.3)
            self.grasp(name)
            self.mechanics.release_storage(self.data, name)
            self.lift(name)
            self.phase("insert", name)
            self.transfer_to(name, self.data.site_xpos[self.mechanics.slot].copy())
            self.seat_with_feedback(
                name,
                self.data.site_xpos[self.mechanics.slot].copy(),
                self.mechanics.seat,
            )
            self.mechanics.lock(self.data, name)
            self.gripper_control(0, delay=150)
            self.retreat()
            self.wait_seconds(0.3)
            self.phase("locked_and_robot_clear", name)

        def retrieve_sample(self, name):
            self.spectrometer.open(self.data)
            self.wait_seconds(0.3)
            self.mechanics.hold = True
            self.wait_seconds(0.6)
            self.grasp(name)
            self.mechanics.unlock(self.data, name)
            self.lift(name)
            self.phase("return", name)
            self.transfer_to(name, STORAGE[name])
            self.seat_with_feedback(
                name, STORAGE[name], self.model.geom(f"storage_{name}_seat").id
            )
            self.mechanics.store(self.data, name)
            self.gripper_control(0, delay=150)
            self.retreat()

        def measure_mass(self, sample_id):
            method = "inertial" if world_name == "orbital" else "static_force"
            self.phase("measure", sample_id)
            self.mechanics.start_measurement(self.data, sample_id, method)
            self.wait_seconds(6.0 if method == "inertial" else 4.0)
            record = self.mechanics.finish_measurement(self.data)
            self.wait_seconds(0.6)
            return record

        def measure_spectrum(self, sample_id):
            self.phase("spectrum", sample_id)
            self.spectrometer.close(self.data)
            self.wait_seconds(0.6)
            return self.spectrometer.measure(self.data, sample_id)

        def execute(self):
            self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span].copy()
            try:
                self.phase("prepare")
                current = self.arm.get_site_pose(self.data)
                self.reposition_directly(
                    Pose(current.pos.copy(), grasp_quaternion()), seconds=1
                )
                if operation == "sample_transfer":
                    self.load_sample("candidate")
                    self.wait_seconds(0.5)
                    self.retrieve_sample("candidate")
                elif operation == "mass_measurement":
                    empty = self.measure_mass(None)
                    self.load_sample("reference")
                    reference = self.measure_mass("reference")
                    self.retrieve_sample("reference")
                    self.load_sample("candidate")
                    candidate = self.measure_mass("candidate")
                    self.retrieve_sample("candidate")
                    self.conclusion = estimate_mass(empty, reference, candidate)
                elif operation == "spectral_measurement":
                    dark = self.measure_spectrum(None)
                    self.load_sample("reference")
                    reference = self.measure_spectrum("reference")
                    self.retrieve_sample("reference")
                    self.load_sample("candidate")
                    first = self.measure_spectrum("candidate")
                    self.wait_seconds(0.2)
                    second = self.spectrometer.measure(self.data, "candidate")
                    self.retrieve_sample("candidate")
                    self.conclusion = analyze_spectrum(dark, reference, [first, second])
                else:
                    raise ValueError("Unknown experiment")
                self.completed = True
                self.phase("complete")
            finally:
                self.mechanics.cancel(self.data)
                self.finish()

    SpaceExperiment.Expert = SpaceExperimentExpert
    return SpaceExperiment


OrbitalTransfer = make_experiment_task("orbital", "sample_transfer")
LunarTransfer = make_experiment_task("lunar", "sample_transfer")
MartianTransfer = make_experiment_task("martian", "sample_transfer")
OrbitalMass = make_experiment_task("orbital", "mass_measurement")
LunarMass = make_experiment_task("lunar", "mass_measurement")
MartianMass = make_experiment_task("martian", "mass_measurement")
OrbitalSpectrum = make_experiment_task("orbital", "spectral_measurement")
LunarSpectrum = make_experiment_task("lunar", "spectral_measurement")
MartianSpectrum = make_experiment_task("martian", "spectral_measurement")
