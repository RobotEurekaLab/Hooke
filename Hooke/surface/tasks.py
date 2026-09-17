"""Wheel-drive exploration and contact-driven collection in both simulators."""

import mujoco
import numpy as np

from expert_common import ExpertMotionMixin, UR5eArm, make_topp_planner, set_gravcomp
from kinematics import Pose
from simulation import Manager
from surface.navigation import WheelNavigation
from surface.profiles import MISSIONS
from surface.progress import SamplingProgress
from surface.scene import ARM_PREFIX, BIN_CENTER, CACHE, build
from task import Expert, Task
from worlds.environment import EnvironmentSystem


class SurfaceEnvironment(EnvironmentSystem):
    def _configure(self, mission):
        self.mission = mission
        super()._configure(mission.environment)

    def report(self):
        return dict(
            profile=self.mission.environment_report(),
            thermal_witness=self.witness.report(),
        )


def make_sampling_task(world):
    mission = MISSIONS[world]

    class SurfaceSampling(Task):
        default_scene = CACHE / f"{world}-sampling.xml"
        default_task = mission.task_name
        time_limit = 120.0
        early_stop = False
        native_physics_options = {"approximate_cylinders": True}

        @classmethod
        def load(cls, scene=None):
            path = scene if scene is not None else build(mission)
            cls.default_scene = path
            return cls.prepare(mujoco.MjSpec.from_file(str(path)))

        @classmethod
        def prepare(cls, spec):
            # The fixed base shares the rover's physical rigid body. Only the
            # moving arm links use compensation, leaving the rover on gravity.
            set_gravcomp(spec.body(ARM_PREFIX + "shoulder_link"))
            return spec

        def __init__(self, spec):
            self.environment = SurfaceEnvironment(mission=mission)
            self.progress = SamplingProgress(mission=mission)
            super().__init__(Manager.from_spec(spec, [self.environment, self.progress]))
            self.arm = UR5eArm(self.model, ARM_PREFIX)
            self.drive = WheelNavigation(self.model, mission)

        def reset(self, seed=None):
            super().reset(seed)
            self.manager.reset(keyframe=0)
            self.drive = WheelNavigation(self.model, mission)
            self.phase_history = []
            if self.arm.ik is not None:
                self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span].copy()
            self.phase_name = "ready"
            self.task_info = dict(
                prefix=mission.label,
                seed=seed,
                camera_mapping={
                    "image": "expedition_overview",
                    "follow": "rover_follow",
                    "sample": "sampling_closeup",
                    "landing": "landing_site",
                },
                display_only=False,
                state_indices=list(range(7))
                + self.drive.state_indices
                + self.arm.state_indices,
                action_indices=self.drive.actuators.tolist() + self.arm.action_indices,
                space_environment=mission.environment_report(),
                mission="surface_sample_collection",
                grasp_attachment=False,
            )
            return self.task_info

        def step_and_log(self, info):
            if self.drive.enabled:
                self.drive.command(self.data)
            super().step_and_log(info)

        def check(self):
            return all(self.mission_checks().values())

        def mission_checks(self):
            return self.progress.checks(self.data)

        def mission_ui(self):
            return dict(
                phase=self.phase_name,
                traveled_m=self.progress.distance_m,
                sample_in_bin=self.progress.inside_bin(self.data),
                rover_position_m=self.data.xpos[self.progress.rover].tolist(),
            )

        def mission_report(self):
            return self.progress.report(self.data)

    class SamplingExpert(SurfaceSampling, Expert, ExpertMotionMixin):
        def __init__(self, spec):
            super().__init__(spec)
            self.period = round(1 / (20 * self.dt))
            self.arm.register_ik(self.data)
            self.planner = make_topp_planner(
                self.arm.dof, self.arm.ik.solve, qc_vel=0.35, qc_acc=0.4
            )

        def phase(self, name):
            self.phase_name = name
            self.phase_history.append(dict(phase=name, time_s=float(self.data.time)))

        def wait_seconds(self, seconds):
            for _ in range(round(seconds / self.dt)):
                self.step_and_log({})

        def navigate(self, target, *, reverse=False):
            self.drive.target = np.asarray(target, dtype=float)
            self.drive.reverse = reverse
            self.drive.parked = False
            deadline = self.data.time + 35
            while self.data.time < deadline:
                position, _ = self.drive.pose(self.data)
                if np.linalg.norm(position - target) < 0.08:
                    self.drive.park(self.data)
                    self.wait_seconds(0.6)
                    return
                self.step_and_log({})
            raise RuntimeError("Wheel-driven navigation did not reach its waypoint")

        def tool_orientation(self):
            _, heading = self.drive.pose(self.data)
            return np.array([0.0, -np.sin(heading / 2), np.cos(heading / 2), 0.0])

        def collect(self):
            self.phase("approach_rock")
            self.gripper_control(0, delay=150)
            # The loose specimen can rotate as it settles. Use the observed
            # mesh centre, with clearance for the lower gripper pad.
            position = self.data.geom_xpos[
                self.model.geom("field_sample_geom").id
            ].copy() + [0, 0, 0.004]
            quat = self.tool_orientation()
            self.move_to(Pose(position + [0, 0, 0.7], quat), 8)
            self.move_to(Pose(position + [0, 0, 0.15], quat), 6)
            self.move_to(Pose(position, quat), 6)
            self.phase("grasp_rock")
            self.gripper_control(255, delay=250)
            if not self.progress.gripped(self.data):
                raise RuntimeError("Rock was not held by both gripper pads")
            self.phase("lift_rock")
            pose = self.arm.get_site_pose(self.data)
            self.move_to(Pose(pose.pos + [0, 0, 0.70], pose.quat), 8)
            self.wait_seconds(0.2)
            if not self.progress.lifted_in_grasp:
                raise RuntimeError("Rock did not lift while held")
            self.phase("stow_sample")
            rotation = self.data.xmat[self.progress.rover].reshape(3, 3)
            side = self.data.xpos[self.progress.rover] + rotation @ np.array(
                [0.10, -0.60, 0.55]
            )
            self.move_to(Pose(side, quat), 6)
            if not self.progress.gripped(self.data):
                raise RuntimeError("Rock lost during transfer around the arm base")
            target = self.data.xpos[self.progress.rover] + rotation @ (
                BIN_CENTER + [0, 0, 0.08]
            )
            self.move_to(Pose(target + [0, 0, 0.15], quat), 8)
            self.move_to(Pose(target, quat), 4)
            self.gripper_control(0, delay=200)
            pose = self.arm.get_site_pose(self.data)
            self.move_to(Pose(pose.pos + [0, 0, 0.18], pose.quat), 4)
            self.wait_seconds(1)
            if not self.progress.collected:
                raise RuntimeError("Sample did not settle in the rover bin")

        def execute(self):
            self.drive.enabled = True
            self.phase("leave_landing_site")
            self.wait_seconds(0.6)
            self.navigate(mission.collection_stop)
            self.collect()
            self.phase("return_to_lander")
            self.navigate(mission.home, reverse=True)
            self.wait_seconds(1)
            self.phase("complete")
            self.finish()

    SurfaceSampling.Expert = SamplingExpert
    return SurfaceSampling


LunarSampling = make_sampling_task("lunar")
MartianSampling = make_sampling_task("martian")
