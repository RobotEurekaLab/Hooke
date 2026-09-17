"""A free-standing G1 authorizes collection by an accompanying rover."""

import math

import mujoco
import numpy as np

from contact_state import body_geoms, touching
from surface.humanoid import HOME, PREFIX, HumanoidNavigation, IndexFingerControl
from surface.profiles import MISSIONS
from surface.tasks import make_sampling_task
from surface.team_scene import attach_team, reset_humanoid


class TeamProgress:
    def __init__(self, model, data, mission):
        self.mission = mission
        self.body = model.body(PREFIX + "pelvis").id
        self.root_va = int(model.joint(PREFIX + "floating_base_joint").dofadr[0])
        self.button_qa = int(model.joint("team_button_joint").qposadr[0])
        # Menagerie fuses the fixed palm into its wrist body when compiling.
        wrist = model.body(PREFIX + "left_wrist_yaw_link").id
        self.hand_geoms = {
            geom
            for geom in body_geoms(model, wrist)
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH
            or model.mesh(int(model.geom_dataid[geom])).name.startswith(
                PREFIX + "left_hand_"
            )
        }
        self.button_geoms = {model.geom("team_button_cap").id}
        self.ground = (
            {model.geom("terrain").id}
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain") >= 0
            else {model.geom("surface").id}
        )
        self.feet = [
            body_geoms(model, model.body(PREFIX + side + "_ankle_roll_link").id)
            for side in ["left", "right"]
        ]
        self.previous = data.xpos[self.body, :2].copy()
        self.previous_time = float(data.time)
        self.traveled_m = 0.0
        self.max_tilt = 0.0
        self.min_height = float(data.xpos[self.body, 2])
        self.foot_contact_s = [0.0, 0.0]
        self.button_contact_s = 0.0
        self.button_pressed = False
        self.visited = False
        self.events = []

    def update(self, data):
        dt = float(data.time) - self.previous_time
        if dt <= 0:
            return
        position = data.xpos[self.body]
        self.traveled_m += float(np.linalg.norm(position[:2] - self.previous))
        self.previous = position[:2].copy()
        self.previous_time = float(data.time)
        self.max_tilt = max(
            self.max_tilt,
            math.acos(np.clip(data.xmat[self.body].reshape(3, 3)[2, 2], -1, 1)),
        )
        self.min_height = min(self.min_height, float(position[2]))
        goal = np.asarray(self.mission.collection_stop) + HOME
        if np.linalg.norm(position[:2] - goal) < 0.3 and not self.visited:
            self.visited = True
            self.events.append(
                dict(
                    event="humanoid_reached_collection_area",
                    simulation_s=float(data.time),
                )
            )
        for i, foot in enumerate(self.feet):
            if touching(data, foot, self.ground):
                self.foot_contact_s[i] += dt
        contact = (
            self.visited
            and touching(data, self.hand_geoms, self.button_geoms)
            and data.qpos[self.button_qa] >= 0.004
        )
        self.button_contact_s = self.button_contact_s + dt if contact else 0.0
        if self.visited and self.button_contact_s >= 0.05 and not self.button_pressed:
            self.button_pressed = True
            self.events.append(
                dict(
                    event="humanoid_physically_confirmed_sampling",
                    simulation_s=float(data.time),
                )
            )

    def checks(self, data):
        return dict(
            humanoid_traveled_at_least_9m=self.traveled_m >= 9,
            humanoid_reached_sampling_area=self.visited,
            humanoid_confirmed_by_contact=self.button_pressed,
            humanoid_remained_upright=self.max_tilt < 0.5 and self.min_height > 0.55,
            humanoid_both_feet_supported=all(t > 0.5 for t in self.foot_contact_s),
            humanoid_returned_within_25cm=bool(
                np.linalg.norm(data.xpos[self.body, :2] - HOME) < 0.25
            ),
            humanoid_base_speed_below_12cm_s=bool(
                np.linalg.norm(data.qvel[self.root_va : self.root_va + 3]) < 0.12
            ),
        )

    def report(self, data):
        return dict(
            robot="Unitree G1 with original dexterous hands",
            role="walk, physically authorize rover sampling, return",
            traveled_m=self.traveled_m,
            position_m=data.xpos[self.body].tolist(),
            max_tilt_rad=self.max_tilt,
            min_pelvis_height_m=self.min_height,
            foot_contact_s=list(self.foot_contact_s),
            sampling_authorized_by_contact=self.button_pressed,
            events=list(self.events),
            floating_base=True,
            external_support=False,
            policy="Unitree RL Gym G1 LSTM; gravity-scaled gains, observations and timing",
        )


def make_cooperative_task(world):
    mission = MISSIONS[world]
    Base = make_sampling_task(world)

    class CooperativeTask(Base):
        default_task = f"space_{world}_humanoid_rover"
        time_limit = 180.0

        @classmethod
        def load(cls, scene=None):
            spec = super().load(scene)
            return attach_team(spec, mission)

        def reset(self, seed=None):
            super().reset(seed)
            reset_humanoid(self.model, self.data)
            self.gait = HumanoidNavigation(self.model, mission.environment.gravity_m_s2)
            self.finger = IndexFingerControl(self.model, self.gait.rate)
            self.hand_home = self.data.ctrl[self.finger.actuators].copy()
            self.team = TeamProgress(self.model, self.data, mission)
            self.escort = False
            for index in range(self.model.njnt):
                joint = self.model.joint(index)
                if joint.name and joint.name.startswith(PREFIX):
                    width = 7 if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE else 1
                    self.task_info["state_indices"].extend(
                        range(int(joint.qposadr[0]), int(joint.qposadr[0]) + width)
                    )
            self.task_info["action_indices"].extend(
                i
                for i in range(self.model.nu)
                if self.model.actuator(i).name.startswith(PREFIX)
            )
            self.task_info["camera_mapping"].update(
                follow="team_follow", sample="team_closeup"
            )
            self.task_info["mission"] = "humanoid_rover_cooperation"
            return self.task_info

        def step_and_log(self, info):
            self.gait.command_joints(self.data)
            self.finger.command_joints(self.data)
            if self.escort and not self.drive.parked:
                humanoid_x = self.data.xpos[self.gait.body, 0] - HOME[0]
                self.drive.target = np.array(
                    [np.clip(humanoid_x, 0, mission.collection_stop[0]), 0.0]
                )
            super().step_and_log(info)
            self.team.update(self.data)
            if self.gait.enabled and (
                self.team.max_tilt >= 0.5 or self.team.min_height <= 0.55
            ):
                raise RuntimeError("G1 lost its upright posture during cooperation")

        def mission_checks(self):
            base = super().mission_checks()
            return (
                base
                if not hasattr(self, "team")
                else dict(base, **self.team.checks(self.data))
            )

        def irreversible_control_failure(self):
            if self.team.max_tilt >= 0.5 or self.team.min_height <= 0.55:
                return dict(
                    check="humanoid_remained_upright",
                    irreversible=True,
                    max_tilt_rad=self.team.max_tilt,
                    min_pelvis_height_m=self.team.min_height,
                )
            return None

        def mission_ui(self):
            report = super().mission_ui()
            report.update(
                scenario="team",
                humanoid_position_m=self.data.xpos[self.gait.body].tolist(),
                humanoid_traveled_m=self.team.traveled_m,
                sampling_authorized=self.team.button_pressed,
            )
            return report

        def mission_report(self):
            report = super().mission_report()
            if hasattr(self, "team"):
                report.update(
                    scenario="team",
                    robot="Unitree G1 + six-wheel sampling rover",
                    humanoid=self.team.report(self.data),
                    checks=self.mission_checks(),
                )
                if self.gait.policy is not None:
                    report["humanoid"]["policy_arrays_sha256"] = self.gait.policy.sha256
                report["limitations"] = [
                    value
                    for value in report["limitations"]
                    if "humanoid walking" not in value
                ]
                report["limitations"].append(
                    "G1 authorizes sampling; rover handles the rock; no humanoid rock grasp or perception-based survey"
                )
            return report

    class CooperativeExpert(CooperativeTask, Base.Expert):
        def navigate(self, target, *, reverse=False):
            self.gait.target = np.asarray(target) + HOME
            self.drive.reverse = reverse
            self.drive.parked = False
            self.escort = True
            deadline = self.data.time + 65
            while self.data.time < deadline:
                rover_position, _ = self.drive.pose(self.data)
                humanoid_position = self.data.xpos[self.gait.body, :2]
                if (
                    np.linalg.norm(rover_position - target) < 0.08
                    and np.linalg.norm(humanoid_position - self.gait.target) < 0.14
                ):
                    self.escort = False
                    self.drive.park(self.data)
                    self.gait.target = self.data.xpos[self.progress.rover, :2] + HOME
                    self.wait_seconds(1)
                    return
                self.step_and_log({})
            raise RuntimeError("Humanoid and rover did not reach their shared waypoint")

        def authorize(self):
            self.phase("humanoid_confirm")
            button = self.model.body("team_button").id
            rotation = self.data.xmat[self.progress.rover].reshape(3, 3)
            rest = self.data.xpos[button] - rotation @ np.array(
                [0, self.data.qpos[self.team.button_qa], 0]
            )
            # Walking leaves different contact poses in each gravity/engine.
            # Stop reaching as soon as actual contact and travel have latched.
            for depth, seconds in ((-0.10, 8), (-0.004, 8), (0.008, 4), (0.014, 4)):
                if self.team.button_pressed:
                    break
                self.finger.target = rest + rotation @ np.array([0, depth, 0])
                deadline = self.data.time + seconds / self.gait.rate
                while self.data.time < deadline and not self.team.button_pressed:
                    self.step_and_log({})
            if not self.team.button_pressed:
                raise RuntimeError(
                    "G1 did not physically press the sampling confirmation button"
                )
            self.finger.target = None
            initial = self.data.ctrl[self.finger.actuators].copy()
            duration = (
                max(4.0, 1.5 * np.max(np.abs(initial - self.hand_home)) / 0.3)
                / self.gait.rate
            )
            for elapsed in np.linspace(0, 1, round(duration / self.dt)):
                amount = elapsed * elapsed * (3 - 2 * elapsed)
                self.data.ctrl[self.finger.actuators] = (
                    1 - amount
                ) * initial + amount * self.hand_home
                self.step_and_log({})

        def execute(self):
            self.gait.enabled = True
            self.drive.enabled = True
            self.phase("leave_landing_site")
            self.wait_seconds(3)
            self.navigate(mission.collection_stop)
            self.authorize()
            self.gait.target = self.data.xpos[self.progress.rover, :2] + [-0.5, -1.3]
            self.wait_seconds(4)
            self.collect()
            self.phase("return_to_lander")
            self.navigate(mission.home, reverse=True)
            self.wait_seconds(2)
            self.phase("complete")
            self.finish()

    CooperativeTask.Expert = CooperativeExpert
    return CooperativeTask


LunarTeam = make_cooperative_task("lunar")
MartianTeam = make_cooperative_task("martian")
