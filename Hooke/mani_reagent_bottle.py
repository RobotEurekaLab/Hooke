"""Pick up a reagent bottle from the bench with a single UR5e arm.

The bottle asset (`model/instrument/reagent_bottle.xml`) was generated via
the GPT-6 Astra + Blender pipeline (`webui/custom_gen.py`) rather than
hand-modeled or sourced from an existing library -- see
private/technical-log.md for the generation session and how the
resulting mesh's axis convention was handled.

This is the first of Hooke's chemistry-lab tasks (as opposed to the
original AutoBio biology-lab instruments) -- deliberately the simplest
possible one (a plain grasp-and-lift, no lid/lever/screw mechanism) to
get a first chemistry asset working end-to-end before attempting
anything with moving parts (e.g. a fume hood sash).
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner


def _grasp_quat(arm_base_pos: np.ndarray, bottle_pos: np.ndarray) -> np.ndarray:
    """Builds a horizontal side-grasp orientation from the actual arm/bottle
    geometry rather than a hand-picked constant: the gripper's approach axis
    (site-local +Z) points from the arm base toward the bottle (horizontal,
    since the bottle sits upright and a parallel gripper closing around its
    cylindrical body needs to approach level, not from above), and the open
    axis (site-local Y) is the horizontal direction perpendicular to that --
    i.e. tangent to the cylinder, so the fingers close around it rather than
    scraping along its side."""
    approach = bottle_pos[:2] - arm_base_pos[:2]
    approach = approach / np.linalg.norm(approach)
    z_axis = np.array([approach[0], approach[1], 0.0])
    y_axis = np.array([-z_axis[1], z_axis[0], 0.0])  # horizontal, perpendicular to z_axis
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = local axes in world frame
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class PickupReagentBottle(Task):
    default_scene = SCENE_ROOT / "mani_reagent_bottle.xml"
    default_task = "pickup_reagent_bottle"
    time_limit = 15.0
    early_stop = True

    @classmethod
    def prepare(cls, mjspec: mujoco.MjSpec) -> mujoco.MjSpec:
        set_gravcomp(mjspec.body('/ur:world'))
        return mjspec

    def __init__(self, mjspec: mujoco.MjSpec):
        manager = Manager.from_spec(mjspec, [])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.bottle_jnt_adr = self.model.joint('reagent_bottle').qposadr.item()
        self.bottle_body = self.model.body('/reagent_bottle:reagent_bottle_body').id
        self.grasp_site = self.model.site('/reagent_bottle:grasp_site').id
        self._initial_bottle_z = None

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        # Small random horizontal jitter on the bottle's resting position.
        jitter = np.random.uniform((-0.03, -0.03), (0.03, 0.03))
        self.data.qpos[self.bottle_jnt_adr:self.bottle_jnt_adr + 2] += jitter
        mujoco.mj_kinematics(self.model, self.data)
        self._initial_bottle_z = self.data.xpos[self.bottle_body][2]

        self.task_info = {
            'prefix': 'pick up the reagent bottle',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        bottle_lifted = self.data.xpos[self.bottle_body][2] > self._initial_bottle_z + 0.05
        gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
        gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
        bottle_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/reagent_bottle:reagent_bottle_body_collision'),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/reagent_bottle:reagent_bottle_neck_collision'),
        }
        gripped = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if (gripper_geom1 in pair or gripper_geom2 in pair) and pair & bottle_geoms:
                gripped = True
                break
        return bottle_lifted and gripped


class PickupReagentBottleExpert(PickupReagentBottle, Expert, ExpertMotionMixin):
    def __init__(self, mjspec: mujoco.MjSpec, freq: int = 20):
        super().__init__(mjspec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / freq))
        self.arm.register_ik(self.data)
        self.planner = make_topp_planner(self.arm.dof, self.arm.ik.solve)

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(delay):
            self.step_and_log({})

    def wait(self, seconds: float):
        for _ in range(int(seconds / self.dt)):
            self.step_and_log({})

    def reposition_directly(self, pose: Pose, seconds: float = 1.5):
        """Solves IK once for `pose` and servos straight there, instead of
        `move_to`'s multi-waypoint slerp-then-IK-per-waypoint path.

        This task's first move is a large reorientation (the arm starts
        facing down at whatever the reset keyframe/perturbation left it at;
        the grasp needs a horizontal side-on approach instead) -- a big
        rotation away from every other task's recipes, which all keep a
        fixed or near-fixed orientation throughout. Interpolating that
        rotation via `move_to`'s per-waypoint IK chain kept landing one
        intermediate waypoint on a hard-to-reach local optimum (residual
        ~1e-3-1e-4, i.e. correct to a few mm/degrees but just missing
        kinematics.py's tolerance) at an unpredictable waypoint/seed
        combination -- solving once for the endpoint sidesteps needing
        every point along the way to be independently reachable."""
        sln = self.arm.ik.solve(pose.pos, pose.quat)
        self.data.ctrl[self.arm.act_span] = sln
        for _ in range(int(seconds / self.dt)):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        # `data.xpos`/`data.site_xpos` rows are views into MuJoCo's own live
        # buffers, not snapshots -- if the bottle gets bumped later in this
        # method, an uncopied reference here would silently follow it,
        # retargeting every downstream move at wherever the bottle ends up
        # rather than where it actually was when the recipe was planned.
        # Copy anything meant to be a fixed target.
        arm_base_pos = self.data.xpos[self.model.body('/ur:base').id].copy()
        bottle_pos = self.data.xpos[self.bottle_body].copy()
        quat = _grasp_quat(arm_base_pos, bottle_pos)

        grasp_pos = self.data.site_xpos[self.grasp_site].copy()
        # Approach from 8cm back along the horizontal approach direction,
        # then move in to the grasp point.
        approach_dir = (bottle_pos[:2] - arm_base_pos[:2])
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        pre_grasp = Pose(grasp_pos - np.array([approach_dir[0] * 0.08, approach_dir[1] * 0.08, 0.0]), quat)

        self.gripper_control(0)

        # reposition_directly commands ctrl straight to the IK solution and
        # lets the (stiff, kp=2000) position servo pull the arm there over
        # a fixed settle time -- fine for a small move, but for the big
        # reorientation this task needs (starting pose to a horizontal
        # side-grasp), doing it anywhere near the bottle's position swept
        # the arm through real 3D space fast enough to fling the bottle off
        # the table entirely (confirmed by rendering/logging its position
        # before and after -- this held even lifted 17cm above table
        # height, since it's the *arm's links*, not just the end-effector
        # target, that sweep through space during a big joint
        # reconfiguration). Reorienting while barely translating at all
        # -- right where the arm already is, before moving anywhere near
        # the bottle -- avoids that: the end-effector's own path stays
        # tiny even though the joint-space move is just as large.
        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        # Orientation is now fixed for the rest of the approach, so these
        # are pure-translation moves -- ordinary interpolated move_to is
        # smooth and safe for those.
        self.move_to(pre_grasp, num_steps=10)

        # The approach can still nudge the bottle in passing -- re-measure
        # its actual current position rather than trusting the pre-move
        # snapshot for the final approach, or a bottle bumped even
        # slightly out of place gets grasped at where it *was*, not where
        # it now is.
        grasp_pos = self.data.site_xpos[self.grasp_site].copy()
        grasp_pose = Pose(grasp_pos, quat)
        self.move_to(grasp_pose, num_steps=5)
        self.gripper_control(255)
        cur_pose = self.arm.get_site_pose(self.data)
        lift_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, 0.15]), cur_pose.quat)
        self.move_to(lift_pose, num_steps=10)
        self.wait(seconds=1.0)
        self.finish()


PickupReagentBottle.Expert = PickupReagentBottleExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = PickupReagentBottle.load()
    expert = PickupReagentBottle.Expert(spec)
    for i in trange(20):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
