"""Press a coin-cell crimper's lever down with a single UR5e arm.

Promotes `coin_cell_crimper_display` (display-only, added in the
display-scene breadth batches -- see private/technical-log.md) to a real
interactive task: the display version already has a real hinge joint on
the lever, this file adds the grasp/actuation recipe and a physical
success check on top of the same asset.

Unlike `mani_fume_hood.py`'s sash (a `slide` joint, so the handle travels
in a straight line and keeps a fixed orientation throughout), this
lever's `hinge` joint means the handle sweeps a genuine arc and its
orientation changes as it rotates -- a rigidly-gripped handle needs the
gripper to rotate along with it, not just translate. Rather than
re-deriving that by hand, this borrows the core trick already used by
`archetypes/lever_lock_centrifuge.py`'s `lever_path`: sample a few qpos
values along the joint's sweep, forward-kinematics each one to get the
handle's actual pose at that angle, and compose a *fixed* relative
grip offset (computed once, from the real grasp orientation at the
start of the reach) onto each sampled pose to get the gripper's own
waypoint. This file doesn't use that archetype directly since it's built
around a lock mechanism this crimper doesn't have (no lid, no `eq_active`
constraint, no `instrument.py` registration) -- just its `FK` +
`qpos_interpolate` building blocks.

**Not a working task, and not catalogued.** The reorientation-in-place
step that every other UR5e task in this codebase uses safely puts the
arm's own forearm/wrist through its mounting table on most seeds here
(confirmed via `mj_contactForce` + geom/body lookup, forces from ~700N up
to ~30kN depending on seed) -- the same root cause diagnosed for
`mani_fume_hood.py`'s `open_fume_hood` (see private/technical-log.md):
this handle's height/approach direction lands in the same kinematically
fragile region of this UR5e's reachable space. Raising the crimper's own
mount height didn't help (the grasp orientation, not the target position,
is what triggers it), and neither did a wider arc-shaped approach path.
Left here as a documented, reproducible instance of an already-known,
still-unsolved class of bug rather than deleted -- see private/TODO.md.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import FK, Pose, mul_pose
from grasp.quat import quatinv, quatcompose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, qpos_interpolate, make_topp_planner

LEVER_OPEN = 0.0
LEVER_CRIMPED = -1.2


def _grasp_quat(arm_base_pos: np.ndarray, handle_pos: np.ndarray) -> np.ndarray:
    """Same shape as mani_fume_hood.py's: the handle knob is a horizontal
    peg (confirmed by querying the compiled model's site orientation, not
    assumed from the XML's own local-frame numbers -- this scene's table
    attach rotation makes local axes an unreliable guide, the same lesson
    logged for every instrument so far), so the approach axis (site-local
    +Z) points horizontally from the arm base to the handle and the open
    axis (site-local Y) is vertical, perpendicular to both."""
    approach = handle_pos[:2] - arm_base_pos[:2]
    approach = approach / np.linalg.norm(approach)
    z_axis = np.array([approach[0], approach[1], 0.0])
    y_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class CrimpCoinCell(Task):
    default_scene = SCENE_ROOT / "mani_coin_cell_crimper.xml"
    default_task = "crimp_coin_cell"
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
        self.lever_jnt_adr = self.model.joint('/coin_cell_crimper:lever_joint').qposadr.item()
        self.grasp_site = self.model.site('/coin_cell_crimper:grasp_site').id

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        self.data.qpos[self.lever_jnt_adr] = LEVER_OPEN
        mujoco.mj_kinematics(self.model, self.data)

        self.fk_lever = FK(1, self.model, self.data, '/coin_cell_crimper:base', '/coin_cell_crimper:grasp_site')

        self.task_info = {
            'prefix': 'press the coin cell crimper lever down',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        return self.data.qpos[self.lever_jnt_adr] < LEVER_CRIMPED + 0.05


class CrimpCoinCellExpert(CrimpCoinCell, Expert, ExpertMotionMixin):
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

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        arm_base_pos = self.data.xpos[self.model.body('/ur:base').id].copy()
        handle_pos = self.data.site_xpos[self.grasp_site].copy()
        quat = _grasp_quat(arm_base_pos, handle_pos)

        approach_dir = (handle_pos[:2] - arm_base_pos[:2])
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        pre_grasp = Pose(handle_pos - np.array([approach_dir[0] * 0.08, approach_dir[1] * 0.08, 0.0]), quat)

        self.gripper_control(0)

        # Reorient in place first (see private/technical-log.md: a big
        # single-step reorientation done while already translating toward
        # an object can sweep the arm's own links through it).
        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        # Re-measure rather than trusting the pre-move snapshot.
        handle_pos = self.data.site_xpos[self.grasp_site].copy()
        grasp_pose = Pose(handle_pos, quat)
        self.move_to(grasp_pose, num_steps=5)

        # The fixed offset between "how the gripper is actually holding the
        # peg right now" and "the peg's own site frame right now" -- computed
        # once, here, from the real achieved grasp, then reused at every
        # angle along the press so the gripper rotates rigidly with the
        # lever instead of the grasp slipping as it swings through its arc.
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[self.grasp_site])
        rel_quat = quatcompose(quatinv(site_quat), quat)

        self.gripper_control(255)

        cur_lever_qpos = np.array([self.data.qpos[self.lever_jnt_adr]])
        target_lever_qpos = np.array([LEVER_CRIMPED])
        qpos_path = qpos_interpolate([cur_lever_qpos, target_lever_qpos], [20])
        gripper_path = []
        for lever_qpos in qpos_path:
            site_pose = self.fk_lever.forward(lever_qpos)
            gripper_path.append(mul_pose(site_pose, Pose(np.zeros(3), rel_quat)))
        self.path_follow(gripper_path)

        self.wait(seconds=0.5)
        self.gripper_control(0)
        cur_pose = self.arm.get_site_pose(self.data)
        retreat_pose = Pose(cur_pose.pos - np.array([approach_dir[0] * 0.1, approach_dir[1] * 0.1, 0.0]), cur_pose.quat)
        self.move_to(retreat_pose, num_steps=10)
        self.wait(seconds=1.0)
        self.finish()


CrimpCoinCell.Expert = CrimpCoinCellExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = CrimpCoinCell.load()
    expert = CrimpCoinCell.Expert(spec)
    for i in trange(10):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
        print(i, expert.check())
