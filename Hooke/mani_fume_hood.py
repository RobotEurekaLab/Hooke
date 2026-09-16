"""Open/close a benchtop fume hood's sliding sash with a single UR5e arm.

Hooke's second chemistry-lab task (after `mani_reagent_bottle.py`) and its
first with a prismatic (slide) interactive joint -- every other task in
this codebase moves a hinge, a screw, or nothing at all. The housing/sash
geometry (`model/instrument/fume_hood.xml`) is plain MJCF primitives, not
a generated or sourced mesh: a rectangular enclosure gets nothing from
the GPT-6 Astra pipeline that hand-authored primitives don't already give
for free (exact collision, no axis-convention surprises to work around).

Shares two lessons learned the hard way while building the reagent-bottle
task (see private/technical-log.md): reorient the gripper in place before
ever moving toward the object (a big single-step reorientation swept
through it there), and re-measure the grasp target after any move that
could have disturbed it rather than trusting a pre-move snapshot.

Only `close_fume_hood` is catalogued. The historical 10/10 result used
final sash position alone: gravity also closes the sash without robot
action. It does not validate manipulation. The versioned episode assessor
in backends.assessment additionally requires handle contact and travel
during contact; see docs/isaac_episode_assessment.md at the repo root. The
class also supports `open_fume_hood` (start closed, slide the sash back
up) but that direction currently fails outright. Originally suspected to
be the sash's glass clipping the gripper on approach; re-diagnosed with
`mj_contactForce` + geom/body lookup (not just a contact-list check) and
that guess was wrong -- the real collision, present from the very first
reorientation step regardless of translation order, is the arm's own
`/ur:forearm_link` slamming into `/vention table` (its own mount), not
anything in the fume hood scene at all. The closed-sash handle sits
lower (z~0.92) than `close_fume_hood`'s open-sash handle (z~1.10), and
`_grasp_quat`'s orientation for that lower, more side-on target apparently
forces an elbow/wrist configuration that self-intersects the mount at
this arm's reset pose -- confirmed reproducible across 10 seeds with
forces from ~1.7kN up to ~110kN, regardless of whether the reorientation
happens before, after, or interleaved with the translation, and
regardless of a high-altitude "go up and over" detour (which additionally
hits UR5e's reach limit at this workspace's arm-base offset). This needs
a genuinely different fix -- likely reworking `_grasp_quat` to avoid the
problematic wrist configuration for low targets, or giving `open_fume_hood`
its own start-of-episode arm pose instead of sharing `close_fume_hood`'s --
not just a smarter waypoint path. Left as a follow-up (see
private/TODO.md) rather than catalogued half-working.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner

SASH_OPEN = 0.18
SASH_CLOSED = 0.0


def _grasp_quat(arm_base_pos: np.ndarray, handle_pos: np.ndarray) -> np.ndarray:
    """Horizontal grasp orientation for a horizontal bar whose own long
    axis runs world-X: the approach axis (site-local +Z) points from the
    arm base toward the handle (horizontal), and the open axis
    (site-local Y) is *vertical* this time -- unlike the reagent bottle's
    vertical-cylinder grasp (open axis horizontal, perpendicular to a
    vertical approach target), a horizontal bar needs the fingers to
    close top-to-bottom, perpendicular to both the approach direction and
    the bar's own length."""
    approach = handle_pos[:2] - arm_base_pos[:2]
    approach = approach / np.linalg.norm(approach)
    z_axis = np.array([approach[0], approach[1], 0.0])
    y_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class OperateFumeHood(Task):
    default_scene = SCENE_ROOT / "mani_fume_hood.xml"
    default_task = "close_fume_hood"
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
        self.sash_jnt_adr = self.model.joint('/fume_hood:sash_slide').qposadr.item()
        self.grasp_site = self.model.site('/fume_hood:grasp_site').id

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        start_sash = SASH_OPEN if self.task == 'close_fume_hood' else SASH_CLOSED
        self.data.qpos[self.sash_jnt_adr] = start_sash
        mujoco.mj_kinematics(self.model, self.data)

        prefix = 'close' if self.task == 'close_fume_hood' else 'open'
        self.task_info = {
            'prefix': f'{prefix} the fume hood sash',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        target = SASH_CLOSED if self.task == 'close_fume_hood' else SASH_OPEN
        return abs(self.data.qpos[self.sash_jnt_adr] - target) < 0.02


class OperateFumeHoodExpert(OperateFumeHood, Expert, ExpertMotionMixin):
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

        # Reorient in place first (see module docstring / mani_reagent_bottle.py).
        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        # Re-measure rather than reusing the pre-move snapshot, in case the
        # approach nudged the sash/handle.
        handle_pos = self.data.site_xpos[self.grasp_site].copy()
        grasp_pose = Pose(handle_pos, quat)
        self.move_to(grasp_pose, num_steps=5)
        self.gripper_control(255)

        # Slide the sash by moving the (now rigidly gripped) handle
        # straight up or down by the joint's own full travel -- the sash
        # is dragged along through the grip, the same way the lever-lock
        # tasks drag a lid through a rigid grasp on its lever.
        target_sash = SASH_CLOSED if self.task == 'close_fume_hood' else SASH_OPEN
        current_sash = self.data.qpos[self.sash_jnt_adr]
        cur_pose = self.arm.get_site_pose(self.data)
        slide_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, target_sash - current_sash]), cur_pose.quat)
        self.move_to(slide_pose, num_steps=15)

        self.gripper_control(0)
        cur_pose = self.arm.get_site_pose(self.data)
        retreat_pose = Pose(cur_pose.pos - np.array([approach_dir[0] * 0.1, approach_dir[1] * 0.1, 0.0]), cur_pose.quat)
        self.move_to(retreat_pose, num_steps=10)
        self.wait(seconds=1.0)
        self.finish()


OperateFumeHood.Expert = OperateFumeHoodExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = OperateFumeHood.load()
    expert = OperateFumeHood.Expert(spec)
    for i in trange(20):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
