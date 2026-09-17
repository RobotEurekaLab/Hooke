"""Contact-driven operation of an ideally counterbalanced fume-hood sash.

The shared MJCF retains sash inertia, guide friction and damping, while
compensating the weight of the pane and handle. Approach the hood's front
normal, grip its transverse handle, slide, release and retreat. Neither
backend receives a sash position command.

Only ``close_fume_hood`` is catalogued. Its versioned assessment requires
final position, handle contact and signed travel during contact. Historical
position-only passes also occurred with gravity and no robot action; they
are not manipulation evidence. Opening remains uncatalogued and unqualified.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner

SASH_OPEN = 0.18
SASH_CLOSED = 0.0


def _grasp_quat(approach: np.ndarray) -> np.ndarray:
    """Point pinch-local +Z into the hood, with fingers closing vertically."""
    z_axis = approach / np.linalg.norm(approach)
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
        handle_pos = self.data.site_xpos[self.grasp_site].copy()
        # The hood front is local -Y. Approach normal to that face, not
        # along the arm-base/handle line (which cuts through the side wall).
        housing = self.model.body('/fume_hood:housing').id
        hood_rotation = self.data.xmat[housing].reshape(3, 3)
        approach = hood_rotation[:, 1].copy()
        quat = _grasp_quat(approach)
        approach_dir = approach[:2]
        pre_grasp = Pose(handle_pos - np.array([approach_dir[0] * 0.08, approach_dir[1] * 0.08, 0.0]), quat)

        self.gripper_control(0)

        # A common ready pose keeps the reorientation within reach across
        # reset perturbations. Reorienting at the initial far-out pinch
        # position can select an elbow branch that intersects the cabinet.
        ready = (handle_pos - approach * 0.28 + hood_rotation[:, 0] * 0.22
                 + np.array([0., 0., 0.10]))
        self.reposition_directly(Pose(ready, quat))

        self.move_to(pre_grasp, num_steps=10)

        # Re-measure rather than reusing the pre-move snapshot, in case the
        # approach nudged the sash/handle.
        handle_pos = self.data.site_xpos[self.grasp_site].copy()
        grasp_pose = Pose(handle_pos, quat)
        self.move_to(grasp_pose, num_steps=5)
        self.gripper_control(255)

        # Do not execute a blind closing motion after a missed grasp.
        handle = self.model.geom('/fume_hood:handle_bar').id
        pads = set()
        for contact in self.data.contact:
            a, b = map(int, contact.geom)
            other = b if a == handle else a if b == handle else None
            if other is not None:
                pads.add(self.model.body(int(self.model.geom_bodyid[other])).name)
        if not {'/ur:2f85:left_pad', '/ur:2f85:right_pad'} <= pads:
            raise RuntimeError('Fume hood grasp failed: both finger pads must contact the handle')

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
