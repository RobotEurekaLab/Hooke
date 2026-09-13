"""Push a vial-filling line's dispensing nozzle down with a single UR5e arm.

Promotes `vial_filling_line_display` (display-only, added in the display-
scene breadth batches -- see private/technical-log.md) to a real
interactive task, picked specifically to avoid both bugs the previous two
promotion attempts hit (see private/technical-log.md's "Trying to promote
two display items" entry):

- A *top-down* grasp (approach straight down, open axis horizontal), like
  the still-unfinished `mani_round_bottom_flask_stand.py` -- confirmed
  there to sidestep the self-collision-with-the-mount-table bug that
  killed `open_fume_hood` and the abandoned `mani_coin_cell_crimper.py`.
- The nozzle itself is a plain 2x2x4cm **box**, not a smooth cylinder --
  picked because the round_bottom_flask's failure traced to a thin round
  neck slipping vertically inside flat gripper pads; a box gives the pads
  full flat-face contact instead of a tangent line, which should hold
  under a rigid drag the same way the (bar-shaped, not round) fume-hood
  sash and reagent-bottle body already do.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner

NOZZLE_UP = 0.0
NOZZLE_DOWN = -0.05


def _grasp_quat() -> np.ndarray:
    """Top-down grasp, identical shape to
    mani_round_bottom_flask_stand.py's: approach axis (site-local +Z)
    straight down (world -Z), open axis (site-local Y) horizontal along
    world Y. The nozzle's cross-section is square (2cm x 2cm, confirmed
    via the compiled model's world half-extent, not just the XML's own
    numbers), so no geometry-dependent axis choice is needed."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class PushFillingNozzle(Task):
    default_scene = SCENE_ROOT / "mani_vial_filling_line.xml"
    default_task = "push_filling_nozzle_down"
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
        self.nozzle_jnt_adr = self.model.joint('/vial_filling_line:nozzle_joint').qposadr.item()
        self.grasp_site = self.model.site('/vial_filling_line:grasp_site').id

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        self.data.qpos[self.nozzle_jnt_adr] = NOZZLE_UP
        mujoco.mj_kinematics(self.model, self.data)

        self.task_info = {
            'prefix': 'push the filling nozzle down',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        pushed = self.data.qpos[self.nozzle_jnt_adr] < NOZZLE_DOWN + 0.01
        gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
        gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
        nozzle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/vial_filling_line:nozzle_body')
        gripped = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if (gripper_geom1 in pair or gripper_geom2 in pair) and nozzle_geom in pair:
                gripped = True
                break
        return pushed and gripped


class PushFillingNozzleExpert(PushFillingNozzle, Expert, ExpertMotionMixin):
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
        quat = _grasp_quat()

        grasp_pos = self.data.site_xpos[self.grasp_site].copy()
        pre_grasp = Pose(grasp_pos + np.array([0.0, 0.0, 0.12]), quat)

        self.gripper_control(0)

        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        grasp_pos = self.data.site_xpos[self.grasp_site].copy()
        self.move_to(Pose(grasp_pos, quat), num_steps=5)
        self.gripper_control(255)

        # Drag the nozzle down through its slide joint's own travel, same
        # trick as mani_fume_hood.py's sash / mani_round_bottom_flask_stand.py.
        cur_pose = self.arm.get_site_pose(self.data)
        push_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, NOZZLE_DOWN - NOZZLE_UP]), cur_pose.quat)
        self.move_to(push_pose, num_steps=15)
        # Deliberately don't release/retreat before finishing: check()
        # verifies the gripper is still in contact with the nozzle, which
        # would trivially fail if we let go first (learned from this
        # task's own first attempt -- see private/technical-log.md).
        self.wait(seconds=1.0)
        self.finish()


PushFillingNozzle.Expert = PushFillingNozzleExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = PushFillingNozzle.load()
    expert = PushFillingNozzle.Expert(spec)
    for i in trange(10):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
        print(i, expert.check())
