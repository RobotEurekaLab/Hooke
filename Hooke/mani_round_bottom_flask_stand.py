"""Lift a round-bottom flask straight up off its ring stand with a single UR5e arm.

Promotes `round_bottom_flask_stand_display` (display-only, added in the
display-scene breadth batches -- see private/technical-log.md) to a real
interactive task: the display version already has a real vertical `slide`
joint on the flask (a `free` joint can't sit there -- see the instrument
XML's own comment -- since the flask isn't a direct child of `<worldbody>`),
this file adds the grasp/actuation recipe and a physical success check.

Deliberately a *top-down* grasp (approach straight down onto the flask's
neck, open axis horizontal) rather than the horizontal side-grasp used by
`mani_reagent_bottle.py`/`mani_fume_hood.py`/the abandoned
`mani_coin_cell_crimper.py` attempt. Those three all aim the gripper
mostly sideways-and-low at a target near the arm's own reset height, which
turned out (see private/technical-log.md's `open_fume_hood` and
coin-cell-crimper diagnoses) to be a kinematically fragile region for this
UR5e's mount -- some target orientations in that region force a wrist/elbow
configuration whose forearm sweeps through the table on the way there.
The top-down approach here does sidestep that specific problem entirely:
across 10 seeds the arm's IK never fails and the end-effector reaches
every intermediate target to within a few mm, every time -- confirmed by
directly comparing the commanded vs. achieved pose at each step, not
assumed from the grasp looking simpler.

**Not yet a working task, though, and not catalogued.** The lift itself
fails on every seed: the gripper does form real, sustained contact with
the flask's neck (confirmed via per-step contact logging through the
whole grip-and-lift sequence), and widening the neck from the display
version's original 0.007m radius to 0.016m turned a clean miss (pads
closing past each other, contact-list showed pad-vs-pad, never
pad-vs-neck) into a real, held two-sided contact -- but the flask's own
slide-joint qpos never advances past ~0 regardless: the arm's
end-effector demonstrably rises the full 0.08m target, while the flask
stays behind, exactly as if the pads were sliding vertically along the
neck's surface under the clamp rather than carrying it. Tried and ruled
out (measured, not guessed): lowering the joint's damping/frictionloss,
raising the neck geom's tangential friction coefficient and contact
stiffness (solref/solimp) severalfold, and using several different
gripper-closing strengths (100 through 255) instead of full close -- none
changed the outcome. The likely culprit is the contact geometry itself: a
flat gripper pad against a smooth, radially-symmetric cylinder gives a
much smaller/less stable contact patch than the flat-sided bars/handles
`mani_fume_hood.py` and the abandoned crimper attempt grip (and the same
"drag it via a rigid grip" trick works fine for those), but this hasn't
been confirmed by fixing it, only by ruling out every simpler alternative
explanation. Left as an open follow-up -- see private/TODO.md."""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner

FLASK_SEATED = 0.0
FLASK_LIFTED = 0.08


def _grasp_quat() -> np.ndarray:
    """Top-down grasp: approach axis (site-local +Z) points straight down
    (world -Z), open axis (site-local Y) horizontal along world Y -- the
    flask's neck is a plain vertical cylinder, radially symmetric, so any
    horizontal open-axis grips it equally well; world Y is simply a fixed,
    arbitrary choice rather than one derived from arm/target geometry (no
    such geometry-dependent choice is needed here, unlike the horizontal
    side-grasp tasks)."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class LiftRoundBottomFlask(Task):
    default_scene = SCENE_ROOT / "mani_round_bottom_flask_stand.xml"
    default_task = "lift_round_bottom_flask"
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
        self.flask_jnt_adr = self.model.joint('/round_bottom_flask_stand:flask_joint').qposadr.item()
        self.grasp_site = self.model.site('/round_bottom_flask_stand:grasp_site').id

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        self.data.qpos[self.flask_jnt_adr] = FLASK_SEATED
        mujoco.mj_kinematics(self.model, self.data)

        self.task_info = {
            'prefix': 'lift the round-bottom flask off its stand',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        lifted = self.data.qpos[self.flask_jnt_adr] > FLASK_LIFTED - 0.01
        gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
        gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
        neck_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/round_bottom_flask_stand:flask_neck')
        gripped = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if (gripper_geom1 in pair or gripper_geom2 in pair) and neck_geom in pair:
                gripped = True
                break
        return lifted and gripped


class LiftRoundBottomFlaskExpert(LiftRoundBottomFlask, Expert, ExpertMotionMixin):
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

        # Reorient in place before translating (private/technical-log.md):
        # safe here regardless since a top-down grasp never asks the wrist
        # to point sideways-and-low in the first place.
        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        # Re-measure rather than trusting the pre-move snapshot.
        grasp_pos = self.data.site_xpos[self.grasp_site].copy()
        self.move_to(Pose(grasp_pos, quat), num_steps=5)
        self.gripper_control(255)

        # Drag the flask up through its slide joint's own travel by moving
        # the (now rigidly gripped) neck straight up by the joint's full
        # range -- same trick as mani_fume_hood.py's sash.
        cur_pose = self.arm.get_site_pose(self.data)
        lift_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, FLASK_LIFTED - FLASK_SEATED]), cur_pose.quat)
        self.move_to(lift_pose, num_steps=15)
        self.wait(seconds=1.0)
        self.finish()


LiftRoundBottomFlask.Expert = LiftRoundBottomFlaskExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = LiftRoundBottomFlask.load()
    expert = LiftRoundBottomFlask.Expert(spec)
    for i in trange(10):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
        print(i, expert.check())
