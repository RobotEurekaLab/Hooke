"""Insert a centrifuge tube into the Eppendorf 5910's rotor (swing-bucket
design), the real prerequisite step before `centrifuge_5910_lid_close`.

Written from scratch, unlike `load_centrifuge_5430.py`'s `InsertCentrifuge5430`
(which this deliberately mirrors in shape wherever possible): the 5910's
rotor is a fundamentally different mechanism from the 5430's. The 5430 has
a fixed annular rotor with 30 numbered `slotNN` sites at known angular
positions (`instrument.Centrifuge_Eppendorf_5430._reload` enumerates them);
the 5910 has 4 independently-hinged swing buckets (`rotor-bucket-0..3`),
each with its own tube adapter mounted on a small internal slide joint, and
`instrument.Centrifuge_Eppendorf_5910` doesn't expose any slot-site
convention at all -- there's nothing to enumerate. Confirmed via the
compiled model (not assumed from the MJCF alone): the buckets have no
damping/stiffness of their own and settle under gravity within a few
hundred physics steps to within ~1 degree of the *top* of their hinge
range (`qpos` ~ -0.009 out of a `[-1.5708, 0]` range) -- i.e. tucked in
against the rotor body, not hanging straight down the way a naive reading
of "swing bucket" might suggest.

This uses bucket 0's `adapter-7x50-0` (sized for a 50 mL tube) as the
single insertion target. `model/object/centrifuge_50ml.xml` is the tube
asset -- a plain single rigid body (no separate screw-cap mechanism, one
free joint), simpler to grasp than `centrifuge_50ml_screw.xml`'s
independently-jointed cap/body split used elsewhere in this codebase, and
sized to match the 7x50 adapter. Its mesh reference had a stale name (`
centrifuge_50ml-visual:` vs. the declared `obj:centrifuge_50ml-visual`) --
fixed as part of this work; the file was otherwise never compiled/used by
any existing scene, so this bug had never surfaced before, and the fix is
real and kept regardless of this task's own status.

**Not a working task, and not catalogued.** Two real bugs were found and
fixed along the way (a wrong rack-mounting height that put the tube
partway through the table, and the mesh typo above), and the grasp
orientation was switched from a horizontal side-grasp (like
`mani_reagent_bottle.py`'s, the first thing tried) to top-down after the
side-grasp hit the same near-unreachable-IK wall documented for
`open_fume_hood`/the abandoned `mani_coin_cell_crimper.py` -- switching
fixed that specific problem outright (see `_grasp_quat`'s docstring). But
two further problems remain, neither resolved:
1. The tube slips substantially in the gripper during the lift (rises
   ~0.03m of a commanded 0.12m) -- the same round-cylinder-vs-flat-pad
   grip-slip failure that stalled `mani_round_bottom_flask_stand.py`
   earlier this session, never solved there either. This is a *free*
   body here (not dragged through a constrained joint the way the
   flask's neck was), so the failure mode isn't identical, but the root
   cause (a smooth round surface between flat parallel pads) looks the
   same.
2. The subsequent large lateral move (rack -> above the rotor, ~0.45m
   horizontally) doesn't reach its target at all (ends up ~0.36m short),
   a distinct, not-yet-diagnosed problem on top of the grip-slip one.

Left in the repo, uncatalogued, with this docstring describing exactly
what's been tried and what's still broken -- same precedent as
`mani_coin_cell_crimper.py`/`mani_round_bottom_flask_stand.py`: the
diagnosis here is real work, re-deriving it from scratch later would
waste it twice. Whoever picks this up next should treat the grip-slip
problem as a standing, cross-task open problem (now hit twice, in two
different task shapes) rather than something specific to this instrument
-- see private/TODO.md.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner
from instrument import Centrifuge_Eppendorf_5910

LID_OPEN = 1.7  # near the joint's own max (1.94); matches the empirically
                 # observed spring-settled "held open" position.


class Centrifuge5910WithBucket(Centrifuge_Eppendorf_5910):
    """Adds bucket/adapter position access on top of the base lid-only
    interface -- `Centrifuge_Eppendorf_5910._reload` doesn't expose any
    slot/bucket convention (see module docstring)."""

    def _reload(self, model: mujoco.MjModel):
        super()._reload(model)
        self.bucket0_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'rotor-bucket-0')
        self.adapter0_body = self.name2id(mujoco.mjtObj.mjOBJ_BODY, 'adapter-7x50-0')

    def get_adapter0_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.adapter0_body].copy(), data.xquat[self.adapter0_body].copy())


def _grasp_quat() -> np.ndarray:
    """Top-down, not the horizontal side-grasp `mani_reagent_bottle.py`
    uses for the same *shape* of object (an upright free cylinder) --
    switched after measuring, not guessing: a horizontal approach at this
    tube's height (~0.83, close to the arm's own base height 0.824) hit
    the same near-unreachable IK wall documented for `open_fume_hood`/
    the abandoned `mani_coin_cell_crimper.py` (residual 0.00228, just over
    kinematics.py's 0.002 tolerance, and untouched by moving the grasp
    height up or down since the problem is the orientation, not the
    position -- see private/technical-log.md). The identical top-down
    quat `push_pull_box.py`/`mani_round_bottom_flask_stand.py` use solved
    cleanly at the same position on the first try. This tube is a free
    body (not constrained by a joint the way the flask's neck was), so
    the grip-slip failure that stalled that task doesn't apply here --
    lifting it off the rack is mechanically closer to
    `mani_reagent_bottle.py`'s already-proven free-body pickup, just
    approached from above instead of the side."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class InsertCentrifuge5910(Task):
    default_scene = SCENE_ROOT / "insert_centrifuge_5910.xml"
    default_task = "insert_centrifuge_5910"
    time_limit = 15.0
    early_stop = True

    @classmethod
    def prepare(cls, mjspec: mujoco.MjSpec) -> mujoco.MjSpec:
        set_gravcomp(mjspec.body('/ur:world'))
        return mjspec

    def __init__(self, mjspec: mujoco.MjSpec):
        self.instrument = Centrifuge5910WithBucket('/centrifuge_eppendorf_5910:')
        manager = Manager.from_spec(mjspec, [self.instrument])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.tube_jnt_adr = self.model.joint('centrifuge_50ml_joint').qposadr.item()
        rack_site = self.model.site('rack/centrifuge_10slot-50ml')
        self.rack_site_id = rack_site.id
        self.rack_row_dir = None  # resolved in reset() from the compiled site frame
        self.rack_col_dir = None

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb(
            lows=(-1.2, -0.2, -0.1, -0.5, -0.2, -0.2),
            highs=(0.0, 0.0, 0.1, 0.2, 0.2, 0.2),
        )
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        # Lid must be open to insert a tube -- the real physical
        # precondition, not an arbitrary choice (confirmed: the adapters
        # sit inside the housing, under where the closed lid covers).
        self.data.qpos[self.instrument.lid_qposadr] = LID_OPEN
        mujoco.mj_kinematics(self.model, self.data)

        rack_pos = self.data.site_xpos[self.rack_site_id].copy()
        tube_quat = self.data.qpos[self.tube_jnt_adr + 3:self.tube_jnt_adr + 7].copy()
        self.data.qpos[self.tube_jnt_adr:self.tube_jnt_adr + 3] = rack_pos
        self.data.qpos[self.tube_jnt_adr + 3:self.tube_jnt_adr + 7] = tube_quat
        mujoco.mj_kinematics(self.model, self.data)

        self.task_info = {
            'prefix': 'insert a centrifuge tube into the 5910 rotor',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        tube_pos = self.data.qpos[self.tube_jnt_adr:self.tube_jnt_adr + 3]
        adapter_pos = self.instrument.get_adapter0_pose(self.data).pos
        distance = np.linalg.norm(tube_pos - adapter_pos)
        # The tube should be standing near-upright (its own +Z close to
        # world +Z), not toppled over, and close to the adapter position.
        tube_quat = self.data.qpos[self.tube_jnt_adr + 3:self.tube_jnt_adr + 7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, tube_quat)
        upright = mat.reshape(3, 3)[2, 2] > 0.8
        return distance < 0.03 and upright


class InsertCentrifuge5910Expert(InsertCentrifuge5910, Expert, ExpertMotionMixin):
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

        tube_pos = self.data.qpos[self.tube_jnt_adr:self.tube_jnt_adr + 3].copy()
        grip_height = tube_pos[2] + 0.06  # grip partway up the tube's body
        pre_grasp = Pose(np.array([tube_pos[0], tube_pos[1], tube_pos[2] + 0.12]), quat)

        self.gripper_control(0)

        # Reorient in place first (private/technical-log.md): a big
        # single-step reorientation done while already translating toward
        # an object can sweep the arm through it. Top-down means this is
        # safe regardless of the arm's own reset height (see _grasp_quat's
        # docstring) -- unlike the horizontal side-grasp shape, there's no
        # awkward wrist configuration to fall into here.
        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        tube_pos = self.data.qpos[self.tube_jnt_adr:self.tube_jnt_adr + 3].copy()
        grasp_pose = Pose(np.array([tube_pos[0], tube_pos[1], tube_pos[2] + 0.06]), quat)
        self.move_to(grasp_pose, num_steps=5)
        self.gripper_control(240)

        cur_pose = self.arm.get_site_pose(self.data)
        self.move_to(Pose(cur_pose.pos + np.array([0.0, 0.0, 0.12]), cur_pose.quat), num_steps=15)

        adapter_pose = self.instrument.get_adapter0_pose(self.data)
        cur_pose = self.arm.get_site_pose(self.data)
        pre_place = Pose(np.array([adapter_pose.pos[0], adapter_pose.pos[1], cur_pose.pos[2]]), cur_pose.quat)
        self.move_to(pre_place, num_steps=15)

        place_pose = Pose(adapter_pose.pos + np.array([0.0, 0.0, 0.02]), cur_pose.quat)
        self.move_to(place_pose, num_steps=15)
        self.gripper_control(0)

        cur_pose = self.arm.get_site_pose(self.data)
        self.move_to(Pose(cur_pose.pos + np.array([0.0, 0.0, 0.1]), cur_pose.quat), num_steps=10)
        self.wait(seconds=1.0)
        self.finish()


InsertCentrifuge5910.Expert = InsertCentrifuge5910Expert

if __name__ == "__main__":
    from tqdm import trange
    spec = InsertCentrifuge5910.load()
    expert = InsertCentrifuge5910.Expert(spec)
    for i in trange(10):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
        print(i, expert.check())
