"""First cross-instrument composite task: two physically distinct
instruments (`vial_filling_line`, `hplc_injector_plunger`) in one shared
scene, operated by task_override the same way `mani_thermal_cycler.py`
switches between `thermal_cycler_close`/`thermal_cycler_open` on one
mechanism -- except here each task_override targets a *different*
instrument, not just a different direction on the same one.

Why this exists: `archetypes/composite_task.py`'s own docstring flags
"cross-instrument composition... needs a combined scene with both
instruments physically present and is not attempted here" as the
concrete next step past same-scene task_override chaining (its only
supported case before this file: thermal_cycler_close <-> open). This is
that combined scene, built from two already-proven `push_pull_box`-shaped
recipes (`push_filling_nozzle_down`, 30/30 seeds; the generated
`hplc_injector_plunger`, 20/20 seeds -- see private/technical-log.md)
rather than new unverified motion code, so any transition failure this
uncovers is attributable to the *chaining*, not to a fragile recipe.

This file intentionally does not go through `push_pull_box.py`'s
`make_task_classes` (which binds one Task/Expert pair to exactly one
instrument) -- a composite episode needs *both* instruments' joints
tracked in the same `reset()` and a single `execute()` that only touches
whichever one `self.task` currently selects, so the recipe logic is
duplicated here per-target (two dicts, not two files) rather than forcing
a premature generalization of `push_pull_box.py` for a single use case.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner

# (instrument_prefix, joint_name, start_qpos, target_qpos) per task_override.
TARGETS = {
    "push_vial_nozzle": ("/vial_filling_line:", "nozzle_joint", 0.0, -0.05),
    "push_hplc_plunger": ("/hplc_injector_plunger:", "part_joint", 0.0, -0.04),
}


def _grasp_quat() -> np.ndarray:
    """Top-down grasp, same shape as push_pull_box.py's -- see that
    module's docstring for why this is the only approach style trusted
    for a new instrument without a full per-target axis investigation."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


class CrossInstrumentComposite(Task):
    default_scene = SCENE_ROOT / "mani_cross_instrument_composite.xml"
    default_task = "push_vial_nozzle"
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
        self.jnt_adr = {}
        self.grasp_site = {}
        for name, (prefix, joint_name, _, _) in TARGETS.items():
            self.jnt_adr[name] = self.model.joint(f"{prefix}{joint_name}").qposadr.item()
            self.grasp_site[name] = self.model.site(f"{prefix}grasp_site").id

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        # Both instruments' joints are reset to their own start_qpos every
        # episode, regardless of which one `self.task` targets first --
        # a composite sequence runs multiple `execute()` calls against one
        # `reset()` (see archetypes/composite_task.py), so this must not
        # depend on which step is "first".
        for name, (_, _, start_qpos, _) in TARGETS.items():
            self.data.qpos[self.jnt_adr[name]] = start_qpos
        mujoco.mj_kinematics(self.model, self.data)

        self.task_info = {
            'prefix': self.task.replace('_', ' '),
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        _, _, start_qpos, target_qpos = TARGETS[self.task]
        qpos = self.data.qpos[self.jnt_adr[self.task]]
        direction_ok = qpos < target_qpos + 0.01 if target_qpos < start_qpos else qpos > target_qpos - 0.01
        gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
        gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
        prefix = TARGETS[self.task][0]
        gripped = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if gripper_geom1 in pair or gripper_geom2 in pair:
                other = con.geom2 if gripper_geom1 in pair else con.geom1
                body = self.model.geom_bodyid[other]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                if bname.startswith(prefix):
                    gripped = True
                    break
        return direction_ok and gripped


class CrossInstrumentCompositeExpert(CrossInstrumentComposite, Expert, ExpertMotionMixin):
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
        _, _, start_qpos, target_qpos = TARGETS[self.task]
        site = self.grasp_site[self.task]

        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        quat = _grasp_quat()

        grasp_pos = self.data.site_xpos[site].copy()
        pre_grasp = Pose(grasp_pos + np.array([0.0, 0.0, 0.12]), quat)

        self.gripper_control(0)

        cur_pose = self.arm.get_site_pose(self.data)
        self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

        self.move_to(pre_grasp, num_steps=10)

        grasp_pos = self.data.site_xpos[site].copy()
        self.move_to(Pose(grasp_pos, quat), num_steps=5)
        self.gripper_control(255)

        cur_pose = self.arm.get_site_pose(self.data)
        delta = target_qpos - start_qpos
        drag_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, delta]), cur_pose.quat)
        self.move_to(drag_pose, num_steps=15)
        # Deliberately don't release/retreat -- check() needs the live
        # grip contact (private/technical-log.md's "check() after
        # releasing" lesson from push_filling_nozzle_down applies here
        # too). A composite sequence's *next* step handles its own
        # reorientation/approach regardless of the gripper's state left
        # behind by this one.
        self.wait(seconds=0.5)
        self.finish()


CrossInstrumentComposite.Expert = CrossInstrumentCompositeExpert

if __name__ == "__main__":
    for name in TARGETS:
        spec = CrossInstrumentComposite.load()
        expert = CrossInstrumentComposite.Expert(spec)
        expert.task = name
        expert.reset(0)
        expert.execute()
        print(name, expert.check())
