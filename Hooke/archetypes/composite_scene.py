"""Generic archetype for a scene holding several physically distinct
instruments, each operable via the `push_pull_box` recipe (top-down grasp,
drag a box-shaped part through a vertical slide joint's range), switched
by `task_override` -- generalizes `mani_cross_instrument_composite.py`'s
hand-written two-instrument class (`vial_filling_line` +
`hplc_injector_plunger`, verified 20/20 full-sequence success both orders,
see private/technical-log.md) to N instruments, the same way
`push_pull_box.py` itself generalized one hand-written recipe
(`push_filling_nozzle_down`) and `lever_lock_centrifuge.py` generalized the
lid-lever family.

This exists so that scaling the `composite` catalog category to more
instrument pairs/triples across more disciplines is "write a spec", not
"copy `mani_cross_instrument_composite.py` and hand-edit it" -- exactly
the duplication problem `push_pull_box.py`'s own docstring already
diagnosed for the single-instrument case.

`mani_cross_instrument_composite.py` itself is left as-is (not migrated to
this archetype) since it already works and isn't broken -- this module is
for every *additional* composite scene from here on.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import mujoco

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner


@dataclasses.dataclass(frozen=True)
class CompositeInstrumentTarget:
    task_name: str          # value assigned to self.task to select this target
    instrument_prefix: str  # e.g. "/vial_filling_line:"
    joint_name: str         # unprefixed joint name on the instrument
    start_qpos: float
    target_qpos: float
    grasp_site_name: str = "grasp_site"


@dataclasses.dataclass(frozen=True)
class CompositeSceneSpec:
    name: str
    scene_file: str
    targets: tuple[CompositeInstrumentTarget, ...]
    time_limit: float = 15.0
    approach_height: float = 0.12

    @property
    def scene_path(self) -> Path:
        return SCENE_ROOT / self.scene_file

    @property
    def default_task(self) -> str:
        return self.targets[0].task_name


def _grasp_quat() -> np.ndarray:
    """Top-down: see push_pull_box.py's identical function for why this is
    the only grasp style trusted for a new instrument without a full
    per-instrument axis investigation."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def make_composite_classes(spec: CompositeSceneSpec) -> tuple[type, type]:
    targets_by_name = {t.task_name: t for t in spec.targets}

    class CompositeSceneTask(Task):
        default_scene = spec.scene_path
        default_task = spec.default_task
        time_limit = spec.time_limit
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
            for name, t in targets_by_name.items():
                self.jnt_adr[name] = self.model.joint(f"{t.instrument_prefix}{t.joint_name}").qposadr.item()
                self.grasp_site[name] = self.model.site(f"{t.instrument_prefix}{t.grasp_site_name}").id

        def reset(self, seed: int | None = None):
            super().reset(seed=seed)
            self.manager.reset(keyframe=0)

            perturbation = self.arm.qpos_perturb()
            self.data.qpos[self.arm.jnt_span] += perturbation
            self.data.ctrl[self.arm.act_span] += perturbation

            # Every instrument's joint is reset to its own start_qpos every
            # episode regardless of which target `self.task` names first --
            # a composite sequence runs multiple execute()s against one
            # reset() (archetypes/composite_task.py), so this can't depend
            # on step order.
            for name, t in targets_by_name.items():
                self.data.qpos[self.jnt_adr[name]] = t.start_qpos
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
            t = targets_by_name[self.task]
            qpos = self.data.qpos[self.jnt_adr[self.task]]
            direction_ok = (
                qpos < t.target_qpos + 0.01 if t.target_qpos < t.start_qpos
                else qpos > t.target_qpos - 0.01
            )
            gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
            gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
            gripped = False
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                pair = {con.geom1, con.geom2}
                if gripper_geom1 in pair or gripper_geom2 in pair:
                    other = con.geom2 if gripper_geom1 in pair else con.geom1
                    body = self.model.geom_bodyid[other]
                    bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                    if bname.startswith(t.instrument_prefix):
                        gripped = True
                        break
            return direction_ok and gripped

    class CompositeSceneExpert(CompositeSceneTask, Expert, ExpertMotionMixin):
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
            t = targets_by_name[self.task]
            site = self.grasp_site[self.task]

            self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
            quat = _grasp_quat()

            grasp_pos = self.data.site_xpos[site].copy()
            pre_grasp = Pose(grasp_pos + np.array([0.0, 0.0, spec.approach_height]), quat)

            self.gripper_control(0)

            cur_pose = self.arm.get_site_pose(self.data)
            self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

            self.move_to(pre_grasp, num_steps=10)

            grasp_pos = self.data.site_xpos[site].copy()
            self.move_to(Pose(grasp_pos, quat), num_steps=5)
            self.gripper_control(255)

            cur_pose = self.arm.get_site_pose(self.data)
            delta = t.target_qpos - t.start_qpos
            drag_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, delta]), cur_pose.quat)
            self.move_to(drag_pose, num_steps=15)
            # Deliberately don't release/retreat -- check() needs the live
            # grip contact (private/technical-log.md's "check() after
            # releasing" lesson). The next step in a composite sequence
            # handles its own reorientation/approach regardless of
            # whatever gripper state this leaves behind.
            self.wait(seconds=0.5)
            self.finish()

    CompositeSceneTask.Expert = CompositeSceneExpert
    return CompositeSceneTask, CompositeSceneExpert
