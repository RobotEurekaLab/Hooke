"""Generic archetype for the "lever + lock" centrifuge-lid family.

Several AutoBio instruments (Eppendorf 5430, Eppendorf 5910, and — with a
still-incomplete recipe upstream — the Tiangen T-Gear mini) share the exact
same interaction *shape*: grip a lid lever, rotate it closed along a
computed lever path, then move the gripper through a small sequence of poses
to engage a physical lock, optionally forcing the lock's equality-constraint
active once the real contact would have engaged it.

What differs between instruments is not the *shape* of the interaction but a
handful of numbers (grip/lock approach offsets, an orientation to hold the
gripper at, which end of the lid joint's range corresponds to "closed", how
long the episode runs) and, in general, the exact ordered sequence of motion
steps. This module represents that shape once (`LeverLockInstrumentMixin`,
`LeverLockExpert`) and takes the per-instrument specifics as data
(`LeverLockSpec` + a `recipe` of step dicts), so a new instrument in this
family is a spec, not a new ~250-line file.

See `autobio/mani_centrifuge_5430.py` / `mani_centrifuge_5910.py` for the
original, hand-duplicated implementations this generalizes, and
`private/technical-log.md` for the equivalence check run when this module
was introduced.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import numpy as np
import mujoco

from kinematics import Pose, FK
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, qpos_interpolate, make_topp_planner
from archetypes.lid_lock import lid_lock_passes

# Shared across every instrument in this family: the wrist orientation held
# while moving through the lock sub-sequence (not instrument-specific -- the
# lock mechanism is always approached the same way relative to the lid site
# frame, only the *offset* from that frame differs per instrument).
LOCK_QUAT = np.array([0.0, 0.7071, 0.7071, 0.0])


@dataclasses.dataclass(frozen=True)
class LeverLockSpec:
    name: str                       # e.g. "centrifuge_5430"
    instrument_cls: type            # base instrument System subclass, from instrument.py
    instrument_prefix: str          # e.g. "/centrifuge_eppendorf_5430:"
    scene_file: str                 # scene XML filename under model/scene/
    task_name: str                  # matches the original `default_task` string
    prompt_prefix: str              # language instruction, e.g. "close the lid of ..."
    time_limit: float
    jntlimit_index: int             # 0 (open end) or 1 (closed end) of the lid joint's range
    jntlimit_offset: float          # closed_qpos = jntlimit[jntlimit_index] + jntlimit_offset
    rel_quat: np.ndarray            # fixed gripper orientation offset (relative to lid site), shape (4,)
    mode_offsets: dict[str, np.ndarray]  # {'1/detach','2/detach','grip','lock_pre','lock'} -> pos offset (3,)
    recipe: list[dict[str, Any]]    # ordered motion steps, see LeverLockExpert._run_recipe
    qpos_perturb_lows: tuple = (-0.1, 0.0, -0.2, -0.1, 0.0, -0.2)
    qpos_perturb_highs: tuple = (0.1, 0.3, 0.2, 0.1, 0.3, 0.2)
    # Which arm class drives this task -- defaults to the original UR5eArm;
    # override (e.g. to PandaArm/XArm7Arm from archetypes/menagerie_arms.py)
    # to run the exact same recipe against a different real (IK-driving)
    # arm, rather than swapping in a robot for visualization only. When the
    # arm's own DOF doesn't match qpos_perturb_lows/highs above (e.g. a
    # 7-DOF arm with this class's 6-element UR5e-shaped defaults), reset()
    # falls back to the arm class's own default perturbation range instead
    # of erroring on a shape mismatch.
    arm_cls: type = UR5eArm
    qc_vel: float = 1.5
    qc_acc: float = 1.0
    completion_check: Callable = lid_lock_passes

    @property
    def scene_path(self) -> Path:
        return SCENE_ROOT / self.scene_file


def _make_instrument_class(spec: LeverLockSpec) -> type:
    """Dynamically builds the per-instrument System subclass, mixing the shared
    lever/lock geometry logic into the instrument's own base class (which
    provides `_reload`'s named joint/site lookups, e.g. `lid_qposadr`)."""

    def _reset(self, data):
        spec.instrument_cls._reset(self, data)
        self.fk_lever = FK(1, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid')

    def fk(self, qpos: np.ndarray) -> Pose:
        return self.fk_lever.forward(qpos)

    def lever_path(self, data: mujoco.MjData, mode: str = '1/close') -> list[Pose]:
        cur_qpos = np.asarray(data.qpos[self.lid_qposadr]).reshape(1)
        if mode != '1/close':
            raise ValueError(f"Unknown lever path mode: {mode}")
        qpos1 = np.array([self.lid_jntlimit[spec.jntlimit_index] + spec.jntlimit_offset])
        qpos_list = qpos_interpolate([cur_qpos, qpos1], [15])
        path = [self.get_eefpose_lever(self.fk(qpos), 'grip') for qpos in qpos_list]
        path.append(self.get_eefpose_lever(self.fk(qpos1), '2/detach'))
        return path

    def get_eefpose_lever(self, sitepose: Pose, mode: str) -> Pose:
        if mode not in spec.mode_offsets:
            raise ValueError(f"Unknown approach mode: {mode}")
        rel_pos = spec.mode_offsets[mode]
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(res_pos, res_quat, sitepose.pos, sitepose.quat, rel_pos, spec.rel_quat)
        return Pose(res_pos, res_quat)

    def get_eef_pose(self, data: mujoco.MjData, loc: str, mode: str, random: bool = False) -> Pose:
        if loc != 'lid':
            raise ValueError(f"Unknown location: {loc}")
        site_pos = data.site_xpos[self.lid_site]
        site_mat = data.site_xmat[self.lid_site]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site_mat)
        return self.get_eefpose_lever(Pose(site_pos, quat), mode)

    return type(
        f"LeverLock_{spec.name}",
        (spec.instrument_cls,),
        dict(
            _reset=_reset, fk=fk, lever_path=lever_path,
            get_eefpose_lever=get_eefpose_lever, get_eef_pose=get_eef_pose,
        ),
    )


class LeverLockMotionMixin(ExpertMotionMixin):
    """Run one lever-lock recipe against the host's actual robot controls."""
    lever_spec: LeverLockSpec

    def _run_recipe(self):
        self._cached_poses = {}
        self._cached_paths = {}
        for step in self.lever_spec.recipe:
            op = step['op']
            getattr(self, f'_step_{op}')(**{k: v for k, v in step.items() if k != 'op'})

    def _step_cache_pose(self, name: str, mode: str):
        self._cached_poses[name] = self.instrument.get_eef_pose(self.data, loc='lid', mode=mode)

    def _step_home_arm(self):
        target = self.model.key_qpos[0,self.arm.jnt_span].copy()
        if self.arm.dof == len(self.lever_spec.qpos_perturb_lows):
            target += (np.asarray(self.lever_spec.qpos_perturb_lows)
                       + np.asarray(self.lever_spec.qpos_perturb_highs))/2
        self.move_joints(target,
                         velocity=self.lever_spec.qc_vel,
                         acceleration=self.lever_spec.qc_acc)
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span].copy()

    def _step_cache_lever_path(self, name: str, mode: str = '1/close'):
        self._cached_paths[name] = self.instrument.lever_path(self.data, mode=mode)

    def _step_move_to_pose(self, mode: str, num_steps: int, quat_override: str | None = None,
                            gripper_before: float | None = None, cached_pose: str | None = None):
        pose = (self._cached_poses[cached_pose] if cached_pose is not None else
                self.instrument.get_eef_pose(self.data, loc='lid', mode=mode))
        if quat_override == 'lock_quat':
            pose.quat = LOCK_QUAT
        if gripper_before is not None:
            self.gripper_control(gripper_before)
        self.move_to(pose, num_steps=num_steps)

    def _step_gripper(self, value: float, delay: int = 300):
        self.gripper_control(value, delay=delay)

    def _step_lever_close(self, mode: str = '1/close', cached_path: str | None = None):
        path = (self._cached_paths[cached_path] if cached_path is not None else
                self.instrument.lever_path(self.data, mode=mode))
        self.path_follow(path[:-1])
        self._lever_end_pose = path[-1]

    def _step_move_to_lever_end(self, num_steps: int):
        assert self._lever_end_pose is not None, "lever_close must run before move_to_lever_end"
        self.move_to(self._lever_end_pose, num_steps=num_steps)

    def _step_force_lock(self):
        self.data.eq_active[self.instrument.lid_lock] = 1

    def _step_wait(self, seconds: float):
        self.wait(seconds, {})


def make_task_classes(spec: LeverLockSpec) -> tuple[type, type]:
    """Builds (Task, Expert) classes for `spec`, matching the structure of the
    original hand-written `Centrifuge{5430,5910}Manipulate(Expert)` classes."""

    instrument_cls = _make_instrument_class(spec)

    class LeverLockTask(Task):
        default_scene = spec.scene_path
        default_task = spec.task_name
        time_limit = spec.time_limit
        early_stop = True

        @classmethod
        def prepare(cls, mjspec: mujoco.MjSpec) -> mujoco.MjSpec:
            set_gravcomp(mjspec.body('/ur:world'))
            return mjspec

        def __init__(self, mjspec: mujoco.MjSpec):
            self.instrument = instrument_cls(spec.instrument_prefix)
            manager = Manager.from_spec(mjspec, [self.instrument])
            super().__init__(manager)
            self.arm = spec.arm_cls(self.model, '/ur:')

        def reset(self, seed: int | None = None):
            super().reset(seed=seed)
            self.manager.reset(keyframe=0)
            if self.arm.dof == len(spec.qpos_perturb_lows):
                perturbation = self.arm.qpos_perturb(spec.qpos_perturb_lows, spec.qpos_perturb_highs)
            else:
                perturbation = self.arm.qpos_perturb()  # arm's own default range for its actual DOF count
            self.data.qpos[self.arm.jnt_span] += perturbation
            self.data.ctrl[self.arm.act_span] += perturbation
            self.task_info = {
                'prefix': spec.prompt_prefix,
                'state_indices': self.arm.state_indices,
                'action_indices': self.arm.action_indices,
                'camera_mapping': {'image': 'table_cam_left', 'wrist_image': '/ur:wrist_cam'},
                'seed': seed,
            }
            return self.task_info

        def check(self):
            return spec.completion_check(self.data, self.instrument)

    class LeverLockExpert(LeverLockTask, Expert, LeverLockMotionMixin):
        lever_spec = spec

        def __init__(self, mjspec: mujoco.MjSpec, freq: int = 20):
            super().__init__(mjspec)
            self.freq = freq
            self.period = int(round(1.0 / self.dt / freq))
            self.arm.register_ik(self.data)
            self.planner = make_topp_planner(self.arm.dof, self.arm.ik.solve,
                                           qc_vel=spec.qc_vel,qc_acc=spec.qc_acc)
            self._lever_end_pose: Pose | None = None


        def execute(self):
            self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
            if self.task == spec.task_name:
                self._run_recipe()
            self.finish()

    LeverLockTask.Expert = LeverLockExpert
    return LeverLockTask, LeverLockExpert
