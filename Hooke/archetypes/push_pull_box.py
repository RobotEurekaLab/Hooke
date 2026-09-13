"""Generic archetype for "grip a box-shaped part from above and drag it
through a vertical slide joint's range" -- the one non-lever interactive
recipe validated this session (`mani_vial_filling_line.py`'s
`push_filling_nozzle_down`, 30/30 seeds; see private/technical-log.md),
generalized the same way `lever_lock_centrifuge.py` generalized the
lid-lever family: the per-instrument specifics become a `PushPullBoxSpec`
instead of a hand-duplicated ~150-line file.

Why *this* recipe and not the horizontal side-grasp one
(`mani_fume_hood.py`/`mani_reagent_bottle.py`'s `_grasp_quat` shape):
that style produced two dead ends this session (`open_fume_hood`, the
abandoned `mani_coin_cell_crimper.py`) via a self-collision-with-the-
mount-table failure mode tied to certain target heights/orientations that
isn't practical to predict in advance for an arbitrary new instrument.
The top-down approach proved reliably collision-free instead (confirmed
across `round_bottom_flask_stand`'s attempt and this recipe's own 30/30),
and restricting the gripped part to a box (not a smooth cylinder) avoids
the separate grip-slip failure that stalled `round_bottom_flask_stand`.
Any protocol step that doesn't fit this shape (round part, non-vertical
motion, a primitive this recipe can't express) falls back to
`static_display.py` instead of forcing a bad fit -- see
`protocol_to_task.py`.

Only a vertical (`axis="0 0 1"`) slide joint is supported, in either
direction (push down or pull up) -- generalizing further (horizontal
drag, hinge rotation) is future work, not something this batch needed.
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
class PushPullBoxSpec:
    name: str                    # e.g. "push_filling_nozzle_down"
    scene_file: str              # scene XML filename under model/scene/
    instrument_prefix: str       # e.g. "/vial_filling_line:"
    joint_name: str              # e.g. "nozzle_joint" (unprefixed)
    grasp_site_name: str = "grasp_site"  # unprefixed site name on the moving body
    start_qpos: float = 0.0
    target_qpos: float = -0.05
    prompt_prefix: str = "push the part down"
    success_tol: float = 0.01
    approach_height: float = 0.12  # pre-grasp offset straight up from the site, meters
    time_limit: float = 15.0

    @property
    def scene_path(self) -> Path:
        return SCENE_ROOT / self.scene_file


def _grasp_quat() -> np.ndarray:
    """Top-down: approach axis (site-local +Z) straight down, open axis
    (site-local Y) horizontal along world Y -- see module docstring."""
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def make_task_classes(spec: PushPullBoxSpec) -> tuple[type, type]:
    """Builds (Task, Expert) classes matching the hand-written structure of
    `mani_vial_filling_line.py`'s `PushFillingNozzle`/`PushFillingNozzleExpert`."""

    prefixed_joint = f"{spec.instrument_prefix}{spec.joint_name}"
    prefixed_site = f"{spec.instrument_prefix}{spec.grasp_site_name}"

    class PushPullBoxTask(Task):
        default_scene = spec.scene_path
        default_task = spec.name
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
            self.part_jnt_adr = self.model.joint(prefixed_joint).qposadr.item()
            self.grasp_site = self.model.site(prefixed_site).id
            self._part_geom = None  # resolved lazily in reset(), needs a compiled model

        def reset(self, seed: int | None = None):
            super().reset(seed=seed)
            self.manager.reset(keyframe=0)

            perturbation = self.arm.qpos_perturb()
            self.data.qpos[self.arm.jnt_span] += perturbation
            self.data.ctrl[self.arm.act_span] += perturbation

            self.data.qpos[self.part_jnt_adr] = spec.start_qpos
            mujoco.mj_kinematics(self.model, self.data)

            self.task_info = {
                'prefix': spec.prompt_prefix,
                'state_indices': self.arm.state_indices,
                'action_indices': self.arm.action_indices,
                'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
                'seed': seed,
            }
            return self.task_info

        def check(self):
            direction_ok = (
                self.data.qpos[self.part_jnt_adr] < spec.target_qpos + spec.success_tol
                if spec.target_qpos < spec.start_qpos else
                self.data.qpos[self.part_jnt_adr] > spec.target_qpos - spec.success_tol
            )
            gripper_geom1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:left_pad1')
            gripper_geom2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, '/ur:2f85:right_pad1')
            gripped = False
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                pair = {con.geom1, con.geom2}
                if (gripper_geom1 in pair or gripper_geom2 in pair):
                    # Any gripper-pad contact with a body belonging to the
                    # instrument's moving part counts -- the archetype
                    # doesn't require a specifically-named collision geom
                    # the way the hand-written version did, since a
                    # generated instrument's part may have more than one
                    # geom under the moving body.
                    other = con.geom2 if gripper_geom1 in pair else con.geom1
                    body = self.model.geom_bodyid[other]
                    bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                    if bname.startswith(spec.instrument_prefix):
                        gripped = True
                        break
            return direction_ok and gripped

    class PushPullBoxExpert(PushPullBoxTask, Expert, ExpertMotionMixin):
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
            pre_grasp = Pose(grasp_pos + np.array([0.0, 0.0, spec.approach_height]), quat)

            self.gripper_control(0)

            cur_pose = self.arm.get_site_pose(self.data)
            self.reposition_directly(Pose(cur_pose.pos.copy(), quat))

            self.move_to(pre_grasp, num_steps=10)

            grasp_pos = self.data.site_xpos[self.grasp_site].copy()
            self.move_to(Pose(grasp_pos, quat), num_steps=5)
            self.gripper_control(255)

            cur_pose = self.arm.get_site_pose(self.data)
            delta = spec.target_qpos - spec.start_qpos
            drag_pose = Pose(cur_pose.pos + np.array([0.0, 0.0, delta]), cur_pose.quat)
            self.move_to(drag_pose, num_steps=15)
            # Deliberately don't release/retreat before finishing -- check()
            # verifies the gripper is still in contact (see
            # private/technical-log.md's "check() after releasing" lesson).
            self.wait(seconds=1.0)
            self.finish()

    PushPullBoxTask.Expert = PushPullBoxExpert
    return PushPullBoxTask, PushPullBoxExpert
