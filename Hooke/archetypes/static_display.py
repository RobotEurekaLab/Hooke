"""Generic "just place it in the scene and render" task, deliberately with
no interaction at all -- for rapidly widening the visual variety of lab
equipment on offer without paying the full engineering cost (real
IK-driven grasp geometry, collision-clearance tuning, per-task physics
verification) that `mani_reagent_bottle.py`/`mani_fume_hood.py` needed.

This is the explicit tradeoff the user asked for after those two: more
scenes, faster, accepting "looks right, sits correctly in a real physics
scene" rather than "a scripted expert can pick it up/operate it" as the
bar for this batch. It mirrors the same tradeoff already made for the
floor-mount robots (Unitree G1, Tiago Dual) added earlier -- visualization,
not a functional substitute -- just applied to instruments instead of
robots.

Every instrument built this way still compiles and renders through real
MuJoCo physics (a real MJCF scene, real collision geoms, a real camera
render) -- "not verified" here means "no scripted-expert interaction was
written or checked," not "not a real simulator."
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco

from task import Task, Expert, Manager, SCENE_ROOT


@dataclasses.dataclass(frozen=True)
class StaticDisplaySpec:
    name: str
    scene_file: str
    prompt: str
    camera: str = "table_cam_front"

    @property
    def scene_path(self) -> Path:
        return SCENE_ROOT / self.scene_file


def make_static_task(spec: StaticDisplaySpec) -> tuple[type, type]:
    """Builds (Task, Expert) classes that just load `spec`'s scene, apply
    its default keyframe, and finish immediately -- no motion, no
    per-instrument Python beyond this."""

    class StaticDisplayTask(Task):
        default_scene = spec.scene_path
        default_task = spec.name
        time_limit = 2.0
        early_stop = True

        def __init__(self, mjspec: mujoco.MjSpec):
            manager = Manager.from_spec(mjspec, [])
            super().__init__(manager)

        def reset(self, seed: int | None = None):
            super().reset(seed=seed)
            self.manager.reset(keyframe=0)
            self.task_info = {
                'prefix': spec.prompt,
                'state_indices': [],
                'action_indices': [],
                'camera_mapping': {'image': spec.camera},
                'seed': seed,
            }
            return self.task_info

        def check(self):
            # No interaction is attempted -- see module docstring for why
            # this isn't a real success criterion, matching the same
            # already-established "no check exists yet for this family"
            # pattern several other tasks have (e.g. the lever-lock family).
            return True

    class StaticDisplayExpert(StaticDisplayTask, Expert):
        def execute(self):
            self.finish()

    StaticDisplayTask.Expert = StaticDisplayExpert
    return StaticDisplayTask, StaticDisplayExpert
