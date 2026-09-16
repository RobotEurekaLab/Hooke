"""Renders a single preview frame (the reset/initial state, not a rollout)
for a Hooke task, for the Step 5 interactive UI's scene picker.

Uses MuJoCo's native OpenGL/EGL renderer (matching render.py's "basic
render" path) rather than the Blender-Cycles path, since this needs to be
fast enough for an interactive UI, not photorealistic -- see AutoBio's own
paper for why the Blender path is ~100x slower.
"""
import io

import mujoco
import numpy as np
from PIL import Image

from archetypes.rotor_variants import generate_rotor_variant
from load_centrifuge_5430 import InsertCentrifuge5430
from backends.source_renderer import mujoco_renderer


def _build_task(entry, variant: int | None):
    if entry.name == "insert_centrifuge_5430" and variant is not None and variant != 30:
        scene_path = generate_rotor_variant(variant)
        task_cls, expert_cls = InsertCentrifuge5430.for_variant(
            scene_path, f"insert_centrifuge_5430_{variant}slot"
        )
        return task_cls
    task_cls, _ = entry.load_classes()
    return task_cls


def render_scene(entry, seed: int = 0, variant: int | None = None, width: int = 480, height: int = 360) -> dict:
    """Returns {"image_png_bytes": bytes, "task_info": dict, "robot": str}."""
    task_cls = _build_task(entry, variant)
    task = task_cls(task_cls.load())
    # task_override must be set *before* reset(): reset() itself branches on
    # self.task to decide both the prompt text and the initial state (e.g.
    # thermal_cycler_open's reset() starts the lid closed, since opening it
    # is the point -- applying the override after reset() would render the
    # "close" task's start state under the "open" task's label).
    if entry.task_override:
        task.task = entry.task_override
    task_info = task.reset(seed=seed)
    mujoco.mj_forward(task.model, task.data)

    camera_name = task_info["camera_mapping"].get("image")
    with mujoco_renderer(task.model, width=width, height=height) as renderer:
        if camera_name:
            renderer.update_scene(task.data, camera=camera_name)
        else:
            renderer.update_scene(task.data)
        image = renderer.render()

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image)).save(buf, format="PNG")

    return {
        "image_png_bytes": buf.getvalue(),
        "task_info": task_info,
        "robot": entry.robot,
    }
