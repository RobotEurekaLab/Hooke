"""Composes a task scene with a chosen robot: either swapping an "arm"-mount
robot in for the task's native one, or adding a "floor"-mount robot standing
nearby (see robot_registry.py for why these are handled differently).

This bypasses the Task/Expert Python classes entirely and compiles the MJCF
directly with `mujoco.MjModel.from_xml_path`, because those classes assume
their original robot's exact joint names (UR5eArm looks up
f'{prefix}shoulder_pan', which doesn't exist on a Panda or a G1). That means
a swapped/added-robot preview shows the scene's *default* keyframe state,
not the task's randomized reset() state -- see scene_render.py for the
native, Task-class-based path that still gives the accurate reset state
when no robot swap is requested.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco

mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')  # some scenes (e.g. pickup) use the thread mechanism plugin

from webui.robot_registry import ROBOTS, RobotEntry

MODEL_ROOT = Path(__file__).parent.parent / "model"
SCENE_ROOT = MODEL_ROOT / "scene"

_UR5E_MODEL_RE = re.compile(r'(<model name="ur5e" file=")\.\./robot/ur5e_gripper\.xml(")')

# Matches either MJCF spelling of a root free joint: <freejoint .../> or
# <joint type="free" .../> (attribute order may vary; the two robots we
# vendor use one spelling each).
_FREEJOINT_RE = re.compile(r'<freejoint\b[^>]*/>|<joint\b(?=[^>]*type="free")[^>]*/>')
# Any <keyframe>...</keyframe> block. Dropped for the same reason: it's
# authored assuming the model's own (now-removed) root freejoint occupies
# the first 7 qpos slots, so it's no longer valid once that joint moves to
# the attaching scene's wrapper body -- and we don't apply per-robot
# keyframes for floor-mount bystanders anyway (see render_robot_preview).
_KEYFRAME_BLOCK_RE = re.compile(r'<keyframe>.*?</keyframe>', re.DOTALL)

# Where a "floor"-mount robot stands, relative to the scene's own origin --
# off to the side of the table, not at the arm's tabletop mount point.
# Approximate for every scene (not tailored per task layout); see
# private/technical-log.md.
#
# This default happens to sit almost exactly at `table_cam_left`'s own
# position (-1.2, 0, 1.65 -- see task_catalog.py) -- fine for the other
# camera (table_cam_front), but for any scene using table_cam_left it
# places the robot right where that camera is, so the render shows
# essentially nothing (the robot is behind/inside the camera). Real bug,
# not a rare one: 4 of the 6 catalog tasks use table_cam_left. Since a
# single offset can't suit both cameras' very different distances from
# the table (verified: an offset good for table_cam_left's wider framing
# ends up far too close to table_cam_front's tighter one, dominating the
# shot instead), branch on which camera the scene actually declares.
FLOOR_STAND_POS = "-1.0 -0.9 0"
FLOOR_STAND_POS_TABLE_CAM_LEFT = "0.6 0.0 0"


def _attachable_floor_robot_path(robot: RobotEntry) -> Path:
    """MuJoCo's <attach> always wraps the attached content in a new body,
    and a body with a freejoint must be a *direct* child of <worldbody> --
    so a model with its freejoint baked into its own root body (every
    humanoid/mobile robot we vendor) can never be <attach>'d directly
    ("free joint can only be used on top level"). The fix (confirmed
    against a MuJoCo maintainer's guidance for this exact error) is to
    strip that model's own freejoint and instead declare a fresh one, at
    the correct top level, on the wrapping body the scene itself provides.
    This writes a `*_attachable.gen.xml` copy of the robot with its root
    freejoint removed, once per robot (cached across calls).
    """
    src = MODEL_ROOT / robot.mjcf_path
    out = src.with_name(src.stem + "_attachable.gen.xml")
    if not out.exists():
        text = src.read_text()
        new_text, n = _FREEJOINT_RE.subn("", text, count=1)
        if n != 1:
            raise RuntimeError(f"Expected exactly one root freejoint in {src}, found {n}")
        new_text = _KEYFRAME_BLOCK_RE.sub("", new_text)
        out.write_text(new_text)
    return out


def compose_scene(base_scene_path: Path, native_robot: str, chosen_robot: str) -> Path:
    """Returns a scene XML path with `chosen_robot` swapped in / added. If
    `chosen_robot == native_robot`, returns `base_scene_path` unchanged."""
    if chosen_robot == native_robot:
        return base_scene_path

    robot = ROBOTS[chosen_robot]
    text = base_scene_path.read_text()

    if robot.mount == "arm":
        if native_robot != "ur5e":
            raise ValueError(f"Arm-mount swap only supported for ur5e-native scenes, got {native_robot!r}")
        new_text, n = _UR5E_MODEL_RE.subn(rf'\g<1>../{robot.mjcf_path}\g<2>', text)
        if n == 0:
            raise ValueError(f"No ur5e model reference found in {base_scene_path}")
    elif robot.mount == "floor":
        attachable_path = _attachable_floor_robot_path(robot)
        rel_path = attachable_path.relative_to(MODEL_ROOT)
        prefix = f"{robot.name}:"
        model_decl = f'<model name="{robot.name}" file="../{rel_path}" content_type="text/xml" />'
        stand_pos = FLOOR_STAND_POS_TABLE_CAM_LEFT if 'name="table_cam_left"' in text else FLOOR_STAND_POS
        attach_block = (
            f'<body name="{prefix}mount" pos="{stand_pos}">'
            f'<joint name="{prefix}root" type="free"/>'
            f'<attach model="{robot.name}" body="world" prefix="{prefix}"/>'
            f'</body>'
        )
        assert "</asset>" in text and "</worldbody>" in text
        new_text = text.replace("</asset>", f"{model_decl}\n    </asset>", 1)
        new_text = new_text.replace("</worldbody>", f"{attach_block}\n  </worldbody>", 1)
    else:
        raise ValueError(f"Robot {chosen_robot!r} has mount={robot.mount!r}, not selectable here")

    out_path = SCENE_ROOT / f"{base_scene_path.stem}_robot_{chosen_robot}.gen.xml"
    out_path.write_text(new_text)
    return out_path


def render_robot_preview(scene_path: Path, camera_name: str | None, width: int = 480, height: int = 360):
    """Compiles `scene_path` directly (no Task class) and renders one frame.

    Applies keyframe 0 if present -- every task scene has exactly one, for
    its native/swapped-in arm's resting pose (MuJoCo's <attach> merges an
    attached model's own <keyframe> into the combined scene automatically,
    prefixed and padded to the full qpos vector, so this also picks up e.g.
    Panda's folded "home" pose instead of an all-zeros default that looks
    like an awkward stretched-out T-pose). A "floor"-mount addition doesn't
    have or need its own keyframe here: its standing position/orientation
    is already set declaratively by compose_scene's wrapping body.
    """
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height, width)
    try:
        if camera_name:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data)
        return renderer.render()
    finally:
        renderer.close()
