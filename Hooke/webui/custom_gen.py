"""Phase I Step 5's "generate a custom task scene" flow: turn a free-text
description (+ optional reference photo) into a simulation-ready asset,
using an LLM to write a Blender (`bpy`) script rather than outputting mesh
data directly -- see private/TODO.md for why (GPT-6 Astra's actual 3D
capability is "write bpy code, run it in real Blender", not native mesh
output; BenchCAD, OpenAI's own benchmark for this, evaluates exactly that).

Security model: the caller's OpenAI API key is used only for the duration
of one `generate_custom_asset()` call (passed as a parameter, read from a
request by the caller in server.py) and is never written to disk, logged,
or held past that call. Separately -- and just as important, since this
step is unavoidably executing LLM-authored Python -- every generated
script is statically checked (`check_script_safety`) before it's ever
handed to Blender; anything using the filesystem outside its own scratch
directory, networking, process/subprocess execution, or dynamic code
execution is rejected outright rather than sandboxed-and-hoped.

Pipeline: text (+ optional image) -> LLM -> bpy script -> safety check ->
run in vendored Blender (headless) in a fresh scratch dir -> exported OBJ
mesh -> wrapped in a minimal MJCF scene -> rendered preview (MuJoCo EGL).
"""
from __future__ import annotations

import base64
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

BLENDER_BIN = Path(__file__).parent.parent.parent / "vendor" / "blender-5.2.1-linux-x64" / "blender"
OUTPUT_OBJ_NAME = "asset.obj"
SCRIPT_TIMEOUT_S = 90


class GenerationError(Exception):
    """Raised with a message meant to be shown directly to the UI user --
    every stage of the pipeline fails through this, so the caller doesn't
    need per-stage exception handling."""


# ---------------------------------------------------------------------------
# 1. Prompting

SYSTEM_PROMPT = """You write a single Python script that runs inside Blender (using the `bpy` \
and `bmesh` modules) to build a simple 3D model of a piece of laboratory equipment or a \
container, based on the user's description (and reference photo, if provided).

Hard requirements:
- Use ONLY the `bpy`, `bmesh`, `math`, and `mathutils` modules. Do not import `os`, `sys`, \
`subprocess`, `socket`, `urllib`, `requests`, `shutil`, or anything else that touches the \
filesystem, network, or another process.
- Start from `bpy.ops.wm.read_factory_settings(use_empty=True)` to get a clean scene.
- Build the geometry with primitive meshes (cubes, cylinders, spheres) combined with modifiers \
or bmesh operations -- do not attempt to load any external file.
- Scale the model in meters, roughly within a 0.05-0.5m bounding box (benchtop lab instrument \
scale), centered near the origin.
- At the end, select all mesh objects and export them with exactly:
  `bpy.ops.wm.obj_export(filepath="{output_name}", export_selected_objects=True)`
- Respond with ONLY one ```python fenced code block containing the complete script -- no prose \
before or after it.
"""


def build_prompt() -> str:
    return SYSTEM_PROMPT.format(output_name=OUTPUT_OBJ_NAME)


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def extract_script(response_text: str) -> str:
    match = _CODE_FENCE_RE.search(response_text)
    if not match:
        raise GenerationError(
            "The model's response didn't contain a fenced ```python code block, so there's no "
            "script to run. Try rephrasing the description."
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# 2. Safety review -- see module docstring. Deny-list, not sandboxing: if a
# script needs any of this, we don't run it, full stop.

_DENYLIST_PATTERNS = [
    (r"\bimport\s+os\b", "imports os"),
    (r"\bimport\s+sys\b", "imports sys"),
    (r"\bimport\s+subprocess\b", "imports subprocess"),
    (r"\bimport\s+socket\b", "imports socket"),
    (r"\bimport\s+shutil\b", "imports shutil"),
    (r"\burllib\b", "references urllib"),
    (r"\brequests\b", "references requests"),
    (r"\bopen\s*\(", "calls open() directly"),
    (r"\bexec\s*\(", "calls exec()"),
    (r"\beval\s*\(", "calls eval()"),
    (r"__import__", "uses __import__"),
    (r"\bos\.\w+", "references os.*"),
    (r"\bsys\.\w+", "references sys.*"),
]


def check_script_safety(script: str) -> list[str]:
    """Returns a list of human-readable violations; empty means the script
    passed. Does not attempt to be a complete Python sandbox analysis --
    it's a deliberately blunt deny-list for the specific things a bpy asset
    -generation script has no legitimate reason to need."""
    violations = []
    for pattern, reason in _DENYLIST_PATTERNS:
        if re.search(pattern, script):
            violations.append(reason)
    if OUTPUT_OBJ_NAME not in script:
        violations.append(f"does not appear to export to the required '{OUTPUT_OBJ_NAME}' filename")
    return violations


# ---------------------------------------------------------------------------
# 3. Running the (approved) script through Blender

def run_blender_script(script_text: str, workdir: Path) -> Path:
    script_path = workdir / "gen_script.py"
    script_path.write_text(script_text)

    try:
        proc = subprocess.run(
            [str(BLENDER_BIN), "--background", "--python", str(script_path)],
            cwd=workdir, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise GenerationError(f"Blender did not finish within {SCRIPT_TIMEOUT_S}s; the generated script may be stuck.") from e

    obj_path = workdir / OUTPUT_OBJ_NAME
    if proc.returncode != 0 or not obj_path.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise GenerationError(f"Blender failed to produce {OUTPUT_OBJ_NAME}.\n{tail}")
    return obj_path


# ---------------------------------------------------------------------------
# 4. Wrapping the exported mesh in a minimal MJCF scene and rendering it

_SCENE_TEMPLATE = """
<mujoco model="custom_asset_preview">
  <asset>
    <mesh name="custom_asset" file="{obj_path}" scale="{scale} {scale} {scale}"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="groundplane"/>
    <body name="asset" pos="0 0 {z_offset}">
      <geom type="mesh" mesh="custom_asset" rgba="0.75 0.75 0.78 1"/>
    </body>
    <camera name="preview_cam" pos="0.5 -0.5 0.4" xyaxes="0.7 0.7 0 -0.3 0.3 0.9"/>
  </worldbody>
</mujoco>
"""


def render_asset_preview(obj_path: Path, width: int = 480, height: int = 360) -> np.ndarray:
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh")
    extent = float(np.max(mesh.extents)) if mesh.extents is not None else 0.2
    # Normalize so the model's largest dimension is ~0.3m, in case the
    # generated script didn't scale to meters correctly.
    scale = 0.3 / extent if extent > 0 else 1.0
    z_offset = -float(mesh.bounds[0][2]) * scale if mesh.bounds is not None else 0.0

    scene_xml = _SCENE_TEMPLATE.format(obj_path=str(obj_path), scale=scale, z_offset=z_offset)
    scene_path = obj_path.parent / "preview_scene.xml"
    scene_path.write_text(scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height, width)
    try:
        renderer.update_scene(data, camera="preview_cam")
        return renderer.render()
    finally:
        renderer.close()


# ---------------------------------------------------------------------------
# 5. LLM call (the only part that needs a real, caller-supplied API key)

def call_llm(description: str, image_bytes: bytes | None, api_key: str, model: str = "gpt-6-astra") -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)
    content = [{"type": "input_text", "text": description}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    response = client.responses.create(
        model=model,
        instructions=build_prompt(),
        input=[{"role": "user", "content": content}],
    )
    return response.output_text


# ---------------------------------------------------------------------------
# Orchestration

def generate_custom_asset(
    description: str, image_bytes: bytes | None, api_key: str,
    model: str = "gpt-6-astra", _script_override: str | None = None,
) -> dict:
    """Runs the full pipeline and returns {"image_png_bytes", "script",
    "warnings"}. `_script_override` bypasses the LLM call entirely (used by
    tests to exercise safety-check -> Blender -> MJCF -> render without a
    real API key; see test_custom_gen.py)."""
    script = _script_override if _script_override is not None else call_llm(description, image_bytes, api_key, model)
    if _script_override is None:
        script = extract_script(script)

    violations = check_script_safety(script)
    if violations:
        raise GenerationError(
            "Generated script failed the safety review and was not run:\n- " + "\n- ".join(violations)
        )

    with tempfile.TemporaryDirectory(prefix="hooke_custom_asset_") as tmpdir:
        workdir = Path(tmpdir)
        obj_path = run_blender_script(script, workdir)
        image = render_asset_preview(obj_path)
        buf_path = workdir / "preview.png"
        Image.fromarray(image).save(buf_path)
        image_bytes_out = buf_path.read_bytes()

    return {"image_png_bytes": image_bytes_out, "script": script, "warnings": []}
