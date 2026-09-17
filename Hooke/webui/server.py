"""Phase I Step 5: interactive task-scene picker.

Landing page offers two paths (per the user's spec, private/step5-ui-notes.md):
1. "Use an existing task scene" -- pick a category, then a task within it,
   then a robot to place in the scene (its native robot, an arm-mount
   alternative, or a floor-mount bystander -- see robot_registry.py), then
   see a rendered preview.
2. "Generate a custom task scene" (upload a photo/video/description) --
   depends on the generative-3D work in private/TODO.md, which is deferred.
   The UI surfaces this option but marks it not-yet-available rather than
   hiding it, so the two-option landing page matches the spec even though
   only one path works yet.

Run with:
    conda activate autobio
    export MUJOCO_GL=egl
    python -m webui.server
Then open http://localhost:8080/
"""
import base64
import dataclasses
import os

from flask import Flask, jsonify, request, send_from_directory

os.environ.setdefault("MUJOCO_GL", "egl")

from archetypes.task_catalog import CATALOG
from webui.robot_registry import ROBOTS, robot_options_for
from webui.robot_scene import compose_scene, render_robot_preview
from webui.scene_render import render_scene
from backends.gpu_lease import GPUBusy
from webui.custom_gen import generate_custom_asset, GenerationError
from PIL import Image
import io

app = Flask(__name__, static_folder="static", static_url_path="")
# 8MB cap on the whole request (mainly to bound the optional reference-photo
# upload for /api/generate_custom) -- not a security boundary by itself,
# just a sane limit for a local dev server.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
from webui.backend_api import bp as backend_blueprint
app.register_blueprint(backend_blueprint)
from webui.space_experiment_api import bp as space_experiment_blueprint
app.register_blueprint(space_experiment_blueprint)
from webui.surface_api import bp as surface_blueprint
app.register_blueprint(surface_blueprint)


def _robot_entry_json(entry) -> dict:
    """`RobotEntry.arm_cls` (added when Panda/xArm7/etc. got real IK) is a
    Python class, not JSON-serializable via plain `dataclasses.asdict` --
    replace it with the one thing the frontend actually cares about:
    whether this robot is functionally controllable or placement-only."""
    d = dataclasses.asdict(entry)
    d["arm_cls"] = None
    d["functional"] = entry.arm_cls is not None
    return d

# The only task with more than one asset variant right now (see Step 2 /
# archetypes/rotor_variants.py). 30 is the original, unmodified rotor.
VARIANTS = {"insert_centrifuge_5430": [10, 15, 20, 24, 30]}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/catalog")
def api_catalog():
    tasks = []
    for entry in CATALOG.values():
        tasks.append({
            "name": entry.name,
            "description": entry.description,
            "category": entry.category,
            "robot": entry.robot,
            "robot_options": robot_options_for(entry.robot),
            "variants": VARIANTS.get(entry.name),
        })
    robots = {name: _robot_entry_json(r) for name, r in ROBOTS.items()}
    return jsonify({"tasks": tasks, "robots": robots})


def _png_base64(image) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/scene")
def api_scene():
    body = request.get_json(force=True)
    task_name = body.get("task")
    if task_name not in CATALOG:
        return jsonify({"error": f"Unknown task '{task_name}'"}), 400

    entry = CATALOG[task_name]
    seed = int(body.get("seed", 0))
    variant = body.get("variant")
    if variant is not None:
        variant = int(variant)
        if variant not in VARIANTS.get(task_name, []):
            return jsonify({"error": f"'{variant}' is not a valid variant for '{task_name}'"}), 400

    robot = body.get("robot") or entry.robot
    if robot not in robot_options_for(entry.robot):
        return jsonify({"error": f"Robot '{robot}' is not offered for task '{task_name}'"}), 400

    try:
        if robot == entry.robot:
            # Native robot: go through the Task class, which gives the
            # task's actual randomized reset() state and prompt text.
            result = render_scene(entry, seed=seed, variant=variant)
            image_b64 = base64.b64encode(result["image_png_bytes"]).decode("ascii")
            task_info = result["task_info"]
        else:
            # Swapped-in or added robot: bypass the Task class (its arm/IK
            # code assumes the native robot's joint names) and render the
            # scene's default keyframe state instead -- see
            # webui/robot_scene.py's module docstring.
            if variant is not None and task_name == "insert_centrifuge_5430":
                from load_centrifuge_5430 import InsertCentrifuge5430
                from archetypes.rotor_variants import generate_rotor_variant
                base_scene = generate_rotor_variant(variant) if variant != 30 else entry.load_classes()[0].default_scene
            else:
                base_scene = entry.load_classes()[0].default_scene
            scene_path = compose_scene(base_scene, entry.robot, robot, camera_name=entry.camera)
            image = render_robot_preview(scene_path, camera_name=entry.camera)
            image_b64 = _png_base64(image)
            task_info = {"prefix": f"[preview only -- {ROBOTS[robot].display_name} placed in the '{task_name}' scene]"}
    except GPUBusy as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "image_png_base64": image_b64,
        "task_info": task_info,
        "robot": _robot_entry_json(ROBOTS[robot]),
    })


@app.post("/api/generate_custom")
def api_generate_custom():
    """Generate a custom asset preview from a text description (+ optional
    reference photo), using the caller's own OpenAI API key.

    The key is read from the request, used only for this one call, and
    never written to disk, logged, or kept past the end of this request --
    see webui/custom_gen.py's module docstring for the full security model
    (this includes the generated script itself being statically rejected
    if it tries to touch the filesystem/network/subprocesses).
    """
    description = (request.form.get("description") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    model = (request.form.get("model") or "gpt-6-astra").strip()
    image_file = request.files.get("image")

    if not description:
        return jsonify({"error": "Please describe the instrument or asset you want to generate."}), 400
    if not api_key:
        return jsonify({"error": "An OpenAI API key is required for this step (used once, not stored)."}), 400

    image_bytes = image_file.read() if image_file and image_file.filename else None

    try:
        result = generate_custom_asset(description, image_bytes, api_key, model=model)
    except GenerationError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        # Deliberately not including api_key in this message -- only ever
        # pass it as a positional argument to the OpenAI client, never into
        # a format string, so it can't leak into an error message either.
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "image_png_base64": base64.b64encode(result["image_png_bytes"]).decode("ascii"),
        "script": result["script"],
    })


@app.post("/api/protocol_to_task")
def api_protocol_to_task():
    """Phase I's protocol-to-task pipeline (archetypes/protocol_to_task.py):
    parse one or more real lab-protocol steps (one per line) into a task
    spec, generate the instrument/scene, and run the automatic-validation
    yield check -- using the caller's own API key, same security model as
    `/api/generate_custom` (used once, never stored).

    Deliberately does NOT write anything into `task_catalog.py` from this
    endpoint -- generating and validating candidate tasks from an HTTP
    request is fine; committing them into the real catalog is a separate,
    deliberate step (`protocol_to_task.promote_passing_to_catalog`, run
    directly, not over the web) so a request can't silently mutate the
    repo's own source files.
    """
    from archetypes.protocol_to_task import run_pipeline

    protocol_text = (request.form.get("protocol_text") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    n_seeds = int(request.form.get("n_seeds") or 10)

    if not protocol_text:
        return jsonify({"error": "Please provide at least one protocol step (one per line)."}), 400
    if not api_key:
        return jsonify({"error": "An LLM API key is required for this step (used once, not stored)."}), 400

    steps = [line.strip() for line in protocol_text.splitlines() if line.strip()]

    try:
        report = run_pipeline(steps, api_key=api_key, n_seeds=n_seeds)
    except Exception as e:
        # Never format api_key into this message -- same rule as
        # /api/generate_custom's error handling.
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    results = []
    for entry in report.entries:
        results.append({
            "step": entry["step"],
            "spec": entry["spec"],
            "error": entry["error"],
            "report": entry["report"],
        })

    return jsonify({
        "total": report.total,
        "parsed": report.parsed,
        "passed": report.passed,
        "parse_yield": report.parse_yield,
        "overall_yield": report.overall_yield,
        "results": results,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hooke scene and backend interface")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    app.run(host=options.host, port=options.port, debug=False)
