"""Phase I Step 5: interactive task-scene picker.

Landing page offers two paths (per the user's spec, private/step5-ui-notes.md):
1. "Use an existing task scene" -- pick a category, then a task within it,
   then see the robot it uses and a rendered preview of the reset state.
   This is what's actually implemented here.
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
import os

from flask import Flask, jsonify, request, send_from_directory

os.environ.setdefault("MUJOCO_GL", "egl")

from archetypes.task_catalog import CATALOG
from webui.scene_render import render_scene

app = Flask(__name__, static_folder="static", static_url_path="")

ROBOTS = {
    "ur5e": {"name": "UR5e + Robotiq 2F-85", "asset_dir": "assets/robot/ur5e"},
    "aloha": {"name": "Aloha (dual-arm)", "asset_dir": "assets/robot/aloha2"},
}

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
            "variants": VARIANTS.get(entry.name),
        })
    return jsonify({"tasks": tasks, "robots": ROBOTS})


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

    try:
        result = render_scene(entry, seed=seed, variant=variant)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "image_png_base64": base64.b64encode(result["image_png_bytes"]).decode("ascii"),
        "task_info": result["task_info"],
        "robot": ROBOTS.get(result["robot"], {"name": result["robot"]}),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
