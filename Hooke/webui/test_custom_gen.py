"""Tests for webui/custom_gen.py that don't need a real OpenAI API key.

`generate_custom_asset(..., _script_override=...)` bypasses the LLM call
entirely, so this exercises everything else end-to-end: the safety
checker (both accepting a clean script and rejecting unsafe ones) and the
full Blender -> OBJ -> MJCF -> render pipeline with a real, known-safe
script. Only the actual `call_llm` network call is untested here -- that
needs a live key, tried manually once one is available (see
private/technical-log.md).

Invoke directly:
    python -m webui.test_custom_gen
"""
from webui.custom_gen import check_script_safety, generate_custom_asset, GenerationError

SAFE_SCRIPT = """
import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.15, location=(0, 0, 0.075))
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.mode_set(mode='OBJECT')

for obj in bpy.context.scene.objects:
    obj.select_set(True)
bpy.ops.wm.obj_export(filepath="asset.obj", export_selected_objects=True)
"""

UNSAFE_SCRIPT = """
import bpy
import os
os.system("echo pwned")
bpy.ops.wm.obj_export(filepath="asset.obj")
"""


def test_safety_checker():
    assert check_script_safety(SAFE_SCRIPT) == [], "safe script should have no violations"
    violations = check_script_safety(UNSAFE_SCRIPT)
    assert violations, "unsafe script should be flagged"
    print("safety checker: OK (clean script passes, os.system script rejected):", violations)


def test_end_to_end_pipeline():
    try:
        generate_custom_asset(
            description="(unused, script override)", image_bytes=None, api_key="unused",
            _script_override=UNSAFE_SCRIPT,
        )
        raise AssertionError("unsafe script should have raised GenerationError")
    except GenerationError as e:
        print("end-to-end rejects unsafe script as expected:", e)

    result = generate_custom_asset(
        description="(unused, script override)", image_bytes=None, api_key="unused",
        _script_override=SAFE_SCRIPT,
    )
    assert len(result["image_png_bytes"]) > 0
    with open("/tmp/custom_gen_test_preview.png", "wb") as f:
        f.write(result["image_png_bytes"])
    print("end-to-end pipeline OK, preview saved to /tmp/custom_gen_test_preview.png")


if __name__ == "__main__":
    test_safety_checker()
    test_end_to_end_pipeline()
    print("\nAll custom_gen checks passed.")
