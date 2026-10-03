"""Fingerprint rendering behavior without importing Blender or starting a renderer."""

import hashlib
from pathlib import Path


def render_source_digest():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "real_labs/render.py",
        "real_labs/finishes.py",
        "microscopy/blender_render.py",
    ):
        content = (package / relative).read_bytes()
        digest.update(relative.encode() + b"\0" + str(len(content)).encode() + b"\0")
        digest.update(content)
    return digest.hexdigest()
