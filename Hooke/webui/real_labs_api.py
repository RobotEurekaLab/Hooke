"""CPU-only inspection and mechanical controls for reference laboratory scenes."""

from collections import OrderedDict
from pathlib import Path
from threading import RLock
import math
import os
import xml.etree.ElementTree as ET

from flask import Blueprint, abort, current_app, jsonify, request, send_file

from real_labs.catalog import scenes


bp = Blueprint("real_labs", __name__)
ROOT = Path(__file__).resolve().parents[2]


class _SessionCache:
    """Serialize native physics access and bound per-application model memory."""

    def __init__(self):
        self.lock = RLock()
        self.items = OrderedDict()

    def get(self, identifier):
        if identifier not in self.items:
            while len(self.items) >= 2:
                self.items.popitem(last=False)
            self.items[identifier] = _build_session(identifier)
        self.items.move_to_end(identifier)
        return self.items[identifier]


def _cache():
    return current_app.extensions.setdefault("real_labs_sessions", _SessionCache())


def _build_session(identifier):
    # No rendering context, GPU worker, background loop or service is created.
    import mujoco
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(identifier)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest)


def _known(identifier):
    if identifier not in scenes():
        abort(404)


def recorded_preview_path(identifier, name="overview.png"):
    """Resolve only published scene views, without loading a model or renderer."""
    if identifier not in scenes() or name not in {
        "overview.png",
        "workstation.png",
        "interior.png",
    }:
        return None
    base = Path(
        os.environ.get("HOOKE_REAL_LABS_OUTPUT", ROOT / "temp/real_labs/latest")
    ).resolve()
    path = (base / identifier / name).resolve()
    return path if path.is_relative_to(base) and path.is_file() else None


def _state(session):
    return {
        **session.inspect(),
        "manifest": session.manifest,
        "time_s": float(session.data.time),
        "render_mode": "recorded_preview",
        "session_scope": "shared scene state; least recently used scenes reset after eviction",
    }


@bp.get("/real-labs")
def page():
    return send_file(Path(__file__).with_name("static") / "real_labs.html")


@bp.get("/api/real-labs")
def catalog():
    return jsonify(
        scenes=list(scenes().values()),
        scope="Reference-informed laboratory prototypes with CPU mechanical controls",
        reconstruction_status="Not surveyed or institution-validated digital twins",
        render_mode="recorded_preview",
    )


@bp.get("/api/real-labs/<identifier>/state")
def state(identifier):
    _known(identifier)
    cache = _cache()
    with cache.lock:
        return jsonify(_state(cache.get(identifier)))


@bp.post("/api/real-labs/<identifier>/command")
def command(identifier):
    _known(identifier)
    body = request.get_json(silent=True)
    required = {"equipment", "control", "value"}
    if (
        not isinstance(body, dict)
        or not required <= body.keys()
        or body.keys() - required - {"duration"}
    ):
        return (
            jsonify(error="Expected equipment, control, value and optional duration"),
            400,
        )
    if not isinstance(body["equipment"], str) or not isinstance(body["control"], str):
        return jsonify(error="Equipment and control must be strings"), 400
    duration = body.get("duration", 2.0)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (body["value"], duration)
    ):
        return jsonify(error="Value and duration must be finite numbers"), 400
    if not 0 < duration <= 5:
        return jsonify(error="Duration must be in (0, 5] seconds"), 400
    cache = _cache()
    with cache.lock:
        session = cache.get(identifier)
        try:
            result = session.command(
                body["equipment"], body["control"], body["value"], duration=duration
            )
        except (KeyError, ValueError) as error:
            return jsonify(error=f"Invalid control request: {error}"), 400
        except RuntimeError as error:
            return jsonify(error=str(error)), 409
        return jsonify(result=result, state=_state(session))


@bp.post("/api/real-labs/<identifier>/reset")
def reset(identifier):
    _known(identifier)
    cache = _cache()
    with cache.lock:
        session = cache.get(identifier)
        session.reset()
        return jsonify(_state(session))


@bp.get("/api/real-labs/<identifier>/preview/<name>")
def preview(identifier, name):
    _known(identifier)
    path = recorded_preview_path(identifier, name)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/png")
