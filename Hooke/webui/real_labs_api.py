"""CPU-only inspection and mechanical controls for reference laboratory scenes."""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Lock, RLock, Thread
from uuid import uuid4
import math
import os
import re
import xml.etree.ElementTree as ET

from flask import Blueprint, abort, current_app, jsonify, request, send_file
import numpy as np

from real_labs.catalog import scenes
from real_labs import operations


bp = Blueprint("real_labs", __name__)
ROOT = Path(__file__).resolve().parents[2]


class _SceneBusy(RuntimeError):
    def __init__(self, message, job=None):
        super().__init__(message)
        self.job = deepcopy(job)


class _SessionEntry:
    def __init__(self, session):
        self.session = session
        self.lock = Lock()
        self.job = None
        self.final_state = None


class _SessionCache:
    """Bound model memory while pinning entries with exclusive physics access.

    The cache lock protects membership and job metadata, never a running route.
    A scene lock protects every read or write of its mutable native physics.
    """

    def __init__(self):
        self.lock = RLock()
        self.items = OrderedDict()

    def get(self, identifier):
        # Callers hold self.lock until they have acquired the entry lock.
        if identifier not in self.items:
            while len(self.items) >= 2:
                idle = next(
                    (
                        key
                        for key, entry in self.items.items()
                        if not entry.lock.locked()
                    ),
                    None,
                )
                if idle is None:
                    raise _SceneBusy(
                        "All scene sessions are busy; retry after a job finishes"
                    )
                del self.items[idle]
            self.items[identifier] = _SessionEntry(_build_session(identifier))
        self.items.move_to_end(identifier)
        return self.items[identifier]

    def acquire(self, identifier):
        with self.lock:
            entry = self.get(identifier)
            if not entry.lock.acquire(blocking=False):
                raise _SceneBusy("Scene physics is busy", entry.job)
            return entry

    @contextmanager
    def access(self, identifier):
        entry = self.acquire(identifier)
        try:
            yield entry.session
        finally:
            entry.lock.release()


def _busy_response(error):
    body = {"error": str(error)}
    if error.job is not None:
        body["job"] = error.job
    return jsonify(body), 409


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
    definition = scenes().get(identifier)
    if definition is None or not re.fullmatch(r"[a-z][a-z0-9_]*\.png", name):
        return None
    views = {"overview", "workstation", "interior"} | definition.get(
        "camera", {}
    ).keys()
    if name[:-4] not in views:
        return None
    base = Path(
        os.environ.get("HOOKE_REAL_LABS_OUTPUT", ROOT / "temp/real_labs/latest")
    ).resolve()
    path = (base / identifier / name).resolve()
    return path if path.is_relative_to(base) and path.is_file() else None


def _state(session):
    state = {
        **session.inspect(),
        "manifest": session.manifest,
        "time_s": float(session.data.time),
        "operations": operations.available_operations(session),
        "control_policy": operations.control_policy(session),
        "render_mode": "recorded_preview",
        "session_scope": "shared process-local scene state; idle scenes reset after eviction",
    }
    invalid = [
        field
        for field in ("time", "qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
        if not np.isfinite(getattr(session.data, field)).all()
    ]
    if invalid:
        state.update(
            state_valid=False,
            invalid_physics_state=True,
            invalid_state_fields=invalid,
            state_error="Nonfinite simulation state; explicit recovery is required",
        )
    failure = getattr(session, "operation_failure", None)
    if failure:
        state.update(
            state_valid=False,
            invalid_physics_state=True,
            operation_failure=failure,
            state_error="An operation failed. Explicitly reset the scene before further motion.",
        )
    return _json_safe(state)


def _json_safe(value, nonfinite=None, path="$"):
    """Preserve diagnostics as browser-readable JSON without changing physics."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, nonfinite, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, nonfinite, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite is not None:
            nonfinite.append(path)
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported JSON value at {path}: {type(value).__name__}")


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
    try:
        with cache.access(identifier) as session:
            return jsonify(_state(session))
    except _SceneBusy as error:
        return _busy_response(error)


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
    try:
        with cache.access(identifier) as session:
            result = session.command(
                body["equipment"], body["control"], body["value"], duration=duration
            )
            return jsonify(result=result, state=_state(session))
    except _SceneBusy as error:
        return _busy_response(error)
    except (KeyError, ValueError) as error:
        return jsonify(error=f"Invalid control request: {error}"), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 409


@bp.post("/api/real-labs/<identifier>/reset")
def reset(identifier):
    _known(identifier)
    cache = _cache()
    try:
        with cache.access(identifier) as session:
            session.reset()
            return jsonify(_state(session))
    except _SceneBusy as error:
        return _busy_response(error)


def _run_operation(cache, entry, operation_id):
    """Finish one finite route without a Flask context or an implicit reset."""
    outcome = {"status": "failed"}
    final_state = None
    try:
        operations.validate_operation(entry.session, operation_id)
        report = operations.run_operation(entry.session, operation_id)
        if isinstance(report, dict):
            outcome["result"] = report
        if not isinstance(report, dict) or not isinstance(report.get("success"), bool):
            raise RuntimeError("Operation returned no explicit boolean success result")
        if report["success"]:
            outcome["status"] = "succeeded"
        else:
            outcome["error"] = str(
                report.get("failure")
                or report.get("error")
                or "Operation reported failure"
            )
    except Exception as error:
        outcome["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            final_state = _state(entry.session)
            if final_state.get("invalid_state_fields"):
                outcome["status"] = "failed"
                outcome["invalid_state_fields"] = final_state["invalid_state_fields"]
                outcome["error"] = (
                    outcome.get("error", "") + "; Nonfinite final simulation state"
                ).lstrip("; ")
        except Exception as error:
            outcome["status"] = "failed"
            outcome["state_error"] = f"{type(error).__name__}: {error}"
            outcome.setdefault("error", "Cannot inspect the final measured state")
        try:
            nonfinite = []
            outcome = _json_safe(outcome, nonfinite)
            if nonfinite:
                outcome["status"] = "failed"
                outcome["nonfinite_report_fields"] = nonfinite
                outcome["error"] = (
                    outcome.get("error", "") + "; Nonfinite operation report values"
                ).lstrip("; ")
        except Exception as error:
            outcome = {
                "status": "failed",
                "error": f"Cannot serialize operation report: {error}",
            }
        with cache.lock:
            entry.final_state = final_state
            entry.job.update(outcome)
            entry.lock.release()


@bp.post("/api/real-labs/<identifier>/operation")
def start_operation(identifier):
    _known(identifier)
    body = request.get_json(silent=True)
    if (
        not isinstance(body, dict)
        or set(body) != {"operation"}
        or not isinstance(body["operation"], str)
        or not body["operation"]
    ):
        return jsonify(error="Expected exactly one operation string"), 400
    cache = _cache()
    try:
        entry = cache.acquire(identifier)
    except _SceneBusy as error:
        return _busy_response(error)
    transferred = False
    try:
        operations.validate_operation(entry.session, body["operation"])
        job = {"id": uuid4().hex, "operation": body["operation"], "status": "running"}
        with cache.lock:
            entry.job = job.copy()
            entry.final_state = None
        worker = Thread(
            target=_run_operation,
            args=(cache, entry, body["operation"]),
            name=f"real-lab-operation-{job['id']}",
            daemon=True,
        )
        try:
            worker.start()
        except Exception as error:
            with cache.lock:
                entry.job.update(
                    status="failed", error=f"Could not start operation: {error}"
                )
                entry.final_state = _state(entry.session)
            return jsonify(error="Could not start operation", job=entry.job), 503
        transferred = True
        return jsonify(job=job), 202
    except (KeyError, ValueError) as error:
        return jsonify(error=f"Invalid operation request: {error}"), 400
    except RuntimeError as error:
        return jsonify(error=str(error)), 409
    finally:
        if not transferred:
            entry.lock.release()


@bp.get("/api/real-labs/<identifier>/operation")
def operation_status(identifier):
    _known(identifier)
    cache = _cache()
    with cache.lock:
        entry = cache.items.get(identifier)
        if entry is None or entry.job is None:
            return jsonify(error="No retained operation job for this scene"), 404
        result = {"job": deepcopy(entry.job)}
        if entry.job["status"] == "running":
            return jsonify(result)
        if not entry.lock.acquire(blocking=False):
            return _busy_response(_SceneBusy("Scene physics is busy", entry.job))
        cache.items.move_to_end(identifier)
        if entry.final_state is not None:
            result["completed_state"] = deepcopy(entry.final_state)
    try:
        # The completion snapshot remains historical after a later scalar command
        # or explicit reset; clients receive the current measured state separately.
        result["state"] = _state(entry.session)
        return jsonify(result)
    finally:
        entry.lock.release()


@bp.get("/api/real-labs/<identifier>/preview/<name>")
def preview(identifier, name):
    _known(identifier)
    path = recorded_preview_path(identifier, name)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/png")
