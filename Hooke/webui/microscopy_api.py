"""LAN controls for a single-owner microscope simulation process."""

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from threading import Lock
import time

from flask import Blueprint, jsonify, request, send_file
from backends.environment import require_isaac_environment, IsaacConfigurationError

from microscopy import OPERATIONS, BACKENDS
from microscopy.artifacts import write_json
from microscopy.recording_catalog import FORMATS, recording_directory

ROOT = Path(__file__).resolve().parents[2]
MEDIA = Path(os.environ.get("HOOKE_MICROSCOPY_MEDIA_ROOT", ROOT / "temp/microscopy_demo")).resolve()
LIVE = MEDIA / "live"
bp = Blueprint("microscopy", __name__)


def live_path(backend):
    if backend not in BACKENDS:
        raise ValueError("Unknown microscopy backend")
    return LIVE if backend == "mujoco" else MEDIA / "live-isaac"


class WorkstationBusy(RuntimeError):
    pass


class Workstation:
    """Own the worker, IPC pipe and lock; Flask threads never own EGL contexts."""

    def __init__(self):
        self.process = None
        self.lock = Lock()
        self.log = None
        self.backend = None

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=50)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdout.close()
            self.process = None
            self.backend = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while selector.select(max(0, deadline - time.monotonic())):
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"Microscope worker exited; see {live_path(self.backend) / 'worker.log'}")
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    # Libraries may write informational lines before the ready message.
                    if time.monotonic() >= deadline:
                        break
        raise TimeoutError("Microscope worker did not respond in time")

    def start(self, backend):
        if self.process is not None and self.process.poll() is None and self.backend == backend:
            return
        # Validate before releasing the running source workstation or deleting its views.
        installation = require_isaac_environment() if backend == "isaac" else None
        gpu = str(installation["gpu"]) if installation else os.environ.get("HOOKE_MICROSCOPY_GPU", "0")
        self.close()
        output = live_path(backend)
        output.mkdir(parents=True, exist_ok=True)
        for view in ("overview", "closeup", "sample", "microscope", "microscope_fluorescence"):
            (output / f"{view}.png").unlink(missing_ok=True)
        write_json(output / "state.json", dict(phase="initializing", backend=backend))
        environment = dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        self.log = (output / "worker.log").open("a")
        self.backend = backend
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "microscopy.worker", "--output", str(output), "--gpu", gpu, "--backend", backend],
            cwd=ROOT / "Hooke", env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
        )
        if not self.receive(30).get("ready"):
            raise RuntimeError("Microscope worker failed to initialize")

    def command(self, body):
        backend = body.get("backend", "mujoco")
        live_path(backend)
        if not self.lock.acquire(blocking=False):
            raise WorkstationBusy("An experiment is running; wait for it to finish")
        try:
            if self.backend is not None and self.backend != backend and body.get("command") not in ("reset", "demo"):
                raise ValueError("Reset the workstation when changing its backend")
            self.start(backend)
            self.process.stdin.write(json.dumps(body, allow_nan=False) + "\n")
            self.process.stdin.flush()
            response = self.receive(900 if backend == "isaac" else 120)
            if not response.get("ok"):
                if response.get("fatal"):
                    raise RuntimeError(response.get("error", "Workstation simulation failed"))
                raise ValueError(response.get("error", "Invalid workstation command"))
            return response
        except IsaacConfigurationError:
            raise
        except (OSError, RuntimeError, TimeoutError):
            self.close()
            raise
        finally:
            self.lock.release()


workstation = Workstation()
atexit.register(workstation.close)


@bp.get("/microscopy")
def page():
    return send_file(Path(__file__).parent / "static" / "microscopy.html")


@bp.get("/api/microscopy/state")
def state():
    try:
        path = live_path(request.args.get("backend", "mujoco")) / "state.json"
    except ValueError as error:
        return jsonify(error=str(error)), 400
    response = jsonify(json.loads(path.read_text()) if path.is_file() else {"phase": "offline"})
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/api/microscopy/control")
def control():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("command") not in ("reset", "step", "autofocus", "demo"):
        return jsonify(error="Unknown workstation command"), 400
    if body.get("backend", "mujoco") not in BACKENDS:
        return jsonify(error="Unknown microscopy backend"), 400
    try:
        return jsonify(workstation.command(body))
    except WorkstationBusy as error:
        return jsonify(error=str(error)), 409
    except IsaacConfigurationError as error:
        return jsonify(error=str(error), environment=error.report), 503
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    except (OSError, RuntimeError, TimeoutError) as error:
        return jsonify(error=str(error)), 503


@bp.get("/api/microscopy/image/<view>")
def image(view):
    if view not in ("overview", "closeup", "microscope", "sample", "objective", "detection"):
        return jsonify(error="Unknown microscope view"), 404
    try:
        output = live_path(request.args.get("backend", "mujoco"))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    state_path = output / "state.json"
    if state_path.is_file() and json.loads(state_path.read_text()).get("phase") == "initializing":
        return jsonify(error="The workstation scene is loading"), 404
    if view == "sample":
        state_path = output / "state.json"
        if not state_path.is_file() or json.loads(state_path.read_text()).get("volume_unit") != "pL":
            return jsonify(error="Sample detail requires the cell experiment"), 404
    if view == "objective":
        if (not state_path.is_file()
                or not json.loads(state_path.read_text()).get("calibration", {}).get("reference_objective")):
            return jsonify(error="Objective detail requires the drawing-reference cell workstation"), 404
    path = output / (view + ".png")
    if view == "microscope":
        channel = request.args.get("channel", "phase_contrast")
        if channel not in ("phase_contrast", "fluorescence"):
            return jsonify(error="Unknown cell imaging channel"), 400
        if channel == "fluorescence":
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            if channel not in state.get("calibration", {}).get("display_channels", []):
                return jsonify(error="Fluorescence requires a cell experiment"), 404
            path = output / "microscope_fluorescence.png"
    if view == "detection":
        if (not state_path.is_file()
                or not json.loads(state_path.read_text()).get("calibration", {}).get("reference_detection_path")):
            return jsonify(error="Detection detail requires the collection-reference cell workstation"), 404
    if not path.is_file():
        return jsonify(error="Initialize the workstation first"), 404
    response = send_file(path, mimetype="image/png", conditional=False)
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.get("/api/microscopy/recordings")
def recordings():
    backend = request.args.get("backend", "mujoco")
    if backend not in BACKENDS:
        return jsonify(error="Unknown microscopy backend"), 400
    try:
        folder, revision = recording_directory(MEDIA, backend)
    except (OSError, ValueError) as error:
        return jsonify(error=str(error)), 503
    response = jsonify(backend=backend, revision=revision, operations={
        operation: [extension for extension in FORMATS if (folder/operation/f"demo.{extension}").is_file()]
        for operation in OPERATIONS})
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.get("/api/microscopy/recording/<operation>")
def recording(operation):
    if operation not in OPERATIONS:
        return jsonify(error="Unknown microscopy experiment"), 404
    backend = request.args.get("backend", "mujoco")
    if backend not in BACKENDS:
        return jsonify(error="Unknown microscopy backend"), 400
    video_format = request.args.get("format", "mp4")
    if video_format not in ("mp4", "webm"):
        return jsonify(error="Unknown recording format"), 400
    try:
        folder, _ = recording_directory(MEDIA, backend)
    except (OSError, ValueError) as error:
        return jsonify(error=str(error)), 503
    path = folder / operation / f"demo.{video_format}"
    if not path.is_file():
        return jsonify(error="This experiment has no recording yet"), 404
    return send_file(path, mimetype=f"video/{video_format}", conditional=True, max_age=0)


@bp.get("/api/microscopy/asset-preview/<name>")
def asset_preview(name):
    if name not in ("zaber-rhf", "smaract-sge17"):
        return jsonify(error="Unknown CAD inspection preview"), 404
    path = ROOT / "temp" / "microscopy_research" / "cad_meshes" / name / "cad-preview.png"
    if not path.is_file():
        return jsonify(error="This CAD has not been inspected and rendered locally"), 404
    return send_file(path, mimetype="image/png", conditional=True)
