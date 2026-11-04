"""Manage an isolated native Isaac process and release partial resources."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from backends.config import isaac_gpu
from backends.environment import inspect_installation, IsaacConfigurationError
from backends.gpu_lease import GPUBusy, GPULease
from backends.ipc import read_message, write_message
from backends.worker_process import OwnedProcessSession


@dataclass(frozen=True)
class WorkerTimeouts:
    startup: float = 180
    load: float = 600
    step: float = 180
    request: float = 180
    shutdown: float = 5
    terminate: float = 10
    kill: float = 10

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= 3600):
                raise ValueError(f"Worker {field.name} timeout must be finite and in (0, 3600] seconds")


class WorkerTransportError(RuntimeError):
    """Terminal IPC failure; a command may have executed and must not be replayed."""


class WorkerRemoteError(RuntimeError):
    """A valid remote error response; the transport remains usable."""


class WorkerCleanupError(RuntimeError):
    """Cleanup failed; a lease is retained while owned processes may remain live."""


class IsaacWorker:
    def __init__(self, output: Path, gpu: int | None = None, *, timeouts: WorkerTimeouts | None = None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = self.connection = self.process = None
        self.directory = self.gpu_lock = self.log = self.process_session = None
        self.closed = self.broken = False
        self.failure = self.cleanup_report = None
        self.timeouts = timeouts if timeouts is not None else WorkerTimeouts()
        if not isinstance(self.timeouts, WorkerTimeouts):
            raise TypeError("timeouts must be WorkerTimeouts")
        self.transport = os.environ.get("HOOKE_ISAAC_TRANSPORT", "binary")
        if self.transport not in ("json", "binary"):
            raise ValueError("HOOKE_ISAAC_TRANSPORT must be json or binary")
        gpu = isaac_gpu() if gpu is None else gpu
        if gpu < 0:
            raise ValueError("Isaac GPU index must be nonnegative")
        installation = inspect_installation()
        if not installation["installation_ready"]:
            raise IsaacConfigurationError(installation)
        install = Path(installation["isaac_path"])
        try:
            self.gpu_lock = GPULease(gpu)
            query = ["nvidia-smi", "-i", str(gpu)]
            memory = int(
                subprocess.check_output(
                    query
                    + ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
            )
            if memory >= 2048:
                raise GPUBusy(
                    f"GPU {gpu} is busy ({memory} MiB); wait for the running job"
                )
            uuid = subprocess.check_output(
                query + ["--query-gpu=uuid", "--format=csv,noheader"], text=True
            ).strip()
            self.directory = tempfile.TemporaryDirectory(prefix="hooke-isaac-")
            self.socket_path = Path(self.directory.name) / "worker.sock"
            env = os.environ.copy()
            for key in ("CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME", "PYTHONEXE"):
                env.pop(key, None)
            env.update(
                CUDA_VISIBLE_DEVICES=uuid,
                HOOKE_ISAAC_RENDER_GPU=str(gpu),
                HOOKE_ISAAC_TRANSPORT=self.transport,
                OMP_NUM_THREADS="8",
                OPENBLAS_NUM_THREADS="1",
                PYTHONUNBUFFERED="1",
            )
            self.log = (self.output / "isaac-worker.log").open("wb")
            self.process = subprocess.Popen(
                [
                    str(install / "python.sh"),
                    str(Path(__file__).with_name("isaac_worker.py")),
                    "--socket",
                    str(self.socket_path),
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.process_session = OwnedProcessSession(self.process)
            deadline = time.monotonic() + self.timeouts.startup
            while not self.socket_path.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f'Isaac worker exited; inspect {self.output / "isaac-worker.log"}'
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Isaac startup exceeded {self.timeouts.startup:g} seconds")
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.connection.settimeout(self.timeouts.startup)
            self.connection.connect(str(self.socket_path))
            self.stream = self.connection.makefile("rwb")
        except BaseException as exc:
            self.broken = True
            self._cleanup_failure(exc)
            raise

    def _cleanup_failure(self, error):
        try:
            self.close()
        except BaseException as cleanup:
            error.add_note(f"Worker cleanup also failed: {cleanup}")

    def _exchange(self, op, kwargs, timeout):
        try:
            self.connection.settimeout(timeout)
            write_message(self.stream, {"op": op, **kwargs}, self.transport)
        except OSError as exc:
            raise WorkerTransportError(f"Worker {op} write failed: {exc}") from exc
        # Local request validation happens before writing bytes. In contrast,
        # an invalid incoming frame makes the response boundary untrustworthy.
        try:
            response = read_message(self.stream, self.transport)
            if response is None:
                raise EOFError("Worker disconnected")
            if not isinstance(response.get("ok"), bool):
                raise ValueError("Worker response has no boolean status")
            if response["ok"]:
                return response["result"]
            error = response["error"]
            trace = response.get("traceback", "")
            if not isinstance(error, str) or not isinstance(trace, str):
                raise ValueError("Worker error response is malformed")
        except (OSError, EOFError, ValueError, TypeError, KeyError) as exc:
            raise WorkerTransportError(f"Worker {op} response failed (budget {timeout:g}s): {exc}") from exc
        raise WorkerRemoteError(error + ("\n" + trace if trace else ""))

    def call(self, op: str, **kwargs):
        if self.closed or self.broken:
            raise WorkerTransportError(self.failure or "Isaac worker is closed; create a new worker")
        if self.stream is None:
            raise RuntimeError("Isaac worker has not connected")
        timeout = {"load": self.timeouts.load, "step": self.timeouts.step,
                   "shutdown": self.timeouts.shutdown}.get(op, self.timeouts.request)
        try:
            return self._exchange(op, kwargs, timeout)
        except (WorkerTransportError, KeyboardInterrupt, SystemExit) as exc:
            self.broken = True
            self.failure = str(exc) or type(exc).__name__
            self._cleanup_failure(exc)
            raise

    def close(self):
        self.closed = True
        graceful = False
        errors = []
        if self.stream is not None and not self.broken:
            try:
                # A subclass's call() policy cannot extend this deadline.
                self._exchange("shutdown", {}, self.timeouts.shutdown)
                graceful = True
            except (RuntimeError, OSError, EOFError, ValueError):
                self.broken = True
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for name in ("stream", "connection"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except OSError as exc:
                    errors.append(exc)
                finally:
                    setattr(self, name, None)
        stopped = self.process is None
        if self.process is not None:
            try:
                # Popen can succeed immediately before constructor cancellation.
                if self.process_session is None:
                    self.process_session = OwnedProcessSession(self.process)
                self.cleanup_report = self.process_session.close(
                    grace=self.timeouts.shutdown if graceful else 0,
                    terminate=self.timeouts.terminate, kill=self.timeouts.kill)
                self.process = self.process_session = None
                stopped = True
            except Exception as exc:
                errors.append(exc)
        for name in ("log", "gpu_lock"):
            if name == "gpu_lock" and not stopped:
                continue
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
                else:
                    setattr(self, name, None)
        if self.directory is not None and stopped:
            try:
                self.directory.cleanup()
            except OSError as exc:
                errors.append(exc)
            else:
                self.directory = None
        if errors:
            raise WorkerCleanupError("; ".join(str(error) for error in errors)) from errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.close()
        else:
            self._cleanup_failure(exc)
