"""Manage an isolated native Isaac process and release partial resources."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from backends.config import isaac_gpu, isaac_installation
from backends.ipc import read_message, write_message


class IsaacWorker:
    def __init__(self, output: Path, gpu: int | None = None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = self.connection = self.process = None
        self.directory = self.gpu_lock = self.log = None
        self.transport = os.environ.get("HOOKE_ISAAC_TRANSPORT", "binary")
        if self.transport not in ("json", "binary"):
            raise ValueError("HOOKE_ISAAC_TRANSPORT must be json or binary")
        gpu = isaac_gpu() if gpu is None else gpu
        if gpu < 0:
            raise ValueError("Isaac GPU index must be nonnegative")
        install = isaac_installation()
        if not (install / "python.sh").is_file():
            raise FileNotFoundError(
                f"Isaac python.sh is missing in {install}; set HOOKE_ISAAC_PATH"
            )
        lock_dir = Path(__file__).resolve().parents[2] / "temp/backend_parity"
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.gpu_lock = (lock_dir / f"gpu-{gpu}.lock").open("a")
            try:
                fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"GPU {gpu} already has a Hooke Isaac job running"
                ) from None
            query = ["nvidia-smi", "-i", str(gpu)]
            memory = int(
                subprocess.check_output(
                    query
                    + ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
            )
            if memory >= 2048:
                raise RuntimeError(
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
            )
            deadline = time.monotonic() + 180
            while not self.socket_path.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f'Isaac worker exited; inspect {self.output / "isaac-worker.log"}'
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("Isaac startup exceeded 180 seconds")
                time.sleep(0.1)
            self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.connection.settimeout(180)
            self.connection.connect(str(self.socket_path))
            self.stream = self.connection.makefile("rwb")
        except BaseException:
            self.close()
            raise

    def call(self, op: str, **kwargs):
        if self.stream is None:
            raise RuntimeError("Isaac worker is closed or has not connected")
        write_message(self.stream, {"op": op, **kwargs}, self.transport)
        response = read_message(self.stream, self.transport)
        if response is None:
            raise RuntimeError("Isaac worker disconnected; inspect its log")
        if not response["ok"]:
            raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
        return response["result"]

    def close(self):
        if self.stream is not None:
            try:
                self.connection.settimeout(5)
                self.call("shutdown")
            except (OSError, EOFError, ValueError, RuntimeError):
                pass
            finally:
                try:
                    self.stream.close()
                except OSError:
                    pass
                self.stream = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        for name in ("log", "gpu_lock"):
            resource = getattr(self, name)
            if resource is not None:
                resource.close()
                setattr(self, name, None)
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
