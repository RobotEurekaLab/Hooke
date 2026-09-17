"""Original MuJoCo rendering under the same lease as native simulation."""

from contextlib import contextmanager
import os
import subprocess

from backends.config import isaac_gpu
from backends.gpu_lease import GPUBusy, GPULease
from backends.render_settings import RenderSettings


@contextmanager
def mujoco_renderer(model, gpu=None, *, width=None, height=None):
    import mujoco

    gpu = isaac_gpu() if gpu is None else gpu
    with GPULease(gpu):
        memory = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if memory >= 2048:
            raise GPUBusy(
                f"GPU {gpu} is busy ({memory} MiB); wait for the running job"
            )
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        settings = RenderSettings.from_environment()
        with mujoco.Renderer(
            model,
            height=settings.height if height is None else height,
            width=settings.width if width is None else width,
        ) as renderer:
            yield renderer
