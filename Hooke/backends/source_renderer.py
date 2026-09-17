"""Original MuJoCo rendering under the same lease as native simulation."""

from contextlib import contextmanager
import os
import subprocess

import numpy as np

from backends.config import isaac_gpu
from backends.gpu_lease import GPUBusy, GPULease
from backends.render_settings import RenderSettings


def center_directional_shadows(renderer, model):
    """Place directional shadow views over the scene, without changing physics.

    A directional light's position only centres its finite OpenGL shadow map.
    Recentring that view avoids clipping artefacts as scenery grows distant.
    """
    for light in renderer.scene.lights[: renderer.scene.nlight]:
        if light.directional and light.castshadow:
            direction = np.asarray(light.dir)
            light.pos[:] = (
                model.stat.center
                - direction / np.linalg.norm(direction) * model.stat.extent
            )


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
            raise GPUBusy(f"GPU {gpu} is busy ({memory} MiB); wait for the running job")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        settings = RenderSettings.from_environment()
        with mujoco.Renderer(
            model,
            height=settings.height if height is None else height,
            width=settings.width if width is None else width,
        ) as renderer:
            yield renderer
