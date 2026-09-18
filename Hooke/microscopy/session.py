"""Own one live experiment, its physics backend and paired observation frames."""

from contextlib import ExitStack
from pathlib import Path
import shutil
from uuid import uuid4

from microscopy.artifacts import workstation_frame, workstation_views, live_microscope_images, write_json
from microscopy.tasks import make_task
from microscopy.control import apply_request, experiment_request
from microscopy import BACKENDS


class ExperimentSession:
    def __init__(self, output, backend, gpu, capture_fps):
        if backend not in BACKENDS:
            raise ValueError("Unknown microscopy backend")
        self.output = Path(output)
        self.backend, self.gpu, self.capture_fps = backend, gpu, capture_fps
        self.task = self.worker = self.adapter = self.renderer = self.visuals = None
        self.resources = ExitStack()
        self.session_id = uuid4().hex
        self.generation = 0
        self.last_capture = -1.
        self.runtime = None
        self.frame_times = []
        self.episode = None

    def close(self):
        try:
            self.resources.close()
        finally:
            if self.worker is not None:
                self.worker.close()
                self.worker = None

    def load(self, operation, seed):
        # Reset can change static cell geometry as well as joint positions.
        # Export a fresh native scene so both engines receive the same seed.
        self.resources.close()
        self.task = self.adapter = self.renderer = self.visuals = None
        write_json(self.output / "state.json", dict(phase="initializing", backend=self.backend,
                   operation=operation, session_id=self.session_id))
        task = make_task(operation)
        task.reset(seed)
        task.task_info.update(backend=self.backend,
                              physics_engine="PhysX" if self.backend == "isaac" else "MuJoCo")
        self.generation += 1
        episode = self.output / "sessions" / self.session_id / f"{self.generation:05d}"
        episode.mkdir(parents=True, exist_ok=False)
        self.episode = episode
        self.frame_times = []
        if self.backend == "isaac":
            from backends.closed_loop import PhysXTaskAdapter
            from backends.worker_client import IsaacWorker
            from backends.visual_state import LiveVisuals

            if self.worker is None:
                self.worker = IsaacWorker(episode.parent, self.gpu)
            self.adapter = self.resources.enter_context(PhysXTaskAdapter(
                task, self.worker, episode, render=True, report_progress=False,
                managed_render=True, physics_options=getattr(task, "native_physics_options", None),
            ))
            self.visuals = LiveVisuals(task, episode / "visuals")
            self.resources.callback(self.visuals.close)
        else:
            from backends.source_renderer import mujoco_renderer

            self.renderer = self.resources.enter_context(
                mujoco_renderer(task.model, self.gpu, width=960, height=720))
        self.task = task
        self.last_capture = -1.
        original_step = task.manager.step

        def step():
            original_step()
            if task.data.time - self.last_capture >= 1 / self.capture_fps:
                self.capture()

        task.manager.step = step

    def public_state(self):
        return dict(self.task.public_state(), backend=self.backend,
                    physics_engine="PhysX" if self.backend == "isaac" else "MuJoCo",
                    world_image_source="Isaac RTX" if self.backend == "isaac" else "MuJoCo EGL",
                    microscope_image_source="synthetic_object_space",
                    session_id=self.session_id, generation=self.generation,
                    native_runtime=self.runtime if self.backend == "isaac" else None)

    def capture(self):
        if self.backend == "isaac":
            result = self.worker.call("render", visuals=self.visuals.snapshot())
            self.runtime = self.worker.call("info")
            for view, camera in workstation_views(self.task):
                camera_id = self.task.model.camera(camera).id
                source = self.adapter.output / "isaac" / f"camera_{camera_id}" / f"{result['frames']-1:05d}.png"
                destination = self.output / f"{view}.png"
                temporary = destination.with_suffix(".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
            live_microscope_images(self.task, self.output)
            write_json(self.output / "state.json", self.public_state())
        else:
            workstation_frame(self.task, self.renderer, self.output, state=self.public_state())
        frame = len(self.frame_times)
        folder = self.episode / "observations" / f"{frame:05d}"
        folder.mkdir(parents=True, exist_ok=False)
        views = ["microscope"]+[view for view, _ in workstation_views(self.task)]
        if "fluorescence" in self.task.microscope.public_calibration().get("display_channels", []):
            views.append("microscope_fluorescence")
        for view in views:
            shutil.copyfile(self.output / f"{view}.png", folder / f"{view}.png")
        write_json(folder / "state.json", self.public_state())
        self.frame_times.append(float(self.task.data.time))
        write_json(self.episode / "frames.json", dict(backend=self.backend, time_s=self.frame_times))
        self.last_capture = float(self.task.data.time)

    def command(self, request, default_operation="push"):
        command = request.get("command")
        if command not in ("reset", "demo", "step", "autofocus"):
            raise ValueError("Unknown workstation command")
        if command in ("reset", "demo"):
            # Validate before replacing the current scene or advancing physics.
            operation, seed = experiment_request(self.task, request, default_operation)
            self.load(operation, seed)
            if command == "demo":
                self.task.execute()
        else:
            if self.task is None:
                self.load(default_operation, 0)
            apply_request(self.task, request)
        self.capture()
        result = self.task.microscopy_report()
        result["execution"] = dict(backend=self.backend, session_id=self.session_id,
                                   generation=self.generation, native_runtime=self.runtime)
        if self.adapter is not None:
            from microscopy.native_trace import archive_native_trace
            result["native_trace"] = archive_native_trace(self.adapter, self.runtime, self.episode)
        write_json(self.output / "result.json", result)
        write_json(self.episode / "result.json", result)
        return dict(ok=True, state=self.public_state(), checks=result["checks"])
