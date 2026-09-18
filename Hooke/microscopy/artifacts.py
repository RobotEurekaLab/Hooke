"""Atomic public state and image outputs for demo replay and the LAN interface."""

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_image(path, image):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def microscope_frame(task, output, index):
    folder = Path(output) / "microscopy"
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / f"{index:05d}.png", task.microscope_image())
    write_json(folder / "state.json", task.public_state())


def live_microscope_images(task, output):
    """Publish display and tracking channels from the same experiment state."""
    folder = Path(output)
    write_image(folder / "microscope.png", task.microscope_image())
    if "fluorescence" in task.microscope.public_calibration().get("display_channels", []):
        state = task.mechanics.image_state(task.data)
        write_image(folder / "microscope_fluorescence.png", task.microscope.render(state, channel="fluorescence"))


def workstation_views(task):
    cameras = set(task.task_info["camera_mapping"].values())
    return [(view, camera) for view, camera in (
        ("overview", "workstation_overview"), ("closeup", "instrument_closeup"),
        ("sample", "cell_detail"), ("objective", "objective_detail"),
        ("detection", "detection_path_detail")) if camera in cameras]


def workstation_frame(task, renderer, output, *, state=None):
    from PIL import Image
    from microscopy.runtime_visuals import update_scene

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    for name, camera in workstation_views(task):
        update_scene(task, renderer, camera)
        write_image(folder / (name + ".png"), Image.fromarray(renderer.render()))
    live_microscope_images(task, folder)
    write_json(folder / "state.json", task.public_state() if state is None else state)
