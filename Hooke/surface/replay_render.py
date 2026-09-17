"""Render a qualified trajectory, optionally refreshing validated appearance."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from backends.config import isaac_gpu
from backends.render_settings import RenderSettings
from backends.run import write_json
from backends.worker_client import IsaacWorker
from surface.evidence import expert_passed
from surface.render_contract import check_physics


def render_replay(folder, output, gpu, worker=None, refresh_appearance=False):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    original = folder / "result.json"
    result = json.loads(original.read_text())
    backend = result.get("backend")
    supported = backend == "isaac" or (refresh_appearance and backend == "mujoco")
    if not supported or not expert_passed(result):
        scope = "surface" if refresh_appearance else "native"
        raise ValueError(
            f"Replay rendering requires a qualified {scope} expert episode"
        )
    if output.exists():
        raise FileExistsError("Choose a fresh replay output to preserve evidence")
    settings = RenderSettings.from_environment()
    output.mkdir(parents=True)
    appearance = None
    if refresh_appearance:
        from archetypes.task_catalog import CATALOG
        from backends.baseline import write_snapshot

        task = CATALOG[result["task"]].make_expert()
        task.reset(result["seed"])
        (output / "source").mkdir()
        write_snapshot(task, output / "source")
        appearance = check_physics(folder / "source", output / "source")
    else:
        shutil.copytree(folder / "source", output / "source")
    shutil.copy2(folder / "trajectory.npz", output / "trajectory.npz")
    with np.load(folder / "trajectory.npz", allow_pickle=False) as record:
        times, qpos, qvel = (record[key].copy() for key in ("time", "qpos", "qvel"))
    with np.load(folder / "source/model.npz", allow_pickle=False) as model:
        initial_qpos, initial_qvel = model["reset_qpos"], model["reset_qvel"]
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Recorded trajectory times must increase")
    frames = int(np.ceil(result["simulation_s"] * settings.frames_per_second))
    max_position_error = max_velocity_error = 0.0
    with ExitStack() as stack:
        if backend == "isaac":
            if worker is None:
                worker = stack.enter_context(IsaacWorker(output / "worker", gpu))
            loaded = worker.call(
                "load",
                source=str(output / "source"),
                output=str(output / "isaac"),
                render=True,
                managed_render=True,
            )
        else:
            import mujoco
            from PIL import Image
            from backends.source_renderer import (
                center_directional_shadows,
                mujoco_renderer,
            )

            model = mujoco.MjModel.from_binary_path(str(output / "source/model.mjb"))
            data = mujoco.MjData(model)
            renderer = stack.enter_context(mujoco_renderer(model, gpu))
            meta = json.loads((output / "source/scene.json").read_text())
            cameras = list(dict.fromkeys(meta["task_info"]["camera_mapping"].values()))
        for index in range(frames):
            time = index / settings.frames_per_second
            sample = max(0, int(np.searchsorted(times, time, side="right")) - 1)
            position, velocity = (
                (initial_qpos, initial_qvel)
                if index == 0
                else (qpos[sample], qvel[sample])
            )
            if backend == "isaac":
                state = worker.call(
                    "reset", qpos=position.tolist(), qvel=velocity.tolist()
                )
            else:
                data.qpos[:], data.qvel[:] = position, velocity
                mujoco.mj_forward(model, data)
                state = {"qpos": data.qpos, "qvel": data.qvel}
            position_error = float(np.max(np.abs(np.asarray(state["qpos"]) - position)))
            velocity_error = float(np.max(np.abs(np.asarray(state["qvel"]) - velocity)))
            if position_error > 1e-5 or velocity_error > 1e-4:
                raise RuntimeError("Native replay pose differs from recorded state")
            max_position_error = max(max_position_error, position_error)
            max_velocity_error = max(max_velocity_error, velocity_error)
            if backend == "isaac":
                worker.call("render", visuals={})
            else:
                for camera in cameras:
                    renderer.update_scene(data, camera=camera)
                    center_directional_shadows(renderer, model)
                    camera_id = meta["names"]["camera"].index(camera)
                    images = output / "mujoco" / f"camera_{camera_id}"
                    images.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(renderer.render()).save(images / f"{index:05d}.png")
                if data.time != 0:
                    raise RuntimeError("Source replay advanced the physics clock")
        if backend == "isaac":
            runtime = worker.call("info")
            if runtime["physics_events_since_load"] != 0 or runtime["frames"] != frames:
                raise RuntimeError(
                    "Replay must render all frames without advancing physics"
                )
    result["runtime"]["frames"] = frames
    result["render_settings"] = settings.report()
    if backend == "isaac":
        result["conversion"] = loaded["conversion"]
    result["render_replay"] = dict(
        visualization_mode=(
            "recorded_native_state_replay"
            if backend == "isaac"
            else "recorded_source_state_replay"
        ),
        physics_episode=str(folder),
        physics_result_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        trajectory_sha256=hashlib.sha256(
            (folder / "trajectory.npz").read_bytes()
        ).hexdigest(),
        physics_events_during_replay=0,
        max_qpos_error=max_position_error,
        max_qvel_error=max_velocity_error,
    )
    if appearance is not None:
        result["render_replay"]["appearance_refresh"] = appearance
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    parser.add_argument("--refresh-appearance", action="store_true")
    args = parser.parse_args()
    result = render_replay(
        args.folder, args.output, args.gpu, refresh_appearance=args.refresh_appearance
    )
    print(json.dumps(result["render_replay"]))


if __name__ == "__main__":
    main()
