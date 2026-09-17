"""Actual source/native collision checks for downloaded terrain crops."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from backends.closed_loop import PhysXTaskAdapter
from backends.config import isaac_gpu
from backends.extended_contracts import fixture
from backends.run import write_json
from backends.source_renderer import mujoco_renderer
from backends.worker_client import IsaacWorker
from PIL import Image
from worlds.build import camera
from worlds.pds_terrain import SOURCE_ROOT, add_heightfield, load_crop
from worlds.profiles import WORLDS


def run_case(world, backend, directory, folder, worker=None, render=False, gpu=6):
    height, metadata = load_crop(directory)
    root = ET.Element("mujoco", model=f"pds_{world}_terrain_probe")
    ET.SubElement(
        root,
        "option",
        timestep=".002",
        gravity=f"0 0 {-WORLDS[world].gravity_m_s2}",
        integrator="implicitfast",
    )
    ET.SubElement(root, "asset")
    body = ET.SubElement(root, "worldbody")
    ET.SubElement(
        body,
        "light",
        directional="true",
        pos="0 0 30",
        dir="0 0 -1",
        diffuse=".8 .8 .8",
    )
    add_heightfield(root, directory)
    probe = ET.SubElement(
        body, "body", name="guided_probe", pos=f"0 0 {float(height.max())+.5}"
    )
    ET.SubElement(
        probe, "joint", name="probe_slide", type="slide", axis="0 0 1", damping=".005"
    )
    ET.SubElement(
        probe,
        "geom",
        name="probe",
        type="sphere",
        size=".12",
        mass=".1",
        rgba=".9 .58 .1 1",
        solref=".005 1",
    )
    center = float(height.mean())
    camera(body, "terrain_overview", (24, -26, center + 24), (0, 0, center), 45)
    task = fixture(ET.tostring(root, encoding="unicode"), folder, f"pds_{world}")
    task.task_info["camera_mapping"] = {"overview": "terrain_overview"}
    adapter = (
        PhysXTaskAdapter(
            task,
            worker,
            folder,
            render=render,
            managed_render=render,
            report_progress=False,
        )
        if backend == "isaac"
        else None
    )
    contact_steps = 0
    rows = []
    for _ in range(3000):
        if adapter:
            adapter.step(task.model, task.data)
        else:
            mujoco.mj_step(task.model, task.data)
        contact_steps += bool(task.data.ncon)
        rows.append(
            [float(task.data.time), float(task.data.qpos[0]), float(task.data.qvel[0])]
        )
    if render:
        if adapter:
            worker.call("render", visuals={})
        else:
            with mujoco_renderer(task.model, gpu) as renderer:
                renderer.update_scene(task.data, camera="terrain_overview")
                folder.joinpath("mujoco/camera_0").mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(
                    folder / "mujoco/camera_0/00000.png"
                )
    finite = bool(
        np.isfinite(task.data.qpos).all() and np.isfinite(task.data.qvel).all()
    )
    result = dict(
        world=world,
        backend=backend,
        passed=finite and contact_steps > 500 and abs(float(task.data.qvel[0])) < 0.01,
        contact_steps=contact_steps,
        finite=finite,
        simulation_s=float(task.data.time),
        final_height_m=float(task.data.xpos[1, 2]),
        final_speed_m_s=float(task.data.qvel[0]),
        source_resolution_m=metadata["source_spacing_m"],
        product_id=metadata["product"]["product_id"],
        source_label_sha256=metadata["label_sha256"],
        crop_sha256=metadata["height_sha256"],
        scope="guided_collision_probe_on_real_pds_crop_not_robot_or_geological_validation",
    )
    if adapter:
        info = worker.call("info")
        result["one_native_event_per_step"] = info["physics_events_since_load"] == 3000
        result["passed"] &= result["one_native_event_per_step"]
    np.savez_compressed(folder / "trajectory.npz", time_qpos_qvel=np.asarray(rows))
    write_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terrain", type=Path, default=SOURCE_ROOT / "assets/space/terrain"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("mujoco", "isaac"),
        default=["mujoco", "isaac"],
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    args = parser.parse_args()
    results = []
    for backend in args.backends:
        context = (
            IsaacWorker(args.output / "worker", args.gpu)
            if backend == "isaac"
            else nullcontext(None)
        )
        with context as worker:
            for world in ("lunar", "martian"):
                result = run_case(
                    world,
                    backend,
                    args.terrain / world,
                    args.output / backend / world,
                    worker,
                    args.render,
                    args.gpu,
                )
                results.append(result)
                write_json(
                    args.output / "summary.json",
                    dict(
                        results=results,
                        passed=all(r["passed"] for r in results),
                        complete=False,
                    ),
                )
                print(json.dumps(result), flush=True)
    pairs = []
    for world in ("lunar", "martian"):
        rows = [r for r in results if r["world"] == world]
        if len(rows) == 2:
            error = abs(rows[0]["final_height_m"] - rows[1]["final_height_m"])
            pairs.append(
                dict(world=world, final_height_error_m=error, passed=error < 0.02)
            )
    summary = dict(
        results=results,
        pairs=pairs,
        passed=all(r["passed"] for r in [*results, *pairs]),
        complete=True,
        qualification="terrain_crop_and_guided_collision_only_not_default_experiment_terrain",
    )
    write_json(args.output / "summary.json", summary)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
