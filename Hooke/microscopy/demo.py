"""Render three contact/flow experiments, videos and a standalone MJCF asset."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from backends.source_renderer import mujoco_renderer
from microscopy.runtime_visuals import update_scene
from microscopy import OPERATIONS
from microscopy.artifacts import workstation_frame, write_image, write_json
from microscopy.tasks import make_task


def composite(task, renderer):
    update_scene(task, renderer, "cell_detail" if task.volume_unit == "pL" else "instrument_closeup")
    view = Image.fromarray(renderer.render())
    micro = task.microscope_image().resize((view.height, view.height))
    image = Image.new("RGB", (view.width + micro.width, view.height + 40), (20, 31, 42))
    image.paste(view, (0, 40))
    image.paste(micro, (view.width, 40))
    ImageDraw.Draw(image).text((16, 14),
        f"HOOKE MICROSCOPY  |  {task.operation}  |  {task.public_state()['phase']}  |  t={task.data.time:.2f}s",
        fill="white")
    return image


def run(output, gpu, seed=0, fps=8, operations=OPERATIONS[:3]):
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(seed=seed, backend="mujoco", experiments={})
    for operation in operations:
        task = make_task(operation)
        (output / (operation+"_workstation.xml")).write_text(task.spec.to_xml())
        with mujoco_renderer(task.model, gpu, width=960, height=720) as renderer:
            task.reset(seed)
            initial = {"cell_injection": "cell_initial", "suction_injection": "suction_initial"}.get(operation, "initial")
            workstation_frame(task, renderer, output / initial)

            folder = output / operation
            folder.mkdir(parents=True, exist_ok=True)
            original_step = task.manager.step
            next_frame = 0.0
            previous_phase = None
            with (folder / "video.log").open("w") as log:
                encoder = subprocess.Popen([
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "1680x760", "-framerate", str(fps), "-i", "pipe:0", "-an",
                    "-c:v", "libx264", "-threads", "2", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(folder / "demo.mp4"),
                ], stdin=subprocess.PIPE, stderr=log)
                def capture():
                    nonlocal next_frame, previous_phase
                    frame = composite(task, renderer)
                    encoder.stdin.write(np.asarray(frame).tobytes())
                    next_frame += 1 / fps
                    phase = task.public_state()["phase"]
                    if phase != previous_phase:
                        write_image(folder / (phase + ".png"), frame)
                        previous_phase = phase
                def step():
                    original_step()
                    if task.data.time >= next_frame:
                        capture()
                task.manager.step = step
                try:
                    capture()
                    task.execute()
                    capture()
                finally:
                    task.manager.step = original_step
                    encoder.stdin.close()
                    if encoder.wait(timeout=30):
                        raise RuntimeError(f"Video encoding failed; see {folder / 'video.log'}")
            workstation_frame(task, renderer, folder)
            report = task.microscopy_report()
            report.update(success=task.check(), simulation_s=float(task.data.time), phases=task.phase_history)
            write_json(folder / "result.json", report)
            summary["experiments"][operation] = report
            print(json.dumps(dict(operation=operation, success=task.check(), checks=report["checks"])), flush=True)
    filename = {("cell_injection",): "cell_summary.json", ("suction_injection",): "suction_summary.json"}.get(operations, "summary.json")
    write_json(output / filename, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--operation", choices=OPERATIONS)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("FPS must be from 1 to 30")
    operations = (args.operation,) if args.operation else OPERATIONS[:3]
    summary = run(args.output.resolve(), args.gpu, args.seed, args.fps, operations)
    raise SystemExit(0 if all(report["success"] for report in summary["experiments"].values()) else 1)


if __name__ == "__main__":
    main()
