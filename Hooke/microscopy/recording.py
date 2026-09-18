"""Encode recorded live observations without running or replaying physics."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from microscopy.artifacts import write_json


def encode_episode(episode, output, fps=4):
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError("Recording FPS must be an integer in 1–30")
    episode, output = Path(episode), Path(output)
    if output.suffix not in (".mp4", ".webm"):
        raise ValueError("Recording output must be MP4 or WebM")
    if output.exists():
        raise FileExistsError(output)
    folders = sorted((episode / "observations").iterdir())
    states = [json.loads((folder / "state.json").read_text()) for folder in folders]
    if not states:
        raise ValueError("The episode has no observations")
    times = np.asarray([state["time_s"] for state in states])
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Observation times must be finite and nondecreasing")
    backend, operation = states[0]["backend"], states[0]["operation"]
    if any((state["backend"], state["operation"], state["generation"], state["session_id"]) !=
           (backend, operation, states[0]["generation"], states[0]["session_id"]) for state in states):
        raise ValueError("A recording must contain observations from one episode")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    codec = (["-c:v", "libx264", "-crf", "22", "-movflags", "+faststart"]
             if output.suffix == ".mp4" else ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                 "-deadline", "realtime", "-cpu-used", "4", "-row-mt", "1"])
    frames = math.ceil((times[-1]-times[0])*fps)+1
    provenance = []
    for folder in folders:
        names = ["overview.png", "microscope.png", "state.json"]
        if (folder/"closeup.png").is_file():
            names.append("closeup.png")
        if (folder/"microscope_fluorescence.png").is_file():
            names.append("microscope_fluorescence.png")
        provenance.append({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                           for name in names})
    log_path = output.with_name(output.name + ".log")
    with log_path.open("w") as log:
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "1680x760", "-framerate", str(fps), "-i", "pipe:0", "-an",
            *codec, "-threads", "2", "-pix_fmt", "yuv420p", str(temporary),
        ], stdin=subprocess.PIPE, stderr=log)
        try:
            for index in range(frames):
                time = min(times[-1], times[0]+index/fps)
                sample = int(np.searchsorted(times, time, side="right")-1)
                state, folder = states[sample], folders[sample]
                image = Image.new("RGB", (1680, 760), (20, 31, 42))
                world = "closeup" if (folder/"closeup.png").is_file() else "overview"
                for name, size, origin in ((world, (960, 720), (0, 40)),
                                           ("microscope", (720, 720), (960, 40))):
                    with Image.open(folder / f"{name}.png") as source:
                        panel = ImageOps.contain(source.convert("RGB"), size)
                    image.paste(panel, (origin[0]+(size[0]-panel.width)//2,
                                       origin[1]+(size[1]-panel.height)//2))
                ImageDraw.Draw(image).text((16, 14),
                    f"HOOKE | {backend} | {operation} | {state['phase']} | observed t={state['time_s']:.3f}s | synthetic microscopy",
                    fill="white")
                fluorescence = folder/"microscope_fluorescence.png"
                if fluorescence.is_file():
                    with Image.open(fluorescence) as source:
                        inset = ImageOps.contain(source.convert("RGB"), (240, 240))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((14, 64, 262, 332), fill=(20, 31, 42), outline=(130, 154, 164))
                    draw.text((20, 70), "Synthetic fluorescence", fill="white")
                    image.paste(inset, (18, 88))
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            if encoder.wait(timeout=30):
                raise RuntimeError(f"Recording failed; see {log_path}")
    temporary.replace(output)
    report = dict(backend=backend, operation=operation, frames=frames, fps=fps,
                  codec="h264" if output.suffix == ".mp4" else "vp9",
                  observation_times_s=times.tolist(), episode=str(episode.absolute()),
                  observation_hashes=provenance, temporal_interpolation=False,
                  fluorescence_inset=any("microscope_fluorescence.png" in entry for entry in provenance),
                  synthetic_microscope=True, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    report.update(volume_unit=states[0].get("volume_unit", "nL"),
                  specimen_kind=states[0].get("sample_scale", {}).get("kind"),
                  specimen_diameters_m=states[0].get("sample_scale", {}).get("diameters_m"))
    write_json(output.with_name(output.name + ".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(encode_episode(args.episode, args.output, args.fps)))


if __name__ == "__main__":
    main()
