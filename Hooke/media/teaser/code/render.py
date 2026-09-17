"""Render all scenes to video, then mux with the audio mix."""
import importlib
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import imageio_ffmpeg
import numpy as np

from common import FPS, W, H, post, smooth
from timeline import SCENES, TOTAL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.dirname(HERE)
PART1 = dict(part=1, grade=((0.0, 0.004, 0.012), (1.0, 1.0, 1.0), (1.06, 1.0, 0.9)), bloom_amt=0.55, flicker=0.012)
PART2 = dict(part=2, grade=((0.0, 0.004, 0.01), (1.0, 1.0, 1.0), (0.95, 1.0, 1.08)), bloom_amt=0.7, chroma=2, weave=False)


def frame_job(args):
    name, start, dur, fi = args
    t = fi / FPS - start
    t = min(max(t, 0.0), dur - 1e-3)
    if name == "end":
        mod = importlib.import_module("scene8")
        arr = mod.render_end(t)
        cfg = PART2
    else:
        mod = importlib.import_module("scene" + name[1:])
        arr = mod.render(t)
        cfg = PART1 if name in ("s1", "s2", "s3", "s4") else PART2
    # scene-boundary treatments
    if name == "s1":
        arr = arr * float(smooth(t / 0.5)) * float(1 - smooth((t - (dur - 0.35)) / 0.35))
    elif name == "s2":
        arr = arr * float(smooth(t / 0.25))
    elif name == "s4":
        arr = arr * float(0.35 + 0.65 * smooth(t / 0.25))
    elif name == "s6":
        arr = arr + math.exp(-t * 12) * np.array([0.3, 0.8, 1.0], np.float32) * 0.8
    elif name == "s8":
        if t < 0.25:
            from scene5 import glitch
            arr = glitch(arr, 1 - t / 0.25, fi)
        arr = arr * float(1 - smooth((t - (dur - 0.5)) / 0.5))
    out = post(arr, fi, **cfg)
    return (out * 255 + 0.5).astype(np.uint8).tobytes()


def encode_scene(name, start, dur, workers):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    seg = os.path.join(HERE, "build", f"seg_{name}.mp4")
    if os.path.exists(seg):
        return seg
    tmp = seg + ".part.mp4"
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                             "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = int(round(TOTAL * FPS))
    f0, f1 = int(round(start * FPS)), min(total, int(round((start + dur) * FPS)))
    jobs = [(name, start, dur, fi) for fi in range(f0, f1)]
    with Pool(workers, maxtasksperchild=40) as pool:
        for buf in pool.imap(frame_job, jobs, chunksize=1):
            proc.stdin.write(buf)
    proc.stdin.close()
    proc.wait()
    os.replace(tmp, seg)
    return seg


def main():
    only = sys.argv[1:]
    workers = int(os.environ.get("WORKERS", "6"))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    t0 = time.time()
    segs = []
    for name, start, dur, _ in SCENES:
        if only and name not in only:
            continue
        segs.append(encode_scene(name, start, dur, workers))
        print(f"{name} done, {time.time() - t0:.0f}s elapsed", flush=True)
    if only:
        return
    lst = os.path.join(HERE, "build", "segs.txt")
    with open(lst, "w") as f:
        for s_ in segs:
            f.write("file '" + s_.replace(os.sep, "/") + "'\n")
    final = os.path.join(OUT_DIR, "Robot_Scientists_Teaser.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", os.path.join(HERE, "build", "mix.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    print("wrote", final, flush=True)


if __name__ == "__main__":
    main()
