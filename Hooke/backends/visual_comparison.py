"""Quantify actual paired RGB images without claiming pixel parity."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def compare(first, second):
    a, b = np.asarray(Image.open(first).convert("RGB"), dtype=float), np.asarray(
        Image.open(second).convert("RGB"), dtype=float
    )
    if a.shape != b.shape:
        raise ValueError("Paired image dimensions differ")
    error = a - b
    mse = float(np.mean(error**2))
    return dict(
        dimensions=list(a.shape),
        mean_absolute_error_255=float(np.mean(abs(error))),
        root_mean_square_error_255=math.sqrt(mse),
        psnr_db=None if mse == 0 else 10 * math.log10(255**2 / mse),
        mean_rgb={
            "mujoco": a.mean(axis=(0, 1)).tolist(),
            "isaac": b.mean(axis=(0, 1)).tolist(),
        },
        image_sha256={
            "mujoco": hashlib.sha256(Path(first).read_bytes()).hexdigest(),
            "isaac": hashlib.sha256(Path(second).read_bytes()).hexdigest(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for image in sorted(args.mujoco.glob("*-camera-*.png")):
        paired = args.isaac / image.name
        if not paired.is_file():
            raise ValueError(f"Missing paired image: {image.name}")
        rows.append(dict(image=image.name, **compare(image, paired)))
    if not rows:
        raise ValueError("No paired images found")
    report = dict(
        kind="paired_actual_rgb_difference",
        pairs=len(rows),
        results=rows,
        pixel_parity_qualified=False,
        scope="Phase endpoints include different actual motion and rendering; error does not isolate shader differences.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                pairs=len(rows),
                mean_absolute_error_255=float(
                    np.mean([r["mean_absolute_error_255"] for r in rows])
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
