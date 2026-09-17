"""Fetch a pinned BSD-3-Clause G1 policy and export its arrays for NumPy.

Torch is needed once for export, not by the simulator or the Web UI.
"""

import argparse
import hashlib
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

from surface.artifacts import GAIT_ROOT
from surface.recurrent import RecurrentPolicy

REVISION = "276801e46c5d433564f24658bac64f254b7d2d4b"
REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_gym"
DESTINATION = GAIT_ROOT
FILES = {
    "LICENSE": (
        "LICENSE",
        "aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1",
    ),
    "motion.pt": (
        "deploy/pre_train/g1/motion.pt",
        "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d",
    ),
    "g1.yaml": (
        "deploy/deploy_mujoco/configs/g1.yaml",
        "73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb",
    ),
}


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    certificate = Path("/etc/ssl/certs/ca-certificates.crt")
    context = ssl.create_default_context(
        cafile=str(certificate) if certificate.is_file() else None
    )
    for name, (relative, digest) in FILES.items():
        path = destination / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        url = f"https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/{REVISION}/{relative}"
        with urllib.request.urlopen(url, context=context, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Pinned G1 asset checksum mismatch: {name}")
        with tempfile.NamedTemporaryFile(
            "wb", dir=destination, delete=False
        ) as temporary:
            temporary.write(data)
        Path(temporary.name).replace(path)


def export(destination):
    import torch

    for name, (_, digest) in FILES.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned G1 asset checksum mismatch: {name}")
    model = torch.jit.load(str(destination / "motion.pt"), map_location="cpu")
    weights = {name: value.detach().numpy() for name, value in model.named_parameters()}
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".npz", dir=destination, delete=False
    ) as temporary:
        np.savez(temporary, **weights)
    candidate = Path(temporary.name)
    policy = RecurrentPolicy(candidate)
    model.hidden_state.zero_()
    model.cell_state.zero_()
    maximum = 0.0
    observations = np.random.default_rng(4).normal(size=(200, 47)).astype(np.float32)
    with torch.no_grad():
        for observation in observations:
            expected = model(torch.from_numpy(observation)[None]).numpy()[0]
            maximum = max(
                maximum, float(np.max(np.abs(policy(observation) - expected)))
            )
    if maximum >= 2e-5:
        candidate.unlink()
        raise ValueError(
            "NumPy G1 actor does not reproduce the source recurrent policy"
        )
    candidate.replace(destination / "policy.npz")
    report = dict(
        repository=REPOSITORY,
        revision=REVISION,
        license="BSD-3-Clause",
        files={name: digest for name, (_, digest) in FILES.items()},
        derived_policy_sha256=hashlib.sha256(
            (destination / "policy.npz").read_bytes()
        ).hexdigest(),
        policy_arrays_sha256=policy.sha256,
        recurrent_comparison_steps=200,
        numpy_torch_max_absolute_error=maximum,
        modification="NumPy array export; simulation uses gravity-scaled timing, observations, gains and friction",
    )
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument(
        "--torch-python", help="Python interpreter with Torch for the one-time export"
    )
    parser.add_argument("--export-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if args.export_only:
        export(destination)
    else:
        fetch(destination)
        subprocess.run(
            [
                args.torch_python or sys.executable,
                "-m",
                "surface.gait_assets",
                "--export-only",
                "--destination",
                str(destination),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )


if __name__ == "__main__":
    main()
