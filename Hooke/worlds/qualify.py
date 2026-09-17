"""Run actual world scenes, pictures and independent ballistic checks."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backends.config import isaac_gpu
from backends.evidence import compare_sources
from backends.run import run
from backends.worker_client import IsaacWorker
from worlds.profiles import WORLDS


def assess(folder, profile):
    result = json.loads((folder / "result.json").read_text())
    with np.load(folder / "source/model.npz") as model, np.load(
        folder / "trajectory.npz"
    ) as trajectory:
        meta = json.loads((folder / "source/scene.json").read_text())
        index = meta["names"]["joint"].index("free_cartridge_joint")
        address = int(model["jnt_qposadr"][index])
        dof = int(model["jnt_dofadr"][index])
        origin = model["reset_qpos"][address : address + 3]
        elapsed = float(trajectory["time"][-1])
        position = trajectory["qpos"][-1, address : address + 3]
        velocity = trajectory["qvel"][-1, dof : dof + 3]
        expected = origin + [
            0.06 * elapsed,
            0,
            -0.5 * profile.gravity_m_s2 * elapsed**2,
        ]
        expected_velocity = np.array([0.06, 0, -profile.gravity_m_s2 * elapsed])
        position_error = float(np.max(abs(position - expected)))
        velocity_error = float(np.max(abs(velocity - expected_velocity)))
        checks = {
            "preview_complete": result["status"] == "PREVIEW_COMPLETE",
            "finite_trajectory": bool(
                np.isfinite(trajectory["qpos"]).all()
                and np.isfinite(trajectory["qvel"]).all()
            ),
            "ballistic_position_within_2_5mm": position_error < 0.0025,
            "ballistic_velocity_within_2mm_s": velocity_error < 0.002,
            "real_half_second": abs(elapsed - 0.5) < 1e-6,
            "no_scientific_success_claim": result["source_success"] is False,
            "thermal_clock_matches_physics": abs(
                result["space_environment"]["thermal_witness"]["elapsed_s"] - elapsed
            )
            < 1e-6,
            "actual_images": len(list(folder.rglob("camera_*/*.png"))) >= 4,
        }
    return {
        "world": profile.name,
        "backend": result["backend"],
        "checks": checks,
        "passed": all(checks.values()),
        "position_m": position.tolist(),
        "velocity_m_s": velocity.tolist(),
        "position_error_m": position_error,
        "velocity_error_m_s": velocity_error,
        "source": str(folder / "source"),
        "result_sha256": hashlib.sha256(
            (folder / "result.json").read_bytes()
        ).hexdigest(),
        "environment": result["space_environment"],
        "runtime": result["runtime"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in ("mujoco", "isaac"):
        with ExitStack() as stack:
            worker = (
                stack.enter_context(IsaacWorker(args.output / "worker", args.gpu))
                if backend == "isaac"
                else None
            )
            for profile in WORLDS.values():
                folder = args.output / backend / profile.name
                options = SimpleNamespace(
                    task=profile.task_name,
                    backend=backend,
                    mode="preview",
                    seed=0,
                    output=folder,
                    gpu=args.gpu,
                    seconds=0.5,
                    max_sim_seconds=2.0,
                    no_render=False,
                    science_model=None,
                )
                result = run(options, worker=worker)
                if result["status"] != "PREVIEW_COMPLETE":
                    raise RuntimeError(
                        f"World startup failed: {backend}/{profile.name}: {result.get('error')}"
                    )
                row = assess(folder, profile)
                rows.append(row)
                (args.output / "progress.json").write_text(
                    json.dumps(
                        {"finished": len(rows), "requested": 6, "last": row}, indent=2
                    )
                )
    pairs = [
        {
            "world": profile.name,
            **compare_sources(
                args.output / "mujoco" / profile.name / "source",
                args.output / "isaac" / profile.name / "source",
            ),
        }
        for profile in WORLDS.values()
    ]
    document = {
        "kind": "space_world_scene_qualification",
        "requested": 6,
        "results": rows,
        "source_pairs": pairs,
        "passed": all(row["passed"] for row in rows)
        and all(pair["equal"] for pair in pairs),
        "scope": "Three world scenes, 0.5 seconds of actual rigid-body motion and real RGB. No E1/E2, gas/pressure seal, grasp/latch, terrain data, full thermal or scientific equivalence qualification.",
        "parity_qualified": False,
        "scientific_process_validated": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"passed": document["passed"], "completed": len(rows)}))
    raise SystemExit(0 if document["passed"] else 1)


if __name__ == "__main__":
    main()
