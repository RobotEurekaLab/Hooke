"""Run complete surface missions with duration-matched no-action controls."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from backends.config import isaac_gpu
from backends.run import run, write_json
from backends.worker_client import IsaacWorker
from archetypes.task_catalog import CATALOG
from surface.profiles import MISSIONS
from surface.evidence import control_passed, expert_passed


def qualify(output, backends, worlds, gpu, render=True, scenario="rover"):
    output = Path(output).resolve()
    folders = [
        output / f"{world}-{backend}{suffix}"
        for backend in backends
        for world in worlds
        for suffix in ("", "-no-action")
    ]
    existing = next((path for path in folders if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"Preserve earlier evidence; choose a fresh output: {existing}"
        )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in backends:
        with ExitStack() as stack:
            worker = (
                stack.enter_context(IsaacWorker(output / "worker", gpu))
                if backend == "isaac"
                else None
            )
            for world in worlds:
                folder = output / f"{world}-{backend}"
                args = SimpleNamespace(
                    task=(
                        f"space_{world}_humanoid_rover"
                        if scenario == "team"
                        else MISSIONS[world].task_name
                    ),
                    backend=backend,
                    mode="expert",
                    seed=0,
                    seconds=2.0,
                    max_sim_seconds=CATALOG[
                        (
                            f"space_{world}_humanoid_rover"
                            if scenario == "team"
                            else MISSIONS[world].task_name
                        )
                    ].max_sim_seconds,
                    gpu=gpu,
                    no_render=not render,
                    output=folder,
                    physics_options=None,
                    science_model=None,
                )
                result = run(args, worker)
                passed = expert_passed(result)
                row = dict(
                    world=world,
                    backend=backend,
                    folder=str(folder),
                    status=result["status"],
                    expert_passed=passed,
                )
                rows.append(row)
                write_json(output / "qualification_runs.json", dict(cases=rows))
                print(f"{world} / {backend}: {result['status']}", flush=True)
                if not passed:
                    continue
                args.mode = "no_action"
                args.output = folder.with_name(folder.name + "-no-action")
                args.seconds = float(result["simulation_s"])
                args.no_render = True
                control = run(args, worker)
                row["control_passed"] = control_passed(result, control, args.output)
                print(
                    f"{world} / {backend}: no-action {control['status']}, rejected={row['control_passed']}",
                    flush=True,
                )
                write_json(output / "qualification_runs.json", dict(cases=rows))
    complete = len(rows) == len(backends) * len(worlds) and all(
        row["expert_passed"] and row.get("control_passed", False) for row in rows
    )
    write_json(
        output / "qualification_runs.json",
        dict(
            cases=rows,
            complete=complete,
            deterministic_layout=True,
            randomized_scene_coverage=False,
        ),
    )
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("mujoco", "isaac"),
        default=["mujoco", "isaac"],
    )
    parser.add_argument(
        "--worlds", nargs="+", choices=tuple(MISSIONS), default=list(MISSIONS)
    )
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    parser.add_argument("--scenario", choices=("rover", "team"), default="rover")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if len(set(args.backends)) != len(args.backends) or len(set(args.worlds)) != len(
        args.worlds
    ):
        parser.error("Choose each backend and world once")
    raise SystemExit(
        0
        if qualify(
            args.output,
            args.backends,
            args.worlds,
            args.gpu,
            not args.no_render,
            args.scenario,
        )
        else 1
    )


if __name__ == "__main__":
    main()
