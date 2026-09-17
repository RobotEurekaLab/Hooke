"""Run continuous experiments, preserve failures and assess private truth here.

Private mass/profile checks belong to this evaluator, not instrument analysis.
Reuse an Isaac process between episodes; each scene receives a fresh load/reset.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import secrets
from types import SimpleNamespace

import numpy as np

from backends.config import isaac_gpu
from backends.run import run, write_json
from backends.worker_client import IsaacWorker
from experiments import OPERATIONS


def assess(result, output):
    report = result.get("space_experiment", {})
    passed = (
        result.get("status") == "TASK_SUCCEEDED"
        and report.get("source_success") is True
    )
    checks = {
        "continuous_task": passed,
        "within_limit": result.get("within_declared_time_limit") is True,
    }
    conclusion = report.get("conclusion")
    metrics = {}
    if report.get("operation") == "mass_measurement" and conclusion:
        # This archived field is read only by the separate release evaluator.
        metadata = json.loads((output / "source/scene.json").read_text())
        body = metadata["names"]["body"].index("sample_candidate")
        with np.load(output / "source/model.npz") as model:
            truth = float(model["body_mass"][body])
        error = abs(conclusion["total_mass_kg"] - truth)
        metrics.update(absolute_mass_error_kg=error, relative_mass_error=error / truth)
        checks["mass_within_3g"] = error <= 0.003
        checks["three_calibration_records"] = len(conclusion["measurement_ids"]) == 3
    if report.get("operation") == "spectral_measurement" and conclusion:
        truth = json.loads((output / "evaluator_truth.json").read_text())[
            "candidate_spectral_profile"
        ]
        expected = None if truth == 3 else f"analytic_profile_{truth}"
        checks["correct_match_or_holdout_abstention"] = conclusion["match"] == expected
        checks["four_spectral_records"] = len(conclusion["measurement_ids"]) == 4
    runtime = result.get("runtime", {})
    if result["backend"] == "isaac":
        checks["one_native_event_per_step"] = runtime.get(
            "physics_events_since_load"
        ) == result.get("steps")
        checks["native_fk_within_5mm"] = (
            result.get("max_fk_position_error_m", 1) < 0.005
        )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        scope="robot_handling_and_declared_spring_or_analytic_spectral_model",
    )


def qualify(
    output, backends, seeds, render, controls, worlds, operations, gpu, campaign=None
):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = next(
        (
            output / backend / f"space_{world}_{operation}" / f"seed_{seed:03d}"
            for backend in backends
            for world in worlds
            for operation in operations
            for seed in seeds
            if (
                output / backend / f"space_{world}_{operation}" / f"seed_{seed:03d}"
            ).exists()
        ),
        None,
    )
    if existing is not None:
        raise FileExistsError(
            f"Preserve existing episode evidence; choose a fresh output: {existing}"
        )
    private_file = output / "private_campaign.json"
    if private_file.is_file():
        campaign_key = json.loads(private_file.read_text())["key"]
        if campaign and campaign_key != json.loads(campaign.read_text())["key"]:
            raise ValueError("Existing output belongs to a different private campaign")
    else:
        campaign_key = (
            json.loads(campaign.read_text())["key"]
            if campaign
            else (os.environ.get("HOOKE_SCIENCE_CAMPAIGN_KEY") or secrets.token_hex(32))
        )
        descriptor = os.open(private_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(dict(key=campaign_key), stream)
    os.environ["HOOKE_SCIENCE_CAMPAIGN_KEY"] = campaign_key
    records = []
    for backend in backends:
        context = (
            IsaacWorker(output / "worker", gpu)
            if backend == "isaac"
            else nullcontext(None)
        )
        with context as worker:
            for world in worlds:
                for operation in operations:
                    task = f"space_{world}_{operation}"
                    for seed in seeds:
                        folder = output / backend / task / f"seed_{seed:03d}"
                        args = SimpleNamespace(
                            task=task,
                            backend=backend,
                            seed=seed,
                            mode="expert",
                            seconds=2,
                            max_sim_seconds=120,
                            gpu=gpu,
                            no_render=not (render and seed == seeds[0]),
                            output=folder,
                            science_model=None,
                            physics_options=None,
                        )
                        result = run(args, worker)
                        assessment = assess(result, folder)
                        write_json(folder / "experiment_assessment.json", assessment)
                        row = dict(
                            task=task,
                            world=world,
                            operation=operation,
                            backend=backend,
                            seed=seed,
                            folder=str(folder),
                            status=result["status"],
                            simulation_s=result.get("simulation_s"),
                            total_wall_s=result.get("total_wall_s"),
                            result_sha256=hashlib.sha256(
                                (folder / "result.json").read_bytes()
                            ).hexdigest(),
                            **assessment,
                        )
                        records.append(row)
                        if controls:
                            control_folder = folder.with_name(
                                folder.name + "_no_action"
                            )
                            args.mode, args.output = "no_action", control_folder
                            args.seconds = max(float(result.get("simulation_s", 2)), 2)
                            args.no_render = True
                            control = run(args, worker)
                            row["no_action"] = dict(
                                folder=str(control_folder),
                                status=control["status"],
                                false_success=bool(control.get("source_success")),
                                simulation_s=control.get("simulation_s"),
                            )
                            row["passed"] &= control[
                                "status"
                            ] == "CONTROL_COMPLETE" and not control.get(
                                "source_success"
                            )
                        write_json(
                            output / "summary.json",
                            dict(
                                schema_version=1,
                                cases=records,
                                passed=sum(r["passed"] for r in records),
                                total=len(records),
                                complete=False,
                                fidelity="dry_rigid_cartridges_spring_encoder_and_analytic_spectral_test_profiles",
                            ),
                        )
                        print(
                            json.dumps(
                                dict(
                                    task=task,
                                    backend=backend,
                                    seed=seed,
                                    passed=row["passed"],
                                    case_count=len(records),
                                )
                            ),
                            flush=True,
                        )
    summary = dict(
        schema_version=1,
        cases=records,
        passed=sum(r["passed"] for r in records),
        total=len(records),
        complete=True,
        fidelity="dry_rigid_cartridges_spring_encoder_and_analytic_spectral_test_profiles",
        limitations=[
            "no_gas_seal_or_pressure_solver",
            "no_real_planetary_composition",
            "no_full_srb_isaac_lab_migration",
            "no_arbitrary_mjcf_or_dynamic_equivalence_claim",
        ],
    )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("mujoco", "isaac"),
        default=["mujoco", "isaac"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--worlds",
        nargs="+",
        choices=("orbital", "lunar", "martian"),
        default=["orbital", "lunar", "martian"],
    )
    parser.add_argument(
        "--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS)
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    parser.add_argument(
        "--campaign",
        type=Path,
        help="Host-only shared campaign key for paired episodes",
    )
    args = parser.parse_args()
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(s < 0 for s in args.seeds)
    ):
        parser.error("Unique nonnegative seeds required")
    summary = qualify(
        args.output,
        args.backends,
        args.seeds,
        args.render,
        args.controls,
        args.worlds,
        args.operations,
        args.gpu,
        args.campaign,
    )
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
