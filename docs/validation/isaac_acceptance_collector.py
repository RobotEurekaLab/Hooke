"""Collect saved acceptance evidence without launching either simulation backend."""

import json
import sys
from datetime import datetime, timezone
from statistics import median
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "temp/backend_parity"
sys.path.insert(0, str(ROOT / "Hooke"))
from backends.evidence import compare_sources, file_sha256
from backends.matrix import summarize


def read(path):
    return json.loads(path.read_text())


def save(name, report):
    path = ROOT / "docs/validation" / name
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def batch(path):
    manifest = read(path / "manifest.json")
    report = read(path / "report.json")
    rows = []
    for row in report["results"]:
        result_path = path / row["evidence"]
        result = read(result_path)
        result["evidence"] = str(result_path.relative_to(ROOT))
        result["result_sha256"] = file_sha256(result_path)
        trajectory = result_path.parent / "trajectory.npz"
        if trajectory.exists():
            result["trajectory_sha256"] = file_sha256(trajectory)
        rows.append(result)
    evidence = dict(
        path=str(path.relative_to(ROOT)),
        parameters=manifest["parameters"],
        provenance={
            k: v for k, v in manifest["provenance"].items() if k != "input_files"
        },
        input_file_count=len(manifest["provenance"]["input_files"]),
        manifest_sha256=file_sha256(path / "manifest.json"),
        report_sha256=file_sha256(path / "report.json"),
    )
    return rows, evidence


def collect(task, paths, control_seconds):
    rows, evidence = [], []
    for path in paths:
        part, proof = batch(path)
        rows.extend(r for r in part if r["task"] == task)
        evidence.append(proof)
    keys = [(r["backend"], r["seed"], r["mode"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate acceptance episode: " + task)
    parameters = dict(
        tasks=[task],
        seeds=list(range(10)),
        modes=["expert", "no_action"],
        backends=["mujoco", "isaac"],
    )
    expected = {
        (b, s, m)
        for b in parameters["backends"]
        for s in parameters["seeds"]
        for m in parameters["modes"]
    }
    if not set(keys) <= expected:
        raise ValueError("Unexpected seed/backend/mode: " + task)
    compact = [
        {
            k: v
            for k, v in r.items()
            if k not in ("conversion", "runtime", "final_qpos", "final_contact_pairs")
        }
        for r in rows
    ]
    report = summarize(compact, parameters)
    checks = []
    for r in rows:
        if r["mode"] == "expert":
            passed = (
                r["status"] == "TASK_SUCCEEDED"
                and r.get("source_success") is True
                and r.get("assessment", {}).get("success_within_time_limit") is True
            )
            if task == "pipette_transfer":
                passed = (
                    passed and r.get("volume_assessment", {}).get("success") is True
                )
        else:
            passed = (
                r["status"] == "CONTROL_COMPLETE"
                and r.get("source_success") is False
                and r.get("assessment", {}).get("success") is False
                and r.get("simulation_s", 0) >= control_seconds - 1e-6
            )
            if task == "pipette_transfer":
                passed = (
                    passed and r.get("volume_assessment", {}).get("success") is False
                )
        checks.append(
            dict(
                backend=r["backend"],
                seed=r["seed"],
                mode=r["mode"],
                passed=bool(passed),
                evidence=r["evidence"],
            )
        )
    inputs = []
    lookup = {(r["backend"], r["seed"], r["mode"]): r for r in rows}
    for seed in range(10):
        for mode in parameters["modes"]:
            pair = [
                lookup.get((backend, seed, mode)) for backend in parameters["backends"]
            ]
            if not all(pair):
                continue
            folders = [ROOT / r["evidence"] for r in pair]
            result = compare_sources(*(p.parent / "source" for p in folders))
            inputs.append(dict(seed=seed, mode=mode, **result))
    stats = []
    for backend in parameters["backends"]:
        for mode in parameters["modes"]:
            selected = [
                r for r in rows if r["backend"] == backend and r["mode"] == mode
            ]
            measurements = {}
            for field in ("simulation_s", "total_wall_s", "steps"):
                values = [
                    r[field] for r in selected if isinstance(r.get(field), (float, int))
                ]
                if values:
                    measurements[field] = dict(
                        count=len(values),
                        minimum=min(values),
                        maximum=max(values),
                        median=median(values),
                    )
            stats.append(dict(backend=backend, mode=mode, **measurements))
    complete = set(keys) == expected
    report.update(
        kind="saved_paired_ten_seed_acceptance",
        date="2026-09-17",
        task=task,
        control_seconds=control_seconds,
        minimum_control_seconds=control_seconds,
        batches=evidence,
        collected_at=datetime.now(timezone.utc).isoformat(),
        measurements=stats,
        collector_sha256=file_sha256(Path(__file__)),
        missing=[
            dict(backend=b, seed=s, mode=m) for b, s, m in sorted(expected - set(keys))
        ],
        episode_checks=checks,
        source_input_pairs=inputs,
        complete=complete,
        acceptance_passed=complete
        and all(c["passed"] for c in checks)
        and all(c["equal"] for c in inputs),
        scientific_process_validated=False,
        scope="Task manipulation and declared time limits; ideal volume if applicable. Frozen runtime versions are reported per batch; no full-catalogue, calibrated science, dynamics, pixel or runtime equivalence claim.",
    )
    return report


def main():
    source = sorted((BASE / "comprehensive_final_source10").glob("group_*"))
    source = [p for p in source if p.is_dir() and (p / "report.json").exists()]
    tasks = {
        "pipette_transfer": (
            [
                BASE / "transfer_endpoint_source10",
                BASE / "transfer_endpoint_native_retry",
                BASE / "transfer_endpoint_native4",
                BASE / "comprehensive_transfer_native5",
                BASE / "comprehensive-final-transfer-native-controls10",
            ],
            45.0,
            "isaac_pipette_transfer_ten_seed_summary.json",
        ),
        "centrifuge_5430_cycle": (
            source + [BASE / "comprehensive-final-cycle-native10"],
            60.0,
            "isaac_centrifuge_cycle_summary.json",
        ),
        "centrifuge_5430_close_lid": (
            source
            + [
                BASE / "comprehensive-final-lids-native-pilot",
                BASE / "comprehensive-final-lids-native8",
            ],
            30.0,
            "isaac_5430_lid_summary.json",
        ),
        "centrifuge_mini_close_lid": (
            source
            + [
                BASE / "comprehensive-final-lids-native-pilot",
                BASE / "comprehensive-final-lids-native8",
            ],
            30.0,
            "isaac_mini_lid_summary.json",
        ),
    }
    if len(sys.argv) > 1:
        tasks = {task: tasks[task] for task in sys.argv[1:]}
    results = []
    for task, (paths, seconds, name) in tasks.items():
        report = collect(
            task, [p for p in paths if (p / "report.json").exists()], seconds
        )
        save(name, report)
        print(
            task,
            report["recorded_episodes"],
            "/",
            report["requested_episodes"],
            report["acceptance_passed"],
            flush=True,
        )
        results.append(report["acceptance_passed"])
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
