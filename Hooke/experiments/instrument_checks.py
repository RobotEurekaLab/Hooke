"""Known-load physics checks, separate from continuous robot episodes."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import mujoco
import numpy as np

from backends.closed_loop import PhysXTaskAdapter
from backends.config import isaac_gpu
from backends.extended_contracts import fixture
from backends.run import write_json
from backends.worker_client import IsaacWorker
from experiments.analysis import oscillation, static_signal


def cases():
    for world, gravity in (("orbital", 0), ("lunar", 1.62), ("martian", 3.73)):
        for mass in (0.0, 0.1, 0.16):
            yield dict(
                name=f"{world}_load_{mass}",
                world=world,
                gravity=gravity,
                mass=mass,
                dt=0.002,
                axis_z=0.0 if world == "orbital" else 1.0,
                locked=True,
            )
    for dt in (0.001, 0.004):
        for world, gravity in (("orbital", 0), ("martian", 3.73)):
            yield dict(
                name=f"{world}_dt_{dt}",
                world=world,
                gravity=gravity,
                mass=0.16,
                dt=dt,
                axis_z=0.0 if world == "orbital" else 1.0,
                locked=True,
            )
    yield dict(
        name="martian_axis_45deg",
        world="martian",
        gravity=3.73,
        mass=0.16,
        dt=0.002,
        axis_z=2**-0.5,
        locked=True,
    )
    yield dict(
        name="orbital_unlocked_negative",
        world="orbital",
        gravity=0,
        mass=0.16,
        dt=0.002,
        axis_z=0.0,
        locked=False,
    )


def run_case(case, backend, folder, worker=None):
    orbital = case["world"] == "orbital"
    k, damping = (20, 0.02) if orbital else (100, 3)
    z = case["axis_z"]
    axis = f"{np.sqrt(max(1-z*z, 0))} 0 {z}"
    sample = ""
    equality = ""
    if case["mass"]:
        sample = f'<body name="sample" pos="0 0 1.05"><freejoint/><geom type="box" size=".023 .02 .035" mass="{case["mass"]}" contype="0" conaffinity="0"/></body>'
        equality = f'<equality><weld body1="tray" body2="sample" active="{str(case["locked"]).lower()}" solref=".004 1" solimp=".9999 .9999 .001"/></equality>'
    xml = f'<mujoco><option timestep="{case["dt"]}" gravity="0 0 {-case["gravity"]}" integrator="implicitfast"/><worldbody><body name="tray" pos="0 0 1"><joint name="slide" type="slide" axis="{axis}" stiffness="{k}" damping="{damping}" range="-.1 .1"/><geom type="box" size=".035 .027 .008" mass=".2" contype="0" conaffinity="0"/></body>{sample}</worldbody>{equality}</mujoco>'
    task = fixture(xml, folder, case["name"])
    model, data = task.model, task.data
    adapter = (
        PhysXTaskAdapter(task, worker, folder, report_progress=False)
        if backend == "isaac"
        else None
    )
    rows = []
    next_sample = 0.0
    duration = 6 if orbital else 4
    for _ in range(round(duration / case["dt"])):
        data.qfrc_applied[0] = 0.15 if orbital and data.time < 0.08 else 0
        if adapter:
            adapter.step(model, data)
        else:
            mujoco.mj_step(model, data)
        if data.time + 1e-10 >= next_sample:
            rows.append([float(data.time), float(data.qpos[0])])
            next_sample += 0.02
    record = dict(
        measurement_id="m0001",
        sample_id="known_load" if case["mass"] else None,
        method="inertial" if orbital else "static_force",
        time_position=rows,
        valid=True,
        encoder_noise_std_m=1e-6,
        gravity_m_s2=case["gravity"] * case["axis_z"],
        spring_stiffness_n_m=k,
    )
    moving_mass = 0.2 + (case["mass"] if case["locked"] else 0)
    if orbital:
        diagnostics = oscillation(record)
        expected = 4 * np.pi**2 * moving_mass / k
        measured = diagnostics["period_squared_s2"]
        downsampled = oscillation(dict(record, time_position=rows[::2]))
        sample_rate_error = abs(downsampled["period_squared_s2"] - measured) / expected
    else:
        diagnostics = static_signal(record)
        expected = moving_mass * record["gravity_m_s2"]
        measured = diagnostics["force_n"]
        sample_rate_error = 0.0
    relative_error = abs(measured - expected) / expected
    checks = dict(
        known_load_response_within_half_percent=relative_error < 0.005,
        sample_rate_change_within_half_percent=sample_rate_error < 0.005,
        finite_state=bool(
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        ),
    )
    if adapter:
        info = worker.call("info")
        checks["one_native_event_per_step"] = info[
            "physics_events_since_load"
        ] == round(duration / case["dt"])
    result = dict(
        case,
        backend=backend,
        passed=all(checks.values()),
        checks=checks,
        measured_signal=measured,
        expected_signal=expected,
        relative_error=relative_error,
        sample_rate_relative_error=sample_rate_error,
        diagnostics=diagnostics,
        simulation_s=float(data.time),
        scope="known_load_instrument_fixture_not_robot_episode",
    )
    write_json(folder / "observations.json", record)
    write_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("mujoco", "isaac"),
        default=["mujoco", "isaac"],
    )
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in args.backends:
        context = (
            IsaacWorker(args.output / "worker", args.gpu)
            if backend == "isaac"
            else nullcontext(None)
        )
        with context as worker:
            for case in cases():
                result = run_case(
                    case, backend, args.output / backend / case["name"], worker
                )
                results.append(result)
                write_json(
                    args.output / "summary.json",
                    dict(
                        results=results,
                        passed=all(r["passed"] for r in results),
                        total=len(results),
                        complete=False,
                    ),
                )
                print(json.dumps(result), flush=True)
    summary = dict(
        results=results,
        passed=all(r["passed"] for r in results),
        total=len(results),
        complete=True,
    )
    write_json(args.output / "summary.json", summary)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
