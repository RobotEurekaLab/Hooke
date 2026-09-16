"""Verify native constructor recovery and 120 seconds of actual contact stability."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from backends.config import isaac_gpu
from backends.extended_contracts import fixture
from backends.baseline import write_snapshot
from backends.worker_client import IsaacWorker


def run(output, worker):
    task = fixture(
        """<mujoco><option timestep=".002"/>
        <worldbody><geom type="plane" size="1 1 .1"/>
        <body name="load" pos="0 0 .3"><freejoint/>
        <geom type="sphere" size=".05" mass=".1"/></body></worldbody></mujoco>""",
        output / "fixture",
        "stable_contact",
    )
    source = output / "source"
    source.mkdir()
    write_snapshot(task, source)
    partial = output / "missing_archive"
    partial.mkdir()
    shutil.copyfile(source / "scene.json", partial / "scene.json")
    error = None
    try:
        worker.call(
            "load",
            source=str(partial),
            output=str(output / "failed_constructor"),
            render=False,
        )
    except RuntimeError as exc:
        error = str(exc)
    if error is None or "model.npz" not in error:
        raise RuntimeError("Missing archive did not reject the constructor as expected")
    loaded = worker.call(
        "load", source=str(source), output=str(output / "recovered"), render=False
    )
    initial = worker.call("info")
    if initial["steps"] != 0 or initial["time"] != 0:
        raise RuntimeError("Recovery initialization advanced physics")
    np.testing.assert_allclose(
        loaded["state"]["qpos"], task.data.qpos, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        loaded["state"]["qvel"], task.data.qvel, atol=1e-6, rtol=0
    )
    max_speed = 0.0
    max_velocity_components = np.zeros(6)
    settled_low = np.full(7, np.inf)
    settled_high = np.full(7, -np.inf)
    contact_steps = 0
    stream = hashlib.sha256()
    reference = mujoco.MjData(task.model)
    for index in range(60000):
        state = worker.call("step", control=[])
        mujoco.mj_step(task.model, reference)
        qpos = np.asarray(state["qpos"])
        qvel = np.asarray(state["qvel"])
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise RuntimeError("Non-finite state during stability run")
        contact_steps += bool(state["contacts"])
        if index >= 5000:
            max_speed = max(max_speed, float(np.max(abs(qvel))))
            max_velocity_components = np.maximum(max_velocity_components, abs(qvel))
            settled_low = np.minimum(settled_low, qpos)
            settled_high = np.maximum(settled_high, qpos)
        stream.update(qpos.tobytes())
        stream.update(qvel.tobytes())
        if index % 5000 == 0:
            print(
                json.dumps({"step": index + 1, "native_time_s": state["time"]}),
                flush=True,
            )
    final = worker.call("info")
    height = float(state["qpos"][2])
    difference = abs(height - float(reference.qpos[2]))
    report = dict(
        kind="native_lifecycle_and_contact_stability",
        physics_options=loaded["conversion"]["physics_options"],
        constructor_error=error,
        constructor_recovery=True,
        initialization_physics_steps=initial["initialization_physics_events"],
        initialization_episode_steps=initial["steps"],
        initialization_state_matches_source=True,
        physics_callback_steps=final["physics_events_total"]
        - initial["physics_events_total"],
        physics_callback_elapsed_s=final["physics_event_elapsed_s"]
        - initial["physics_event_elapsed_s"],
        physics_steps=final["steps"],
        simulation_s=final["time"],
        contact_steps=contact_steps,
        max_velocity_after_10s=max_speed,
        max_velocity_components_after_10s=max_velocity_components.tolist(),
        settled_qpos_range=(settled_high - settled_low).tolist(),
        final_qvel=qvel.tolist(),
        native_height_m=height,
        source_height_m=float(reference.qpos[2]),
        height_error_m=difference,
        trajectory_sha256=stream.hexdigest(),
        passed=bool(
            final["steps"] == 60000
            and final["physics_events_total"] - initial["physics_events_total"] == 60000
            and final["time"] == 120.0
            and abs(
                final["physics_event_elapsed_s"]
                - initial["physics_event_elapsed_s"]
                - 120.0
            )
            < 1e-3
            and contact_steps > 50000
            and max_speed < 0.002
            and difference < 0.005
        ),
        scope="Isolated contact fixture; catalogue task stability is reported separately.",
        parity_qualified=False,
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True)
    with IsaacWorker(output, args.gpu) as worker:
        result = run(output, worker)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
