"""Actual paired physics for body wrenches, captured welds and static terrain."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from backends.baseline import write_snapshot
from backends.closed_loop import PhysXTaskAdapter
from backends.config import isaac_gpu
from backends.source_forces import update_kinematics
from backends.worker_client import IsaacWorker


def fixture(xml, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        spec=spec,
        model=model,
        data=data,
        task=name,
        default_scene=folder / "fixture.xml",
        task_info={"camera_mapping": {}},
    )


def run(output, worker):
    results = []
    folder = output / "body_wrench"
    task = fixture(
        """<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body name="load" pos="0 0 1" quat=".70710678 0 0 .70710678"><freejoint/>
      <geom name="load" type="sphere" pos=".02 .01 .03" size=".04" mass="1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>""",
        folder,
        "body_wrench",
    )
    model, data = task.model, task.data
    data.xfrc_applied[1] = [0.2, 0.1, 0.05, 0.001, 0.002, 0.003]
    reference = mujoco.MjData(model)
    reference.xfrc_applied[:] = data.xfrc_applied
    mujoco.mj_forward(model, reference)
    adapter = PhysXTaskAdapter(task, worker, folder, report_progress=False)
    for _ in range(100):
        mujoco.mj_step(model, reference)
        adapter.step(model, data)
    position = float(np.linalg.norm(data.qpos[:3] - reference.qpos[:3]))
    velocity = float(np.max(abs(data.qvel - reference.qvel)))
    result = dict(
        fixture="world_body_wrench_at_offset_com",
        position_error_m=position,
        velocity_error=velocity,
        max_fk_position_error_m=adapter.max_fk_position_error,
        passed=bool(position < 1e-4 and velocity < 0.002),
    )
    results.append(result)
    print(json.dumps(result), flush=True)

    folder = output / "captured_weld"
    task = fixture(
        """<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body name="rotor" pos="0 0 1"><joint name="rotor" axis="0 0 1"/>
      <geom type="cylinder" size=".03 .01" mass="1" contype="0" conaffinity="0"/></body>
      <body name="load" pos=".1 0 1"><freejoint/><geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body></worldbody>
      <equality><weld name="holder" body1="rotor" body2="load" active="false" solref=".005 1" solimp=".9999 .9999 .001 .5 2"/></equality>
      <actuator><motor joint="rotor"/></actuator></mujoco>""",
        folder,
        "captured_weld",
    )
    model, data = task.model, task.data
    adapter = PhysXTaskAdapter(task, worker, folder, report_progress=False)
    # Capture at the actual loaded state, with zero initial constraint residual.
    rotor = model.body("rotor").id
    load = model.body("load").id
    p, q = np.zeros(3), np.zeros(4)
    invp, invq = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(invp, invq, data.xpos[rotor], data.xquat[rotor])
    mujoco.mju_mulPose(p, q, invp, invq, data.xpos[load], data.xquat[load])
    model.eq_data[0, 3:6] = p
    model.eq_data[0, 6:10] = q
    data.eq_active[0] = True
    reference = mujoco.MjData(model)
    reference.eq_active[0] = True
    data.ctrl[:] = reference.ctrl[:] = 0.01
    mujoco.mj_forward(model, reference)
    for _ in range(200):
        mujoco.mj_step(model, reference)
        adapter.step(model, data)
    relative = np.zeros(3)
    orientation = np.zeros(4)
    mujoco.mju_negPose(invp, invq, data.xpos[rotor], data.xquat[rotor])
    mujoco.mju_mulPose(
        relative, orientation, invp, invq, data.xpos[load], data.xquat[load]
    )
    retained = float(np.linalg.norm(relative - p))
    rotation = float(data.qpos[0])
    error = float(np.max(abs(data.qpos - reference.qpos)))
    result = dict(
        fixture="captured_weld_on_rotating_load",
        retention_error_m=retained,
        actual_rotation_rad=rotation,
        source_native_qpos_error=error,
        passed=bool(retained < 5e-5 and rotation > 0.1 and error < 0.002),
    )
    results.append(result)
    print(json.dumps(result), flush=True)

    folder = output / "heightfield"
    task = fixture(
        """<mujoco><option timestep=".002"/><asset><hfield name="terrain" nrow="5" ncol="5" size="1 1 .2 .1"
      elevation="0 0 0 0 0  0 1 1 1 0  0 1 1 1 0  0 1 1 1 0  0 0 0 0 0"/></asset>
      <worldbody><geom type="hfield" hfield="terrain"/><body name="load" pos="0 0 .6"><freejoint/>
      <geom name="load" type="sphere" size=".05" mass=".1"/></body></worldbody></mujoco>""",
        folder,
        "heightfield",
    )
    model, data = task.model, task.data
    mujoco.mj_forward(model, data)
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    adapter = PhysXTaskAdapter(task, worker, folder, report_progress=False)
    native_contact_steps = 0
    for _ in range(1500):
        mujoco.mj_step(model, reference)
        adapter.step(model, data)
        native_contact_steps += bool(data.ncon)
    error = float(abs(data.qpos[2] - reference.qpos[2]))
    result = dict(
        fixture="static_heightfield_drop",
        source_height_m=float(reference.qpos[2]),
        native_height_m=float(data.qpos[2]),
        height_error_m=error,
        native_contact_steps=native_contact_steps,
        passed=bool(
            error < 0.005 and 0.23 < data.qpos[2] < 0.26 and native_contact_steps > 500
        ),
    )
    results.append(result)
    print(json.dumps(result), flush=True)
    report = dict(
        kind="extended_native_physics_contracts",
        passed=all(r["passed"] for r in results),
        results=results,
        parity_qualified=False,
        scope="Bounded fixtures; no arbitrary MJCF or weld compliance parity claim.",
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=isaac_gpu())
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with IsaacWorker(args.output, args.gpu) as worker:
        report = run(args.output, worker)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
