"""Measure closed-gripper pose and actual instrument touch across contact settings."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
from archetypes.task_catalog import CATALOG
from backends.closed_loop import PhysXTaskAdapter
from backends.worker_client import IsaacWorker
from backends.native_contracts import run as check_contracts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    reference=np.load(args.reference/'trajectory.npz');results=[]
    with IsaacWorker(output) as worker:
        contracts=check_contracts(output/'contracts',worker)
        if not contracts['passed']:raise RuntimeError('Native physical contract failed; inspect its result before calibrating the gripper')
        variants=[('rigid_loop',{}),('soft_loop',{'soft_connect_constraints':True}),
                  ('soft_loop_stiff',{'soft_connect_constraints':True,'connect_impedance_fraction':1.}),
                  ('soft_loop_and_pads',{'soft_connect_constraints':True,'compliant_impedance_fraction':0.})]
        for name,variant in variants:
            task=CATALOG['thermal_mixer'].make_expert();task.reset(0);mujoco.mj_forward(task.model,task.data)
            folder=output/name;folder.mkdir(exist_ok=True)
            options={'compliant_contact_scale':1.,**variant}
            with PhysXTaskAdapter(task,worker,folder,physics_options=options) as adapter:
                task.data.ctrl[-1]=255
                for _ in range(500):task.step_and_log({})
                closed=task.data.qpos[6:].copy()
                state=worker.call('reset',qpos=reference['qpos'][-1].tolist(),qvel=np.zeros(task.model.nv).tolist())
                adapter.apply_state(state);task.data.ctrl[:]=reference['control'][-1]
                peaks=np.zeros(task.model.nsensordata)
                for _ in range(1000):
                    task.step_and_log({});peaks=np.maximum(peaks,task.data.sensordata)
                row={'variant':name,'physics_options':options,'closed_gripper_qpos':closed.tolist(),
                     'sensor_peaks':peaks.tolist(),'final_sensors':task.data.sensordata.tolist(),
                     'instrument_state':asdict(task.instrument.ui_state),
                     'final_qpos':task.data.qpos.tolist()}
                (folder/'state.json').write_text(json.dumps(worker.call('observe')))
                results.append(row);print(json.dumps(row),flush=True)
    (output/'result.json').write_text(json.dumps({'scope':'Held-pose gripper/touch diagnostic; not an original task episode.','results':results},indent=2))


if __name__=='__main__':main()
