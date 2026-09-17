"""Load/reset/step/render archived scenes in one native Isaac process.

This establishes runtime compatibility, not task or physics equivalence.
"""
from __future__ import annotations
from backends.config import isaac_gpu
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import mujoco
from backends.worker_client import IsaacWorker
from backends.source_forces import passive_residual, update_kinematics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',action='append')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--gpu',type=int,default=isaac_gpu())
    parser.add_argument('--simple-only',action='store_true')
    args=parser.parse_args()
    args.catalog=args.catalog.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.catalog/'export_results.json').read_text())['results']
    if args.task:rows=[r for r in rows if r['task'] in args.task]
    if args.simple_only:rows=[r for r in rows if r.get('plugins',0)==0 and r.get('max_joints_per_body',0)==1 and 0 not in r.get('joint_types',[])]
    results=[]
    # Plugin instances in archived MJBs use the same original library as Task.
    mujoco.mj_loadPluginLibrary(str(Path(__file__).resolve().parents[1]/'libmjlab.so.3.3.0'))
    with IsaacWorker(args.output,args.gpu) as worker:
        for row in rows:
            name=row['task'];source=args.catalog/name;out=args.output/name
            start=time.perf_counter();result={'task':name,'status':'RUNNING','parity_qualified':False}
            try:
                loaded=worker.call('load',source=str(source),output=str(out),render=args.render)
                if loaded['conversion']['instantiated_source_joints']!=row['joints']:
                    raise RuntimeError('Source joint count differs from instantiated joints')
                with np.load(source/'model.npz') as archive:ctrl=archive['reset_ctrl'].tolist()
                model=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));data=mujoco.MjData(model)
                state=loaded['state']
                max_fk_error=0.
                for _ in range(args.steps):
                    data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];data.time=state['time']
                    update_kinematics(model,data)
                    for body,pose in state.get('body_poses',{}).items():
                        error=float(np.linalg.norm(data.xpos[int(body)]-pose[:3]))
                        max_fk_error=max(max_fk_error,error)
                        if error>.005:raise RuntimeError(f'Body {body} and source FK disagree by {error} m')
                    state=worker.call('step',control=ctrl,extra_forces=passive_residual(model,data).tolist())
                if not np.isfinite(state['qpos']).all() or not np.isfinite(state['qvel']).all():
                    raise RuntimeError('Non-finite final PhysX state')
                data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel']
                update_kinematics(model,data)
                for body,pose in state.get('body_poses',{}).items():
                    error=float(np.linalg.norm(data.xpos[int(body)]-pose[:3]))
                    max_fk_error=max(max_fk_error,error)
                    if error>.005:raise RuntimeError(f'Final body {body} and source FK disagree by {error} m')
                info=worker.call('info')
                if args.render and info['frames']<1:raise RuntimeError('No rendered frames')
                result.update(status='LOAD_STEP_RENDER_OK' if args.render else 'LOAD_STEP_OK',
                              source_geoms=row['geoms'],exported_geoms=loaded['conversion']['geom_count'],
                              source_joints=row['joints'],scalar_joints=loaded['conversion']['joint_count'],
                              free_joints=len(loaded['conversion']['free_joints']),
                              instantiated_source_joints=loaded['conversion']['instantiated_source_joints'],
                              conversion_version=loaded['conversion']['conversion_version'],
                              physics_options=loaded['conversion']['physics_options'],
                              max_fk_position_error_m=max_fk_error,
                              simulation_s=state['time'],frames=info['frames'],steps=info['steps'],
                              finite=bool(np.isfinite(state['qpos']).all() and np.isfinite(state['qvel']).all()))
            except Exception as exc:
                result.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
            result['wall_s']=time.perf_counter()-start
            results.append(result)
            out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2))
            report={'scope':'Scene loading and finite PhysX stepping; original controls held at reset values. Does not qualify experts or task success.',
                    'total_requested':len(rows),'completed':len(results),'results':results}
            temporary=args.output/'batch_results.tmp'
            temporary.write_text(json.dumps(report,indent=2));temporary.replace(args.output/'batch_results.json')
            print(json.dumps({k:v for k,v in result.items() if k!='traceback'}),flush=True)
    raise SystemExit(1 if any(r['status']=='ERROR' for r in results) else 0)


if __name__=='__main__':main()
