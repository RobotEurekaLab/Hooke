"""Run the same catalogue task through either engine for the LAN interface."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
import faulthandler
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from PIL import Image
from backends.task_result import task_result
from backends.assessment import EpisodeAssessment


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def instrument_state(task):
    state=getattr(getattr(task,'instrument',None),'ui_state',None)
    return asdict(state) if is_dataclass(state) else None


def run(args,worker=None):
    from archetypes.task_catalog import CATALOG
    from backends.baseline import write_snapshot
    from backends.closed_loop import PhysXTaskAdapter
    from backends.worker_client import IsaacWorker
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    result={'status':'STARTING','task':args.task,'backend':args.backend,'seed':args.seed,
            'mode':args.mode,'parity_qualified':False}
    result_path=output/'result.json';write_json(result_path,result)
    started=time.perf_counter();task=None;steps=0;contact_steps=0;frames=0;physics_wall=0.;rows=[]
    renderer=None;adapter=None;original_step=None;original_manager_step=None;visuals=None;assessment=None;volume_assessment=None
    try:
        entry=CATALOG[args.task]
        task=entry.make_expert();task.reset(args.seed);mujoco.mj_forward(task.model,task.data)
        assessment=EpisodeAssessment(task,args.task)
        if getattr(task,'liquid_transfer',None) is not None:
            from backends.volume_assessment import PipetteVolumeAssessment
            volume_assessment=PipetteVolumeAssessment(task.liquid_transfer)
        task.task_info['simulation_backend']=args.backend
        task.task_info['physics_engine']='PhysX' if args.backend=='isaac' else 'MuJoCo'
        if args.mode=='expert':task.set_serializer(log_root=output/'task_log',log_name='episode')
        display_only=any(cls.__name__=='StaticDisplayTask' for cls in type(task).__mro__)
        cameras=list(dict.fromkeys(task.task_info.get('camera_mapping',{}).values()))
        camera_ids=[mujoco.mj_name2id(task.model,mujoco.mjtObj.mjOBJ_CAMERA,name) for name in cameras]
        result.update(display_only=display_only,cameras=[{'id':i,'name':n} for i,n in zip(camera_ids,cameras)],
                      declared_time_limit_s=task.time_limit)
        with ExitStack() as stack:
            if args.backend=='isaac':
                if worker is None:worker=stack.enter_context(IsaacWorker(output,args.gpu))
                adapter=stack.enter_context(PhysXTaskAdapter(task,worker,output,render=not args.no_render,
                                                           physics_options=getattr(args,'physics_options',None),
                                                           report_progress=False,managed_render=not args.no_render))
            else:
                source=output/'source';source.mkdir(exist_ok=True);write_snapshot(task,source)
                if not args.no_render:renderer=mujoco.Renderer(task.model,height=480,width=640)
            if not args.no_render:
                from backends.visual_state import LiveVisuals
                visuals=LiveVisuals(task,output/'visuals')
            result['status']='RUNNING';write_json(result_path,result)

            def capture():
                nonlocal frames
                if visuals is None:return
                visual_state=visuals.snapshot()
                if adapter:
                    worker.call('render',visuals=visual_state)
                if renderer:
                    for i,name in zip(camera_ids,cameras):
                        folder=output/'mujoco'/f'camera_{i}';folder.mkdir(parents=True,exist_ok=True)
                        renderer.update_scene(task.data,camera=name)
                        visuals.apply_mujoco(renderer,visual_state)
                        destination=folder/f'{frames:05d}.png';temporary=destination.with_suffix('.tmp')
                        Image.fromarray(renderer.render()).save(temporary,format='PNG');temporary.replace(destination)
                frames+=1
            capture()
            original_step=mujoco.mj_step
            def step(model,data,nstep=1):
                nonlocal steps,contact_steps,physics_wall
                if model is not task.model or data is not task.data:raise RuntimeError('Task replaced its active scene')
                for _ in range(nstep):
                    if data.time>=args.max_sim_seconds:raise TimeoutError('Simulation duration limit reached')
                    control=data.ctrl.copy();tick=time.perf_counter();original_step(model,data)
                    physics_wall+=time.perf_counter()-tick;steps+=1;contact_steps+=bool(data.ncon)
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise RuntimeError('Non-finite state')
                    if not adapter:rows.append({'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),'control':control,'time':float(data.time)})
            mujoco.mj_step=step
            original_manager_step=task.manager.step
            def manager_step():
                original_manager_step()
                assessment.update()
                if volume_assessment is not None:volume_assessment.update()
                # The instrument/liquid systems update after physics. Render
                # their current state, with the same ordering for both engines.
                if steps%max(1,round(.05/task.dt))==0:capture()
                if steps%100==0:
                    write_json(output/'progress.json',{'steps':steps,'simulation_s':float(task.data.time),'contacts':task.data.ncon,
                                                      'instrument_state':instrument_state(task)})
            task.manager.step=manager_step
            try:
                if args.mode in ('preview','no_action') or display_only:
                    for _ in range(round(args.seconds/task.dt)):task.step_and_log({})
                    if display_only and args.mode=='expert':task.finish()
                else:task.execute()
            finally:
                mujoco.mj_step=original_step;original_step=None
                task.manager.step=original_manager_step;original_manager_step=None
            check=task_result(task.check());result.update(check);source_check=check['source_success']
            status='PREVIEW_COMPLETE' if args.mode=='preview' else 'CONTROL_COMPLETE' if args.mode=='no_action' else 'DISPLAY_COMPLETE' if display_only else 'TASK_SUCCEEDED' if source_check else 'TASK_FAILED'
            result.update(status=status,source_success=source_check,steps=steps,contact_steps=contact_steps,
                          simulation_s=float(task.data.time),within_declared_time_limit=bool(task.data.time<=task.time_limit),
                          final_qpos=task.data.qpos.tolist(),
                          final_contact_pairs=sorted({tuple(sorted(map(int,c.geom))) for c in task.data.contact}))
            if adapter:
                result.update(runtime=worker.call('info'),conversion=adapter.loaded['conversion'],
                              acceleration_source='Finite difference of actual PhysX generalized velocities; source spatial kinematics.',
                              max_fk_position_error_m=adapter.max_fk_position_error,
                              max_fk_rotation_error_rad=adapter.max_fk_rotation_error)
            else:result.update(runtime={'frames':frames,'physics_wall_s':physics_wall})
    except KeyboardInterrupt:
        result.update(status='STOPPED',error='Interrupted before the expert completed')
        raise
    except Exception as exc:
        result.update(status='TIME_LIMIT' if isinstance(exc,TimeoutError) else 'ERROR',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if original_step is not None:mujoco.mj_step=original_step
        if original_manager_step is not None:task.manager.step=original_manager_step
        if visuals:
            result['runtime_visuals']={'display_texture_updates':visuals.updates,'liquid_surface_drawing':bool(visuals.liquid_extra),
                                      'liquid_fit_fallbacks':visuals.liquid_fit_fallbacks}
            visuals.close()
        if renderer:renderer.close()
        trajectory=adapter.rows if adapter else rows
        if trajectory:np.savez_compressed(output/'trajectory.npz',**{k:np.asarray([r[k] for r in trajectory]) for k in trajectory[0]})
        result['total_wall_s']=time.perf_counter()-started
        if adapter is not None:
            result.update(max_fk_position_error_m=adapter.max_fk_position_error,
                          max_fk_rotation_error_rad=adapter.max_fk_rotation_error,
                          physics_options=adapter.loaded['conversion']['physics_options'])
        if task is not None:
            if assessment is not None:result['assessment']=assessment.report()
            if volume_assessment is not None:result['volume_assessment']=volume_assessment.report()
            if getattr(task,'phase_history',None) is not None:result['expert_phases']=task.phase_history
            result['final_qpos']=task.data.qpos.tolist()
            result['final_contact_pairs']=sorted({tuple(sorted(map(int,c.geom))) for c in task.data.contact})
            if task.model.nsensor:
                result['final_sensors']={task.model.sensor(i).name:task.data.sensordata[
                    task.model.sensor_adr[i]:task.model.sensor_adr[i]+task.model.sensor_dim[i]].tolist()
                    for i in range(task.model.nsensor)}
            state=instrument_state(task)
            if state is not None:result['instrument_state']=state
        result.update(steps=steps,simulation_s=float(task.data.time) if task is not None else 0.)
        write_json(result_path,result);print(json.dumps(result),flush=True)
    return result


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--backend',choices=['mujoco','isaac'],required=True)
    parser.add_argument('--mode',choices=['preview','expert','no_action'],default='expert',
                        help='no_action holds reset controls while physics and instrument systems run')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--max-sim-seconds',type=float,default=120.)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--no-render',action='store_true',help='Physics/controller validation without RGB')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=args.max_sim_seconds<=120:parser.error('Invalid duration')
    result=run(args)
    raise SystemExit(1 if result['status'] in ('ERROR','TIME_LIMIT','TASK_FAILED') else 0)


if __name__=='__main__':main()
