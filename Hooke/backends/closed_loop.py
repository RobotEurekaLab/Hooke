"""Run an original Hooke expert against actual state returned by native PhysX.

MuJoCo provides source-model names and kinematic calculations for existing IK
code. It never advances physics while this adapter is active. Contact-based
checks receive PhysX contacts, not contacts predicted by a second simulation.
"""
from __future__ import annotations
from backends.config import isaac_gpu
import argparse
import json
from pathlib import Path
import time
import traceback
import faulthandler
import mujoco
import numpy as np
from backends.baseline import write_snapshot
from backends.worker_client import IsaacWorker
from backends.source_forces import body_wrench_forces, passive_residual, update_kinematics
from backends.task_result import task_result


class PhysXTaskAdapter:
    def __init__(self, task, worker, output: Path, render=False,physics_options=None,report_progress=True,managed_render=False):
        self.task,self.worker,self.output=task,worker,output
        self.report_progress=report_progress
        self.rows=[];self.contact_steps=0
        self.max_fk_position_error=0.
        self.max_fk_rotation_error=0.
        source=output/'source';source.mkdir(parents=True,exist_ok=True)
        write_snapshot(task,source)
        loaded=worker.call('load',source=str(source.resolve()),output=str((output/'isaac').resolve()),render=render,physics_options=physics_options,managed_render=managed_render)
        self.loaded=loaded
        self._native_contacts=[]
        self.apply_state(loaded['state'])

    def apply_state(self, state):
        model,data=self.task.model,self.task.data
        velocity=np.asarray(state['qvel']);current_time=float(state['time'])
        previous_time=getattr(self,'_previous_time',current_time)
        elapsed=current_time-previous_time
        data.qacc[:]=(velocity-self._previous_velocity)/elapsed if elapsed>0 and hasattr(self,'_previous_velocity') else 0.
        self._previous_time=current_time;self._previous_velocity=velocity.copy()
        data.qpos[:]=state['qpos'];data.qvel[:]=velocity;data.time=current_time
        # Forward calculations support existing IK and site queries. Position /
        # velocity integration and all actual contacts are owned by PhysX.
        update_kinematics(model,data)
        observed=state.get('body_poses',{})
        if observed:
            ids=np.asarray(list(observed),dtype=np.int32)
            poses=np.asarray(list(observed.values()),dtype=float)
            errors=np.linalg.norm(data.xpos[ids]-poses[:,:3],axis=1)
            self.max_fk_position_error=max(self.max_fk_position_error,float(errors.max()))
            orientations=poses[:,3:]/np.linalg.norm(poses[:,3:],axis=1)[:,None]
            dots=np.clip(abs(np.sum(data.xquat[ids]*orientations,axis=1)),0.,1.)
            self.max_fk_rotation_error=max(self.max_fk_rotation_error,float((2*np.arccos(dots)).max()))
            if np.any(errors>.005):
                index=int(np.argmax(errors));body=int(ids[index]);position_error=float(errors[index])
                raise RuntimeError(f'Native body/FK disagreement at t={data.time:.4f}, body={body} ({model.body(body).name}): {position_error:.6f} m; source={data.xpos[body].tolist()}, native={poses[index,:3].tolist()}')
        # These contacts were solved by PhysX. They have no MuJoCo constraint
        # Jacobian/force entries; never leave the old solver dimensions active.
        data.nefc=0
        data.ncon=0
        self._native_contacts=state['contacts']
        self._contact_arrays={}
        count=len(self._native_contacts)
        if count:
            geoms=np.asarray([[c['geom1'],c['geom2']] for c in self._native_contacts],dtype=np.int32)
            points=np.asarray([c['pos'] for c in self._native_contacts],dtype=float)
            normals=np.asarray([c['normal'] for c in self._native_contacts],dtype=float)
            forces=np.asarray([c['force'] for c in self._native_contacts],dtype=float)
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            helper=np.zeros_like(normals);helper[np.arange(count),(abs(normals[:,0])>=.9).astype(int)]=1.
            tangent=np.cross(normals,helper);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
            frames=np.stack((normals,tangent,np.cross(normals,tangent)),axis=1).reshape(count,9)
            # Reserve contact storage through the public API, then fill its
            # writable array views in bulk. Preserve every actual point.
            empty=mujoco.MjContact();empty.dim=3;empty.efc_address=-1
            empty.flex[:]=-1;empty.elem[:]=-1;empty.vert[:]=-1
            for _ in range(count):
                if mujoco.mj_addContact(model,data,empty):raise RuntimeError('Source contact buffer could not accept PhysX contacts')
            data.contact.geom[:]=geoms;data.contact.pos[:]=points;data.contact.frame[:]=frames
            # MuJoCo 3.3 retains separate deprecated geom1/geom2 fields.
            # They are not aliases of geom[:, 0/1]; original task predicates
            # still read them, so both representations must be populated.
            data.contact.geom1[:]=geoms[:,0];data.contact.geom2[:]=geoms[:,1]
            data.contact.dist[:]=[c['distance'] for c in self._native_contacts]
            friction=np.maximum(model.geom_friction[geoms[:,0]],model.geom_friction[geoms[:,1]])
            data.contact.friction[:]=friction[:,[0,0,1,2,2]]
            self._contact_arrays={'geoms':geoms,'points':points,'forces':forces}
        self.update_touch_sensors()

    def update_touch_sensors(self):
        """Retain the source sensor zone/ray rule using actual native forces.

        Rule reference: MuJoCo 3.3 engine_sensor.c, mjSENS_TOUCH.
        """
        model,data=self.task.model,self.task.data
        for sensor,kind in enumerate(model.sensor_type):
            if kind!=mujoco.mjtSensor.mjSENS_TOUCH:continue
            site=int(model.sensor_objid[sensor]);body=int(model.site_bodyid[site]);value=0.
            for contact in self._native_contacts:
                a=int(model.geom_bodyid[contact['geom1']]);b=int(model.geom_bodyid[contact['geom2']])
                if body not in (a,b):continue
                normal=np.asarray(contact['normal'],float)
                force=float(np.dot(normal,contact['force']))
                if force<=0:continue
                direction=-normal if body==b else normal
                hit=mujoco.mju_rayGeom(data.site_xpos[site],data.site_xmat[site],model.site_size[site],
                                       np.asarray(contact['pos'],float),direction,int(model.site_type[site]))
                if hit>=0:value+=force
            cutoff=float(model.sensor_cutoff[sensor])
            data.sensordata[model.sensor_adr[sensor]]=min(value,cutoff) if cutoff>0 else value

    def step(self, model, data, nstep=1):
        if model is not self.task.model or data is not self.task.data:
            raise RuntimeError('Task changed its model/data while the PhysX adapter was active')
        for _ in range(nstep):
            control=data.ctrl.copy()
            force = data.qfrc_applied+passive_residual(model,data)
            if np.any(data.xfrc_applied):
                force += body_wrench_forces(model,data)
            state=self.worker.call('step',control=control.tolist(),extra_forces=force.tolist(),
                                   eq_active=data.eq_active.tolist(),
                                   eq_data=model.eq_data.tolist())
            self.apply_state(state)
            self.rows.append({'control':control,'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),
                              'time':float(data.time),'contacts':len(state['contacts'])})
            self.contact_steps+=bool(state['contacts'])
            # Task checks can mutate task history (for example pipetting).
            # The physics adapter must not add calls absent from MuJoCo.
            if len(self.rows)%250==0 and self.report_progress:
                progress=self.output/'progress.json';temporary=progress.with_suffix('.tmp')
                temporary.write_text(json.dumps({'steps':len(self.rows),'simulation_s':data.time,
                    'contacts':data.ncon}))
                temporary.replace(progress)

    def post_constraint(self, model, data):
        if model is not self.task.model:raise RuntimeError('Unexpected model in force query')
        # Refresh spatial accelerations from observed native velocity changes.
        # Liquid systems query mj_objectAcceleration; leaving cacc at reset
        # would detach their dynamics from the moving PhysX container.
        count=data.ncon;data.ncon=0
        try:getattr(self,'original_post',mujoco.mj_rnePostConstraint)(model,data)
        finally:data.ncon=count
        # Preserve actual external contact wrenches for existing observations.
        # Torque is about the root-subtree COM, as in MuJoCo cfrc_ext.
        data.cfrc_ext[:]=0
        # Source body wrenches are world-frame force/torque at the body COM.
        for body in np.flatnonzero(np.any(data.xfrc_applied,axis=1)):
            if body == 0:continue
            force = data.xfrc_applied[body,:3]
            torque = data.xfrc_applied[body,3:]+np.cross(
                data.xipos[body]-data.subtree_com[model.body_rootid[body]],force)
            data.cfrc_ext[body] = np.r_[torque,force]
        if self._contact_arrays:
            arrays=self._contact_arrays;bodies=model.geom_bodyid[arrays['geoms']]
            roots=model.body_rootid[bodies]
            forces=arrays['forces'][:,None,:]*np.array([-1.,1.])[None,:,None]
            torques=np.cross(arrays['points'][:,None,:]-data.subtree_com[roots],forces)
            np.add.at(data.cfrc_ext,bodies.reshape(-1),np.concatenate((torques,forces),axis=2).reshape(-1,6))

    def contact_force(self, model, data, contact_id, result):
        if model is not self.task.model:raise RuntimeError('Unexpected model in contact force query')
        result[:]=0
        if 0<=contact_id<len(self._native_contacts):
            frame=data.contact[contact_id].frame.reshape(3,3)
            result[:3]=frame@np.asarray(self._native_contacts[contact_id]['force'])

    def __enter__(self):
        self.original_step=mujoco.mj_step;self.original_force=mujoco.mj_contactForce
        self.original_post=mujoco.mj_rnePostConstraint
        mujoco.mj_step=self.step;mujoco.mj_contactForce=self.contact_force
        mujoco.mj_rnePostConstraint=self.post_constraint
        return self

    def __exit__(self,*_):
        mujoco.mj_step=self.original_step;mujoco.mj_contactForce=self.original_force
        mujoco.mj_rnePostConstraint=self.original_post


def run(task_name: str, seed: int, output: Path, render: bool, gpu: int):
    from archetypes.task_catalog import CATALOG
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    result={'status':'RUNNING','task':task_name,'seed':seed,'physics_engine':'PhysX',
            'control':'original expert with actual PhysX state feedback','parity_qualified':False}
    result_file=output/'result.json';result_file.write_text(json.dumps(result))
    start=time.perf_counter();adapter=None
    try:
        task=CATALOG[task_name].make_expert();task.reset(seed);mujoco.mj_forward(task.model,task.data)
        task.set_serializer(log_root=output/'task_log',log_name='episode')
        with IsaacWorker(output,gpu) as worker:
            adapter=PhysXTaskAdapter(task,worker,output,render)
            with adapter:task.execute()
            check=task_result(task.check());result.update(check);success=check['source_success']
            result.update(status='TASK_SUCCEEDED' if success else 'TASK_FAILED',source_success=success,
                          simulation_s=float(task.data.time),steps=len(adapter.rows),
                          contact_steps=adapter.contact_steps,
                          declared_time_limit_s=task.time_limit,within_declared_time_limit=bool(task.data.time<=task.time_limit),
                          final_qpos=task.data.qpos.tolist(),runtime=worker.call('info'),
                          conversion=adapter.loaded['conversion'],gains=adapter.loaded['gains'])
            result.update(max_fk_position_error_m=adapter.max_fk_position_error,
                          max_fk_rotation_error_rad=adapter.max_fk_rotation_error)
    except Exception as exc:
        result.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if adapter is not None and adapter.rows:
            np.savez_compressed(output/'trajectory.npz',**{k:np.asarray([r[k] for r in adapter.rows]) for k in adapter.rows[0]})
        result['total_wall_s']=time.perf_counter()-start
        result_file.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='hplc_injector_plunger')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=isaac_gpu())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    result=run(args.task,args.seed,args.output,args.render,args.gpu)
    raise SystemExit(0 if result['status']=='TASK_SUCCEEDED' else 1)


if __name__=='__main__':main()
