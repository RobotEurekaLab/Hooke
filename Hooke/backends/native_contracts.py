"""Run small physical regression fixtures in the installed Isaac runtime.

Usage (from Hooke/): ../.venv/bin/python -m backends.native_contracts --output ../temp/native-contracts
These tests actuate real PhysX joints; they never copy the leader's pose to its follower.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from backends.baseline import write_snapshot
from backends.source_forces import update_kinematics
from backends.worker_client import IsaacWorker


def fixtures():
    for name,leader,follower,coefficient,offset,target in (
        ('prismatic_mimic','slide','slide',1.,0.,.03),
        ('revolute_mimic','hinge','hinge',-.5,.1,.6),
        ('mixed_mimic','hinge','slide',.05,.002,.6),
        ('fixed_tendon_mimic','slide','slide',1.,0.,.03),
    ):
        xml=f'''<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>
        <worldbody><body name="base"><geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
        <body name="leader" pos=".1 0 0"><joint name="leader" type="{leader}" axis="1 0 0"/>
        <geom type="sphere" size=".01" mass=".2" contype="0" conaffinity="0"/></body>
        <body name="follower" pos="-.1 0 0"><joint name="follower" type="{follower}" axis="1 0 0"/>
        <geom type="sphere" size=".01" mass=".2" contype="0" conaffinity="0"/></body></body></worldbody>
        <equality><joint joint1="follower" joint2="leader" polycoef="{offset} {coefficient} 0 0 0"/></equality>
        <actuator><position joint="leader" kp="500" kv="20"/></actuator></mujoco>'''
        if name=='fixed_tendon_mimic':
            xml=xml.replace('<actuator><position joint="leader"',
                '<tendon><fixed name="linkage"><joint joint="leader" coef=".5"/><joint joint="follower" coef=".5"/></fixed></tendon><actuator><position tendon="linkage"')
        yield name,xml,target,coefficient,offset


def run(output,worker,include_experimental=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,xml,target,coefficient,offset in fixtures():
        folder=output/name;source=folder/'source';source.mkdir(parents=True,exist_ok=True)
        spec=mujoco.MjSpec.from_string(xml);model=spec.compile();data=mujoco.MjData(model)
        data.qpos[1]=offset;data.ctrl[0]=target;mujoco.mj_forward(model,data)
        task=SimpleNamespace(spec=spec,model=model,data=data,task=name,
                             default_scene=source/'scene.xml',task_info={'camera_mapping':{}})
        write_snapshot(task,source)
        loaded=worker.call('load',source=str(source),output=str(folder/'isaac'),render=False);state=loaded['state']
        max_error=0.;max_settled_error=0.;max_fk=0.
        for step in range(200):
            state=worker.call('step',control=[target]);q=np.asarray(state['qpos'])
            max_error=max(max_error,abs(q[1]-coefficient*q[0]-offset))
            if step>=100:max_settled_error=max(max_settled_error,abs(q[1]-coefficient*q[0]-offset))
            data.qpos[:]=q;data.qvel[:]=state['qvel'];update_kinematics(model,data)
            for body,pose in state['body_poses'].items():
                max_fk=max(max_fk,float(np.linalg.norm(data.xpos[int(body)]-pose[:3])))
        result={'fixture':name,'final_qpos':state['qpos'],'max_coupling_error':max_error,
                'max_settled_coupling_error':max_settled_error,'settling_time_s':.2,
                'max_fk_position_error_m':max_fk,
                'implicit_tendon_drives':loaded['conversion'].get('implicit_tendon_drives',{}),
                'passed':bool(abs(q[0]-target)<.002 and max_settled_error<5e-5 and max_fk<5e-6)}
        if name=='fixed_tendon_mimic' and not result['implicit_tendon_drives']:result['passed']=False
        results.append(result);print(json.dumps(result),flush=True)
    source=output/'soft_connect/source';source.mkdir(parents=True,exist_ok=True)
    spec=mujoco.MjSpec.from_string('''<mujoco><option timestep=".002"/><worldbody><body name="mass" pos="0 0 1"><freejoint/><geom type="sphere" size=".05" mass=".1" contype="0" conaffinity="0"/></body></worldbody><equality><connect body1="mass" anchor="0 0 0" solref=".02 1" solimp=".95 .95 .001"/></equality></mujoco>''')
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    task=SimpleNamespace(spec=spec,model=model,data=data,task='soft_connect',default_scene=source/'scene.xml',task_info={'camera_mapping':{}})
    write_snapshot(task,source)
    worker.call('load',source=str(source),output=str(output/'soft_connect/isaac'),physics_options={'soft_connect_constraints':True})
    for _ in range(1000):
        mujoco.mj_step(model,data);state=worker.call('step',control=[])
    error=abs(data.qpos[2]-state['qpos'][2])
    result={'fixture':'soft_connect','source_sag_m':float(1-data.qpos[2]),'native_sag_m':float(1-state['qpos'][2]),
            'error_m':float(error),'passed':bool(error<2e-5 and state['qpos'][2]<.99999)}
    results.append(result);print(json.dumps(result),flush=True)
    # All 165 current catalogue scenes use noslip=2. Keep the additional
    # noslip=0 diagnostic available; it is not yet within its error gate.
    for noslip in ((2,0) if include_experimental else (2,)):
        name='static_friction' if noslip else 'regularized_friction'
        source=output/name/'source';source.mkdir(parents=True,exist_ok=True)
        # Match the catalogue's noslip pass: the source deliberately removes
        # friction regularization creep. With noslip=0, MuJoCo itself creeps.
        spec=mujoco.MjSpec.from_string(f'''<mujoco><option timestep=".002" integrator="implicitfast" noslip_iterations="{noslip}"/><worldbody><body pos="0 0 1"><joint type="slide" axis="0 0 1" frictionloss=".3" damping="1"/><geom type="box" size=".015 .015 .015" mass=".027" contype="0" conaffinity="0"/></body></worldbody></mujoco>''')
        model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        task=SimpleNamespace(spec=spec,model=model,data=data,task=name,default_scene=source/'scene.xml',task_info={'camera_mapping':{}})
        write_snapshot(task,source)
        loaded=worker.call('load',source=str(source),output=str(output/name/'isaac'))
        for _ in range(1000):
            mujoco.mj_step(model,data);state=worker.call('step',control=[])
        held=float(state['qpos'][0]);source_held=float(data.qpos[0])
        # Above the friction bound, both engines must allow motion.
        data.qfrc_applied[0]=-.1
        for _ in range(100):
            mujoco.mj_step(model,data);state=worker.call('step',control=[],extra_forces=[-.1])
        moved=float(state['qpos'][0]);source_moved=float(data.qpos[0])
        result={'fixture':name,'noslip_iterations':noslip,'source_hold_qpos':source_held,'native_hold_qpos':held,
                'source_loaded_qpos':source_moved,'native_loaded_qpos':moved,
                'passed':bool(abs(held-source_held)<.0001 and (not noslip or abs(held)<2e-6) and abs(moved-source_moved)<.0005 and moved<-.001
                              and loaded['conversion']['implicit_friction_joints'])}
        results.append(result);print(json.dumps(result),flush=True)
    # The original vortex task produces non-unit free-joint quaternions.
    # Exercise the reset path with an offset COM and angular velocity too:
    # using the unnormalized quaternion for the velocity frame is also wrong.
    name='nonunit_free_reset';source=output/name/'source';source.mkdir(parents=True,exist_ok=True)
    spec=mujoco.MjSpec.from_string('''<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" pos=".02 .01 .03" size=".04" mass=".1" contype="0" conaffinity="0"/></body></worldbody></mujoco>''')
    model=spec.compile();data=mujoco.MjData(model)
    data.qpos[3:7]=np.array([1.,0.,0.,1.])*np.sqrt(2.)
    data.qvel[:]=[.1,.2,.3,.2,.1,.4];mujoco.mj_forward(model,data)
    task=SimpleNamespace(spec=spec,model=model,data=data,task=name,default_scene=source/'scene.xml',task_info={'camera_mapping':{}})
    write_snapshot(task,source)
    state=worker.call('load',source=str(source),output=str(output/name/'isaac'))['state']
    expected=data.qpos.copy();expected[3:7]/=np.linalg.norm(expected[3:7])
    initial_pose_error=float(np.max(np.abs(np.asarray(state['qpos'])-expected)))
    initial_velocity_error=float(np.max(np.abs(np.asarray(state['qvel'])-data.qvel)))
    for _ in range(100):
        mujoco.mj_step(model,data);state=worker.call('step',control=[])
    position_error=float(np.linalg.norm(np.asarray(state['qpos'][:3])-data.qpos[:3]))
    orientation_error=float(min(np.linalg.norm(np.asarray(state['qpos'][3:7])-data.qpos[3:7]),
                                np.linalg.norm(np.asarray(state['qpos'][3:7])+data.qpos[3:7])))
    result={'fixture':name,'source_reset_quaternion_norm':2.,'initial_pose_error':initial_pose_error,
            'initial_velocity_error':initial_velocity_error,'position_error_m':position_error,
            'orientation_quaternion_error':orientation_error,
            'passed':bool(initial_pose_error<1e-6 and initial_velocity_error<1e-6 and position_error<5e-5 and orientation_error<1e-5)}
    results.append(result);print(json.dumps(result),flush=True)
    report={'scope':'Native joint coupling, compliant closure, catalogue noslip friction and free-joint reset fixtures.',
            'includes_experimental_noslip_zero':include_experimental,
            'passed':all(r['passed'] for r in results),'results':results}
    (output/'result.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--include-experimental',action='store_true',help='Also check the unqualified noslip=0 approximation')
    args=parser.parse_args()
    with IsaacWorker(args.output,args.gpu) as worker:report=run(args.output,worker,args.include_experimental)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
