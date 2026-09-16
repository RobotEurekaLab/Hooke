"""Check original liquid updates using an already recorded native trajectory.

This is a state-propagation diagnostic, not a live engine or contact test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from archetypes.task_catalog import CATALOG
from backends.closed_loop import PhysXTaskAdapter
from liquid import ContainerSystem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();task=CATALOG[args.task].make_expert();task.reset(0)
    adapter=PhysXTaskAdapter.__new__(PhysXTaskAdapter);adapter.task=task
    adapter.max_fk_position_error=adapter.max_fk_rotation_error=0.
    qpos=task.data.qpos.copy()
    for joint,kind in enumerate(task.model.jnt_type):
        if kind==mujoco.mjtJoint.mjJNT_FREE:
            address=task.model.jnt_qposadr[joint]+3;qpos[address:address+4]/=np.linalg.norm(qpos[address:address+4])
    adapter.apply_state({'qpos':qpos,'qvel':task.data.qvel.copy(),'time':0.,'contacts':[]})
    systems=task.manager.systems_by_type.get(ContainerSystem,[])
    if not systems:raise ValueError('Task has no liquid container systems')
    maximum=0.;samples=0;initial_volumes=[s.container.volume for s in systems]
    with np.load(args.trajectory) as trajectory:
        for qpos,qvel,control,time in zip(trajectory['qpos'],trajectory['qvel'],trajectory['control'],trajectory['time']):
            task.data.ctrl[:]=control
            adapter.apply_state({'qpos':qpos,'qvel':qvel,'time':float(time),'contacts':[]})
            adapter.post_constraint(task.model,task.data)
            for system in systems:
                system.update(task.data);liquid=system.container.liquid
                if liquid is not None:
                    values=np.r_[liquid.surface_normal,liquid.surface.distance,liquid.surface.center,liquid.volume]
                    if not np.isfinite(values).all():raise RuntimeError(f'Non-finite liquid state at {time}')
                acceleration=np.zeros(6)
                mujoco.mj_objectAcceleration(task.model,task.data,mujoco.mjtObj.mjOBJ_GEOM,system.geom_id,acceleration,False)
                maximum=max(maximum,float(np.linalg.norm(acceleration[3:])))
            samples+=1
    report={'scope':'Recorded PhysX trajectory through original liquid systems with native-velocity-derived acceleration; no physics integration or contact validation.',
            'task':args.task,'trajectory':str(args.trajectory.resolve()),'passed':True,'samples':samples,
            'simulation_s':float(task.data.time),'max_container_proper_acceleration_m_s2':maximum,
            'initial_volumes_m3':initial_volumes,'final_volumes_m3':[s.container.volume for s in systems],
            'parity_qualified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
