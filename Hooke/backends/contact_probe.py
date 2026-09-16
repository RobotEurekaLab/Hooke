"""Isolated contact calibration, separate from original expert validation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from backends.baseline import write_snapshot
from backends.worker_client import IsaacWorker


def run(output,grasp,worker):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    static=[]
    for mass in (.1,10.):
        folder=output/f'sphere_{mass:g}';source=folder/'source';source.mkdir(parents=True,exist_ok=True)
        xml=f'''<mujoco><option timestep=".002"/><default><geom solref=".02 1" solimp=".95 .95 .001"/></default><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .1"><freejoint/><geom type="sphere" size=".1" mass="{mass}"/></body></worldbody></mujoco>'''
        spec=mujoco.MjSpec.from_string(xml);model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        task=SimpleNamespace(spec=spec,model=model,data=data,task='compliance_sphere',default_scene=source/'scene.xml',task_info={'camera_mapping':{}})
        write_snapshot(task,source)
        worker.call('load',source=str(source),output=str(folder/'isaac'),physics_options={'compliant_contact_scale':1.})
        for _ in range(1000):
            mujoco.mj_step(model,data);state=worker.call('step',control=[])
        row={'mass_kg':mass,'source_depth_m':float(.1-data.qpos[2]),'native_depth_m':float(.1-state['qpos'][2])}
        row['error_m']=abs(row['source_depth_m']-row['native_depth_m']);row['passed']=row['error_m']<5e-5
        static.append(row);print('STATIC',json.dumps(row),flush=True)
    archive=np.load(Path(grasp)/'trajectory.npz');q=archive['qpos'][-1].tolist();ctrl=archive['control'][-1].tolist()
    source=(Path(grasp)/'source').resolve();results=[]
    for scale in (0.,.1,1.,10.):
        folder=output/f'grasp_scale_{scale:g}'
        worker.call('load',source=str(source),output=str(folder),physics_options={'compliant_contact_scale':scale})
        worker.call('reset',qpos=q,qvel=np.zeros(archive['qvel'].shape[1]).tolist())
        for _ in range(300):state=worker.call('step',control=ctrl)
        pairs=sorted({tuple(sorted((c['geom1'],c['geom2']))) for c in state['contacts']})
        row={'compliant_scale':scale,'height_m':state['qpos'][17],'finger_qpos':state['qpos'][6:8],
             'required_contact':(41,87) in pairs,'contact_pairs':pairs,
             'source_predicate_at_probe_end':state['qpos'][17]>.925 and (41,87) in pairs}
        (folder/'state.json').write_text(json.dumps(state));results.append(row);print('GRASP',json.dumps(row),flush=True)
    report={'scope':'Static calibration and held-grasp diagnostic; not an original expert episode.',
            'static':static,'grasp_results':results}
    (output/'result.json').write_text(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grasp',type=Path,required=True);parser.add_argument('--gpu',type=int,default=6);args=parser.parse_args()
    with IsaacWorker(args.output,args.gpu) as worker:run(args.output,args.grasp,worker)


if __name__=='__main__':main()
