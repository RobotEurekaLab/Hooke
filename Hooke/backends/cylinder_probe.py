"""Measure native cylinder cooking and timing without changing source assets.

This diagnostic does not enable convex cylinders in the normal runtime.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from backends.worker_client import IsaacWorker


def run(source,output,worker):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with np.load(source/'model.npz') as archive:
        types=archive['geom_type'];sizes=archive['geom_size'];control=archive['reset_ctrl'].tolist()
        ids=np.flatnonzero(types==5)[:32].tolist()
    rows=[]
    for approximate in (False,True):
        folder=output/('convex' if approximate else 'analytic')
        worker.call('load',source=str(source),output=str(folder),physics_options={'approximate_cylinders':approximate})
        profile=worker.call('profile_steps',control=control,steps=100)
        (folder/'profile.txt').write_text(profile['profile'])
        cooked=worker.call('collider_info',geoms=ids)
        bounds=[]
        for item in cooked:
            radius,height=map(float,sizes[item['geom'],:2]);row={'geom':item['geom'],'radius_m':radius,'half_height_m':height}
            if len(item['meshes'])==1 and 'vertices' in item['meshes'][0]:
                mesh=item['meshes'][0];vertices=np.asarray(mesh['vertices']);planes=np.asarray(mesh['planes'])
                radial=np.linalg.norm(vertices[:,:2],axis=1)
                sides=planes[np.abs(planes[:,2])<1e-5]
                matching=bool(np.max(abs(radial-radius))<1e-5 and abs(np.max(abs(vertices[:,2]))-height)<1e-5)
                if len(sides) and matching:
                    apothems=abs(sides[:,3])/np.linalg.norm(sides[:,:2],axis=1)
                    row.update(vertices=len(vertices),radial_error_bound_m=float(max(abs(apothems-radius).max(),abs(radial-radius).max())))
                else:row['geometry_bound']='Unknown: cooked dimensions or planes differ from the source primitive'
            else:row['geometry_bound']='Unknown: no single cooked convex mesh is exposed'
            bounds.append(row)
        result={'approximate_cylinders':approximate,'steps':100,'wall_s':profile['wall_s'],
                'final_qpos':profile['state']['qpos'],'contacts':len(profile['state']['contacts']),
                'cylinders':bounds,'parity_qualified':False}
        (folder/'profile.txt').write_text(profile['profile'])
        (folder/'cooked.json').write_text(json.dumps(cooked))
        rows.append(result);print(json.dumps(result),flush=True)
    report={'scope':'100 reset-control steps; geometry bound where cooking data is exposed. Not an expert or full dynamics comparison.','results':rows}
    (output/'result.json').write_text(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with IsaacWorker(args.output) as worker:run(args.source,args.output,worker)


if __name__=='__main__':main()
