"""Execute catalogue experts with an isolated process and bounded runtime.

Loading/rendering evidence is deliberately separate from task outcomes. Every
result includes the real exit code, including native crashes with stale JSON.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from backends.run import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['mujoco','isaac'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',action='append')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-sim-seconds',type=float,default=120.)
    parser.add_argument('--wall-seconds',type=float,default=3600.,help='Maximum wall time per original expert')
    args=parser.parse_args()
    if not 0<args.wall_seconds<=86400:parser.error('Invalid wall-time allowance')
    root=Path(__file__).resolve().parents[2];output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((root/'temp/backend_parity/inventory.json').read_text())['entries']
    names=args.task or [r['task'] for r in inventory if not r['display_only']]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    results=[]
    for name in names:
        folder=output/name;folder.mkdir(exist_ok=True)
        command=[sys.executable,'-m','backends.run','--backend',args.backend,'--task',name,'--seed',str(args.seed),
                 '--output',str(folder),'--no-render','--max-sim-seconds',str(args.max_sim_seconds)]
        started=time.perf_counter()
        timed_out=False
        with (folder/'run.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=root/'Hooke',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=args.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out=True
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        try:result=json.loads((folder/'result.json').read_text())
        except (OSError,ValueError):result={'status':'ERROR','error':'Missing result file'}
        if result.get('status') in ('STARTING','RUNNING'):
            result.update(status='ERROR',error=f'Process ended without a final result, exit={process.returncode}')
        if timed_out:result.update(status='TIME_LIMIT',error=f'Expert exceeded {args.wall_seconds:g} seconds of wall time')
        result.update(task=name,backend=args.backend,exit_code=process.returncode,process_wall_s=time.perf_counter()-started,
                      parity_qualified=False,source_check_constant_true=next(r['check_constant_true'] for r in inventory if r['task']==name))
        write_json(folder/'result.json',result)
        results.append({k:v for k,v in result.items() if k not in ('conversion','final_qpos','traceback','cameras')})
        write_json(output/'task_results.json',{'scope':'Original expert execution; a true source predicate is not complete parity qualification.',
                    'backend':args.backend,'seed':args.seed,'total_requested':len(names),'completed':len(results),'results':results})
        print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
