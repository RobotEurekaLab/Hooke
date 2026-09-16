"""Run physical fixtures and original experts in one isolated Isaac worker."""
from __future__ import annotations

import argparse
import faulthandler
from contextlib import redirect_stdout
from pathlib import Path

from backends.native_contracts import run as contracts
from backends.run import run, write_json
from backends.worker_client import IsaacWorker


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',action='append',required=True)
    parser.add_argument('--seed',action='append',type=int)
    parser.add_argument('--render-task',action='append',default=[])
    parser.add_argument('--max-sim-seconds',type=float,default=120.)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--skip-contracts',action='store_true')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    seeds=args.seed or [0];rows=[]
    with IsaacWorker(output,args.gpu) as worker:
        if not args.skip_contracts:
            checked=contracts(output/'contracts',worker)
            if not checked['passed']:raise RuntimeError('Native physical fixture failed; see contracts/result.json')
        for seed in seeds:
            for name in args.task:
                folder=output/name if len(seeds)==1 else output/f'seed_{seed}'/name
                folder.mkdir(parents=True,exist_ok=True)
                options=argparse.Namespace(output=folder,task=name,backend='isaac',mode='expert',
                    seed=seed,seconds=2.,max_sim_seconds=args.max_sim_seconds,gpu=args.gpu,
                    no_render=name not in args.render_task)
                with (folder/'run.log').open('w') as log,redirect_stdout(log):result=run(options,worker)
                row={k:v for k,v in result.items() if k not in ('conversion','final_qpos','final_contact_pairs','traceback','cameras')}
                rows.append(row)
                write_json(output/'task_results.json',{'scope':'Original expert predicates, separate from physics equivalence.',
                    'total_requested':len(args.task)*len(seeds),'completed':len(rows),'results':rows})
                print(name,seed,result['status'],result.get('simulation_s'),flush=True)
    raise SystemExit(1 if any(r['status'] in ('ERROR','TIME_LIMIT','TASK_FAILED') for r in rows) else 0)


if __name__=='__main__':main()
