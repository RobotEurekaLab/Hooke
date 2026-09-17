"""Export reset catalogue scenes without executing experts or changing assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback


def export_task(name: str, output: Path, seed: int = 0):
    import mujoco
    from archetypes.task_catalog import CATALOG
    from backends.baseline import write_snapshot
    entry = CATALOG[name]
    cls, _ = entry.load_classes()
    task = cls(cls.load())
    if entry.task_override:
        task.task = entry.task_override
    task.reset(seed)
    mujoco.mj_forward(task.model, task.data)
    output.mkdir(parents=True, exist_ok=True)
    meta = write_snapshot(task, output)
    meta.update(catalog_key=name,
                model_options={k: (getattr(task.model.opt, k).tolist()
                                  if hasattr(getattr(task.model.opt, k), 'tolist')
                                  else int(getattr(task.model.opt, k))
                                  if k in ('disableflags','enableflags','integrator','cone','solver')
                                  else float(getattr(task.model.opt, k)))
                               for k in ('disableflags','enableflags','integrator','cone','solver','timestep','gravity','impratio')})
    (output/'scene.json').write_text(json.dumps(meta, indent=2))
    return {'task': name, 'status': 'EXPORTED', 'source': str(output),
            'joint_types': sorted(set(map(int,task.model.jnt_type))),
            'max_joints_per_body': int(max(task.model.body_jntnum)),
            'plugins': int(task.model.nplugin), 'sensors': int(task.model.nsensor),
            'meshes': int(task.model.nmesh), 'geoms': int(task.model.ngeom),
            'cameras': int(task.model.ncam), 'joints': int(task.model.njnt)}


def main():
    from archetypes.task_catalog import CATALOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', action='append', choices=sorted(CATALOG))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.task and not args.all:
        parser.error('Choose --task NAME or --all')
    names = list(CATALOG) if args.all else args.task
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        start = time.perf_counter()
        try:
            row = export_task(name, args.output/name, args.seed)
        except Exception as exc:
            row = {'task': name, 'status': 'ERROR', 'error': str(exc), 'traceback': traceback.format_exc()}
        row['wall_s'] = time.perf_counter()-start
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'traceback'}), flush=True)
        (args.output/'export_results.json').write_text(json.dumps({'results': rows}, indent=2))
    raise SystemExit(0 if all(r['status'] == 'EXPORTED' for r in rows) else 1)


if __name__ == '__main__':
    main()
