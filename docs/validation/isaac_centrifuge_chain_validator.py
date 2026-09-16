"""Measure one-reset insertion and closure against the selected real engine."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from archetypes.task_catalog import CATALOG
from archetypes.lid_lock import lid_lock_passes
from backends.assessment import EpisodeAssessment
from backends.baseline import write_snapshot
from backends.closed_loop import PhysXTaskAdapter
from backends.matrix import provenance
from backends.worker_client import IsaacWorker

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=('mujoco', 'isaac'), required=True)
parser.add_argument('--seeds', nargs='+', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=('expert', 'no_action'), default='expert')
args = parser.parse_args()
args.output.mkdir(parents=True)
report = dict(kind='actual_one_reset_insert_close_protocol', backend=args.backend,
    runtime=provenance(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    protocol_seconds=45., atomic_task_seconds=15., spin_executed=False, results=[])
for seed in args.seeds:
    output = args.output / f'seed_{seed}'
    output.mkdir()
    task = CATALOG['composite_insert_centrifuge_5430'].make_expert()
    task.reset(seed)
    mujoco.mj_forward(task.model, task.data)
    started = time.perf_counter()
    observer = EpisodeAssessment(task, 'composite_insert_centrifuge_5430')
    trajectory = []
    phases = []
    with ExitStack() as stack:
        if args.backend == 'isaac':
            worker = stack.enter_context(IsaacWorker(output / 'worker', 6))
            adapter = stack.enter_context(PhysXTaskAdapter(task, worker, output, render=False))
        else:
            (output / 'source').mkdir()
            write_snapshot(task, output / 'source')
        original = task.manager.step
        def step():
            if task.data.time >= 45.:
                raise TimeoutError('Continuous insert-close protocol exceeded 45 simulated seconds')
            control = task.data.ctrl.copy()
            original()
            observer.update()
            trajectory.append(dict(control=control, qpos=task.data.qpos.copy(), qvel=task.data.qvel.copy(),
                                   time=float(task.data.time)))
        task.manager.step = step
        if args.mode == 'expert':
            task.set_serializer(log_root=output / 'task_log', log_name='episode')
            for variant in ('insert_centrifuge_5430_step', 'close_lid_step'):
                task.task = variant
                task.arm.ik.initial_qpos = task.data.qpos[task.arm.jnt_span].copy()
                start = float(task.data.time)
                if variant == 'insert_centrifuge_5430_step':
                    task.planner = task._insert_planner
                    task._execute_insert()
                else:
                    task.planner = task._lever_planner
                    task._execute_close_lid()
                passed = bool(task.check())
                phases.append(dict(task=variant, start_s=start, end_s=float(task.data.time),
                                   success=passed, within_atomic_15s=task.data.time-start <= 15.))
                if not passed:
                    break
            task.finish()
        else:
            while task.data.time < 45.:
                task.manager.step()
        task.task = 'insert_centrifuge_5430_step'
        final_insertion = bool(task.check())
        closed = bool(lid_lock_passes(task.data, task.instrument))
        actual = observer.report()
        row = dict(seed=seed, mode=args.mode, reset_count=1, phases=phases,
            simulation_s=float(task.data.time), total_wall_s=time.perf_counter()-started,
            final_insertion_geometry_pass=final_insertion, final_insertion_assessment=actual,
            lid_closed_error_rad=abs(float(task.data.qpos[task.instrument.lid_qposadr])-task.instrument.lid_qpos_min),
            lid_lock_active=bool(task.data.eq_active[task.instrument.lid_lock]), lid_geometry_pass=closed,
            sequence_success=len(phases)==2 and all(p['success'] for p in phases) and final_insertion
                and actual['success'] is True and closed and task.data.time <= 45.,
            spin_executed=False, scientific_process_validated=False,
            max_fk_position_error_m=adapter.max_fk_position_error if args.backend=='isaac' else None)
    row['total_wall_s'] = time.perf_counter() - started
    row['wall_scope'] = 'After task reset through native worker closure; includes native startup, controls, recording and assessment, excludes task compilation/reset.'
    np.savez_compressed(output / 'trajectory.npz', **{key:np.asarray([r[key] for r in trajectory]) for key in trajectory[0]})
    (output / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    report['results'].append(row)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(seed, row['sequence_success'], row['simulation_s'], row['lid_closed_error_rad'], flush=True)
