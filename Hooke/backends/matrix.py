"""Paired, resumable multi-seed regression with isolated episode timeouts.

Run from Hooke/: python -m backends.matrix --task close_fume_hood
    --seeds 0 1 2 --mode expert --mode no_action --output ../temp/matrix
JSON, CSV and Markdown are updated after every episode. Reuse requires
--resume and exactly matching code, scene configuration and run parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from backends.assessment import VERSION

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'TASK_SUCCEEDED', 'TASK_FAILED', 'CONTROL_COMPLETE', 'DISPLAY_COMPLETE'}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def provenance():
    paths = subprocess.check_output(
        ['git', 'ls-files', '-co', '--exclude-standard', '-z', '--', 'Hooke', 'webui', 'tests'],
        cwd=ROOT).decode().split('\0')
    hashes = {}
    for name in sorted(set(paths)):
        path = ROOT / name
        if path.is_file() and (path.suffix in ('.py', '.xml', '.json') or '.so' in path.suffixes):
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'inputs_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'input_files': hashes, 'assessment_version': VERSION,
            'python': sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'scipy')},
            'isaac_task_threads': os.environ.get('HOOKE_ISAAC_TASK_THREADS', '8')}


def compact(result, evidence):
    keys = ('task', 'backend', 'seed', 'mode', 'status', 'source_success', 'source_check_kind',
            'source_check', 'within_declared_time_limit', 'assessment', 'steps',
            'simulation_s', 'total_wall_s', 'max_fk_position_error_m')
    row = {key: result[key] for key in keys if key in result}
    row['evidence'] = evidence
    return row


def summarize(rows, requested):
    """Failures/errors remain in denominators; absence is never agreement."""
    groups = []
    for task in requested['tasks']:
        for mode in requested['modes']:
            for backend in requested['backends']:
                selected = [r for r in rows if (r['task'], r['mode'], r['backend']) == (task, mode, backend)]
                completed = [r for r in selected if r['status'] in TERMINAL]
                groups.append({'task': task, 'mode': mode, 'backend': backend,
                    'requested': len(requested['seeds']), 'recorded': len(selected),
                    'completed': len(completed), 'status_counts': dict(Counter(r['status'] for r in selected)),
                    'legacy_successes': sum(r.get('source_success') is True for r in completed),
                    'legacy_constant_count': sum(r.get('assessment', {}).get('legacy_constant') is not None for r in completed),
                    'v2_assessed': sum(r.get('assessment', {}).get('success') is not None for r in completed),
                    'v2_successes': sum(r.get('assessment', {}).get('success') is True for r in completed),
                    'v2_successes_within_time': sum(r.get('assessment', {}).get('success_within_time_limit') is True for r in completed),
                    'failure_reasons': dict(Counter(reason for r in completed for reason in r.get('assessment', {}).get('failure_reasons', [])))})
    pairs = []
    lookup = {(r['task'], r['seed'], r['mode'], r['backend']): r for r in rows}
    if set(requested['backends']) == {'mujoco', 'isaac'}:
        for task in requested['tasks']:
            for seed in requested['seeds']:
                for mode in requested['modes']:
                    a = lookup.get((task, seed, mode, 'mujoco'))
                    b = lookup.get((task, seed, mode, 'isaac'))
                    ready = bool(a and b and a['status'] in TERMINAL and b['status'] in TERMINAL)
                    legacy = ready and all(r.get('assessment', {}).get('legacy_constant') is None for r in (a, b))
                    audited = ready and all(r.get('assessment', {}).get('success') is not None for r in (a, b))
                    pairs.append({'task': task, 'seed': seed, 'mode': mode, 'completed': ready,
                                  'legacy_agreement': a['source_success'] == b['source_success'] if legacy else None,
                                  'v2_agreement': a['assessment']['success'] == b['assessment']['success'] if audited else None})
    return {'groups': groups, 'pairs': pairs, 'parity_qualified': False,
            'requested_episodes': len(requested['tasks']) * len(requested['seeds']) * len(requested['modes']) * len(requested['backends']),
            'recorded_episodes': len(rows), 'results': rows}


def write_report(output, rows, manifest):
    report = summarize(rows, manifest['parameters'])
    report['provenance'] = {k: v for k, v in manifest['provenance'].items() if k != 'input_files'}
    atomic_json(output / 'report.json', report)
    fields = ['task', 'mode', 'backend', 'requested', 'recorded', 'completed', 'legacy_successes',
              'legacy_constant_count', 'v2_assessed', 'v2_successes', 'v2_successes_within_time']
    with (output / 'report.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['groups'])
    lines = ['# Paired multi-seed regression', '',
             f"Assessment: `{VERSION}`. Recorded {len(rows)}/{report['requested_episodes']} episodes.", '',
             'Legacy outcomes and v2 assessments use different criteria. Agreement includes joint failures and does not establish physics equivalence.',
             'A v2 dash means unaudited. No-action successes are negative-control failures. All counts use the requested seeds as denominator.', '',
             '| Task | Mode | Backend | Completed | Legacy success | Constant checks | v2 success | v2 within time |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for g in report['groups']:
        n = g['requested']
        v2 = f"{g['v2_successes']}/{n}" if g['v2_assessed'] else '—'
        timed = f"{g['v2_successes_within_time']}/{n}" if g['v2_assessed'] else '—'
        lines.append(f"| {g['task']} | {g['mode']} | {g['backend']} | {g['completed']}/{n} | {g['legacy_successes']}/{n} | {g['legacy_constant_count']} | {v2} | {timed} |")
    lines += ['', '## Failure reasons', '']
    for g in report['groups']:
        if g['failure_reasons']:
            lines.append(f"- {g['task']} / {g['mode']} / {g['backend']}: {json.dumps(g['failure_reasons'], sort_keys=True)}")
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def episode(args, task, seed, mode, backend, folder):
    folder.mkdir(parents=True)
    command = [sys.executable, '-m', 'backends.run', '--task', task, '--seed', str(seed),
               '--backend', backend, '--mode', mode, '--seconds', str(args.control_seconds),
               '--max-sim-seconds', str(args.max_sim_seconds), '--gpu', str(args.gpu),
               '--no-render', '--output', str(folder)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.perf_counter()
    with (folder / 'process.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT / 'Hooke', env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=args.wall_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # The native worker is in this process group; ensure no orphan
            # survives even if the episode parent exited before its child.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if isinstance(exc, KeyboardInterrupt):
                raise
            timed_out = True
    path = folder / 'result.json'
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        result = {}
    result.update(task=task, seed=seed, mode=mode, backend=backend)
    if timed_out or result.get('status') not in TERMINAL | {'ERROR', 'TIME_LIMIT', 'STOPPED'} or process.returncode not in (0, 1):
        result.update(status='WALL_TIMEOUT' if timed_out else 'PROCESS_ERROR',
                      total_wall_s=time.perf_counter() - started, process_returncode=process.returncode)
        atomic_json(path, result)
    return compact(result, str(path.relative_to(args.output)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', action='append', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--backend', action='append', choices=['mujoco', 'isaac'])
    parser.add_argument('--mode', action='append', choices=['expert', 'no_action'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--control-seconds', type=float, default=2.)
    parser.add_argument('--max-sim-seconds', type=float, default=120.)
    parser.add_argument('--wall-seconds', type=float, default=600.)
    args = parser.parse_args()
    if not 0 < args.control_seconds <= args.max_sim_seconds <= 120 or args.wall_seconds <= 0:
        parser.error('Invalid duration')
    from archetypes.task_catalog import CATALOG
    if any(name not in CATALOG for name in args.task):
        parser.error('Unknown catalogue task')
    args.output = args.output.resolve()
    parameters = {'tasks': sorted(set(args.task)), 'seeds': sorted(set(args.seeds)),
                  'backends': args.backend or ['mujoco', 'isaac'], 'modes': args.mode or ['expert'],
                  'gpu': args.gpu, 'control_seconds': args.control_seconds,
                  'max_sim_seconds': args.max_sim_seconds, 'wall_seconds': args.wall_seconds,
                  'render': False}
    parameters['backends'] = list(dict.fromkeys(parameters['backends']))
    parameters['modes'] = list(dict.fromkeys(parameters['modes']))
    manifest = {'parameters': parameters, 'provenance': provenance()}
    manifest_path = args.output / 'manifest.json'
    if args.resume:
        prior = json.loads(manifest_path.read_text())
        # A commit of identical tested code is compatible; changed bytes are not.
        if prior['parameters'] != parameters or any(prior['provenance'][k] != manifest['provenance'][k]
                for k in ('inputs_sha256', 'assessment_version', 'python', 'packages', 'isaac_task_threads')):
            parser.error('Resume refused: parameters, code or scene configuration changed')
        manifest = prior
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('Output is not empty; use a new directory or --resume')
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    rows = []
    write_report(args.output, rows, manifest)
    for backend in parameters['backends']:
        for task in parameters['tasks']:
            for seed in parameters['seeds']:
                for mode in parameters['modes']:
                    folder = args.output / backend / task / f'seed_{seed}' / mode
                    cached = folder / 'matrix-result.json'
                    if cached.exists():
                        row = json.loads(cached.read_text())
                    else:
                        if folder.exists():
                            # Preserve interrupted evidence instead of overwriting it.
                            folder.rename(folder.with_name(mode + f'.interrupted-{time.time_ns()}'))
                        row = episode(args, task, seed, mode, backend, folder)
                        atomic_json(cached, row)
                    rows.append(row)
                    write_report(args.output, rows, manifest)
                    print(backend, task, seed, mode, row['status'], flush=True)
    failed = any(r['status'] not in TERMINAL or (r['mode'] == 'expert' and
                 (r.get('assessment', {}).get('success') is False or r.get('source_success') is False)) or
                 (r['mode'] == 'no_action' and r.get('assessment', {}).get('success') is True) for r in rows)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
