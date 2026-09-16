"""Check whether the source fume-hood task succeeds with unchanged reset controls."""
import argparse
import json
from pathlib import Path
import mujoco


def no_action(seed: int, seconds: float, counterbalance: bool = False):
    from archetypes.task_catalog import CATALOG
    entry = CATALOG['close_fume_hood']
    if counterbalance:
        from expert_common import set_gravcomp
        cls, expert_cls = entry.load_classes()
        spec = cls.load(); set_gravcomp(spec.body('/fume_hood:sash'))
        task = expert_cls(spec)
    else:
        task = entry.make_expert()
    task.reset(seed); mujoco.mj_forward(task.model, task.data)
    initial = float(task.data.qpos[task.sash_jnt_adr])
    first_success = None
    for _ in range(round(seconds/task.dt)):
        task.manager.step()
        if task.check() and first_success is None:
            first_success = float(task.data.time)
    return {'seed': seed, 'expert_executed': False, 'control': 'unchanged reset controls',
            'variant': 'experimental_gravity_compensated_sash' if counterbalance else 'original',
            'initial_sash_m': initial, 'final_sash_m': float(task.data.qpos[task.sash_jnt_adr]),
            'first_source_success_s': first_success, 'source_success': bool(task.check()),
            'simulation_s': float(task.data.time)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--counterbalance-fume-hood', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        parser.error('episodes and seconds must be positive')
    rows = [no_action(i, args.seconds, args.counterbalance_fume_hood) for i in range(args.episodes)]
    result = {'status': 'FAIL_SOURCE_TASK_CAUSALITY' if any(r['source_success'] for r in rows) else 'NO_FALSE_SUCCESS_OBSERVED',
              'no_action_successes': sum(r['source_success'] for r in rows), 'episodes': args.episodes,
              'results': rows, 'scope': 'A no-action failure is necessary, but does not prove correct robotic grasping.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['no_action_successes'] else 0)


if __name__ == '__main__':
    main()
