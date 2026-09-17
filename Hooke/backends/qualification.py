"""Fail-closed parity gate: importing an asset or a true task predicate is insufficient."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# These are evidence requirements, not claims that the prototype implements them.
REQUIRED = (
    'source_task_action_is_necessary',
    'geometry_and_units',
    'mass_inertia_and_initial_state',
    'constraints_and_actuators',
    'contact_and_task_dynamics',
    'closed_loop_task_success_across_seeds',
    'camera_and_material_visuals',
    'runtime_performance_budget',
)


def evaluate(evidence: dict) -> dict:
    checks = evidence.get('checks', {})
    unresolved = [name for name in REQUIRED
                  if not isinstance(checks.get(name), dict)
                  or checks[name].get('passed') is not True
                  or not checks[name].get('artifact')]
    return {'schema_version': 1, 'status': 'QUALIFIED' if not unresolved else 'UNQUALIFIED',
            'unresolved': unresolved, 'checks': checks,
            'scope': 'All required evidence must pass for the same task, scene variant, engine versions and configuration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.evidence.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'QUALIFIED' else 1)


if __name__ == '__main__':
    main()
