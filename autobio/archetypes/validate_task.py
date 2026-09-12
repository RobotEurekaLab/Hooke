"""Generic automatic-validation harness for AutoBio/Hooke tasks.

Phase I of the research program (see private/proposal.tex) plans to scale
the task/asset library well beyond hand-curation, which only works if every
generated (or refactored) task can be automatically checked for basic
usability -- does it compile, does the simulation stay numerically stable,
does its own scripted expert actually succeed at it -- rather than relying
on a human staring at a video. `archetypes/validate_rotor_variants.py`
was a first, narrowly-scoped version of this idea (built while validating
Step 2's rotor-capacity variants); this module generalizes it to any
Task/Expert pair, not just that one family, so it can be reused for the
lever-lock centrifuges (Step 1), rotor variants (Step 2), and whatever
Step 4's task-synthesis pipeline eventually produces.

Usage:
    from archetypes.validate_task import validate_task
    result = validate_task(task_cls, expert_cls, episodes=10)

Or from the command line, against anything in task.py's registry:
    python -m archetypes.validate_task centrifuge_5430_close_lid centrifuge_5910_lid_close --episodes 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def validate_task(task_cls, expert_cls, episodes: int = 10, seed0: int = 0, name: str | None = None) -> dict:
    """Runs `expert_cls`'s own scripted policy for `episodes` seeds and reports
    compile/stability/success statistics. Does not raise on a per-episode
    failure (IK errors, MuJoCo warnings, etc.) -- those count as
    divergences, which is the point of this harness."""
    name = name or getattr(task_cls, "__name__", str(task_cls))

    try:
        spec = task_cls.load()
    except Exception as e:
        return {
            "name": name, "compiled": False, "error": str(e),
            "episodes": episodes, "successes": 0, "divergences": 0, "success_rate": None,
        }

    successes = 0
    divergences = 0
    errors = []
    for i in range(episodes):
        seed = seed0 + i
        try:
            expert = expert_cls(task_cls.load())
            expert.reset(seed)
            expert.execute()
        except Exception as e:
            divergences += 1
            errors.append(f"seed={seed}: {type(e).__name__}: {e}")
            continue
        if np.any(expert.data.warning.number):
            divergences += 1
            errors.append(f"seed={seed}: MuJoCo warning flagged")
        elif expert.check():
            successes += 1

    return {
        "name": name,
        "compiled": True,
        "episodes": episodes,
        "successes": successes,
        "divergences": divergences,
        "success_rate": successes / episodes,
        "errors": errors,
    }


def validate_by_name(task_name: str, episodes: int = 10, seed0: int = 0) -> dict:
    """Convenience wrapper for anything resolvable via task.py's registry."""
    from task import get_task_class
    task_cls = get_task_class(task_name)
    return validate_task(task_cls, task_cls.Expert, episodes=episodes, seed0=seed0, name=task_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_names", nargs="+", help="Task names resolvable via task.get_task_class")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    results = []
    for name in args.task_names:
        print(f"=== validating '{name}' ===")
        try:
            result = validate_by_name(name, episodes=args.episodes)
        except Exception as e:
            result = {"name": name, "compiled": False, "error": f"{type(e).__name__}: {e}"}
        results.append(result)
        if result.get("compiled"):
            print(f"  success_rate={result['success_rate']}, divergences={result['divergences']}")
        else:
            print(f"  FAILED TO RESOLVE/COMPILE: {result.get('error')}")

    print("\n" + json.dumps(results, indent=2))
