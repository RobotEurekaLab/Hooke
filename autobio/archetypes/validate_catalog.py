"""Runs archetypes.validate_task's checks against every entry in
archetypes.task_catalog.CATALOG, respecting each entry's `task_override`
(for classes like ThermalCyclerManipulate that serve more than one named
task through the same Task/Expert pair). This is what actually certifies the
catalog Step 4 hands to the LLM is runnable, not just importable.

    python -m archetypes.validate_catalog --episodes 5
"""
import argparse
import json

import numpy as np

from archetypes.task_catalog import CATALOG


def validate_entry(entry, episodes: int, seed0: int = 0) -> dict:
    try:
        task_cls, expert_cls = entry.load_classes()
        task_cls.load()  # cheap compile check before spending episodes
    except Exception as e:
        return {"name": entry.name, "compiled": False, "error": str(e)}

    successes = 0
    divergences = 0
    errors = []
    for i in range(episodes):
        seed = seed0 + i
        try:
            expert = entry.make_expert()
            expert.reset(seed)
            expert.execute()
        except Exception as e:
            divergences += 1
            errors.append(f"seed={seed}: {type(e).__name__}: {e}")
            continue
        if np.any(expert.data.warning.number):
            divergences += 1
        elif expert.check():
            successes += 1

    return {
        "name": entry.name, "compiled": True, "episodes": episodes,
        "successes": successes, "divergences": divergences,
        "success_rate": successes / episodes, "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    results = [validate_entry(entry, args.episodes) for entry in CATALOG.values()]
    for r in results:
        if r["compiled"]:
            print(f"{r['name']}: success_rate={r['success_rate']}, divergences={r['divergences']}")
        else:
            print(f"{r['name']}: FAILED TO COMPILE: {r['error']}")
    print("\n" + json.dumps(results, indent=2))
