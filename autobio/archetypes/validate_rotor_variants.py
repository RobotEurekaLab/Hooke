"""Automatic validation harness for generated rotor-capacity variants.

For each requested slot count, generates the variant (if not already on
disk), then runs the scripted expert for `episodes` seeds and reports:
  - compile rate: did the MJCF load and did `instrument.num_slots` match
    the requested count (always checked, should be 100% by construction).
  - simulation stability: did every episode finish without a MuJoCo
    warning (NaN/divergence)?
  - task success rate: how often the scripted expert's own `check()`
    reports success (this is the number that matters for whether a
    variant is actually usable, not just physically stable -- e.g. the
    `slot_window` reachability assumption inherited from the 30-slot
    original may not hold at a different spacing).

This is the Step 3 ("automatic validation") idea applied narrowly to Step
2's first asset family, ahead of building the general harness -- see
private/technical-log.md.
"""
import argparse
import json

import numpy as np

from load_centrifuge_5430 import InsertCentrifuge5430
from archetypes.rotor_variants import generate_rotor_variant


def validate(num_slots: int, episodes: int, seed0: int = 1000):
    scene_path = generate_rotor_variant(num_slots)
    task_cls, expert_cls = InsertCentrifuge5430.for_variant(scene_path, f"insert_centrifuge_5430_{num_slots}slot")

    compiled = True
    detected_slots = None
    try:
        probe = task_cls(task_cls.load())
        detected_slots = probe.instrument.num_slots
    except Exception as e:
        compiled = False
        print(f"  [{num_slots} slots] FAILED TO COMPILE: {e}")

    successes = 0
    divergences = 0
    if compiled:
        for i in range(episodes):
            seed = seed0 + i
            expert = expert_cls(task_cls.load())
            expert.reset(seed)
            try:
                expert.execute()
            except Exception as e:
                divergences += 1
                print(f"  [{num_slots} slots] seed={seed} raised: {e}")
                continue
            if np.any(expert.data.warning.number):
                divergences += 1
            elif expert.check():
                successes += 1

    return {
        "num_slots": num_slots,
        "compiled": compiled,
        "detected_slots_match": detected_slots == num_slots,
        "episodes": episodes,
        "successes": successes,
        "divergences": divergences,
        "success_rate": successes / episodes if compiled else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_slots", type=int, nargs="+")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    results = []
    for n in args.num_slots:
        print(f"=== validating {n}-slot rotor variant ===")
        result = validate(n, args.episodes)
        results.append(result)
        print(f"  success_rate={result['success_rate']}, divergences={result['divergences']}, "
              f"detected_slots_match={result['detected_slots_match']}")

    print("\n" + json.dumps(results, indent=2))
