"""Rotor-capacity-variant-specific wrapper around the generic
`archetypes.validate_task` harness: generates each requested variant (if not
already on disk) and confirms `instrument.num_slots` actually matches the
request, in addition to the generic compile/stability/success-rate checks.

See `archetypes/validate_task.py` for the reusable harness this is built on,
and `private/technical-log.md` for how this was used to validate the
10/15/20/24-slot variants.
"""
import argparse
import json

from load_centrifuge_5430 import InsertCentrifuge5430
from archetypes.rotor_variants import generate_rotor_variant
from archetypes.validate_task import validate_task


def validate(num_slots: int, episodes: int, seed0: int = 1000):
    scene_path = generate_rotor_variant(num_slots)
    task_cls, expert_cls = InsertCentrifuge5430.for_variant(scene_path, f"insert_centrifuge_5430_{num_slots}slot")

    detected_slots_match = None
    try:
        probe = task_cls(task_cls.load())
        detected_slots_match = probe.instrument.num_slots == num_slots
    except Exception:
        pass  # surfaced as compiled=False by validate_task below

    result = validate_task(task_cls, expert_cls, episodes=episodes, seed0=seed0, name=f"{num_slots}slot")
    result["num_slots"] = num_slots
    result["detected_slots_match"] = detected_slots_match
    return result


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
