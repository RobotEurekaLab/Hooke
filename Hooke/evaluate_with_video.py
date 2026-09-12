"""Evaluate a policy over a few episodes and keep the MuJoCo state logs so they
can be rendered into videos afterwards with render.py / render.bash.

This is a thin wrapper around evaluate.py's serial evaluation path that calls
task.set_serializer(...) before each episode, mirroring how demonstration
trajectories are logged in the *.py task-definition scripts.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from task import create_task
from evaluator import Evaluator
from evaluate import make_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--image_history", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_root", type=Path, required=True, help="Where to write per-episode MuJoCo state logs")
    parser.add_argument("--save", type=str, default=None, help="Optional path to dump per-episode success flags")
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**32 - 1, size=args.num_episodes).tolist()

    policy = make_policy(args.host, args.port)
    task = create_task(args.task)
    evaluator = Evaluator(task, image_history=args.image_history)

    results = []
    for i, seed in enumerate(tqdm(seeds)):
        task.reset(seed=seed)
        task.set_serializer(log_root=args.log_root, log_name=f"ep{i:03d}_seed{seed}")
        success = evaluator.evaluate(policy)
        results.append(bool(success))
        print(f"episode {i} (seed={seed}): {'SUCCESS' if success else 'FAIL'}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f)
    print(f"Success rate: {np.mean(results) * 100:.1f}% ({sum(results)}/{len(results)})")


if __name__ == "__main__":
    main()
