"""Export scripted-expert rollouts as training demonstrations, the other
missing piece (alongside `policy_rollout.py`) for eventually fine-tuning
π0.5 on Hooke's own tasks (private/proposal.tex Phase I item 3, "expert
trajectory synthesis").

**Deliberately stops one step short of a real LeRobot dataset.** The
actual `lerobot` package (the format `openpi`'s training configs expect,
e.g. `openpi/examples/ur5/README.md`'s `LeRobotUR5DataConfig`) pulls in
torch, torchvision, datasets, accelerate, diffusers, transformers, wandb
-- multi-GB, and a real risk of resolver conflicts with this project's
`autobio` conda env (which already needed careful handling once this
session, when `openpi-client`'s own numpy<2.0 pin conflicted with the
numpy 2.4.6 mujoco/jax need -- see `policy_client.py`'s docstring; that
was a small, safe vendoring fix, this is a much heavier package with no
equivalent workaround). Not worth that risk to build the LAST conversion
step before it's actually needed for a training run the user explicitly
said isn't imminent.

This module instead exports each successful episode as one self-
contained `.npz` (images, per-step state/action, and the language prompt)
using the *exact same* observation construction as `policy_rollout.py`
(`policy_client.CameraRenderer`), so an exported demo and a live policy
rollout can't silently diverge in how they render the same task. Turning
a directory of these into a real `LeRobotDataset` later is a short,
mechanical script (outlined at the bottom of this docstring) to run once
`lerobot` is actually installed -- in its own environment, not `autobio`.

Only *successful* episodes are exported (`check() == True`) -- training
on a scripted expert's own failures would teach the model to fail the
same way; AutoBio's own Phase I deliverable language calls for
"expert-solvability" filtering for exactly this reason.

## Later conversion to a real LeRobot dataset (not run here)

```python
# In a fresh env with `pip install lerobot` (NOT autobio):
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np, glob

ds = LeRobotDataset.create(
    repo_id="your-username/hooke_push_filling_nozzle_down",
    fps=20,
    features={
        "observation.state": {"dtype": "float32", "shape": (7,)},
        "observation.images.base_rgb": {"dtype": "video", "shape": (3, 224, 224)},
        "observation.images.wrist_rgb": {"dtype": "video", "shape": (3, 224, 224)},
        "action": {"dtype": "float32", "shape": (7,)},
    },
)
for path in sorted(glob.glob("logs/demos/push_filling_nozzle_down/*.npz")):
    ep = np.load(path)
    for t in range(len(ep["action"])):
        ds.add_frame({
            "observation.state": ep["state"][t],
            "observation.images.base_rgb": ep["base_rgb"][t],
            "observation.images.wrist_rgb": ep["wrist_rgb"][t],
            "action": ep["action"][t],
        }, task=str(ep["prompt"]))
    ds.save_episode()
```
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np

from archetypes.policy_client import CameraRenderer
from archetypes.task_catalog import CatalogEntry

DEMO_ROOT = Path(__file__).parent.parent / "logs" / "demos"


@dataclasses.dataclass
class ExportStats:
    attempted: int
    exported: int

    @property
    def yield_rate(self) -> float:
        return self.exported / self.attempted if self.attempted else 0.0


def export_episode(entry: CatalogEntry, seed: int, camera: CameraRenderer, control_freq: int = 20) -> dict | None:
    """Runs one scripted-expert episode, recording the exact same
    observation shape `policy_rollout.py` would see at eval time, plus
    the action actually commanded at each control tick. Returns the
    recorded arrays, or `None` if the episode didn't succeed (see module
    docstring on why failures aren't exported)."""
    expert = entry.make_expert()
    expert.reset(seed)

    arm = expert.arm
    period = max(1, int(round(1.0 / expert.dt / control_freq)))

    states, base_rgbs, wrist_rgbs, actions = [], [], [], []
    prompt = expert.task_info.get("prefix", expert.task)

    # Wrap step_and_log so every control tick this Expert already takes
    # (via move_to/gripper_control/wait -- see expert_common.py) gets
    # recorded, instead of re-deriving the control loop for export the
    # way policy_rollout.py has to (that one *replaces* the scripted
    # motion; this one just observes it).
    tick_counter = {"n": 0}
    orig_step_and_log = expert.step_and_log

    def recording_step(info):
        result = orig_step_and_log(info)
        tick_counter["n"] += 1
        if tick_counter["n"] % period == 0:
            obs = camera.get_observation(expert)
            states.append(obs["state"])
            base_rgbs.append(obs["base_rgb"])
            wrist_rgbs.append(obs["wrist_rgb"])
            actions.append(np.concatenate([
                expert.data.ctrl[arm.act_span].copy(),
                [expert.data.ctrl[arm.gripper_id]],
            ]).astype(np.float32))
        return result

    expert.step_and_log = recording_step
    try:
        expert.execute()
    except Exception as e:
        # A scripted expert can raise mid-episode on a bad seed (e.g. an
        # IK-unreachable assertion) -- treat that the same as a failed
        # check(): skip this seed, don't crash the whole batch. Printed
        # (not silent) so a task that excepts on *every* seed is still
        # obviously broken rather than reporting a quiet 0/N.
        print(f"  seed {seed}: expert.execute() raised {type(e).__name__}: {e}")
        return None

    if not expert.check() or not actions:
        return None

    return {
        "prompt": prompt,
        "state": np.stack(states),
        "base_rgb": np.stack(base_rgbs),
        "wrist_rgb": np.stack(wrist_rgbs),
        "action": np.stack(actions),
    }


def export_demos(entry: CatalogEntry, seeds: list[int], out_dir: Path | None = None, img_size: int = 224) -> ExportStats:
    """Exports every successful seed in `seeds` for `entry` as one .npz
    per episode under `out_dir` (default `logs/demos/<entry.name>/`)."""
    out_dir = out_dir or (DEMO_ROOT / entry.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = CameraRenderer(img_size=img_size)

    exported = 0
    for seed in seeds:
        episode = export_episode(entry, seed, camera)
        if episode is None:
            continue
        np.savez_compressed(out_dir / f"episode_{seed:05d}.npz", **episode)
        exported += 1

    return ExportStats(attempted=len(seeds), exported=exported)


if __name__ == "__main__":
    import argparse
    from archetypes.task_catalog import CATALOG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="a key in archetypes.task_catalog.CATALOG")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()

    entry = CATALOG[args.task_name]
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    stats = export_demos(entry, seeds)
    print(f"{args.task_name}: exported {stats.exported}/{stats.attempted} ({stats.yield_rate:.0%}) to {DEMO_ROOT / entry.name}")
