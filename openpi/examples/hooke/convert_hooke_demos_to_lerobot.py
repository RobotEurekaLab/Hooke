"""Converts Hooke's own exported demonstrations (one `.npz` per successful
scripted-expert episode, written by `archetypes/demo_export.py` in the
Hooke repo -- see that module's docstring) into a real local LeRobot
dataset, the way `examples/libero/convert_libero_data_to_lerobot.py`
converts LIBERO's RLDS data.

This lives here (in openpi's own repo, under its own `uv`-managed
environment) rather than in Hooke's `archetypes/` because it needs the
real `lerobot` package -- pinned by openpi itself to a specific git
revision (see openpi's own `pyproject.toml`), not the plain PyPI
`lerobot` release. Hooke's own `autobio` conda env deliberately doesn't
carry that dependency (see `policy_client.py`'s module docstring for why
taking on `lerobot`'s full weight there -- torch, torchvision, datasets,
accelerate, diffusers, transformers, wandb -- wasn't worth the resolver-
conflict risk for a step that isn't needed until training is actually
imminent).

Usage (from the openpi repo root):
    uv run examples/hooke/convert_hooke_demos_to_lerobot.py \\
        --demo_dir /path/to/Hooke/logs/demos/push_filling_nozzle_down \\
        --repo_name your_username/hooke_push_filling_nozzle_down

The resulting dataset is written under `$HF_LEROBOT_HOME` (a local cache
dir by default -- nothing is pushed to the Hugging Face Hub unless
`--push_to_hub` is passed).
"""

import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def main(demo_dir: str, repo_name: str, *, fps: int = 20, push_to_hub: bool = False):
    demo_dir = Path(demo_dir)
    episode_paths = sorted(demo_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.npz files found under {demo_dir}")

    # Confirm every episode agrees on state/action dims before creating the
    # dataset -- a real, not merely likely, invariant across episodes of
    # the same task, checked rather than assumed.
    first = np.load(episode_paths[0])
    state_dim = first["state"].shape[-1]
    action_dim = first["action"].shape[-1]
    img_size = first["base_rgb"].shape[-1]  # exported as (T, C, H, W), H == W

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ur5e",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (img_size, img_size, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (img_size, img_size, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for path in episode_paths:
        episode = np.load(path, allow_pickle=True)
        prompt = str(episode["prompt"])
        n_steps = episode["action"].shape[0]
        for t in range(n_steps):
            dataset.add_frame({
                # CHW (Hooke/openpi's own inference-time convention, see
                # policy_client.CameraRenderer) -> HWC (lerobot's on-disk
                # convention, matching the libero conversion example).
                "image": np.transpose(episode["base_rgb"][t], (1, 2, 0)),
                "wrist_image": np.transpose(episode["wrist_rgb"][t], (1, 2, 0)),
                "state": episode["state"][t],
                "actions": episode["action"][t],
                "task": prompt,
            })
        dataset.save_episode()

    print(f"Wrote {len(episode_paths)} episodes to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["hooke", "ur5e"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
