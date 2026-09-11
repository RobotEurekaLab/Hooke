import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AutoBioInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding images for pi0/pi0.5, not pi0-FAST (mirrors libero_policy.py convention).
        mask_padding = self.model_type != _model.ModelType.PI0_FAST

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        wrist_image_2_mask = data["observation/wrist_image_2"] is not None
        if wrist_image_2_mask:
            wrist_image_2 = _parse_image(data["observation/wrist_image_2"])
        else:
            wrist_image_2 = np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": wrist_image_2,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": wrist_image_2_mask if mask_padding else np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AutoBioRepack(transforms.DataTransformFn):
    """Like transforms.RepackTransform, but tolerates AutoBio tasks whose dataset has no
    'wrist_image_2' column (single-arm tasks only log 'image' + 'wrist_image')."""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        return {
            "observation/image": flat["image"],
            "observation/wrist_image": flat["wrist_image"],
            "observation/wrist_image_2": flat.get("wrist_image_2"),
            "observation/state": flat["state"],
            "actions": flat["actions"],
            "prompt": flat["prompt"],
        }


@dataclasses.dataclass(frozen=True)
class AutoBioOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
