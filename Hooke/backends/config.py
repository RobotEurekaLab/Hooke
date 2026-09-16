"""Machine-specific installation settings stay outside version control."""

import json
import os
from pathlib import Path


def local_settings():
    local = Path(__file__).resolve().parents[2] / "temp/isaac_local.json"
    if not local.is_file():
        return {}
    value = json.loads(local.read_text())
    if not isinstance(value, dict):
        raise ValueError("Isaac local settings must be a JSON object")
    return value


def isaac_installation():
    override = os.environ.get("HOOKE_ISAAC_PATH")
    if override:
        return Path(override).expanduser()
    value = local_settings().get("isaac_path")
    if value:
        return Path(value).expanduser()
    return Path("/opt/isaacsim")


def isaac_gpu():
    value = int(os.environ.get("HOOKE_ISAAC_GPU", local_settings().get("gpu", 6)))
    if value < 0:
        raise ValueError("Isaac GPU index must be nonnegative")
    return value
