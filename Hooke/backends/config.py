"""Machine-specific installation settings stay outside version control."""
import json
import os
from pathlib import Path


def isaac_installation():
    override=os.environ.get('HOOKE_ISAAC_PATH')
    if override:
        return Path(override).expanduser()
    local=Path(__file__).resolve().parents[2]/'temp/isaac_local.json'
    if local.is_file():
        value=json.loads(local.read_text()).get('isaac_path')
        if value:
            return Path(value).expanduser()
    return Path('/opt/isaacsim')
