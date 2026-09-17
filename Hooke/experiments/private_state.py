"""Host-only reproducible sample assignment, separate from public episode seeds."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets


def host_key():
    return os.environ.get("HOOKE_SCIENCE_CAMPAIGN_KEY") or secrets.token_hex(32)


def persistent_host_key(path):
    """Called under the web job lock; persist paired-engine sample assignments."""
    configured = os.environ.get("HOOKE_SCIENCE_CAMPAIGN_KEY")
    if configured:
        return configured
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(dict(key=secrets.token_hex(32)), stream)
    return json.loads(path.read_text())["key"]


def sample_seed(key, public_seed):
    digest = hmac.new(
        key.encode(), str(public_seed or 0).encode(), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:16], "big")
