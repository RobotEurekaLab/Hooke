"""Client for NVIDIA BioNeMo NIM microservices.

Uses the standard library's urllib rather than adding `requests` as a
dependency, for the same reason as integrations/alchemi.py.

Only GenMol (constrained small-molecule generation) is implemented here, on
purpose: it's the one BioNeMo NIM that pairs directly with
integrations/alchemi.py -- GenMol proposes candidate molecules from a
scaffold, ALCHEMI's BGR/BMD NIMs then physically relax and simulate the ones
worth keeping. The schema below is taken from NVIDIA's published docs:
  https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html

Every other BioNeMo NIM (AlphaFold2, ESMFold, RFdiffusion, MolMIM, DiffDock,
...) follows the same self-hosted POST-JSON container shape
(https://docs.nvidia.com/nim/bionemo/); add them here the same way, once
there's an actual Hooke task that needs one. NVIDIA also hosts some BioNeMo
NIMs (confirmed for AlphaFold2 and MolMIM) at
https://integrate.api.nvidia.com/v1 under an nvapi- key -- if GenMol is
hosted there too by the time you read this, point GENMOL_URL at that base
plus whatever path NVIDIA's current model catalog lists; that exact hosted
path isn't in the docs this file was written against, so it isn't guessed
here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import urllib.error
import urllib.request


class BioNeMoTransportError(RuntimeError):
    """The HTTP request failed before a response was received (network, timeout, DNS)."""


class BioNeMoRemoteError(RuntimeError):
    """The service responded, but reported failure or returned an unexpected schema."""


@dataclass(frozen=True)
class BioNeMoConfig:
    genmol_url: str = field(default_factory=lambda: os.environ.get("GENMOL_URL", "http://localhost:8000"))
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("BIONEMO_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    )
    timeout_s: float = 120.0

    def __post_init__(self):
        # default_factory only runs when genmol_url is omitted, so validation
        # has to live here to also cover a URL passed explicitly by a caller.
        if not self.genmol_url.startswith(("http://", "https://")):
            raise ValueError(f"genmol_url must be an http(s) URL, got {self.genmol_url!r}")
        object.__setattr__(self, "genmol_url", self.genmol_url.rstrip("/"))
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")


@dataclass(frozen=True)
class GeneratedMolecule:
    smiles: str
    score: float | None = None


@dataclass(frozen=True)
class GenMolResult:
    molecules: list[GeneratedMolecule]
    status: str


def generate_molecules(
    config: BioNeMoConfig,
    scaffold_smiles: str,
    *,
    num_molecules: int = 5,
    temperature: float = 1.0,
    noise: float = 1.0,
    scoring: str = "QED",
) -> GenMolResult:
    """Propose candidate molecules around a scaffold.

    `scaffold_smiles` is a SMILES string with a masked fragment GenMol is
    free to redesign, written as NVIDIA's own example shows it:
    "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1O.[*{15-15}]" (the [*{15-15}] span is
    the masked region, here fixed at exactly 15 atoms).

    POST {genmol_url}/generate.
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload = {
        "smiles": scaffold_smiles,
        "num_molecules": num_molecules,
        "temperature": temperature,
        "noise": noise,
        "scoring": scoring,
    }
    url = config.genmol_url + "/generate"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise BioNeMoRemoteError(f"POST {url} returned HTTP {exc.code}: {body_text[:500]}") from exc
    except urllib.error.URLError as exc:
        raise BioNeMoTransportError(f"POST {url} failed: {exc.reason}") from exc
    if status != 200:
        raise BioNeMoRemoteError(f"POST {url} returned HTTP {status}: {body[:500]!r}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BioNeMoRemoteError(f"POST {url} returned a non-JSON body: {body[:500]!r}") from exc
    if parsed.get("status") != "success":
        raise BioNeMoRemoteError(f"GenMol reported failure: {parsed.get('error', parsed.get('status'))}")
    molecules = [GeneratedMolecule(smiles=m["smiles"], score=m.get("score")) for m in parsed.get("molecules", [])]
    return GenMolResult(molecules=molecules, status=parsed["status"])
