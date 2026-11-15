"""Client for NVIDIA ALCHEMI's Batched Geometry Relaxation (BGR) and Batched
Molecular Dynamics (BMD) NIM microservices.

Uses the standard library's urllib rather than adding `requests` as a
dependency -- this is ~30 lines of wire-protocol code, the same reasoning
archetypes/policy_client.py gives for vendoring instead of installing a real
package.

Both NIMs' own published examples are self-hosted Docker containers reached
at http://localhost:8000 with no Authorization header at all -- an NGC
personal API key authenticates the image pull, not the per-request HTTP
call. This client follows that: it only attaches an Authorization header if
an API key is actually configured, so it stays correct whether you're
talking to a local container or (if NVIDIA later exposes these behind a
hosted gateway) a keyed endpoint. Request/response shapes below are taken
directly from NVIDIA's published API reference -- nothing here is guessed:
  https://docs.nvidia.com/nim/alchemi/alchemi-bgr/api-reference
  https://docs.nvidia.com/nim/alchemi/alchemi-bmd/api-reference

Both NIMs take one or more atomic structures (element numbers + flat
cartesian coordinates, optionally a periodic cell) and a machine-learned
interatomic potential (MACE / TensorNet / AIMNet2, selected when the
container is launched) does the actual physics:
  - BGR relaxes a structure to its nearest local energy minimum (geometry
    optimization) and reports its energy, forces and whether it converged.
  - BMD runs a short molecular-dynamics trajectory (NVT/NVE/NPT) from a
    structure and reports the resulting positions/velocities/energies over
    time.

Neither NIM predicts "the outcome of mixing reagent A and B" as a single
call -- that would be a reactive-MD setup (both reactants' atoms placed in
one cell, left to evolve under BMD) or isn't something these two NIMs do at
all. See integrations/examples/chemistry_property_demo.py for a concrete,
runnable worked example against one real molecule, and
integrations/README.md for how far this is from full task-catalog wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any
import urllib.error
import urllib.request


class AlchemiTransportError(RuntimeError):
    """The HTTP request failed before a response was received (network, timeout, DNS)."""


class AlchemiRemoteError(RuntimeError):
    """The service responded, but reported failure or returned an unexpected schema."""


@dataclass(frozen=True)
class AlchemiConfig:
    """Connection settings for the two ALCHEMI NIMs.

    Defaults assume local self-hosted containers on two different ports
    (NVIDIA's own examples all default to 8000; run BGR and BMD on separate
    ports if both run on the same host). Override with the ALCHEMI_BGR_URL /
    ALCHEMI_BMD_URL environment variables if they run elsewhere on your
    network. api_key is optional and unused unless set.
    """

    bgr_url: str = field(default_factory=lambda: os.environ.get("ALCHEMI_BGR_URL", "http://localhost:8000"))
    bmd_url: str = field(default_factory=lambda: os.environ.get("ALCHEMI_BMD_URL", "http://localhost:8001"))
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("ALCHEMI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    )
    timeout_s: float = 120.0

    def __post_init__(self):
        # default_factory only runs when a field is omitted, so URL validation
        # has to live here to also cover a URL passed explicitly by a caller.
        for name in ("bgr_url", "bmd_url"):
            value = getattr(self, name)
            if not value.startswith(("http://", "https://")):
                raise ValueError(f"{name} must be an http(s) URL, got {value!r}")
            object.__setattr__(self, name, value.rstrip("/"))
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")


@dataclass(frozen=True)
class AtomicStructure:
    """One atomic structure, in the flat [x0,y0,z0,x1,y1,z1,...] layout both NIMs expect."""

    coord: list[float]
    numbers: list[int]
    cell: list[float] | None = None  # 9 floats, row-major 3x3; periodic systems only
    pbc: list[bool] = field(default_factory=lambda: [True, True, True])
    charge: int = 0
    mult: int = 1
    mass: list[float] | None = None
    velocity: list[float] | None = None
    active_mask: list[bool] | None = None  # BGR only: which atoms are allowed to move
    structure_id: str | None = None

    def __post_init__(self):
        n_atoms = len(self.numbers)
        if len(self.coord) != 3 * n_atoms:
            raise ValueError(f"coord must have 3*{n_atoms}={3 * n_atoms} entries, got {len(self.coord)}")
        if self.cell is not None and len(self.cell) != 9:
            raise ValueError("cell must be exactly 9 floats (a row-major 3x3 matrix)")
        if len(self.pbc) != 3:
            raise ValueError("pbc must have exactly 3 entries")
        if self.active_mask is not None and len(self.active_mask) != n_atoms:
            raise ValueError("active_mask must have one entry per atom")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "coord": self.coord,
            "numbers": self.numbers,
            "pbc": self.pbc,
            "charge": self.charge,
            "mult": self.mult,
        }
        for key, value in (
            ("cell", self.cell),
            ("mass", self.mass),
            ("velocity", self.velocity),
            ("active_mask", self.active_mask),
            ("structure_id", self.structure_id),
        ):
            if value is not None:
                payload[key] = value
        return payload


def _request(method: str, url: str, payload: dict | None, api_key: str | None, timeout_s: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise AlchemiRemoteError(f"{method} {url} returned HTTP {exc.code}: {body_text[:500]}") from exc
    except urllib.error.URLError as exc:
        raise AlchemiTransportError(f"{method} {url} failed: {exc.reason}") from exc
    if status != 200:
        raise AlchemiRemoteError(f"{method} {url} returned HTTP {status}: {body[:500]!r}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise AlchemiRemoteError(f"{method} {url} returned a non-JSON body: {body[:500]!r}") from exc


def health(config: AlchemiConfig, service: str) -> dict[str, Any]:
    """GET /v1/health/ready on the given service ("bgr" or "bmd"). Raises if unreachable."""
    if service not in ("bgr", "bmd"):
        raise ValueError('service must be "bgr" or "bmd"')
    base = config.bgr_url if service == "bgr" else config.bmd_url
    return _request("GET", base + "/v1/health/ready", None, config.api_key, config.timeout_s)


@dataclass(frozen=True)
class RelaxedStructure:
    coord: list[float]
    numbers: list[int]
    energy: float
    forces: list[float]
    converged: bool
    optimizer_nsteps: int
    structure_id: str | None = None
    cell: list[float] | None = None
    stress: list[float] | None = None


@dataclass(frozen=True)
class BGRResult:
    structures: list[RelaxedStructure]
    status: str
    info: str | None = None


def relax_geometry(
    config: AlchemiConfig,
    structures: list[AtomicStructure],
    *,
    force_tolerance: float = 0.001,
    pressure_tolerance: float = 0.001,
    optimize_cell: bool = False,
) -> BGRResult:
    """Relax one or more atomic structures via the Batched Geometry Relaxation NIM.

    POST {bgr_url}/v1/infer; batches across `structures` in one request, the
    way the NIM is designed to be used for throughput.
    """
    if not structures:
        raise ValueError("structures must be non-empty")
    payload = {
        "atoms": [s.to_payload() for s in structures],
        "opttol": force_tolerance,
        "opttol_pressure": pressure_tolerance,
        "cellopt": optimize_cell,
    }
    body = _request("POST", config.bgr_url + "/v1/infer", payload, config.api_key, config.timeout_s)
    if body.get("status") != "Success":
        raise AlchemiRemoteError(f"BGR relaxation did not succeed: {body.get('info') or body.get('status')}")
    relaxed = [
        RelaxedStructure(
            coord=a["coord"],
            numbers=a["numbers"],
            energy=a["energy"],
            forces=a["forces"],
            converged=a["converged"],
            optimizer_nsteps=a["optimizer_nsteps"],
            structure_id=a.get("structure_id"),
            cell=a.get("cell"),
            stress=a.get("stress"),
        )
        for a in body["atoms"]
    ]
    return BGRResult(structures=relaxed, status=body["status"], info=body.get("info"))


@dataclass(frozen=True)
class MDConfig:
    """A subset of the BMD NIM's `config` object; see the API reference for every field."""

    temperature_k: float = 300.0
    timestep_fs: float = 1.0
    md_time_ps: float = 10.0
    save_interval_steps: int = 100
    ensemble: str = "nvt"
    pressure: float = 0.0

    def __post_init__(self):
        if self.ensemble not in ("nvt", "npt", "nve"):
            raise ValueError('ensemble must be "nvt", "npt" or "nve"')

    def to_payload(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature_k,
            "dt": self.timestep_fs,
            "md_time_max": self.md_time_ps,
            "save_interval": self.save_interval_steps,
            "nvt": self.ensemble == "nvt",
            "npt": self.ensemble == "npt",
            "pressure": self.pressure,
        }


@dataclass(frozen=True)
class TrajectoryFrame:
    coord: list[float]
    velocity: list[float]
    energy: float
    istep: int
    md_time: float
    stress: list[float] | None = None


@dataclass(frozen=True)
class BMDResult:
    trajectory: list[TrajectoryFrame]
    status: str
    info: str | None = None

    @property
    def final_energy(self) -> float:
        if not self.trajectory:
            raise ValueError("Empty trajectory has no final energy")
        return self.trajectory[-1].energy


def run_molecular_dynamics(
    config: AlchemiConfig, structure: AtomicStructure, md_config: MDConfig | None = None
) -> BMDResult:
    """Run one batched-MD trajectory via the Batched Molecular Dynamics NIM.

    POST {bmd_url}/v1/infer. Unlike BGR this NIM takes a single structure
    per request (its own docs: "Run a molecular dynamics simulation on a
    single atomic structure").
    """
    md_config = md_config or MDConfig()
    payload = {"atoms": structure.to_payload(), "config": md_config.to_payload()}
    body = _request("POST", config.bmd_url + "/v1/infer", payload, config.api_key, config.timeout_s)
    frames = [
        TrajectoryFrame(
            coord=f["coord"],
            velocity=f["velocity"],
            energy=f["energy"],
            istep=f["istep"],
            md_time=f["md_time"],
            stress=f.get("stress"),
        )
        for f in body["trajectory"]
    ]
    return BMDResult(trajectory=frames, status=body.get("status", "unknown"), info=body.get("info"))
