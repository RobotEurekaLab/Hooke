# External NVIDIA integrations

Clients for the two services the project wants NVIDIA's help connecting to
chemistry/biology task simulation: **ALCHEMI** (chemistry/materials physics)
and **BioNeMo** (biomolecular AI). Written against NVIDIA's published API
references, not guessed:

- `alchemi.py` — Batched Geometry Relaxation (BGR) and Batched Molecular
  Dynamics (BMD) NIMs.
  https://docs.nvidia.com/nim/alchemi/alchemi-bgr/api-reference
  https://docs.nvidia.com/nim/alchemi/alchemi-bmd/api-reference
- `bionemo.py` — GenMol (constrained small-molecule generation) only, so
  far. https://docs.nvidia.com/nim/bionemo/genmol/latest/endpoints.html

## What's real right now

Both clients build the exact request bodies and parse the exact response
bodies NVIDIA documents, with validation and typed results. Nothing here
has been run against a live NIM in this sandbox (no containers running,
no GPU allocated to them here) — `tests/test_alchemi_client.py` and
`tests/test_bionemo_client.py` verify the request/response handling against
mocked HTTP, not a live service. Before trusting this against your own
deployment, run `integrations/examples/chemistry_property_demo.py` against
real containers and confirm it produces sane output.

## What's NOT here yet

**Nothing is wired into Hooke's task catalog.** The catalog's chemistry
tasks (`filter_cartridge_housing`, `hplc_injector_plunger`, ...) are about
manipulating lab hardware — they carry no molecular-level reagent data
(atomic numbers, coordinates) to hand to ALCHEMI, and no biological-sequence
data to hand to BioNeMo. Getting from "a client that can call these NIMs"
to "a Hooke task result that includes an ALCHEMI/BioNeMo prediction" needs,
in order:

1. A decision on where molecular/sequence data lives in the task-catalog
   schema (`archetypes/task_catalog.py`) — per task? per reagent/instrument
   asset shared across tasks?
2. Real structures for whichever reagents the first wired-up tasks actually
   use (not stand-ins like the demo's water molecule).
3. A call site in `backends/run.py`'s result assembly (see how
   `science_model.report()` is already threaded through `manager_step` for
   the existing microscopy/space scientific-process hooks — same shape of
   integration point) that calls `alchemi.relax_geometry` /
   `bionemo.generate_molecules` and attaches the result, with the same kind
   of explicit "scope"/"limitations" fields the rest of this codebase's
   `conversion`/`assessment` payloads already use, so a real prediction
   doesn't get silently conflated with a scripted placeholder.

## Configuration

Both NIM families are typically run as **self-hosted containers** — NVIDIA's
own published examples all call `http://localhost:8000`, and an NGC
personal API key authenticates the Docker image pull, not the per-request
HTTP call. Both clients only attach an `Authorization: Bearer` header if an
API key is actually configured, so the same code works unmodified if NVIDIA
later exposes a keyed hosted gateway for these.

| Env var | Default | Used by |
|---|---|---|
| `ALCHEMI_BGR_URL` | `http://localhost:8000` | Geometry relaxation |
| `ALCHEMI_BMD_URL` | `http://localhost:8001` | Molecular dynamics |
| `ALCHEMI_API_KEY` | unset | Both ALCHEMI NIMs, if the deployment needs one |
| `GENMOL_URL` | `http://localhost:8000` | GenMol |
| `BIONEMO_API_KEY` | unset | GenMol, if the deployment needs one |
| `NVIDIA_API_KEY` | unset | Fallback for either `*_API_KEY` above if set instead |

Before pointing either client at a real deployment, confirm with
`integrations.alchemi.health(config, "bgr")` / `"bmd"` that the container is
reachable — it hits `GET /v1/health/ready` and raises if not.
