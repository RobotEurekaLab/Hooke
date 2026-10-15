# University laboratory fidelity and acceptance

## Current scope: 15 October 2026

The 12-item catalogue acceptance checklist is complete at its declared scope.
The catalogue contains **120 reference-informed laboratory scenes**. Runtime
checks, raw controls, bounded operations, visual fidelity and scientific models
have separate acceptance criteria; a passing category does not certify the others.
These results cover `real_labs.catalog`; the separate `real_lab` collection added
upstream is outside this acceptance cohort.

| Check | Verified result |
| --- | --- |
| Native loading and non-control gates | 120/120 scenes pass all 11 gates |
| Original scalar-control diagnostics | **546/555 pass; nine failures retained**; 118/120 scenes pass all 12 raw gates |
| Raw native accounting | 514,700 public steps, 834,860 native steps, 860 observed resets |
| Complete bounded CPU routes | **120/120**: 88 full replay records and 32 complete-route testcases |
| Serialized CPU route accounting | 4,312,378 executed and checked steps; excludes testcase-only counters |
| Guarded native operations | **Four** complete operations across NASA, Rochester and Manchester |
| Native visual review | **120 scenes / 240 captures**; one brightness flag inspected and retained |
| Regression | 153 selected complete modules / 1,059 cases; nine default-environment skips passed separately |
| Final promotion regression | 124 focused cases pass without skips |
| Review and export metadata | 120 scoped review records; 120/120 current metadata bindings |

Regression totals combine completed module records and targeted rechecks;
they are not a single run of every test on one source snapshot.

## Supported native operations

| Scene | Qualified operation | Executed steps |
| --- | --- | --- |
| NASA planetary | Pickup, traversal, staging, release and withdrawal | 181,216 public / 724,864 native |
| NASA planetary | Settle, clearance lift and pi/2 yaw | Separately recorded bounded operation |
| Manchester collaborative robotics | Guarded empty-tool inspection and return | 17,659 native |
| Rochester genomics | Complete 31-phase plate loading and retrieval | 55,800 native |

The full NASA route completes 90.608 simulated seconds with all 181,216 source
checks executed. Maximum direct overlap is **38.751 micrometres against a
50-micrometre limit**. Final sample XY error is **0.513 mm**, height error
**0.061 micrometres**, support **0.245236 N**, and tool clearance **84.986 mm**.
The instrumented audit took 141.2 minutes; it does not qualify real-time or
training throughput. All native workers owned by this validation have exited.

The nine scalar failures remain: four NASA wheel commands, reset-state arm yaw,
both individual jaws, and two Rochester loader commands. Their coupled actions
use guarded operations with separate native evidence. Operation success does
not change an original scalar target, tolerance or recorded failure.

NASA's automatic profile uses explicit affine actuation, four native substeps,
bounded parking feedback and a named static-seat contact offset. Reported effort
is the applied simulation input verified against the native input buffer, not a
measured motor reaction. Passive forces and constraints remain separate;
implicit-drive motor effort remains unavailable. The source route, actuator
limits, geometry and post-step guards are preserved.

Production export matches the executed NASA archive for five physical/manifest
files and all 412 compiled/reset arrays. The 65 backend files and effective
profile match. A separately reviewed post-run preflight correction rejects
invalid sample/seat collision interfaces; all 120 selected profiles are unchanged.

## Reconstruction and scientific limits

Each scene retains its university / subject / laboratory classification and
reference provenance. Assess these dimensions independently:

| Area | Required evidence |
| --- | --- |
| Reference fidelity | Source URLs, observed features, dimension basis and explicitly estimated layouts |
| Equipment and assembly | Geometry provenance, physical support, mounting, access and unintended-overlap checks |
| Mechanical usability | Observed motion, reset, contact geometry, inactive-axis stability and a bounded operating envelope |
| Executable tasks | Complete routes with observations, actions, guards and success criteria |
| Scientific models | Separately documented and validated experiment outputs |

The scenes are estimated reconstructions, not surveyed or institution-validated
digital twins. Indiana remains a representative microfluidics scene because its
source is a title card; Waterloo remains category-level because its panorama
cannot establish the modeled floor plan. Institution names identify references,
not endorsement. Manufacturer resemblance does not establish OEM CAD equivalence,
hardware accuracy or redistribution rights; retain each asset's license and
provenance. Restricted manufacturer reference files are not distributed here.

CPU route coverage and the four native operations do not establish 120 complete
native scientific workflows. Fluid, thermal, optical, biological and material
responses require their own model-specific validation. NASA represents an
Earth-gravity test facility with rigid terrain, not validated planetary soil or
flight-rover performance. Recorded previews are separate from live simulation.

## Reproduce and inspect

From the repository root, with the existing Python environment, export models
and CPU mechanical checks into a fresh ignored output directory:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.export \
  --all --output temp/real_labs/recheck
```

Use `--scene rochester_genomics` instead of `--all` for one scene. Review the
exported manifest and validation records; export alone does not reproduce the
native acceptance audit or prove full task success. An established CPU operation:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.manchester_motion \
  --output temp/manchester_inspection.json
```

See [reference scene usage](real_labs.md#build-and-inspect) for Blender rendering,
[Isaac setup](isaac_setup.md) for installation discovery and
[backend operations](isaac_operations.md) for native configuration and diagnostics.
Use an available GPU and a configured installation for native execution; the CPU
commands above do not start an Isaac worker. No simulator or large asset download
is needed to inspect the source or existing local evidence.

The web route `/real-labs` presents recorded previews and live MuJoCo mechanical
controls. The reviewed 240 images are local generated artifacts and must be
regenerated or supplied separately after cloning; no fixed LAN address is required.

## Immutable local evidence

Detailed records and media remain under the ignored directory
`temp/university/optimization_120_20261004/native_completion_20261015/`.
These paths identify local evidence, not files bundled in a fresh clone:

- `review/final_acceptance_v1.json`: finite checklist and hashed evidence bindings.
- `catalog/final_raw120_summary_v2.json`: all 555 original diagnostics and failures.
- `catalog/final_actual_control_coverage_v3.json`: separate guarded-operation coverage.
- `workflows/current120_final_v1.json`: complete bounded CPU route reconciliation.
- `nasa/production_promotion_v1/summary.md` and `export_equivalence.json`: native route and source transfer.
- `review/native_visual_v5_nasa_v8_120/completion_review_v3.json`: visual and log review.

The raw-summary SHA-256 is
`11a3ba0cad029a23ca7632135bb18c2b265bcc01e3d40a1d8de655174eef7cf1`;
the CPU-route summary is
`067a083f097a85240bb7cd705c2228b9d64eb8a5912030c1d2ea0c00bbcc5370`.
Historical reports retain their original paths, hashes, settings and failures.
The complete pre-publication journal is preserved locally at
`temp/publication_20261016/university_optimization_v1/university_optimization.historical_draft.md`
with SHA-256 `60ab5850c39b4ec7844fd2eb2c930211bb6bec1a6a869fda283777b1cef8727f`.
