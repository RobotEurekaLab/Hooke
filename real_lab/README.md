# Real Lab Reconstruction — QS 101-250 pilot

Status: **Phase 1 (candidate identification/research) in progress.**

This directory follows the process in
`private/real_lab_reconstruction_collaboration_guide.md` (gitignored,
not in this tree — ask the project coordinator for a copy). It does
**not** yet contain any 3D scenes, equipment assets, or simulation
files — that's Sections 9-11 of the guide, a separate phase that needs
real CAD/manufacturer research per piece of equipment and isn't
something this pass produces. See `project_config.yaml` for the exact
scope line.

## What's here

- `project_config.yaml` — Template A (project config) from the guide.
- `schools.csv` — the 150 schools ranked 101-250 in the QS World
  University Rankings 2025, seeded with `search_status: pending`.
  Columns follow the guide's Section 3.4 field list.
- `labs/<school_id>.md` — one Lab Candidate Card (Template B) per
  school, written only from real, sourced web research. A school with
  no verifiable research lab found this round is still written, with
  `筛选结论` (screening conclusion) marked "not included this round"
  per the guide's own category, rather than silently omitted.
- `search_log.md` — running search log (Section 8.3 format), one entry
  per research batch.

## Discipline scope this round

Per the guide's Section 3.1 keyword table: molecular biology &
genomics, microscopy & cell manipulation, chemistry & synthesis, plus
any other real experimental lab encountered (unrestricted beyond that
— "other/all" was explicitly selected for this pass).

## Honesty rules actually enforced in these cards

- Every claim traces to a URL actually fetched during research.
- No invented equipment, photos, or room layouts.
- Uncertain or unverified fields say so explicitly rather than guessing.
- A school without a verifiable lab is marked as such, not skipped.
