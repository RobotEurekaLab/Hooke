# Expansion: 100 additional university laboratories

## Scope and counting

This local review batch contains **100 additional distinct laboratory prototypes** at **76 institutions** in the pinned QS World University Rankings 2027 overall top 100. The 20 pre-existing scenes are frozen in `Hooke/university/batches/qs2027_100_labs_20261003.json` and cannot be recounted. Several rooms or views from one laboratory still count as one laboratory. Research leads, camera angles and exported file formats do not count as completed laboratories.

The classification stays **University → Subject → Laboratory → Scene / task**, with stable institution and laboratory IDs. New geometry and controls stay in modular `Hooke/real_labs/university_*_wave*.py` extensions; scene definitions stay in `Hooke/real_labs/scenes/`. `Hooke/university/scene_index.json` binds the implementation to its institutional identity. Source evidence and rights remain in the research register. The homepage layout and existing videos are not part of this expansion.

## Evidence and reconstruction

Each candidate requires an inspected official photograph, plan or actual experiment video showing its apparatus or room. A facility list, campus exterior, CAD advertisement, logo, or unrelated stock photograph does not qualify. Record the actual campus, room or local workstation; separate adjoining rooms when the source does not establish their adjacency. The project exclusion remains University of Toronto Yu Sun / AMNL.

Reconstruction preserves visible spatial and equipment features: for example, a recessed circular wave basin, a long towing tank, an optical bay behind curtains, a glovebox aisle or a suspended fabrication gantry. Hidden surfaces, unreported dimensions and added control mechanisms are explicitly authored estimates. A compact convection tank must not inherit the dimensions of a different flume described on the same page.

Equipment includes supported visible components, an identifiable specimen or carrier, real bounded mechanical controls, collision geometry and reset behavior. Validation exercises targets in both directions and checks intermediate travel and joint-limit combinations. Source dimensions receive instrument-specific checks where available.

These are **reference-informed mechanical prototypes**. Joint actuation does not validate fluid mechanics, optics, material failure, chemical processes, centrifuge loading or full scientific experiments. A static water surface does not produce waves. Exported USD/GLB geometry does not establish Isaac runtime or dynamics validation. Each scene records its narrower capabilities and gaps.

The [laboratory simulation preflight](laboratory_simulation_preflight.md) adds a structured object/physics/camera inventory and an optional archive for the existing Isaac bridge. It follows the NVIDIA scene-preparation workflow supplied during the expansion, while keeping local structural checks separate from official SimReady and runtime validation.

## Review stages

The batch report separates:

1. **Evidence:** an actual laboratory source was visually inspected and linked to the implementation.
2. **Implemented:** a distinct laboratory has source code, scene definition and university binding.
3. **Mechanical:** the current generated MJCF passes exported control checks.
4. **Rendered:** editable and visual assets exist, with three nonblank views corresponding to the current source geometry.
5. **Visually reviewed:** explicit inspection notes refer to the current scene hash and all three image hashes.
6. **Ready for user review:** at least 100 new, distinct, in-scope laboratories satisfy these stages.

The report does not confer institutional approval, scientific validation or distribution rights. Source photographs and generated media stay under ignored `temp/`. The source release includes all 120 laboratory prototypes and explicitly marks specialist validation as pending.

## Reproduce the local review

Run from the repository root:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.export \
  --batch Hooke/university/batches/qs2027_100_labs_20261003.json \
  --output temp/university/expansion_100_20261003/scenes
```

Render completed scenes using the installed Blender and CPU; do not start Isaac services. Example for one scene:

```bash
CUDA_VISIBLE_DEVICES='' temp/visual_upgrade/tools/blender-4.5.14-linux-x64/blender \
  -b --python Hooke/real_labs/render.py -- \
  --input temp/university/expansion_100_20261003/scenes \
  --scene kyoto_yokokawa_nanometrics \
  --samples 32 --width 1120 --height 700 --threads 8
```

After the selected batch has current renders, assemble its gallery:

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.review \
  --input temp/university/expansion_100_20261003/scenes \
  --batch Hooke/university/batches/qs2027_100_labs_20261003.json
```

The gallery preserves three equal-aspect views per scene. Contact sheets paginate ten scenes at a time. Scene packages contain MJCF, Blender, GLB, a USD visual snapshot, source metadata, floor plans and mechanical reports. A reviewed subset can use its own manifest without changing the 100-lab completion target.

Explicit visual inspection records live in the local `visual_reviews.json`, keyed by scene ID. Each record includes `reviewer`, `observations`, `source_urls`, `passed`, `scene_sha256`, and an `image_sha256` mapping for `overview`, `interior`, and `workstation`. Record a pass only after inspecting the actual images. Changing geometry or an image invalidates the previous review.

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.batches \
  --batch Hooke/university/batches/qs2027_100_labs_20261003.json \
  --input temp/university/expansion_100_20261003/scenes \
  --visual-reviews temp/university/expansion_100_20261003/visual_reviews.json \
  --output temp/university/expansion_100_20261003/progress.json
```

Missing exports or reviews are reported as pending. Duplicate laboratories, baseline recounting, out-of-scope schools and inconsistent classification are rejected. A source change requires fresh export/render checks; matching old files alone cannot pass the current-source check.

## Local review result — 2026-10-04

| Measure | Result |
| --- | --- |
| Additional distinct laboratories | 100 |
| Universities in the pinned ranking scope | 76 |
| Source image records checked against local hashes | 125 |
| Mechanically checked controls in the new scenes | 246 |
| Current CPU-rendered screenshots | 335: 300 primary views and 35 additional views |
| Primary views explicitly inspected | 300, across all 100 new scenes |
| Paginated contact sheets | 20, plus one compact 100-scene overview |
| Current portable native archives and structural inventories | 120, including the 20 baseline scenes |
| Relevant automated checks | 230 Python tests and 7 JavaScript tests passed |

The screenshot-only offline gallery and archive are generated under
`temp/university/expansion_100_20261003/`. The full local media revision is
`temp/real_labs/revision5_20261004/`. Generated images, logs and archives remain
ignored by Git. The source release includes scene generators, equipment,
classification, evidence metadata and checks; specialist validation remains pending.

The count includes local workstations where photographs do not establish a
whole-room layout. These remain reference-informed mechanical prototypes with
authored dimensions, masses and hidden structures. Complete scientific
processes, autonomous robot task performance, photometric calibration and
Isaac runtime behavior have not been qualified by this batch.
