# Microscopy asset reconstruction and offline rendering

The microscope uses a single geometry source for MuJoCo, the Isaac scene
export and optional Blender presentation renders. The Blender snapshot also
contains the observed tool poses, cell shapes, pneumatic hoses and cameras.
Rendering does not advance the experiment or replace its physics backend.

## Physical reference

The stand is independently reconstructed from Nikon's **TE2000-S** photographs
and dimensional drawings, specifically pages 3 and 24 of the
[official brochure](https://www.nikonusa.com/fileuploads/pdfs/TE2000_brochure.pdf).
The 25° eyepiece inclination follows
[Nikon's product announcement](https://www.microscope.healthcare.nikon.com/about/news/nikons-eclipse-te2000-inverted-research-microscope-sets-a-new-standard-for-live-cell-microscopy).
The model retains the 195 mm base width, 476.4 mm base depth, 611 mm overall
height and 449.4 mm eyepoint height. The observer elbow, transmitted-light
housing and condenser follow the photographed silhouette. The plain S front
casting excludes the front output port found on other TE2000 variants.

Unmarked curves, accessory interfaces, condenser clearance and the retained
290 mm sample height remain estimates. This is neither manufacturer CAD nor
a verified complete replica. Vendor photographs and logos are not embedded.
See [commercial-reference provenance](microscopy_commercial_reference.md),
[parallel-manipulator assets](microscopy_parallel.md) and
[cell-image formation](microscopy_imaging.md) for their separate sources and
calibration boundaries.

The [manufacturer drawing inventory](microscopy_oem_references.md) records
the original Sutter and Nikon dimensions and installation references used by
the optional [Sutter reference workstation](microscopy_sutter_reference.md).
The parallel-manipulator installation below describes the earlier profile.

The holder's visible adapter is a short side fork attached to its clamp,
leaving the pneumatic inlet clear. Added supports, fasteners and holder
finishes are visual geometry. An invisible axial proxy preserves the original
adapter's numerical inertia; it does not represent a physical part.
The monitor displays a labelled static **optical alignment target**. Actual
cell observations are shown in the application's microscope view.

## Manipulator installation

The parallel mechanisms stand upright on independent bench supports; only the
pipette adapters set the holding and injection elevations to 30° and 35°.
Their broad feet, square towers, boxed offset beams, gussets and open motor
cradles replace the earlier slender cantilever. The cradles leave room for
the motor projecting below each mechanism. Bench screws align with the
modelled breadboard grid. The layout follows the broad independent platforms
seen in [Sutter MT70/71](https://www.sutter.com/stages/mt70) and published
[robotic cell-injection apparatus](https://www.nature.com/articles/s41467-024-53108-5).

These are independently modelled adapters for the imported parallel mechanism,
not Sutter hardware or a verified commercial assembly. Some real systems do
use posts; support geometry and unsupported span matter more than that label.
The simulation fixes the supports rigidly. Collision-clearance checks cannot
establish real stiffness, vibration isolation, clamp preload or hardware fit.

## Create an editable Blender scene

Blender is optional and is not imported by the application. The renderer was
exercised with Blender 4.5 LTS. Configure the desired microscopy asset profile
as described in the [live-backend guide](microscopy_live_backends.md), then run
from the repository root:

```bash
PYTHONPATH=Hooke .venv/bin/python - <<'PY'
from microscopy.blender_export import export_snapshot
from microscopy.tasks import make_task

task = make_task('suction_injection')
task.reset(0)
task.autofocus()
export_snapshot(task.model, task.data, 'temp/microscopy_render/snapshot',
                task.runtime_visuals())
PY

blender --background --factory-startup \
  --python Hooke/microscopy/blender_render.py -- \
  --snapshot temp/microscopy_render/snapshot \
  --output temp/microscopy_render/beauty \
  --camera hero --width 1600 --height 1200 --samples 192
```

The default uses eight CPU threads. For an available NVIDIA GPU, prepend
`CUDA_VISIBLE_DEVICES=<physical-index>` and add `--device CUDA`. Select a free
GPU explicitly on shared servers. The application does not download Blender
or modify the driver.

Outputs include `hero.png`, an editable `workstation.blend` with packed image
textures, and `render.json`. Select `--camera instrument_closeup` to use the
exported simulation camera. Keep separate output directories for different
views because each render saves its camera and settings in `workstation.blend`.
Generated media and scene files belong under ignored `temp/`.

For a previously saved compiled model and state, the exporter also accepts:

```bash
PYTHONPATH=Hooke .venv/bin/python -m microscopy.blender_export \
  --model temp/capture/workstation.mjb --state temp/capture/state.npz \
  --visuals temp/capture/visuals.json --output temp/capture/blender
```

This records hashes of the source inputs. Geometry uses metres and compiled
local-to-world transforms; textures retain their compiled pixels and UVs.
Unsupported geometry fails explicitly. The exported file is a visual scene,
not a portable physics checkpoint or an optical ray-tracing instrument.

## Comparing images

Use the same exported camera, resolution and lighting for before/after
comparisons. Blender uses Cycles, area lighting and AgX colour management.
Named simulator materials receive approximate physically based finishes;
they are not measured reflectance data. MuJoCo's legacy material defaults
are not blindly treated as metal. These rendering choices are recorded in
`render.json`, so a presentation render remains distinguishable from an
Isaac RTX or MuJoCo EGL frame.

## Validation — 2026-09-19

The revised stand, holders and imaging model completed all five public cell
tasks on both MuJoCo and native Isaac: 124 workflow checks passed. The native
traces contain 52,851 actual PhysX steps. All 826 recorded images decoded
successfully. The associated regression suite passed 127 cases; 11 cases
require optional offline geometry dependencies and were skipped.

The new turned optical parts also pass the existing strict source-snapshot
round-trip checks. They use local-axis geometry and explicit assembly poses
to avoid numerical changes when nearly circular meshes are serialized.
Mesh, camera, UV, material and runtime-visual export contracts are covered
by tests that run without Blender.

The local recording release contains 20 decoded MP4/WebM files. It samples
observations at 1 FPS and holds those frames in the 4 FPS output without
interpolating motion. Historical recordings remain unchanged. Detailed
captures, checksums and comparison renders are kept under the ignored
`temp/visual_upgrade/20260919T055432Z/` directory; these simulation checks do
not establish manufacturing fit or biological accuracy.

### Mounting and focus follow-up — 2026-09-19

The upright installation completed all five tasks on both engines: 124 workflow
checks passed. Independent interaction-archive checks passed 199 consistency
criteria for geometry, membrane response, dose, focus and native joint traces.
The source trajectories cover 49,051 physics steps; mesh checks found no
unintended support/CAD or cross-assembly intersections along those trajectories.
The smallest conservative support/CAD separation was 0.646 mm, excluding the
declared fixed clamp's intended contact surface. This is a geometry check,
not a structural stiffness or full-travel assessment.

The native suspended-cell run delivered 0.500 pL, with maximum model penetration
3.590 µm. During commanded delivery, absolute needle defocus averaged 0.127 µm
and peaked at 0.441 µm. Cells still use the reduced membrane model; these values
are simulated observations rather than measured instrument accuracy.

The follow-up recording release captures cell and suction injection at 4 FPS
and the other tasks at 2 FPS, retaining observed frames without interpolation.
All 20 MP4/WebM files decoded successfully, and 80 historical videos retained
their hashes. Detailed results and review images are under ignored
`temp/microscopy_mount_focus/20260919T064656Z/`.
