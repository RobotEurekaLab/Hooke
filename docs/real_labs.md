# Reference laboratory scenes

Hooke includes 120 laboratory prototypes reconstructed from public facility
descriptions, tour stills and equipment specifications. They provide room layouts,
workbenches, reusable equipment and directly operable mechanical controls.

These are **reference-informed, estimated layouts**, not surveyed or
institution-validated digital twins. A scene's institution identifies its reference
source; it does not imply endorsement or an exact inventory match.

## University expansion

The latest batch adds 100 distinct laboratories at 76 universities to the
20-scene baseline. All 120 scene generators and their equipment definitions are
included in the source release. See the [100-laboratory review](university_expansion_100_20261003.md)
for source evidence, mechanical checks and screenshot reproduction.

**Bounded catalogue acceptance is complete for these 120 scenes.** All pass
native loading and the 11 non-control runtime gates, and all have complete
bounded MuJoCo mechanical routes. Raw scalar diagnostics retain nine failures
out of 555 controls; four guarded native operations have separate passing
evidence. Complete scientific-process models and 120 complete native experiment
workflows remain unqualified.

The [fidelity and usability report](university_optimization.md) records the
validated scope, source limitations and reproduction steps. These results apply
to the `real_labs` catalogue; they do not qualify the separate `real_lab` scene
collection. Generated previews and asset archives remain local outputs.

### Earlier ten-laboratory expansion

The second batch adds ten laboratories from nine QS 2027 top-100 institutions.
Its source-informed cleanrooms, vacuum facilities, optical workstations,
materials suite and aquarium use distinct architectures and equipment.
See [the batch review](university_expansion_20261003.md) for the new scenes,
mechanical controls, evidence limits and reproducible screenshot commands.
The [university catalogue](university_labs.md) groups all scenes by institution,
subject and laboratory.

The following revision history describes earlier versions of the original
ten-scene batch. Equipment configurations and counts below are historical;
the linked acceptance report describes the current implementation.

## Spatial revision: what changed and what remains uncertain

The first version reused the same small room and three-counter arrangement too
often. Its joint tests did not validate the room layout. Revision 2 separates
equipment definitions from room architecture and uses distinct arrangements:

| Facility reference | Current spatial arrangement | Evidence boundary |
| --- | --- | --- |
| Rochester | Narrow facing wood-front bench runs, overbench stock, floor sequencer, cold storage | Bench/cabinet cues observed; envelope and coordinates estimated |
| UW IsoLab | Separate analysis and preparation zones, vacuum line, balance area and restrained gas | Separate real rooms documented; compact adjacency and glazed cutaway partition inferred |
| Penn pathology | Long parallel benches, paired overhead cupboards, workstations and rear preparation counter | Aisle and storage pattern observed; dimensions and exact inventory unverified |
| Purdue | Crop chamber, conveyor rows, imaging corridor, irrigation services and drainage | Facility zones and equipment arrangement informed by tour photographs |
| Waterloo | Open glazed robotics hall, overhead gantry, peripheral diagnostics and maglev area | Hall features observed; 2 × 3 m maglev footprint sourced; hall envelope estimated |
| NASA Glenn | High bay with two long soil tanks, separate inclined bed and preparation area | Nominal tank dimensions sourced; building envelope and placement inferred |
| Indiana | Shaded microscopy area, optical table, separate preparation and acquisition desks | Workflow-based design; room plan not recovered |
| EPFL | Optical/mechatronics platform, electronics bench and routed services | Workflow-based design; existing magnetic instrument remains a substitute |
| UVA | Process skid, substrate preparation, floor pump/chiller and service rack | Workflow-based design; deposition work documented, installation geometry unverified |
| ETH | Sampling counter, extraction canopy, gas supply and separate gravimetry table | Workflow-based design; sparse imagery does not establish full layout |

These revisions improve spatial plausibility and reference-specific features;
they do **not** establish high-fidelity reconstruction. None of the ten room
plans has been surveyed, registered to calibrated reference cameras, or validated
by the original laboratory. In particular, making rooms look different is not
evidence that their layouts match the real facilities.

Each export includes a dimensioned `floorplan.svg` and `layout_audit.json`. The
audit checks rotated bench boundaries and overlaps, a one-metre bench-free
doorway approach, and camera/equipment origins. The one-metre approach is a
screening convention, not a building-code assessment. Full equipment envelopes,
furnishings, door swings, maintenance access and human reach require further
review. The elevated NASA rover is explicitly reported for manual support review;
its authored service stand is separate from the floor-mount check.

Grey plan overlays also project the compiled collision geometry between 0.12 and
1.8 m above the floor. They include tanks, partitions and other fixtures omitted
from the bench-only drawing. Their conservative bounding rectangles are checked
against doorway approach zones and reported separately. This caught an inferred
NASA structural column blocking the loading opening; the column was relocated.
These projected boxes still do not establish exact walking or maintenance
clearance around complex shapes.

For the current Flex-size pipetting surrogate, the design also allows at least
20 cm behind and beside the housing, following the manufacturer's
[installation requirements](https://docs.opentrons.com/flex/installation/requirements/).
This checks an installation constraint for the surrogate, not the identity of
the robot photographed in Rochester.

## Reference-detail revision

Revision 3 concentrates on Rochester, Penn and Purdue, with a smaller UW service
detail update. It improves original geometry against inspected references;
it does not change the six reference-informed/four workflow-designed evidence
classes or establish an exact digital twin.

| Scene | Added reference-specific detail | Remaining mismatch |
| --- | --- | --- |
| Rochester | Broad glazed work enclosure, sash and upper housing, pale blue walls, tube/wipe stock, dark pipetting-robot fascia, covered gantry and tapered tips | Cabinet type and robot identity unverified; Flex-sized surrogate retains a sliding door rather than the manufacturer's hinge |
| Penn | Longitudinal diffuser rows, clear framed cupboard panes and visible stock, task seating, rounded microtome shell, specimen clamp, blade guard and waste tray | BIOCUT is a shape reference, not a verified Penn device; retained motorized axes differ from its manual mechanism |
| Purdue | Curved folded leaf meshes, tapered pots and carriers, transport profiles and motors, reflector lamps, conduits, irrigation and wash services | Maize geometry is representative; real images also show cereal plants. Counts, envelope and adjacency remain estimates |
| UW | Vacuum-controller cases and cables, cylinder regulators and gauge faces | Gauge needles are static context; instruments and scientific readings remain surrogates |

The Rochester original photograph reveals enclosure details that were not
readable in the earlier thumbnail. Instrument references include the
[Flex hardware description](https://docs.opentrons.com/flex/system-description/robot/)
and [BIOCUT product information](https://www.leicabiosystems.com/us/histology-equipment/microtomes/histocore-biocut/).
Purdue components follow the existing inspected
[official facility-tour stills](https://ag.purdue.edu/aapf/virtual-tour.html).
These sources support visible features and specified dimensions; none provides
the complete CAD assembly or a calibrated floor plan used by this reconstruction.

`evidence/*.json` records observed, implemented, inferred and unresolved details.
`catalog.py` attaches these records to the exported manifest and gallery, so
additional detail does not silently raise the reported fidelity classification.
Reference media remain local inspection material and are not shipped as textures.

The renderer adds original wood/metal finishes, millimetre-scale edge radii and
diffuser lighting. These are uncalibrated presentation materials. The previous
invented institutional wall signs have been removed. Close instrument views and
Purdue's crop-oriented `reference` camera supplement the room views. The latter
retains the roof, like the human-height interior camera.

Visual review also checks fit: the imaging specimen is sized to clear its booth,
and the conveyor stops before the rear backdrop instead of driving the plant
through it. Mechanical smoke tests, sampled instrument endpoint checks and
plant-envelope tests serve different purposes; none validates biological,
chemical or optical performance.

## Original batch

| Scene ID | Reference facility | Main equipment | Equipment placements / controls |
| --- | --- | --- | --- |
| `rochester_genomics` | University of Rochester Genomics Shared Resource | Pipetting gantry, sequencer loading bays, centrifuge | 6 / 9 |
| `indiana_microfluidics` | Indiana University Jacobson group | Inverted microscope stage, focus and microfluidic valve interface | 5 / 7 |
| `uw_isolab` | University of Washington IsoLab | Water-isotope analyzer, autosampler, balance, furnace | 5 / 5 |
| `uva_deposition` | University of Virginia deposition facility | Vacuum chamber, substrate platen, spin coater | 5 / 6 |
| `purdue_phenotyping` | Purdue automated plant phenotyping facility | Imaging booth, camera traverse, plant carrier and growth shelves | 4 / 5 |
| `epfl_microbiorobotics` | EPFL MicroBioRobotic Systems laboratory | Representative magnetic workstation, sample stage, readout instruments | 5 / 5 |
| `waterloo_robohub` | University of Waterloo RoboHub | Articulated arm, mobile platform, research hall | 4 / 12 |
| `penn_pathology` | University of Pennsylvania pathology facility | Microtome-style specimen fixture, feed and blade carriage | 5 / 7 |
| `eth_air_quality` | ETH Zurich environmental engineering laboratory | Aerosol inlet, filter holder, monitor and balance | 5 / 4 |
| `nasa_planetary` | NASA Glenn SLOPE ground testing facility | Rover wheel and arm mechanisms, soil bed, preparation bench | 4 / 18 |

There are 48 equipment placements: 12 newly authored primary instruments and
36 placements reused from existing Hooke assets. The 78 controls include access
doors, drawers, instrument axes, valves, knobs and wheel drives. A control count
is not a count of completed experiments or robot skills.

All twenty facilities are on Earth and use gravity of 9.81 m/s². The NASA scene
represents a terrestrial test facility, not the lunar surface. Its default rover
is mounted on a service stand for independent wheel and arm inspection. The
rover and mobile-platform asset builders also accept `free_base=True`; that mode
needs a separately configured navigation or terrain task.

## Build and inspect

From the repository root, using the existing Python environment:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.export \
  --all --output temp/real_labs/latest
```

To build one scene, replace `--all` with `--scene rochester_genomics`.
The command compiles the MJCF, tests measured motion of every control, checks
supported specimen carriers and exports geometry snapshots. It exits with an
error when mechanical validation fails. No renderer or GPU context is created.

Each scene directory contains:

- `scene.xml` and `assets/`: portable MuJoCo model with relative mesh paths.
- `manifest.json`: device controls, source URLs, estimated geometry, adaptations
  to reused assets and scientific limitations.
- `validation.json`: observed control positions or velocities, tracking errors
  and specimen placement checks.
- `snapshot/`: geometry used for offline visual export.

With Blender already installed, render the actual exported geometry on the CPU:

```bash
CUDA_VISIBLE_DEVICES='' /path/to/blender -b \
  --python Hooke/real_labs/render.py -- \
  --input temp/real_labs/latest --samples 24 --threads 6
```

This adds equally sized `overview.png`, `workstation.png` and `interior.png` images, plus
`laboratory.blend`, `laboratory.glb`, `laboratory.usdc` and `render.json`. Open
`temp/real_labs/latest/index.html` for the static gallery. The script reuses
Hooke's microscopy geometry snapshot and Blender rendering utilities.

Scenes with additional authored cameras also export `instrument.png` or
`reference.png`; the static gallery includes every camera in their manifest.

`interior.png` uses a standing-height camera and the full room enclosure.
Overview and detail renders cut away the front/right walls and roof to expose
the equipment. The physics model retains those walls. Use the `interior` camera
when inspecting the complete enclosure in a native MuJoCo viewer. Procedural
floor, paint and soil finishes are original presentation materials; soil remains
rigid in physics. Blender preserves those procedural finishes; interchange
formats may simplify them.

**MJCF contains the mechanics.** The Blender, GLB and USD files are visual
snapshots with cameras and lighting. They do not contain a validated Isaac/PhysX
articulation. Importing USD into Isaac requires a subsequent collision, joint,
drive and unit validation pass.

## Web controls

When running the normal Hooke web application, select a Research Laboratories
entry from the existing task selector, or open:

```text
/real-labs?scene=rochester_genomics
```

The page provides equipment control ranges and observed joint states. It uses
MuJoCo CPU stepping only, without a rendering worker or continuous simulation
loop. Its images are explicitly labelled recorded previews and do not change
when a control moves. The homepage preview also uses recorded images; it returns
a setup message if the images have not been generated.

Set `HOOKE_REAL_LABS_OUTPUT` to use another export directory. By default the page
reads `temp/real_labs/latest`. Main homepage layout is unchanged.

For programmatic control:

```python
from real_labs.runtime import InstrumentSession

session = InstrumentSession.from_files(
    "temp/real_labs/latest/rochester_genomics/scene.xml",
    "temp/real_labs/latest/rochester_genomics/manifest.json",
)
result = session.command("liquid_handler", "gantry_x", 0.10)
print(result["target"], result["measured"], result["error"])
```

Commands apply bounded servo targets and advance the physical simulation; they
do not assign joint positions directly. Slide positions use metres, hinge
positions radians, and wheel velocities radians per second. The web API limits
one command to five simulated seconds.

The web application caches two scene sessions. Their state is shared by browser
clients using the same application process; an evicted scene resets when opened
again. This is an inspection tool, not isolated multi-user experiment execution.

## Fidelity and provenance

`Hooke/real_labs/catalog.json` distinguishes observed cues, inferred layouts,
representative instruments and per-scene limitations. Equipment metadata adds
manufacturer references and the dimensions used in each reconstruction.

Examples of the evidence boundary:

- Rochester: tour stills support the pipetting enclosure and wood-front benches;
  the robot model is unconfirmed. The current gantry uses Flex-style dimensions.
- Indiana: the tour transcript supports vacuum-actuated microfluidics with
  fluorescence microscopy; a calibrated room layout was not available.
- UW: preparation and analysis occur in separate real rooms; this prototype
  combines selected workstations into one compact bay.
- UVA: descriptions support the deposition apparatus, but the source equipment
  image was inaccessible during collection. Geometry remains an engineering
  surrogate.
- Purdue: growth conveyors and the enclosed imaging station follow inspected
  photographs; the compact floor plan is estimated.
- EPFL: the inspected real microsurgery assembly does not establish that the
  modelled magnetic-coil workstation is installed there. That workstation is a
  representative substitute and needs replacement for a literal reconstruction.
- NASA: the scene now uses nominal 12 × 3 m fixed tanks and a 6 × 5 m tilted tank
  from the [NASA facility listing](https://ares.jsc.nasa.gov/projects/simulants/dust-testing-facilities/glenn-research-center.html).
  The 20 × 16 m high-bay envelope and coordinates remain inferred. The bed is
  fixed at an authored tilt and does not simulate granular soil deformation.

Public source photographs are references only; they are not included as model
textures or redistributed in the asset pack. No original manufacturer CAD files
were obtained for these new instruments. Their geometry is authored from public
descriptions and dimensions, with estimates marked explicitly. Reused Hooke
assets retain their upstream provenance; inspect applicable notices before
redistributing an exported pack. Public access alone is not a redistribution
licence for the reference media.

## What is functional

### ANU probe registration

The ANU annulus has a supported vertical profiler and an authored open reference
ring. Revision 2 retains the original rails and stylus, extends the upper height
limit from 70 to 170 mm, and replaces the unreachable off-axis target with a
coaxial opening. The ring is an inspection aid, not photographed ANU hardware.

After exporting `anu_climate_fluid_physics`, run its measured CPU operation:

```python
import json
from pathlib import Path
import mujoco
from real_labs.runtime import InstrumentSession
from real_labs.annulus_probe_motion import inspect_annulus_probe

package = Path("temp/real_labs/latest/anu_climate_fluid_physics")
session = InstrumentSession(
    mujoco.MjModel.from_xml_path(str(package / "scene.xml")),
    json.loads((package / "manifest.json").read_text()),
)
report = inspect_annulus_probe(session)
assert report["status"] == "succeeded", report
```

Run Python with `PYTHONPATH=Hooke`. The operation starts at resting home, raises
the probe, approaches 8 mm above the datum, passes 15 mm through the opening,
repeats at another tank angle, and physically returns. It records actual joint
motion, tip height, bore registration and per-step clearance, including pairs
omitted by native parent-body collision filtering. No state reset occurs during
the route; `session.reset()` remains a separate explicit action. The probe and
ring rotate together. Water, heat transport and sensor response are not simulated.

### Penn specimen loading

For an exported `penn_pathology` session, `PennProcessor(session).run()` from
`real_labs.penn_processor_motion` opens the powered door, conveys a free inert
carrier into the enclosure, closes and reopens access, then returns the carrier
to its external tray. Contact geometry carries the specimen throughout the
operation; there is no attachment toggle or live pose placement. The 90 × 70 ×
25 mm, 60 g carrier is a mechanical surrogate, not a clinical histology cassette.

The representative dry processor replaces the previous mismatched microtome in
this scene. Its four old microtome controls are explicitly retired. The guarded
operation checks door sequencing, support, containment, contact and actual return.
It does not implement tissue processing, off-tray pickup or clinical data routing.
The source still establishes the room's parallel benches and glazed cupboards,
but does not identify this authored processor as installed Penn equipment.
Generic web sliders are still independent axis commands; integration of the
guarded workflow into those controls remains pending.

### Rochester plate loading

For a current `rochester_genomics` session, use
`real_labs.rochester_operations.run_plate_loading(session)` for the ordered
contact-driven load, release, close/reopen, retrieval and return sequence.
It validates the scene binding and rejects interrupted or disturbed starting
states rather than resetting them. The external XYZ loader is authored task
equipment, not identified Rochester or OEM hardware. The parked aisle is 0.805 m;
loading temporarily occupies its reserved working space.

`InstrumentSession.command()` and the web command endpoint now apply the same
preflight to scalar inspection. Independent loader X/Y requests are rejected
before state mutation; full physical strokes remain available to the ordered
operation. The controls panel exposes **Load and retrieve plate** and replaces
independent X/Y sliders with operation guidance. The deliberately raw axis
diagnostic still fails for two of twelve controls; rejected public commands are
not counted as successful physical motion.

### Leeds contact recovery

`real_labs.leeds_operations.validate_command(...)` checks the dry scalar envelope
before a caller mutates control or state. Deep approach requires the existing
loaded-slide operation. `run_retract(session)` unloads measured contact over
2.5 simulated seconds while holding the current specimen position, without reset.
Full 25 mm normal travel remains; dry scalar approach is bounded to 12 mm.

The shared `real_labs.operations` registry binds the scene model before dispatch.
Public session commands and the web endpoint apply the dry-command guard. The
controls panel offers **Run loaded sliding** and **Retract contact fixture**;
its scalar approach range describes dry inspection without changing physical
joint travel. Neither authored contact parameters nor the recovery envelope are
calibrated real-device limits.

### NASA grounded sample staging

The current default scene has a free grounded six-wheel rover, steering drives,
and a free 25 g inert sample. Its measured CPU route performs parked observation,
physical grasp, driven traversal, shelf placement and tool withdrawal. The current
route completed 182,159 steps; fifteen focused mast/task tests passed. Direct
clearance checks are sampled, while native contacts are inspected each step.
This is an authored rigid aisle under Earth gravity, not validated soil mechanics.

Revision `grounded_staging_v2` places the rover body-frame origin at floor level
and offsets its child frames inversely. Geometry, mass centers and cameras retain
their world placement; all six tyres remain supported. Canonical export and
reload now pass. The full route was rerun on the new XML for 180,679 steps.
Historical v1 state vectors retain their original XML and cannot be replayed
directly into v2.

The independent-axis diagnostic remains 16/24: six individual wheel commands
against the other held wheels and two sample-contacting jaws do not meet its
tracking conditions. This diagnostic is separate from the successful coordinated
route. NASA operations are not yet exposed through the new web operation registry.
Native Isaac execution and complete scene acceptance remain unverified.

### Complete operations over HTTP

Scene state includes `operations` and `control_policy` metadata. Supported scenes
currently provide Rochester plate loading and Leeds loaded sliding/retraction.
Start one finite CPU operation with:

```http
POST /api/real-labs/rochester_genomics/operation
Content-Type: application/json

{"operation":"plate_loading"}
```

The response is HTTP 202 with a job ID. Poll the same URL with GET until its
`job.status` becomes `succeeded` or `failed`. Terminal responses distinguish
current `state` from historical `completed_state`. Failure preserves measured
physics; it never performs an automatic reset. The existing reset endpoint is
an explicit episode reset.

Commands, resets and competing operations on a busy scene return HTTP 409.
Other scenes remain accessible. Storage is shared within one server process,
with at most two loaded scenes; busy scenes are retained, while idle sessions
and their last job records can be evicted. Restarting the process loses jobs.
This local preview is intended for a single server process, not distributed
workers. Images remain recorded renders rather than live operation video.

### General capabilities

Implemented and checked:

- Scene compilation, rigid contact surfaces and articulated instrument controls.
- Seven free specimen carriers that settle under gravity onto their supports.
- Sample proximity and measured access-door or tray interlocks where specified.
- Relative resource paths that survive moving an exported scene directory.
- Navigation and CPU API integration, including command validation.

Not yet implemented for these scenes:

- Autonomous robot workflows, task success metrics or benchmark episodes.
- Sequencing chemistry, cell biology, microfluidic flow, magnetic propulsion,
  deposition physics, calibrated isotope/aerosol readings or tissue cutting.
- Room reconstruction calibrated against camera poses, measured dimensions or
  institutional review.
- Live rendered views of control motion or tested cross-engine physical parity.

The optional `measure()` API returns a labelled `synthetic_geometry_proxy` after
placement checks. It is not a scientific measurement. Scene-picker adapters
never report task success merely because a layout loaded; autonomous experts
raise an explicit unsupported-workflow error.

## Development

- Add room layouts and evidence in `catalog.json`.
- Revise facility layouts in `layouts/*.json`; `catalog.py` merges these patches
  with the base inventory and rejects duplicate scene patches or unknown devices.
- Keep shared walls, real openings, ceiling and bench construction in
  `architecture.py`; add facility-specific features in the corresponding
  `*_features.py` module.
- Add reusable mechanisms through `equipment.add_equipment`; keep laboratory
  arrangement separate from equipment geometry and control metadata.
- Adapt existing assets in `reuse.py` without changing the source models.
- `furnishings.py` retains the initial layout cues for legacy definitions.
- Keep runtime controls independent of offline rendering.
- Keep reference-detail evidence in `evidence/*.json` and uncalibrated Blender
  finishes in `finishes.py`; neither should alter scientific success criteria.

Run the targeted CPU suite from the repository root:

```bash
PYTHONPATH=Hooke .venv/bin/python -m unittest discover \
  -s tests -p 'test_real_labs*.py'
```

Generated media, portable packs and detailed logs belong under ignored `temp/`.
