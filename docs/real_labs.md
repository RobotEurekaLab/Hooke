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

**Specialist validation is pending.** Scientific processes, complete robot
experiment workflows and Isaac runtime behavior have not been qualified for
these prototypes. Generated previews and asset archives are local outputs;
follow the export and rendering commands to recreate them after cloning.

### Earlier ten-laboratory expansion

The second batch adds ten laboratories from nine QS 2027 top-100 institutions.
Its source-informed cleanrooms, vacuum facilities, optical workstations,
materials suite and aquarium use distinct architectures and equipment.
See [the batch review](university_expansion_20261003.md) for the new scenes,
mechanical controls, evidence limits and reproducible screenshot commands.
The [university catalogue](university_labs.md) groups all scenes by institution,
subject and laboratory.

The following revision history describes the original ten-scene batch.

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
