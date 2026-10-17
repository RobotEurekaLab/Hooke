# University laboratory expansion — 3 October 2026

This batch adds ten laboratory scenes to the university catalogue. It is separate from the ten earlier `real_labs` scenes: none of the earlier scenes counts towards the requested ten additions. The batch spans **nine universities** because NTU contributes two different facilities.

The models provide reference-informed rooms, original equipment geometry and directly operable mechanical controls. Room dimensions and unseen geometry remain estimates. The institutions have not validated the reconstructions, and the scenes do not yet implement complete scientific experiments or autonomous robot policies.

## Scope and classification

The scope uses the repository's versioned [QS 2027 roster](../Hooke/university/qs_world_2027.json), with institutional rank at or below 100. This is the QS global institutional ranking, not a subject ranking. The source selection, rendered evidence and mechanical checks are separate from QS eligibility.

Each scene is bound through [the university scene index](../Hooke/university/scene_index.json):

```text
institution → subject → laboratory → scene
```

The geometry stays in `real_labs`; `university` owns institutional classification and research records. A photograph, a research candidate and a working scene have different statuses. New scenes do not silently promote all laboratories of a university to reconstructed status.

The authoritative list and original-scene baseline are in [the expansion batch](../Hooke/university/batches/qs2027_labs_20261003.json).

## Ten laboratories

| University / laboratory | Scene ID | Source-supported character | Mechanically operable scope | Unimplemented process or important uncertainty |
| --- | --- | --- | --- | --- |
| Harvard — CNS, Cambridge LISE | `harvard_cns_cambridge` | Long corridor, instrument bays and a yellow cleanroom section visible in the facility preview | Two wafer-positioning stations; XY carriers and vertical alignment heads | Generic aligner mechanisms; no mask exposure, lithography or verified AB-M geometry. The selected three-bay arrangement is estimated. |
| Caltech — Stoltz, Schlinger Laboratory | `caltech_stoltz_schlinger` | Glazed fume-hood row, tall stools, supported glassware, bright windows and glovebox context | Three hood sashes and three under-deck cassette drawers | No reaction chemistry, ventilation, containment or glovebox atmosphere. Published photographs are historical; the exact combined bay layout is unverified. |
| Cornell — Schlom Group | `cornell_schlom_mbe` | Metal vacuum chambers, flange connections, support frames and service infrastructure | Load-lock access hatch and a retained substrate-transfer rod | No vacuum, molecular-beam flux, crystal growth or in-situ diagnostic solver. Exterior photographs do not establish a surveyed floor plan. |
| Stanford — Biomechatronics Laboratory | `stanford_biomechatronics_gait` | Treadmill station, overhead support, electronics and operator workstation | Independent left/right drive-roller velocities and unloaded fixture-support lift | Belt surfaces are static. No walking contact, human-subject model, biomechanics or force-plate measurements. The mechanical leg fixture is an authored surrogate. |
| ANU — SHRIMP facility | `anu_shrimp_geochronology` | Curved vacuum/ion-optical apparatus, control racks and services visible in the tour preview | Load-lock access hatch, retained mineral-mount translation and mount rotation | No ion beam, mass separation, isotope-ratio calibration or geochronology. The facility envelope and concealed machine internals are estimated. |
| Oxford — Bonilla Lab | `oxford_bonilla_semiconductor` | Probe-contact workstation, sample stage and laboratory instrument references | Inert-coupon translation and two independent probe-clearance axes | Probe tips remain above the sample; no contact force, semiconductor transport, electrical characterization or optical lifetime measurements. Shared cleanroom/microscopy facilities are not combined into one room. |
| NTU — SGSR characterization suite | `ntu_sgsr_characterization` | A public labelled plan separates laboratory areas; facility inventory documents material-test and AFM equipment | Material-tester crosshead and upper grip; AFM coarse XY positioning and head clearance | No material stress/strain or nanometre AFM feedback. Plan scale is unreadable and inventory/plan dates may differ. |
| NTU — Singapore Membrane Technology Centre | `ntu_smtc_membranes` | Campus industrial research aisle, open instrument frames, membrane housings and suspended services | Two isolation-valve handles and two retained water-vial carriages | No filtration, pressure, membrane transport or water-quality solver. Tuas outdoor pilot equipment is excluded from this campus scene. |
| Tsinghua — RUSH3D workstation | `tsinghua_rush3d` | Large custom microscope, perforated optical table, overhead structure, cables, curtains and side monitor | Sample XY positioning and objective-clearance axis with a visible inert calibration target | No mesoscopic imaging, live-cell process, beam-path reconstruction or optical transfer function. Hidden components are estimated from one apparatus view. |
| UQ — Moreton Bay Research Station aquarium | `uq_moreton_aquarium` | Large round tanks, three yellow aquarium racks, wet-room plumbing and overhead services | Traverse and immersion axes on an added probe bridge | The motorized bridge is an original task fixture, not equipment verified at UQ. Water is a static display surface; no organisms, flow, buoyancy or calibrated sensor measurements. |

### Official references

These links identify the source institutions and images used for reference. They do not establish an open redistribution licence for the source media.

- **Harvard:** [CNS virtual visit](https://cns1.rc.fas.harvard.edu/virtual-visit/). The inspected dollhouse preview supports broad zoning. A full measured building scan was not imported.
- **Caltech:** [Stoltz laboratory tour](https://www.stoltz2.caltech.edu/labtour2.html). Hood, lab-bay and glovebox photographs provide separate observations; their combination and exact adjacency are estimates.
- **Cornell:** [Schlom laboratory tour](https://schlom.mse.cornell.edu/labtour). Equipment exterior evidence is stronger than room-perimeter evidence; a legacy embedded room-tour link was unavailable during source selection.
- **Stanford:** [Biomechatronics Laboratory](https://biomechatronics.stanford.edu/). Two experiment photographs support the local gait-station arrangement. Source photographs of human subjects are not packaged or used as textures.
- **ANU:** [SHRIMP 360 facility tour](https://science.anu.edu.au/study/360-virtual-tours/sensitive-high-resolution-ion-microprobe-shrimp-360deg-facility-tour). The preview provides instrument and rack cues, not a calibrated dimensional survey.
- **Oxford:** [Bonilla Lab infrastructure](https://interface.web.ox.ac.uk/infrastructure). The instrument inventory covers different in-house and shared locations; the model preserves that distinction.
- **NTU SGSR:** [Facilities inventory and plan](https://www.ntu.edu.sg/mse/research/sgsr/facilities). Linked equipment sheets support selected external dimensions; exact installed configuration and operating performance remain unverified.
- **NTU SMTC:** [Official facilities brochure](https://www.ntu.edu.sg/docs/librariesprovider88/newri-domains/smtc/smtc-brochure_compressede6c80970-b0ba-4414-90e7-bcb662e94af7.pdf?sfvrsn=87821fc1_3). Campus analytical, industrial and fabrication photographs are captioned separately from the Tuas pilot site.
- **Tsinghua:** [RUSH3D university feature](https://www.tsinghua.edu.cn/info/3155/119959.htm). The workstation is based on this apparatus reference, not the different earlier RUSH system.
- **UQ:** [Moreton Bay research laboratories and aquariums](https://moreton-bay.research.uq.edu.au/facilities/research-laboratories-aquariums). The facility specifies two 4,000 L tanks, one 10,000 L tank and three racks of twelve standard tanks. Those are nominal source capacity classes, not a calibrated fluid-volume claim for the authored geometry.

## What is implemented

The batch contains **11 controllable equipment types, 15 equipment instances and 37 declared controls**, counted from the ten constructed scene manifests. This counts mechanical interfaces, not scientific processes, robot task successes or independently reconstructed OEM products.

- Different spatial structures and equipment arrangements for different laboratory types, including cleanroom bays, a chemistry bay, vacuum-instrument rooms, precision optical/mechanical stations, membrane rigs and a wet aquarium room.
- Metre-based geometry under Earth gravity, `9.81 m/s²`.
- Visible inert wafers, coupons, cassette/vial carriers, calibration targets or instrument fixtures where declared by each equipment manifest.
- Namespaced joints, actuator limits and semantic control handles. A command advances the MuJoCo simulation and reads measured joint state.
- Real architectural openings, floor/bench support declarations and inspectable authored floor plans.
- Three consistent views per scene: `overview`, `interior` and `workstation`.
- MJCF packages and editable Blender scenes; GLB and USD snapshots for visual asset review.

Static contextual apparatus is intentionally distinguished from controllable equipment. For example, a visible pressure gauge does not produce pressure measurements, and a decorative glovebox is not listed as a controllable atmosphere system.

### Acceptance results for this revision

| Check | Result | Scope |
| --- | --- | --- |
| Python regression suites | 68 tests passed | Laboratory geometry/runtime, university catalogue/expansion and task navigation |
| Node picker tests | 7 tests passed | School/subject/lab filtering and dropdown state handling |
| Bidirectional mechanical control check | 37 controls × 3 targets = 111 target commands passed; reset checks passed | Targets at 15%, 85% and 25% of each declared range, with measured simulation state |
| Rendered images | 33 images, each 1120 × 700 pixels | 30 standard views and 3 additional geometry views; Blender Cycles CPU, 32 samples |
| Artifact association | Scene SHA256 matches manifest, render and validation metadata | All ten current scene packages |

The three additional images are `ntu_sgsr_characterization/afm_detail.png`, `cornell_schlom_mbe/source.png` and `anu_shrimp_geochronology/source.png`. The two `source.png` images are **reference-oriented renders of the authored geometry**, not original photographs or evidence of a photogrammetric reconstruction.

Both overview and instrument-detail contact sheets were independently inspected. Each is 1800 × 3310 pixels, with ten consistently sized 8:5 tiles in two columns. The scene labels match the pictured laboratories, text stays within its title bands, and no missing or blank tiles were observed. The full-resolution scene images remain necessary for close assessment of apparatus geometry; a contact sheet is not a photorealism or institutional-fidelity certification.

Detailed regression output stays in `temp/university/expansion_20261003/logs/final_regression.log` and `picker_tests.log`. Passing these checks does not add scientific capability beyond the mechanical scope documented above.

### Benchmark boundary

These additions contribute laboratory assets and instrument-control interfaces. Joint-motion checks do not establish a successful L1 robot skill, L2 experimental procedure or L3 long-horizon investigation. Autonomous policies, semantic task success conditions, calibrated scientific state and suitable baselines need separate implementation and evaluation.

Likewise, an exported USD file does not prove Isaac/PhysX articulation parity. The validated control path for this batch is MuJoCo on the CPU; Blender/GLB/USD represent a snapshot of its visual geometry.

## Local review artifacts

Generated images, logs and complete experiment records remain under the ignored directory:

```text
temp/university/expansion_20261003/
  scenes/
    review.html
    overview_summary.jpg
    workstation_summary.jpg
    review_manifest.json
    <scene_id>/
      overview.png
      interior.png
      workstation.png
      afm_detail.png  # NTU SGSR only
      source.png      # Cornell/ANU reference-oriented renders only
      scene.xml
      manifest.json
      validation.json
      layout_audit.json
      floorplan.svg
      render.json
      laboratory.blend
      laboratory.glb
      laboratory.usdc
  logs/
```

The overview and instrument contact sheets use identical 8:5 image tiles. Each laboratory keeps the same three view names and image dimensions. The gallery opens local image files and requires no running Isaac service.

The review tool requires each scene XML digest to match the equipment manifest, render and validation metadata. It also checks that the declared images exist, have the declared resolution and contain nonblank image data. These checks detect stale or missing artifacts; a human still needs to compare layout and apparatus appearance with the references.

## Reproduce exports, renders and review

Run these commands from the repository root with the existing environment. Choose a fresh output directory when comparing revisions. No new simulator installation, asset download or background web/GPU service is required.

### 1. Export the ten-scene batch and run mechanical smoke checks

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.export \
  --batch Hooke/university/batches/qs2027_labs_20261003.json \
  --output temp/university/expansion_20261003/scenes
```

The exporter writes `validation.json` per scene and a batch report. It checks the authored room layout, compiles MJCF, advances the mechanical controls and exports a geometry snapshot. Inspect the per-scene files for the actual scope and outcomes.

### 2. Render all declared views with the existing Blender installation on the CPU

```bash
temp/visual_upgrade/tools/blender-4.5.14-linux-x64/blender \
  -b --python Hooke/real_labs/render.py -- \
  --input temp/university/expansion_20261003/scenes \
  --samples 32 --width 1120 --height 700 --threads 8
```

Omitting `--views` renders every declared camera, including the three additional views, for 33 images in this batch. Adding `--views overview interior workstation` limits output to the 30 standard images; it excludes `afm_detail` and the two `source` views. The custom view names are not choices accepted by the explicit `--views` argument, so omit that argument to reproduce them.

Use `--scene <scene_id>` after `--input` for one scene. `--preview-only` renders images without replacing editable exports, so it should not be used for the first full delivery of a scene. The standard full rendering path produces the Blender, GLB and USD visual artifacts expected by the review tool.

### 3. Assemble the screenshot summary

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.review \
  --input temp/university/expansion_20261003/scenes \
  --batch Hooke/university/batches/qs2027_labs_20261003.json
```

Open `temp/university/expansion_20261003/scenes/review.html`, or open the two contact-sheet JPEGs directly. Review the three original PNGs when a contact-sheet tile is too small to assess details.

### 4. Export the institution / subject / laboratory hierarchy

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.export \
  --all --output temp/university/expansion_20261003/classified \
  --previews-from temp/university/expansion_20261003/scenes
```

This command includes all constructed scenes in the catalogue, including earlier scenes and separately classified research institutes. For only one institution, replace `--all` with `--institution <institution_id>`. Research candidates without an implemented scene are not exported.

### 5. Run regression checks

```bash
PYTHONPATH=Hooke .venv/bin/python -m unittest discover \
  -s tests -p 'test_university*.py'

PYTHONPATH=Hooke .venv/bin/python -m unittest discover \
  -s tests -p 'test_real_labs*.py'

PYTHONPATH=Hooke .venv/bin/python -m unittest discover \
  -s tests -p 'test_task_navigation.py'

node --test tests/test_university_picker.js
```

The independent mechanical/scientific claim audit is recorded locally in `temp/university/expansion_20261003/scientific_scope_audit.json`; it covers all ten scene IDs and their source, task and instrument metadata. That record is a scope review, not a substitute for running tests.

Record test outcomes only after executing the current revision. Physics checks, source-fidelity assessment, screenshot review and university classification are separate acceptance requirements.

## Contribution and publication boundaries

Source photographs remain local research references. Original procedural models and source links do not imply permission to rehost institutional photographs, trademarks, virtual tours or OEM CAD. Public release still requires checking the rights of any media selected for inclusion.

Keep raw renders and full reports under `temp/`. Only explicitly selected documentation illustrations belong in `docs/assets/`. All changes remain local pending the user's screenshot review and explicit instruction to submit to GitHub.
