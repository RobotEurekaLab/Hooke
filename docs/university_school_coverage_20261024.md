# QS school coverage: October 24, 2026

## Scope and result

The pinned QS 2027 overall-rank ≤100 roster contains **102 institutions**, including ties and corrections. Before this batch, 84 had constructed scenes. This batch adds **18 laboratories at 18 previously uncovered institutions**, closing the school-level implementation gap to **102/102**.

The reference-laboratory catalogue now contains **138 scenes: 137 university scenes and one NASA scene**. Of the university scenes, 133 belong to the pinned QS scope; the four existing out-of-scope university scenes remain available. The previous 120-scene baseline and its historical acceptance records remain unchanged.

These are reference-informed mechanical prototypes of selected laboratory bays and workstations. They do not establish surveyed room dimensions, complete equipment inventories, scientific response, or institutional endorsement. In particular, this coverage count is not a count of validated long-horizon scientific tasks.

## Added laboratories

Every entry has an inspected installed-apparatus photograph, institution/physical-site attribution, source URLs, image hashes, explicit geometry limitations, and an executable dry mechanical inspection recipe. Original procedural geometry is used; source photographs are kept in ignored local storage.

| Institution | Reconstructed laboratory / workstation | Bounded mechanical controls |
| --- | --- | --- |
| Adelaide University | EXTERRES room-temperature vacuum chamber | Door rotation; retained specimen translation |
| Durham University | Fixed-ground wind-tunnel test section | Model yaw; probe height |
| Institut Polytechnique de Paris | LULI2000 MILKA at École polytechnique, Palaiseau | External diagnostic fixture translation and height |
| King's College London | MIC STELLARIS microscope workstation | Specimen X/Y positioning |
| Korea University | ACE glazed wind-tunnel bridge-section rig | Model heave and pitch |
| London School of Economics | Behavioural Lab EyeLink workstation | Chin-rest height; eye-camera tilt |
| Universiti Malaya | High-voltage laboratory cage and console | Unenergized original electrode fixture gap and height |
| University of Melbourne | HRNBLWT probe workstation | Probe span and height |
| Monash University | GeoExtremes fixed seismic platform | Original retained specimen fixture translation and yaw |
| Peking University | Zhao synthesis laboratory rotary evaporator | Glass lift; flask rotation |
| University of Tokyo | Kurumizaka chromatography workstation | Nozzle height; fraction-collector angle |
| University of Toronto | WB233 Advanced Membranes crossflow bench | Two retained cell-cover lifts |
| Trinity College Dublin | CRANN AML installed Titan bay | Holder translation and rotation |
| Universidad de Buenos Aires | FIUBA chemistry teaching laboratory, **2008 reference** | Clamp height; dry stopcock rotation |
| University College Dublin | Conway Institute SiriusXT installation | Retained cover lift; specimen rotation |
| UCL | SWC optics-training breadboard | Specimen X/Y positioning |
| University of Amsterdam | Speech Lab Studio 2 | Stimulus-arm yaw; screen pitch |
| University of Zurich | Otology specimen-preparation workstation | Specimen X/Y positioning |

The Toronto selection excludes Yu Sun / AMNL as requested. LSE and Amsterdam are actual human-subject experiment facilities; they are not represented as wet laboratories. Adelaide is a terrestrial test facility with Earth gravity, not a simulated spacecraft interior.

## Validation

- All 18 new scenes build and compile in MuJoCo, with 36 bounded mechanical controls, layout/support checks, target tracking and reset checks.
- All 18 source-bound dry inspection recipes pass: 72 phases and **594,707 executed/checked physics steps**, including return-to-home registration. Every recorded phase's model signature matches the final exported model. The UBA descendant-bearing gravity compensation omission was repaired and rechecked.
- The three geometry suites check 1,770 sampled joint configurations in total. Direct separation checks additionally catch interfaces that ordinary parent-child contact filtering can hide; Korea, Malaya and Trinity bearing/guide issues were repaired without weakening the guard.
- The university regression run passed 229 tests, and the inspection-recipe suite passed eight tests. Affected focused tests were rerun after geometry repairs.
- All 120 original scenes produce identical XML with the original and extended module registries. Their 120 institution bindings, 78 existing inspection recipes and 128 existing candidate records are unchanged.
- Blender CPU exports contain 66 images, including all 54 required overview/interior/workstation views at 960 × 600. The 54 main views were inspected against references; obstructed Korea and Melbourne workstation cameras were corrected. The visual-review pass checks visibility and reference cues at the prototype level; it does not certify high fidelity.
- Final batch acceptance reports 18/18 evidence, mechanical, rendered and visually reviewed records. The refreshed web service opens all 18 new scene states successfully; all 54 served main-view images match their reviewed file hashes. The homepage layout is unchanged.

The full checks and source/image hashes are local under `temp/university/coverage_20261024/`; concise batch acceptance is in `progress.json`. Native Isaac has not been run for these 18 additions, and no new GPU renderer or simulator worker was started.

The generated gallery is `temp/university/coverage_20261024/packages/review.html`. Serve that directory locally to inspect the views. Source photographs, rendered previews and raw audit records remain in ignored local storage.

## Evidence boundaries

- Room and instrument scales are estimated unless a specific published dimension is recorded. Melbourne's test-section cross section, Monash's fixed deck footprint and Durham's published test-section dimensions have separate supporting records; this does not verify their complete room layouts.
- Monash's source shows a **fixed platform**. Its deck is fixed in the model; a small original specimen fixture provides the bounded inspection motions.
- Malaya's source establishes the yellow/black safety cage and external console. The apparatus inside is largely obscured; the modeled sphere-gap fixture is explicitly an original proxy.
- The UBA reconstruction uses a dated 2008 institutional photograph, not evidence of the present-day room.
- Apparatus silhouettes and surrounding furnishings remain simplified. MILKA port geometry, instrument enclosures, cables and hidden internals need further detail before high-fidelity claims.
- No flow, high-voltage discharge, vacuum pumping, electron optics, X-ray acquisition, chromatography, chemical reaction, human response or biological process is qualified by these mechanical routes.
- Public accessibility is distinct from redistribution rights. Source photographs are not added to Git. The source ledger preserves individual rights statements, including the credited CC BY-SA UBA photograph.

## Integration and reproducibility

The batch is defined in `Hooke/university/batches/qs2027_school_coverage_20261024.json`. Its `required_institution_ids` gate counts schools uniquely; multiple laboratories at one school cannot close a different school's gap. The catalogue exposes `scope_institutions_without_scenes`, now zero. University, subject and laboratory labels continue to use the existing English alphabetical picker and stable task URLs.

Source and geometry modules are split into `university_coverage_asia`, `university_coverage_europe` and `university_coverage_global`. They use the existing builder, runtime, export and source-bound inspection interfaces. The homepage layout is unchanged.

All new reference downloads, raw operation records, screenshots, editable exports and review manifests are under ignored `temp/university/coverage_20261024/`. The local `packages/` directory contains per-scene MJCF, Blender, GLB and USD visual exports. A USD visual export alone does not establish native Isaac operation.

Rebuild the batch with the existing local environment:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.export \
  --batch Hooke/university/batches/qs2027_school_coverage_20261024.json \
  --output temp/university/coverage_20261024/packages

PYTHONPATH=Hooke .venv/bin/python -m university.batches \
  --batch Hooke/university/batches/qs2027_school_coverage_20261024.json \
  --input temp/university/coverage_20261024/packages \
  --visual-reviews temp/university/coverage_20261024/visual_reviews.json \
  --output temp/university/coverage_20261024/progress.json
```

The second command requires locally generated renders and an explicit visual-review record. Fresh clones do not include `visual_reviews.json` or the raw audit archive: render the exported geometry with `Hooke/real_labs/render.py`, inspect the reference and generated views, then create current review records before evaluating this gate. Re-render and re-inspect changed geometry before reusing visual-review records: source and image hashes must match. Native Isaac and specialist scientific validation for these 18 new scenes are separate pending gates.
