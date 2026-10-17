# University laboratory scenes

## Organization

The homepage's **Lab scene from university** panel uses this hierarchy:

**University → Subject → Laboratory → Scene / task**

`Hooke/university/` stores institutional identity, research evidence and bindings to scenes. Existing geometry, instruments and controls remain in `Hooke/real_labs/`; they are not copied into a second implementation. Stable scene IDs, existing task IDs and `/real-labs?scene=...` URLs stay valid. One laboratory can contain multiple scene variants; their labels include the scene title when needed.

The UI lists schools and labs alphabetically in English. Only constructed university scenes appear as labs. A university with no constructed scene shows an empty state and a disabled launch button; selecting it never falls back to another school's tasks. NASA remains available in Research Laboratories and is classified as a research institute, not a university. Existing task picking, custom generation and the Examples/video layout retain their upstream behavior.

## Versioned research scope

The current assignment scope is **QS World University Rankings 2027, published overall rank ≤100**, including ties and corrected entries. The [official table](https://www.topuniversities.com/qs-top-uni-wur) contains 102 qualifying institutions; the [correction log](https://www.topuniversities.com/rankings-release-summaries/world-university-rankings-2027-release-summary) explains why positions need not be renumbered after corrections. This is a research-priority scope, not a restriction on additional university scenes.

The directory uses versioned, sourced institution records and actual scene bindings. The first expansion established a **20-scene baseline: 19 university scenes and one research-institute scene**, including 15 in-scope scenes across 14 universities. The subsequent [100-laboratory expansion](university_expansion_100_20261003.md) adds distinct laboratories at 76 universities beyond that frozen baseline, bringing the reference-laboratory catalogue to **120 scene prototypes**. Mechanical and primary-view checks have passed; specialist validation of scientific processes, complete robot tasks and Isaac runtime remains pending. The API reports current catalogue counts. Rochester, Indiana Bloomington, Virginia and Waterloo remain outside the pinned scope; NASA is a separate research institute. Ranking updates require a new verified roster and explicit migration; do not overwrite ranks from another edition or infer rank from a subject ranking.

The project-specific search exclusion applies to **University of Toronto Yu Sun / AMNL**, not to the whole university. University membership does not automatically establish affiliation for an overseas campus or independently incorporated institute.

## Source of truth

| File | Responsibility |
| --- | --- |
| `Hooke/university/qs_world_2027.json` | Official institution IDs, names, ranking facts, edition and evidence URLs |
| `Hooke/university/additional_institutions.json` | Other universities and non-university institutes with explicit provenance |
| `Hooke/university/research_candidates.json` | Reviewed leads, actual image-review status, equipment clues, remaining gaps, rights status and rejected/held sources |
| `Hooke/university/scene_index.json` | Scene → institution, subject and laboratory mapping |
| `Hooke/university/batches/*.json` | Explicit expansion scope, original scene baseline and required new lab IDs |
| `Hooke/university/batches.py`, `review.py` | Distinct-lab accounting, current-source checks, visual-review records and paginated screenshot galleries |
| `Hooke/real_labs/catalog.json`, `scenes/*.json`, `layouts/`, `evidence/` | Actual scene definitions, spatial layouts and evidence limitations |
| `Hooke/real_labs/university_extensions.py`, `university_*.py` | Modular instrument geometry, reference records, mechanical interfaces and distinctive room features |

`GET /api/university` exposes constructed scenes and coverage metadata without building models. `GET /api/catalog` includes the same directory under `university`. Neither request starts physics or rendering. Research candidates are not silently promoted into the task catalogue.

## October 2026 reconstruction batch

The batch `qs2027_labs_20261003` adds ten **reference-informed prototypes**. Every institution has published overall rank ≤100 in the pinned QS 2027 roster; NTU contributes two different laboratories. These are authored reconstructions from inspected official images or plans, with estimated dimensions and incomplete inventories. They are not surveyed, scientifically validated or institution-endorsed digital twins.

| University | QS 2027 rank | Laboratory scene | Reconstructed scope |
| --- | ---: | --- | --- |
| Harvard | 5 | `harvard_cns_cambridge` | CNS cleanroom bay and selected instrument workstation |
| Caltech | 7 | `caltech_stoltz_schlinger` | Stoltz chemistry bay, hoods and selected apparatus |
| Cornell | =16 | `cornell_schlom_mbe` | Schlom vacuum deposition workstation and support equipment |
| Stanford | =2 | `stanford_biomechatronics_gait` | Treadmill, support structure, control racks and mechanical leg surrogate |
| Australian National University | 29 | `anu_shrimp_geochronology` | SHRIMP-inspired large instrument and surrounding service equipment |
| Oxford | 4 | `oxford_bonilla_semiconductor` | Semiconductor characterization workstation |
| NTU Singapore | 12 | `ntu_sgsr_characterization` | Compartmented suite, Instron 68TM frame and Dimension Icon AFM |
| NTU Singapore | 12 | `ntu_smtc_membranes` | Membrane test facility and selected apparatus |
| Tsinghua | 14 | `tsinghua_rush3d` | RUSH3D optical assembly, light-controlled enclosure and motorized carrier |
| Queensland | =40 | `uq_moreton_aquarium` | Moreton Bay aquarium laboratory and water-system geometry |

Each scene retains its source URLs, observed cues, placement assumptions, estimated dimensions and mechanism limitations. The candidate records use `reference_informed_prototype` for this reconstruction stage. The university API uses `mechanical_prototype` for the constructed runtime category; neither value implies scientific validity. Candidate rights and release-authorization fields are unchanged by scene construction.

The mechanical interfaces cover selected positioning, access, rotation or carrier motions. They do not establish full real-instrument operation. In particular, the Stanford roller controls do not implement moving-belt contact or human biomechanics; RUSH3D does not implement an optical transfer function; the AFM controls are coarse positioning, not nanometre scanning. Read each scene's manifest for its narrower supported behavior.

## Adding or improving a scene

1. Choose the institution's stable ID and verify the actual lab and campus. Record an official lab/facility URL; do not substitute a university homepage or a campus promotional image.
2. Add or update its research-candidate entry. Separate official-page review, inspected images, room evidence, workstation evidence, measured dimensions, operations and distribution rights. Record access failures and stale tours.
3. Reconstruct the actual room arrangement and selected instruments in `real_labs`, following its existing composition helpers. Add complete definitions under `scenes/*.json`; keep instrument and room-feature implementations in a dedicated extension module. Reuse an asset only when its dimensions, role and support geometry fit. Keep evidence-backed placements separate from estimates; a stock product image does not locate a device in the room.
4. Add a `scene_index.json` binding. The scene ID must already exist; unknown institutions, missing scene bindings, stale bindings and conflicting identities are rejected. Multiple rooms/scenes may share a lab ID and consistent lab identity.
5. Register the actual task adapter and usable controls. Classify success honestly: a moving joint is not a completed scientific experiment. Define short skills, medium workflows and long experiments separately, with observable success/failure and reset behavior.
6. Run relevant CPU validation and compare authored views with source evidence. Qualify MuJoCo mechanics and Isaac dynamics separately; a visual USD snapshot does not establish Isaac support. Keep source photographs and generated evidence under ignored `temp/`; publish only permission-cleared curated media. Construction does not change source-media permissions; submission to GitHub requires the user's explicit instruction after review.

Example binding:

```json
"purdue_phenotyping": {
  "institution_id": "purdue",
  "subject_id": "plant_science",
  "subject_label": "Plant Science",
  "lab_id": "aapf",
  "lab_name": "Ag Alumni Seed Phenotyping Facility"
}
```

## Export and review

From the repository root, use the existing environment:

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.export \
  --all --output temp/university/qs2027_compatibility \
  --previews-from temp/real_labs/latest
```

This exports the existing scenes, asset files, classification/evidence manifests, floor plans, geometry snapshots and mechanical checks. It optionally copies previously recorded overview images. It does not download resources, create new models, start a GPU renderer, or certify new scientific behavior.

```text
temp/university/qs2027_compatibility/
  index.html
  index.json
  universities/<institution>/<subject>/<lab>/<scene>/
  institutes/<institution>/<subject>/<lab>/<scene>/
```

Use `--institution purdue` or `--scene purdue_phenotyping` instead of `--all` for a subset. Use a separate output directory for each export batch: the root index describes that invocation's selection. Candidate-only schools cannot be exported as constructed scenes.

Refresh the research report after editing candidate metadata:

```bash
PYTHONPATH=Hooke .venv/bin/python -m university.report --output docs/university_research.md
PYTHONPATH=Hooke .venv/bin/python -m unittest discover -s tests -p 'test_university*.py'
node --test tests/test_university_picker.js
```

Read the [research register](university_research.md), [NTU SGSR reconstruction brief](university_ntu_sgsr_reconstruction.md), and [scene implementation guide](real_labs.md). The register separates school coverage, reviewed sources and constructed prototypes; none of these alone establishes a high-fidelity, institution-validated digital twin.
