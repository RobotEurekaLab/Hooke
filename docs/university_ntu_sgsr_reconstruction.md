# NTU SGSR: reconstruction implementation and evidence

**Status:** reference-informed prototype implemented, 2026-10-03. Scene ID: `ntu_sgsr_characterization`; institution: `ntu_singapore`; QS World University Rankings 2027 overall rank: 12. Geometry and selected mechanical controls are implemented. The layout is not surveyed, the scientific behavior is not validated, and the laboratory has not endorsed or reviewed this model.

## 1. Evidence and scope

The [official facilities page](https://www.ntu.edu.sg/mse/research/sgsr/facilities) places a [floor-plan image](https://www.ntu.edu.sg/media/images/librariesprovider121/research/sgsr/facilities/lab1.jpg?sfvrsn=1f529923_3) beneath **Lab 1: 02-06, Research Wing, Characterization Laboratory**. The inspected image is only 699 × 392 pixels and spans several named areas. Its relationship to the present room boundary and the inventory's later sections requires confirmation.

The implemented suite follows this plan's compartment ordering and provides two selected characterization instruments. It does not import equipment from Lab 2, Lab 3 or Lab 4. The page separately lists CREATE Shared Laboratory, and the relationship between that inventory and the depicted suite remains unresolved. Its dated inventory sections and historical lists do not establish the plan's revision date.

The reference stays in ignored `temp/research/qs_asia_visual_checks/ntu_sgsr.jpg`; its hash is recorded in that directory's manifest. No source image is included with this document, and no media redistribution permission has been established.

## 2. What the plan actually resolves

Directions below refer to the image, not geographic north. Adjacency means visible relative placement, not a surveyed distance.

| Plan feature | Readable evidence | Modeling boundary |
| --- | --- | --- |
| Meeting room | Enclosed yellow area at left | Trace its footprint; dimensions and finishes are unknown. |
| Dry lab | Central-left open area with repeated equipment islands and perimeter rectangles | Preserve island arrangement provisionally; most device text is unreadable. |
| Optical table | Labelled rectangle near the lower edge of the dry lab | Retain its relation to the islands; no verified instrument model. |
| BSC Room 1 / Room 2 | Adjacent blue rectangles near the center | Keep literal names; do not infer what BSC means or what equipment is inside. |
| Battery lab | Bounded area below the BSC pair | Preserve the zone; its label alone does not justify adding particular battery machines. |
| Analytical lab | Large bounded region at upper right | Trace perimeter and visible equipment footprints. |
| Wet Lab / Confocal / AFM | Labelled compartments beneath the analytical area | Preserve relative ordering; do not guess hidden entrances or furnishings. |
| XRD | Labelled lower-right compartment | Separate from the open dry lab; exact doorway width is unresolved. |

Door-swing arcs and partition openings are visible around the left vestibule and right-hand suite. Corridor space and gaps between equipment islands can be traced approximately. Overprinted labels obscure some connections: do not assign a complete door count, escape route, accessibility clearance, or robot traversability from this image. No reliable scale, ceiling height, north arrow, window schedule, or service routing has been established.

## 3. Annotation-to-inventory correspondence

| Association | What can be asserted | What remains unconfirmed |
| --- | --- | --- |
| AFM area ↔ high-resolution AFM | The page locates an AFM in an AFM Room; its linked equipment sheet identifies Bruker Dimension Icon | Exact footprint, installation configuration and current placement. |
| XRD area ↔ Empyrean XRD | The inventory assigns Empyrean to an XRD Room | Which outline depicts it; revision consistency between plan and inventory. |
| Optical table ↔ optical instruments | Both appear in the evidence | No direct match to the listed Zeiss Smartproof 5 or other optical instrument. |
| Dry/characterization area ↔ Instron 68TM, TA 850 | These devices are listed for the characterization facility | Neither can be mapped to a particular plan rectangle. |
| Colored numbers | Allocation annotations with a separate legend | They are not an OEM model key; do not convert numbers into device identities. |

Represent location confidence independently from device identity confidence. An inventory entry must not become a precise coordinate without a photograph, readable plan key, or institutional confirmation.

## 4. Implemented scene and controls

The definition is in `Hooke/real_labs/scenes/university_precision.json`; geometry and room features are in `Hooke/real_labs/university_precision.py`. The existing room/bench builders and namespaced instrument builder are reused. Instrument geometry is newly authored from the references, rather than presented as an imported manufacturer's CAD file.

### Room and supports

- The authored envelope is **19.6 × 9.8 × 3.1 m**. These dimensions are estimates; the source plan has no verified scale.
- Solid internal partitions retain the meeting zone, dry-lab islands, BSC pair, southern battery zone and eastern analytical/wet/confocal/AFM/optics/XRD compartments. Openings are actual gaps in collision geometry, with inferred positions and widths where the source is unclear.
- Nine benches/tables provide the dry-lab islands, rear counters, lower optical table, separate tester support, AFM support, analytical counter, battery-zone counter and meeting table. Their supports and placement are authored estimates.
- The source's unverified BSC, battery, confocal and XRD contents remain room/counter context. The scene does not invent a full equipment inventory from room labels.

### Instruments

| Instrument | Evidence used | Implemented mechanical interface | Unimplemented behavior |
| --- | --- | --- | --- |
| Instron 68TM family | The [university-linked OEM brochure](https://www.ntu.edu.sg/media/docs/librariesprovider121/research-/sgsr/facilities/lab1/6800series_brochurev1_-2021.pdf?sfvrsn=4dbfbba2_3) provides twin-column photographs and a standard frame of 0.760 × 0.715 × 1.640 m; the NTU load capacity/configuration is unconfirmed | `crosshead`, `upper_grip`; lower-clamped inert coupon, frame columns, guide rods, load-cell-shaped housing and side dashboard | Tensile force, strain, deformation, load-cell response, OEM interlocks and a complete material-test experiment |
| Bruker Dimension Icon AFM | The [official equipment sheet](https://www.ntu.edu.sg/media/docs/librariesprovider121/research-/sgsr/facilities/lab1/afm.pdf?sfvrsn=cbf338ea_3) identifies manufacturer/model and shows the installed apparatus; a 210 mm specimen chuck is documented | `coarse_x`, `coarse_y`, `coarse_head`; original scan-head housing, machined stages, cables, chuck and inert disk | Piezo scan dynamics, tip–surface interaction, feedback, nanometre precision, vibration isolation and surface measurements |

The Instron frame is manufacturer-sized, but grips and dashboard placement are representative. AFM overall dimensions and hidden components are estimated from the photo. Device identity confidence is stronger than placement confidence: neither instrument's exact coordinates in the low-resolution floor plan have been established. AFM coarse-stage controls must not be interpreted as the device's microscopic scanning axes.

### Local verification and limitations

CPU checks exercise **five controls**, with three target positions per control. Combined endpoint configurations, individual-axis swept poses, specimen scale, bench layout and conservative doorway collision bounds are checked by the local validation and `tests/test_university_precision_geometry.py`. These checks concern the authored model; they do not verify institutional fidelity, structural load ratings, accessibility, scientific performance or real-device safety.

The evidence files and detailed local results are retained under ignored `temp/research/university_precision/`. Defined review cameras include overview, interior, tester workstation and AFM detail views. This document does not embed original source photographs or claim permission to distribute them.

MuJoCo supports the declared CPU mechanical controls. USD, GLB and Blender exports are visual snapshots; they do not establish validated Isaac articulation or scientific behavior. No real-world experimental protocol or operating settings are supplied.

## 5. Four review gates

1. **Room:** the partitioned prototype exists. Next, compare its projected plan with the source, obtain scale references, review the inferred openings and confirm plan age and administrative room scope. Local clearance checks do not resolve those uncertainties.
2. **Equipment:** the two selected instruments have device-specific exterior geometry and separate evidence records. Next, confirm installed configurations, hidden surfaces, full support footprints and service-access envelopes with additional photographs or dimensioned drawings.
3. **Controls:** CPU positioning and authored geometry checks cover the five declared axes. Continue reviewing coupled motions, specimen access and task-specific failure cases; separately validate any future Isaac dynamics or scientific process model.
4. **Tasks:** the current scene provides short mechanical positioning/inspection actions. Medium specimen-transfer sequences, longer multi-station experiments and autonomous robot completion remain future work; they require actual controllers, reset/failure criteria and evidence beyond these joint checks.

Each gate records evidence reviewed and unresolved issues. Institutional review is a separate milestone; none of these local checks authorizes an “institution-validated digital twin” claim.

## 6. Evidence to request from the laboratory

- Current plan revision and confirmation of which depicted areas belong to Lab 1 / room 02-06.
- One dimensioned floor plan, two independent reference distances, ceiling height, bench heights/depths, door widths and swing directions.
- Overlapping room photographs from each doorway and corner; both sides of each central equipment island; photos linking the optical table, BSC rooms, battery zone and eastern suite.
- Current device inventory with model/configuration and assigned room; front, rear, side and sample-access views for the first AFM, XRD or mechanical-test workstation.
- OEM installation drawings or CAD with applicable reuse terms; access envelopes, intended supports and non-sensitive utility interfaces.
- Clarification of obscured openings, numbered annotations and relocated equipment; permission boundaries for publishing images, models and institutional names.

Until these arrive, publish only an explicitly labelled reference-informed prototype with its uncertainties and validation scope.
