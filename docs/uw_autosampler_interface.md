# UW IsoLab autosampler interface, version 3

The `uw_isolab` scene keeps its analysis/preparation room, other instruments,
and original overview, interior and workstation cameras. Its autosampler now
uses an original model informed by the UW-hosted Picarro assembly manual. The
analyzer cabinet is 432 × 432 × 191 mm; the free inert vial is 12 mm in diameter
and 32 mm high. The installed analyzer generation remains uncertain.

The stand, open XYZ guides, 96-position tray, vaporizer and services are
estimated geometry. This is a dry geometric inspection model. It does not
inject liquid, measure isotopes, model vacuum, or reproduce an OEM protocol.

## Explicit action migration

The equipment ID stays `isotope_analyzer`. The old `carousel`, `gantry_x`,
`gantry_y` and `injection_z` keys are rejected. They are not automatically
mapped: version 3 has different coordinate origins, travel and vial geometry.

| Control | Version 3 range | Unit |
| --- | --- | --- |
| `v3_gantry_x` | −0.190 to +0.190 | m |
| `v3_gantry_y` | 0 to 0.285 | m |
| `v3_injection_z` | 0 to 0.180, downward | m |

The former Z range was 0–0.230 m. This is an explicit mechanism revision, not
an adjustment to hide an overlap. The candidate's complete 180 mm stroke is
retained, as are its drive limits and physical collision shapes.

## Checked dry inspection

```python
from real_labs.runtime import InstrumentSession
from real_labs.uw_isolab_motion import IsolabInspection

session = InstrumentSession.from_files("scene.xml", "manifest.json")
task = IsolabInspection(session)
report = task.run()
assert report["status"] == "succeeded"
```

Use a current export of `uw_isolab`. Construction rejects modified model,
control, camera, specimen or source metadata and foreign simulation state.
The task starts near the measured home pose, allows the free vial to settle,
travels through raised XY corridors, approaches its actual cap and the static
vaporizer datum, retracts, checks the full Z stroke in the clear front corridor,
and returns without resetting the scene.

A successful observation means approximately 4.5 mm of measured noncontact
stand-off and less than 50 µm of lateral error in this model. It is not syringe
calibration or evidence of image contrast. The visible vial cap and the enabled
solid vial surrogate share the same top plane; the ray record identifies which
of those coincident surfaces was selected.

`task.move([x, y, z])` accepts only the declared raised travel and measured
approach corridors. It rejects out-of-range targets and unqualified low XY
moves before applying commands. The full rectangular joint box contains a
wash-support obstruction. Raw `InstrumentSession.command` remains a low-level
joint interface, so it does not certify a collision-free path.

Every executed step checks enabled moving-assembly/world pairs, including
parent-filtered pairs, other held instrument controls, tracking and the free
vial's support. Free-vial/static contact is checked separately; this is not an
all-world collision certificate. Failures retain actual state, request a hold
and distinguish executed steps from completed checks. No hidden weld,
teleportation or episode reset is used to retain the vial.

## Validation boundaries

The guide clearances, inertial properties, drive parameters, rack pitch and
contact coefficients are authored assumptions. The room uses MuJoCo's
`implicitfast` solver and a 0.5 ms timestep; the earlier isolated study used
Euler integration. Full-room execution therefore needs its own evidence.
Isaac parity, fluid analysis, instrument safety and full source fidelity remain
unqualified. Local run reports and images are stored under ignored `temp/`.

Reference media are not redistributed. Source references are the
[UW-hosted Picarro assembly manual](https://isolab.ess.washington.edu/resources/manuals/Picarro_AutosamplerTraining_06-13-2018.pdf)
and the [Picarro L2120i datasheet hosted by Universidade de Vigo](https://cactiweb.uvigo.es/docs/manuales/Analizador-Picarro-2120i.pdf).
