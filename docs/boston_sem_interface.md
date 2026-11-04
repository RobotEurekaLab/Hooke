# Boston SEM mechanical interface, revision 3

The `boston_photonics_pml_sem` scene now uses a compact, supported door stage
with seven actuated coordinates. It provides physical specimen access and dry
surface registration. It does not simulate electron imaging, vacuum, sealing,
or calibrated SEM measurement.

## Interface and migration

The scene ID and equipment ID `sem_station` are unchanged. The equipment
manifest declares `mechanism_revision.version = 3` and
`automatic_legacy_mapping = false`. This is an incompatible mechanism update:
saved actions must be replanned against the new geometry and joint frames.

| Control key | Compiled travel | Role |
| --- | --- | --- |
| `door_slide_outer` | 0–0.210 m | First supported telescopic slide |
| `door_slide_inner` | 0–0.210 m | Second slide; combined extension reaches 0.420 m |
| `stage_x` | −0.065–0.065 m | Specimen translation |
| `stage_y` | −0.065–0.065 m | Specimen translation |
| `stage_z` | 0–0.050 m | Specimen elevation |
| `stage_tilt` | −3–70° | Specimen tilt; commands use radians |
| `stage_rotation` | −π–π rad | Finite rotation range, not continuous rotation |

Read exact serialized limits from the manifest. These individual ranges do
not form a collision-free Cartesian product. Some coupled poses intersect the
chamber; the task rejects obstructed paths without reducing the joint ranges
or disabling contacts.

The former `specimen_exchange`, `chamber_access`, and `specimen_azimuth` keys
are retired. Constructing the old `SpecimenAccess` task raises a migration
error. There is no automatic mapping of its 600 mm access coordinate to the
new two-slide mechanism. The retained specimen remains a 30 mm diameter,
12 mm high inert stub; the new `stub_surface` site lies on its actual surface.

## Full access, observation and return

From the repository root, with the project environment active:

```python
from real_labs.export import export_scene
from real_labs.runtime import InstrumentSession
from real_labs.boston_sem_motion import StageInspection

folder = "temp/boston_sem_revision3/boston_photonics_pml_sem"
export_scene("boston_photonics_pml_sem", folder, validate=True)
session = InstrumentSession.from_files(
    f"{folder}/scene.xml", f"{folder}/manifest.json"
)
report = StageInspection(session).run()
assert report["status"] == "succeeded", report
```

Use `PYTHONPATH=Hooke` when invoking Python directly. The task starts from a
measured near-home state and integrates each movement through the actuators:

1. Extend both door slides through their full 210 mm strokes. Verify that the
   entire retained stub reaches the outside of the real front opening.
2. Close both slides.
3. Raise the stage by 45 mm and measure the physical 8 mm pole-to-stub gap.
   Register the actual surface at the center, at a 3 mm X offset, and again
   at the center.
4. Lower the specimen and return physically to the measured initial pose.

The specimen stays clamped throughout. The surface observation is a geometric
ray and coordinate check against the real collider, not a rendered SEM image
or a focus calibration. The separate `route()` method is a longer diagnostic
sweep; it is not needed for this access cycle.

## Validation and failures

The task binds the scene, compiled model, data ownership, public controls and
source files. Each actual physics step checks simulation time, native warning
events, measured tracking, held axes and direct collision distances, including
pairs ordinarily removed by parent-body filtering. Conservative bounds omit
only pairs proven farther than the distance cap over the declared domain.
Reported minima are capped lower bounds; discrete checks do not prove
continuous-time collision clearance.

A failed step is counted as executed but receives no completed-check credit.
Failure requests a bounded measured-position hold when the task still owns
the state. It does not explicitly reset the simulation. Native numerical
resets and ambiguous time failures are reported separately; prior bad-state
warning history requires explicit session recovery before reuse.

Local evidence for the production migration is written under
`temp/university/optimization_120_20261004/boston_sem_production_v3/`:
source snapshots, candidate-preservation checks, focused tests, the full-state
operation trace, and exact measured checkpoint exports. Historical revision 2
and isolated-candidate records retain their original hashes and are not
relabeled as executions of revision 3 production code.

The recorded production access cycle before the illumination addition completed
107,544 executed and checked
steps across nine phases, with no reset. Replay verified every recorded step
time and phase boundary, full exterior access of the retained stub, and all
three 8 mm physical surface registrations. Its smallest capped clearance
lower bound was approximately 0.600 mm. Nine focused tests and seven regional
tests passed; these include obstructed poses and actual failure cases, not
only the successful route. These results qualify the recorded mechanical
path, not every coupled configuration or camera view. Its source hashes remain
unchanged in the historical record.

## Chamber observation

The current scene adds roof-supported camera and illumination hardware. The
[Supra-family technical sheet](https://www.ntnu.edu/documents/140082/1269041159/EM-lab-ProdInfoFE-SEM.pdf/6fb1a853-fa92-4c7b-a653-62fff3fdadf3)
documents a CCD camera with infrared illumination. Fixture dimensions, positions
and intensity here are authored estimates; the installed BU camera configuration
has not been verified. The manifest identifies this as observation revision 1.

The `chamber_scope` render uses an emissive white-light approximation to make the
actual retained specimen, support and pole visible. It does not model infrared
spectral response, detector gain/noise or electron imaging. The chamber walls and
camera position remain unchanged. Lighting has not been calibrated to hardware.

Independent checks preserve the original geometry, physics and six cameras, and
establish at least 48.978 mm separation between the new fixtures and the padded
joint envelope of the recorded access cycle. Two thousand additional actual
steps, including a small X excursion at home and at the registered pose, match
the preceding model bit for bit. Historical checkpoint states are restored into
the illumination derivative for image comparison; this is not a new execution
of the full 107,544-step route. The original unrestricted joint box remains
obstructed. Evidence is under
`temp/university/optimization_120_20261004/boston_chamber_observation/`.

## Source fidelity and limits

The BU facility photograph informs the exterior and console arrangement.
Supra-family documentation informs the nominal 330 mm chamber width, 270 mm
height and five-axis stage envelope. Hidden telescopes, bearings, masses,
gains, cabinet details and detector dimensions are authored estimates, not OEM
CAD or a measured reproduction of the BU installation. The current exterior
adds connected source-informed ports and detector hardware, while room and
apparatus detail remain incomplete.

The door is not vacuum-sealed; the stub cannot be freely picked up or loaded
by a robot. Electron optics, detector signals, charging, material removal and
scientific measurement are absent. Physical opacity is retained. The observation
approximation improves visible geometry but does not establish calibrated
camera measurement or SEM imaging. Validation here uses CPU
MuJoCo and does not establish equivalent Isaac/PhysX dynamics.
