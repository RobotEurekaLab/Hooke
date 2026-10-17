# Duke guarded mechanical inspection

`duke_joint_mechanical_testing` contains a reference-informed approximation of
the installed TestResources 830LE-AT and its nearby controls and shelving. The
official photograph supports the overall arrangement, clear specimen enclosure,
tall upper drive, separate control cabinet and open shelves. It does not establish
the concealed mechanism, precise room layout or installed options.

## Mechanical interface

The `biaxial_frame` equipment has these controls:

| Control | Range | Scope |
| --- | --- | --- |
| `axial_chuck_position` | 0–0.20 m, downward | Original frame, drive and range preserved |
| `lower_chuck_rotation` | −0.60–0.60 rad | Original frame, drive and range preserved |
| `guard_opening` | 0–100° | New authored hinged access mechanism and ideal servo |

The equipment metadata identifies mechanical revision 2. The original two
controls and original cameras remain compatible. The new `specimen_access`
camera and `specimen_surface` site supplement the retained `inert_coupon`
reference datum. The surface site is on material; the old datum is slightly in
front of the material and is not an exact working-distance reference.

The source-informed enclosure has enabled contact geometry, open ram and
torsion bearing bores, connected supports and a full opening sweep. Ideal joints
carry the mechanism; modeled running gaps do not validate bearing preload,
stiffness or rated machine loads. The chosen 550 mm clear column spacing is an
interpretation of an 830L-family option, not a verified installed dimension.

## Measured dry inspection

```python
import xml.etree.ElementTree as ET
import mujoco
from real_labs.builder import build_scene
from real_labs.duke_frame_motion import GuardedInspection
from real_labs.runtime import InstrumentSession

root, manifest = build_scene("duke_joint_mechanical_testing")
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
session = InstrumentSession(model, manifest)
report = GuardedInspection(session).run()
```

Begin at resting home. The operation opens the guard, lowers the unloaded upper
grip, indexes the lower fixture in both directions, returns the index, raises
the upper grip and closes the guard. It uses measured actuator motion without
resetting or writing live joint positions. Every completed physics step is
checked for tracking and enabled moving/world clearances, including pairs
normally omitted by parent filtering. A failed step requests a hold and reports
partial coverage; a failed leg has no whole-leg clearance claim.

The closest upper-grip/coupon gap is **55 mm**. This operation does not load the
coupon and is not a tensile, torsional, fatigue or fracture experiment. It does
not perform sample pickup or transfer. First-hit camera rays identify the actual
coupon in the open configuration; glass transmission, contrast and image
readability need separate image inspection. Discrete checks do not prove
continuous collision clearance.

## Local validation record

The recorded seven-phase operation completed 92,606 observed steps over
185.212 simulated seconds. It is preserved under its executed task digest. A
subsequent task-only fix preserves exact requested subdivision endpoints, so a
legal rotational limit cannot become an out-of-range value through floating-point
roundoff. Public limits remain strict. Six recorded phase plans remain bitwise
identical; the seventh replaces a final residual of at most 2.78e−17 with the
exact requested zero. The historical long record is not relabeled as a new run.

Current focused tests cover legal-boundary rejection/acceptance, real support
connections, full-travel direct distances, model/state binding and disturbed
failure. Raw traces, immutable source snapshots, actual-state exports and
independent review records remain under the ignored
`temp/university/optimization_120_20261004/duke_fidelity/` directory.

## References and remaining fidelity

- [Duke installed-machine photograph](https://mems.duke.edu/wp-content/uploads/sites/10/2024/08/Test-Resources-830LE-AT_300x400p.png)
- [Manufacturer 830L-family brochure](https://testresources.com/docs/systems/830L.pdf)

The installed option, guard actuation, concealed bearing dimensions, electronic
controls, service routing and room dimensions remain estimates. No OEM CAD,
institutional endorsement or calibrated material-process model is claimed.
