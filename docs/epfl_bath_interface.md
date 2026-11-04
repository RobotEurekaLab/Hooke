# EPFL hood bath positioning

The `epfl_lspn_chemistry` scene reconstructs selected apparatus in the historical
[LSPN twin-hood photograph](https://www.epfl.ch/labs/lspn/group/lab-tour/).
Each hood has a supported sash, a grounded clamp rack, fixed glassware, and a
torque-driven two-stage scissor jack carrying a hotplate and open bath. Left and
right hoods use separate controls and object names.

## Interface revision 2

| Control | Meaning | Range |
| --- | --- | --- |
| `sash_height` | Vertical sash displacement | 0–0.480 m |
| `bath_height` | Measured bath rim height above that hood's worktop | 0.245–0.375 m |

The flask and condenser stay clamped to the rack while the bath moves. The old
`flask_lift` command is retired and returns an explanatory error. It is not an
alias for bath height. The flask sample site remains fixed.

The screw motor applies torque through eight mechanical coordinates and four
loop/transmission constraints. The deck has no hidden vertical slide. The
controller uses measured shaft position, estimated gravity load and projected
inertia. It holds both jacks while either sash or bath is commanded.

Commands use a smooth height trajectory. A complete 130 mm stroke takes at
least 15 simulated seconds, followed by one second of settling. A shorter
requested duration is extended and reported in `trajectory_duration_s` and
`simulated_seconds`; it does not request an unqualified fast stroke. A response
reports actual height and `target_reached`, not just the requested target.

## Python and web controls

Use `InstrumentSession.command()` for movement and `InstrumentSession.advance()`
for additional simulation. These paths maintain the torque controllers at every
step. Direct `mujoco.mj_step()` calls bypass this controller and are not a
supported way to operate the assembled jacks.

```python
import xml.etree.ElementTree as ET
import mujoco
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession

root, manifest = build_scene("epfl_lspn_chemistry")
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
session = InstrumentSession(model, manifest)
session.command("left_hood", "sash_height", 0.48)
result = session.command("left_hood", "bath_height", 0.375)
assert result["target_reached"]
session.command("left_hood", "bath_height", 0.245)
session.reset()
```

The existing university laboratory page exposes both controls through its
equipment selector. Its images remain recorded previews, as labeled on the
page; control responses update measured state rather than a live rendered image.
Export settling uses the same holding controller. Exported geometry does not
include an automatically portable Isaac controller or establish PhysX behavior.

## Geometry and model limits

- Hood dimensions, internal fittings, masses and inertias are authored estimates.
  The photograph is dated 2019; this is not a surveyed room or identified OEM CAD.
- The deck and thermal payload use estimated masses of 0.5 kg and 1.8 kg.
  The ideal 1.25 mm screw lead has no thread friction or self-locking model.
- Glassware is statically attached. The flask's visible hollow shell has
  conservative convex wall colliders, rather than a solid sphere filling its
  cavity. This does not validate insertion through the installed condenser,
  fluid containment, glass fracture or frictional clamp retention.
- Previously recorded isolated probe trials used a 200 microsecond timestep.
  They do not qualify contact with arbitrary tools at the room's 2 ms timestep.
- Additional photographed hotplates, secondary jacks and hidden service routing
  remain incomplete. Heating, chemistry and extraction airflow are not modeled.

The generic scalar-joint audit explicitly leaves the coupled bath mechanism for
dedicated validation. Scene, dynamic and rendering evidence belongs under
`temp/university/optimization_120_20261004/epfl_hood_fidelity/production_v1/`.
Full-room fidelity, complete scientific experiments and native Isaac validation
remain outside the completed evidence.
