# PolyU tribometer: revision 3

The default `polyu_materials_tribology` scene now uses a physically supported **free coupon**, two opening guards, opposing jaws and low longitudinal retaining lips. The original X/Z joint frames, full ranges, gains, body inertias, contact settings, 40 × 40 × 4 mm specimen envelope and four cameras are retained.

The installed PolyU photograph supports the instrument silhouette and local bench placement. Bruker-family patent figures inform the split transparent cover and rectangular holder concept. Hidden bearings, dimensions, powered door/clamp actions and the low lips are authored mechanisms; they are not verified features of the exact installed model. Source photographs are not redistributed.

## Public operations

```python
import xml.etree.ElementTree as ET
import mujoco

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.polyu_access_motion import CouponAccess, FreeCouponReciprocation

root, manifest = build_scene("polyu_materials_tribology")
session = InstrumentSession(
    mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
    manifest,
)

access = CouponAccess(session).run()
assert access["status"] == "succeeded"
loaded = FreeCouponReciprocation(session).run()
assert loaded["status"] == "succeeded"
```

Both operations use the caller's actual `InstrumentSession`. They do not create a substitute live model or reset the state between operations. The task contract is bound to the current canonical scene, metadata and owned model/data. The current finite-face calculations deliberately support this scene's world-aligned fixture; they do not promise arbitrary rotated scene reuse.

`CouponAccess.run()` clamps and checks retention, raises the counterface, opens both guards, releases the supported coupon, observes its support, reclamps, closes and returns the counterface. It exposes a loading corridor; it does **not** pick up, replace or transfer the sample.

`FreeCouponReciprocation.run()` clamps, approaches the specimen, performs a −10 mm / +10 mm / center traverse under measured contact, and retracts. Loaded phases require both guards closed, both physical jaw contacts and actual counterface contact. The 150 µm nominal side clearances allow bounded sample movement; exact sample repositioning is not claimed.

## Controls and specimen references

| Key | Contract |
| --- | --- |
| `coupon_x` | Original −65 to +65 mm carriage travel; useful loaded path is narrower. |
| `probe_height` | Original −100 to +80 mm range; sufficiently low commands physically contact or collide with the fixture. |
| `guard_left`, `guard_right` | Authored 0–100° opening proxies with physical hinges and contact surfaces. |
| `clamp_front`, `clamp_rear` | Authored −0.2 to +15 mm closure/release travel. |
| `loading_center` | Fixed loading datum on the translated support; used by the existing strict specimen registrar. |
| `tribology_coupon` | Actual top site attached to the 17.28 g free coupon. |

The sample has one free joint and no attachment equality. Support, jaw and lip contacts provide its retention. The 17.28 g mass is an authored inert material approximation, not an identified experimental specimen. Direct scalar commands retain the full physical ranges and can produce unsafe combinations; the checked operation wrappers provide the stated interlocks.

## Version migration

The default scene requires explicit `interface_revision=3`. The former `polyu_contact_motion.LoadedReciprocation` is a revision-two **fixed-coupon** task and rejects the new interface. Use the two classes above. There is no silent mapping of old fixed-coupon assumptions to free-body behavior.

The revision-two builder/task source and archived source-bound packages remain historical evidence. The new geometry reuses the former exterior and load-head construction, with separate access/fixture geometry and a local contact-checking module. Shared specimen registration, runtime, physics defaults and rendering are unchanged.

## Validation and limits

The isolated revision-three fixture completed 20,750 checked loaded-operation steps and 11,800 checked access steps. Independent review covered full pair sets at extra poses, exact finite-face depth calculations, recorded state/force replay and disturbed failures. Production integration passed 16,333 named property comparisons and a continuous access → loaded reciprocation → access episode: **44,350 actual, fully checked steps across 26 phases**, with exact complete-state handoffs and no reset. Eight focused tests cover contact, source/state binding, nonzero task construction, disturbances and native reset reporting; five regional tests also pass. Generated traces, screenshots and exact source hashes remain under ignored `temp/`.

In that production episode the measured loaded relative path was 39.12 mm, peak normal force 0.128 N, peak tangential force 0.0422 N and maximum lateral specimen offset 0.155 mm. These are simulation observations, not calibrated experimental results. A fresh task constructor computes its private geometric bounds without changing the caller's positions, velocities, controls, warm start, forces or clock. Production-specific independent review and actual-state image review are recorded separately from the isolated fixture review.

Each actual step checks direct geometry, including parent-filtered pairs, free-body retention, intended contact depth, time and native warnings. Failed steps are not counted as fully checked. Failures request measured-position hold and report partial coverage and any detected native reset; they do not silently reset the scene or guarantee arrest of an externally disturbed free body. Capped clearances are lower bounds, and discrete checks do not prove continuous collision freedom.

The original 8 ms numerical contact response, friction and ideal gravity compensation are simulation assumptions. There is no calibrated force controller, material-specific friction, wear, deformation, scientific result or autonomous loading robot.
