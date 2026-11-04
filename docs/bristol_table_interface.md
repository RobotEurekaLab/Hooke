# Bristol EQUALS mechanical interface, version 2

The Bristol scene now uses eight telescopic actuators connected to a free rigid
platform. Version 2 replaces the earlier independent XY-slide approximation.
The equipment ID remains `shake_platform`; the old `table_x` and `table_y`
actions are retired and have **no automatic mapping**.

## Evidence and estimates

[Bristol's facility description](https://www.bristol.ac.uk/research/groups/earthquakegeo/equip/)
specifies a 3 m square platform, eight actuators with 300 mm stroke and 70 kN
dynamic capacity each, and four vertical static support sections. It reports
up to 15 tonnes payload, about 5g depending on load, and ±150 mm displacement.
Those performance figures are source context, not simulated qualifications.

The model follows the photographed recessed pit, yellow bolted test
frame, surrounding guardrails, actuator hardware and stored-cylinder rack.
Hidden anchors, pit depth, bearing construction and the frame's identity remain
estimates. The model uses a 2,000 kg rigid platform/frame assembly, estimated
leg masses, explicit servo stiffness/damping and passive preload springs for
the static-support sections. These are engineering assumptions, not OEM data
or hydraulic pressure dynamics. The frame does not deform or fail.

## Controls and operation

The eight position-control keys are:

- `v2_vertical_0` through `v2_vertical_3`;
- `v2_horizontal_x_0` and `v2_horizontal_x_1`;
- `v2_horizontal_y_0` and `v2_horizontal_y_1`.

Each actuator retains ±150 mm travel and a ±70 kN force cap. The eight lengths
are redundant constraints on a six-degree-of-freedom platform. Arbitrary
combinations are therefore **not a valid workspace**. The low-level runtime's
scalar command interface is not a coordinated-motion safety interface.

Use `real_labs.bristol_table_motion.TableInspection` or
`inspect_bristol_table(session)` for the authored inert inspection. The task
commands compatible actuator lengths while measuring actual platform motion.
It checks the physical links, full moving/world pair set including
parent-filtered pairs, spherical-constraint closure, finite state, native
warnings and limits after every actual step. It returns using the actuators;
it never explicitly resets or teleports the platform. A native engine reset
or warning aborts the route, requests a measured hold, and is reported
separately from completed collision checks. Nonfinite entry time is rejected.

The inspection covers the full positive/negative single-axis displacement
targets and a bounded combined translation/rotation. Coupled corners that
exceed any actuator stroke are rejected. Plan samples are not continuous
collision certification; executed coverage is at the simulator timestep.
Ideal angular joints and spherical constraints transfer loads across actual
open bearing clearances. Numerical compliance and tracking error remain
observable. The task tolerance is 3 mm position and 2 mrad orientation; this
does not establish laboratory positioning accuracy.

The optional `coordinated_inspection_targets` metadata supplies one complete
compatible vector to the generic smoke test. Every declared control must move
and meet the unchanged tracking tolerance. This small smoke check does not
replace the separate full-travel task.

## Limits

This is a cold, rigid, inert mechanical surrogate. It does not reproduce a
seismic waveform, hydraulic flow, pressure regulation, structural stress,
damage, real bearing loads or facility safety procedures. The photographs and
documentation do not establish an exact surveyed room or hidden mechanism.
