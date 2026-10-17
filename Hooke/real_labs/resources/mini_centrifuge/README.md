# Mini centrifuge mechanical adapter resources

These are final physical asset resources derived from the existing Hooke miniature centrifuge meshes. Original files under `Hooke/model` and `Hooke/assets/instrument` remain unchanged. The repository MIT license and RobotEurekaLab copyright apply; retain the top-level LICENSE notice. No third-party photographs or new downloads are included.

`contacts.npz` stores named local convex vertex sets, body ownership and offsets. The 548 rotor parts retain the real loading bores. The lid has 417 clipped cover parts and 316 source-sectional hinge parts; 117 source-sectional fixed hinge parts preserve the open bearing cavity. The original four main-body proxies and original inertia meshes remain. Runtime requires NumPy and MuJoCo, not CoACD, Manifold, Blender or the generation logs.

`lid_visual.obj` matches the local contact fit repair and preserves original split normals on surviving surfaces. Local material removal is approximately 1.827%; original mass/inertia and frames are deliberately preserved simulation parameters. This is an authored mechanical fit correction, not certified OEM CAD.

`provenance.json` records schema, resource/source hashes, decomposition settings, source-derived port datums, numerical requirements and limitations. `names`, `bodies`, `vertices` and `offsets` are the complete archive schema; no pickle/object arrays are used. Raw decomposition logs and experiment captures belong under ignored `temp/`.

## Mechanical inspection

The ordinary Rochester, Indiana and Penn scenes remain unloaded. A separate,
persistable variant adds a supported parking cradle and a free 2 g inert gauge:

```python
from real_labs.mini_centrifuge_fixture import create_inspection_session
from real_labs.mini_centrifuge_motion import inspect_mini_centrifuge

session = create_inspection_session("rochester_genomics", port=0)
report = inspect_mini_centrifuge(session)
assert report["status"] == "succeeded", report["failure"]
```

Run with `Hooke` on the Python module path. `export_inspection_scene` writes the
variant's portable XML and manifest for `InstrumentSession.from_files`.

The operation lifts the gauge with a bounded external handling wrench, inserts
and releases it, closes the original lid, commands both signed inspection
speeds, stops, reopens, extracts the free gauge and returns it to its cradle.
Handling is explicit; there is no modeled robot gripper. The gauge is neither
welded nor teleported. Every executed step checks enabled instrument contacts,
held controls and pose constraints; reports distinguish executed steps from
completed checks. A failed operation requests a hold without resetting state.

The copied instrument requires `ccd_tolerance=1e-10` for its small convex contact
features. The adapter tightens this setting only in models containing this
asset; the existing 2 ms scene timestep, joint frames, inertia and drive limits
remain unchanged. The direct monitor uses conservative separating bounds,
independent of MuJoCo's parent-body collision filter. Its 0.1 micrometre
instrument overlap limit and 0.2 mm allowed gauge-contact limit are authored
mechanical checks, not calibrated instrument specifications. Ports 0 and 4
have standalone loaded evidence; the integrated full-route evidence currently
qualifies port 0. Other port choices remain unqualified until actually run.

This operation models dry geometry and mechanics only. It does not simulate
scientific separation, biological processing, calibrated imbalance sensing,
an automatic lid interlock or manufacturer operating speeds.
