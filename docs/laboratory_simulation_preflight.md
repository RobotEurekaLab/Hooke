# Laboratory simulation preparation

University laboratories retain the existing **University → Subject → Laboratory → Scene / task** classification. Their simulation package also needs an explicit contract for objects, physical behavior, perception and validation.

The workflow follows the inspection, semantic authoring, physics, visual-review and validation stages described in [NVIDIA's scene preparation article](https://developer.nvidia.com/blog/how-to-use-ai-agents-to-prepare-3d-scenes-for-simulation/). NVIDIA's [SimReady Foundation guide](https://github.com/NVIDIA/simready-foundation/blob/main/nv_core/sr_specs/docs/guides/getting_started.md) describes USD-specific profiles and validation. Hooke's local inventory is **not** a replacement for that validator or an Isaac runtime test.

## Current artifacts and their scope

| Artifact | What it establishes |
| --- | --- |
| Source register and scene manifest | Institutional identity, inspected references, estimates and equipment roles |
| `scene.xml` | Portable MuJoCo geometry, bodies, joints, actuators and contact parameters |
| `validation.json` and geometry tests | Bounded mechanical motion and the specifically tested collision/clearance cases |
| Three rendered views and explicit image reviews | Current geometry is readable from selected review viewpoints |
| `laboratory.blend`, `.glb`, `.usdc` | Editable or portable visual snapshots; physical USD behavior is unverified |
| `simulation_preflight.json` | Structured inventory, existing native-bridge structural checks, and missing simulation data |
| Optional `native_source/` | Compiled source archive accepted as input by the existing `backends.usd_scene.SceneBridge` interface; native execution remains unqualified |

The inventory records body hierarchy, fixed versus moving bodies, mass and inertia, gravity compensation, collision flags and masks, contact friction, visual materials, named equipment controls, sample attachment and review-camera transforms/FOV. Authored numerical values are not measured equipment specifications. Material colors do not establish a physical material class.

A clamped sample is explicitly **not freely graspable**. A target site on a moving machine does not make the sample a free rigid body. A review camera has no assumed runtime resolution, sampling rate or verified robot-target visibility. These distinctions prevent a rendered apparatus from silently becoming a claimed robot task.

## Generate a report and native source archive

Export the scene normally first. Run from the repository root:

```bash
PYTHONPATH=Hooke .venv/bin/python -m real_labs.preflight \
  --input temp/university/expansion_100_20261003/scenes \
  --scene uwa_ngcf_c72 \
  --native-archive
```

For a fully exported selection, replace `--scene` with `--batch path/to/batch.json`; the batch contains a `scene_ids` list. Omit `--native-archive` for the inventory alone.

This command uses CPU MuJoCo and existing project code. It starts no simulator service or renderer and installs no packages. Reports and native archives stay alongside the selected package under ignored `temp/`.

The native archive reuses `backends.baseline.write_snapshot` rather than introducing another physics conversion format. It contains the numeric compiled model, named hierarchy, canonical XML, assets and reset state. Its existing export checks reload XML and compare masses, inertias, joint ranges, actuator gains and geometry ownership. Native source IDs can then survive the existing USD bridge's naming normalization.

## Remaining acceptance stages

1. Author task semantics into USD prims while retaining source identifiers and university bindings.
2. Map physical materials and justify contact, mass and inertia values for each intended task.
3. Define task sensors, resolution/rate and target frames; verify target visibility and clipping from those sensors.
4. Convert with the existing Isaac bridge and test contact behavior, joints, reset and actual task interactions.
5. Select and pin an appropriate SimReady profile and version, inspect the resulting USD APIs and run its official validator.
6. Repeat the affected physical and visual checks after repairs; preserve hashes tying evidence to the reviewed artifact.

The local report keeps `isaac_runtime_validated`, `simready_validated`, `robot_task_validated` and `scientific_process_validated` false. A structurally supported source can attempt conversion; it has not passed these later stages. Neither a rendered liquid nor a moving test fixture establishes a scientific process model.
