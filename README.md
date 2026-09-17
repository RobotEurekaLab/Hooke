# Hooke

Hooke is a simulation and benchmark platform for robotic automation in biology
laboratories, built on top of [AutoBio](https://arxiv.org/abs/2505.14030)
(originally from [autobio-bench/AutoBio](https://github.com/autobio-bench/AutoBio)).

## Layout

- `Hooke/` — the MuJoCo-based simulator, task definitions, and demonstration
  data generation/rendering pipeline. See `Hooke/README.md`.
- `openpi/` — VLA training/inference stack (forked from
  [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)),
  upgraded to support fine-tuning **pi0.5** on AutoBio tasks.
- `openpi-pi0-legacy/` — the original pi0-only openpi setup, kept for reference.
- `RoboticsDiffusionTransformer/` — RDT baseline, forked from
  [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer).

## Status

Actively evolving beyond the original AutoBio release — expect the layout and
tooling here to diverge over time as new features land.

## MuJoCo / Isaac backends

The `/backends` web page can open catalogue scenes and run their original
controllers with either MuJoCo or an isolated Isaac Sim 4.5 / PhysX process.
The adapter reuses source assets and feeds actual PhysX state and contacts back
to the task. In the recorded baseline, all 165 catalogue scenes passed a short
load, physics-step, and RGB check. All 22 original experts completed at seed 0; Isaac reproduced
the 18 source predicate successes (15 also within declared time limits).
Full physical, visual, and performance equivalence remains unqualified.

See [setup, validation evidence, and limitations](docs/isaac_scene_parity.md)
for the existing server configuration, commands, and LAN comparison page.
See the [completion metrics and remaining work](docs/isaac_status.md) for the
experimental branch assessment and a committed seed-0 validation snapshot.
See [shared dual-arm control and ideal pipette volume accounting](docs/isaac_shared_processes.md)
for the reusable process modules, separate assessments, and validation scope.
See [full two-container transfer and actual rendered keyframes](docs/isaac_pipette_transfer.md)
for the 45-second task and historical validation. The
[ten-seed transfer report](docs/validation/isaac_pipette_transfer_ten_seed_summary.json)
records ten successful experts in each backend and their frozen runtime versions.
See [continuous centrifuge insertion, closure, and locking](docs/isaac_centrifuge_chain.md)
for the earlier protocol, and [the complete centrifuge cycle](docs/isaac_centrifuge_cycle.md)
for measured rotor control, braking and safe unlocking.
See [operations and configuration](docs/isaac_operations.md),
[actual visual comparisons](docs/isaac_visual_configuration.md), and
[the seven acceptance areas](docs/isaac_completion_plan.md) for the current implementation,
evidence and reduced-model limits.

## Space experiment workstations

The `/space-worlds` page shows orbital, lunar and Martian workstations with
shared robot/instrument assets and explicit environmental limits. Gravity is
applied by the selected engine; gas pressure remains a reference condition.
See [scenes, assets, environment assumptions and actual checks](docs/space_worlds.md).
See [open scene and asset sources](docs/space_asset_sources.md) for the researched
ISS interior, lunar terrain, Martian terrain and experiment asset candidates.

The `/space-assets` page compares three optional space scenes with attributed
NASA/SRB assets in MuJoCo and Isaac. See [reuse, research terms and validation](docs/space_asset_reuse.md).

The `/space-experiments` page provides nine continuous robot tasks across the
orbital, lunar and Martian worlds: sample loading/return, calibrated spring
mass measurement and mechanically gated analytic spectral measurements.
Both engines use the same robot, scene and controller. Actual videos, public
measurement curves and acceptance records are described in
[space experiments](docs/space_experiments.md). Real LROC/HiRISE terrain crops
have separate provenance and collision checks; the default robot workspaces
retain procedural ground.
