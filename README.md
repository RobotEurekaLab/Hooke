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
to the task. All 165 catalogue scenes have passed a short load, physics-step,
and RGB check. All 22 original experts completed at seed 0; Isaac reproduced
the 18 source predicate successes (15 also within declared time limits).
Full physical, visual, and performance equivalence remains unqualified.

See [setup, validation evidence, and limitations](docs/isaac_scene_parity.md)
for the existing server configuration, commands, and LAN comparison page.
See the [completion metrics and remaining work](docs/isaac_status.md) for the
experimental branch assessment and a committed seed-0 validation snapshot.
