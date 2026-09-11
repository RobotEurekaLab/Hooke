# Hooke

Hooke is a simulation and benchmark platform for robotic automation in biology
laboratories, built on top of [AutoBio](https://arxiv.org/abs/2505.14030)
(originally from [autobio-bench/AutoBio](https://github.com/autobio-bench/AutoBio)).

## Layout

- `autobio/` — the MuJoCo-based simulator, task definitions, and demonstration
  data generation/rendering pipeline. See `autobio/README.md`.
- `openpi/` — VLA training/inference stack (forked from
  [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)),
  upgraded to support fine-tuning **pi0.5** on AutoBio tasks.
- `openpi-pi0-legacy/` — the original pi0-only openpi setup, kept for reference.
- `RoboticsDiffusionTransformer/` — RDT baseline, forked from
  [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer).

## Status

Actively evolving beyond the original AutoBio release — expect the layout and
tooling here to diverge over time as new features land.
