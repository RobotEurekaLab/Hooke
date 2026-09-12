"""Phase II infrastructure: chain atomic tasks into one continuous episode
and score it with graded success metrics, instead of resetting the
simulation between sub-tasks.

This exists to reproduce and extend AutoBio's own trajectory-concatenation
finding (private/proposal.tex Phase II; the original paper's Table 4:
Close/Open thermal-cycler-lid solved individually at 96-100%, but the
concatenated Close->Open task collapses to 3-4% for both pi0 and RDT). The
paper's own protocol is explicit about the mechanics: "execute one subtask
to completion before switching prompts" -- i.e. no reset() between steps,
the next sub-task starts from wherever the previous one's actions actually
left the scene.

IMPORTANT SCOPING NOTE: this module only runs the *scripted expert*
end-to-end, not a trained VLA policy. That's deliberate -- the interesting
phenomenon Table 4 identified (a *trained* policy memorizing per-task
trajectories rather than composing them) can only be reproduced by chaining
a real checkpoint's rollouts, which needs the GPU training this session
explicitly deferred (see the 2026-09-12 conversation). What this harness
*can* establish now, without training anything, is whether the chaining
mechanics themselves are sound: does an episode that switches sub-tasks
mid-simulation (no reset) behave correctly, and does the scripted policy
reliably complete both halves back to back? That's a necessary (not
sufficient) precondition for later re-measuring the trained-policy
collapse on this same harness.

Cross-instrument composition (e.g. pickup_centrifuge_tube ->
centrifuge_5430_close_lid) needs a combined scene with both instruments
physically present and is not attempted here -- see private/TODO.md. Only
compositions where every step shares one already-built Task/Expert class
(today: thermal_cycler_close <-> thermal_cycler_open, via
task_override) are supported.
"""
from __future__ import annotations

import dataclasses

from archetypes.task_catalog import CatalogEntry


@dataclasses.dataclass(frozen=True)
class CompositeStep:
    entry: CatalogEntry
    task_override: str  # which of the entry's task_override variants this step is


def run_composite_scripted(steps: list[CompositeStep], seed: int) -> dict:
    """Runs `steps` as one continuous episode: a single reset(), then each
    step's scripted Expert.execute() in turn (no reset in between), stopping
    at the first failed or errored step. Returns graded results: which
    steps were attempted, which succeeded, and where (if anywhere) it broke.
    """
    if not steps:
        raise ValueError("steps must be non-empty")

    task_cls, expert_cls = steps[0].entry.load_classes()
    for step in steps[1:]:
        other_cls, _ = step.entry.load_classes()
        if other_cls is not task_cls:
            raise ValueError(
                "run_composite_scripted only supports steps that share one Task/Expert "
                "class (i.e. the same scene, switched via task_override) today -- "
                f"got both {task_cls.__name__} and {other_cls.__name__}. Cross-instrument "
                "composition needs a combined scene asset; see private/TODO.md."
            )

    expert = expert_cls(task_cls.load())
    # reset() itself branches on self.task to decide the *initial* state
    # (e.g. thermal_cycler_open's reset() forces the lid closed, since
    # opening it is the point) -- so the first step's task_override must be
    # set *before* reset(), exactly the bug fixed in webui/scene_render.py
    # for the single-task preview path. Getting this wrong here silently
    # starts the sequence from the wrong initial state instead of raising.
    expert.task = steps[0].task_override
    expert.reset(seed=seed)  # the *only* reset for the whole sequence

    per_step_success: list[bool] = []
    errors: list[str] = []
    failed_at_transition = False

    for i, step in enumerate(steps):
        expert.task = step.task_override
        try:
            expert.execute()
        except Exception as e:
            errors.append(f"step {i} ({step.task_override}) raised {type(e).__name__}: {e}")
            failed_at_transition = i > 0
            break
        success = bool(expert.check())
        per_step_success.append(success)
        if not success:
            failed_at_transition = i > 0
            break

    completed = len(per_step_success)
    full_sequence_success = completed == len(steps) and all(per_step_success)

    return {
        "steps": [s.task_override for s in steps],
        "per_step_success": per_step_success,
        "completed_steps": completed,
        "full_sequence_success": full_sequence_success,
        # True only when the failure happened on a step *after* the first --
        # i.e. a step that (per its own standalone success rate) usually
        # works, but broke specifically when chained after another step
        # without a reset. Mirrors the paper's own framing of the failure
        # as happening "at the seam", not within either task alone.
        "failed_at_transition": failed_at_transition,
        "errors": errors,
    }


def run_composite_batch(steps: list[CompositeStep], seeds: list[int]) -> dict:
    """Runs run_composite_scripted for each seed and aggregates."""
    results = [run_composite_scripted(steps, seed) for seed in seeds]
    n = len(results)
    return {
        "steps": [s.task_override for s in steps],
        "n_episodes": n,
        "full_sequence_success_rate": sum(r["full_sequence_success"] for r in results) / n,
        "transition_failure_rate": sum(r["failed_at_transition"] for r in results) / n,
        "per_episode": results,
    }
