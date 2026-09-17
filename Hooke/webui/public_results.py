"""Public experiment outcomes exclude model archives and evaluator state."""

from archetypes.task_catalog import CATALOG

EXPERIMENT_FIELDS = frozenset(
    (
        "status",
        "task",
        "backend",
        "seed",
        "mode",
        "parity_qualified",
        "render_settings",
        "display_only",
        "cameras",
        "declared_time_limit_s",
        "source_check",
        "source_check_kind",
        "source_success",
        "steps",
        "contact_steps",
        "simulation_s",
        "within_declared_time_limit",
        "runtime",
        "total_wall_s",
        "space_experiment",
        "surface_mission",
        "space_environment",
        "assessment",
        "expert_phases",
        "max_fk_position_error_m",
        "max_fk_rotation_error_rad",
        "acceleration_source",
        "error",
    )
)


def public_result(result, task_name=None):
    entry = CATALOG.get(task_name or result.get("task"))
    if entry is None or entry.completion_rule not in ("space_experiment", "surface_sampling"):
        return result
    return {key: value for key, value in result.items() if key in EXPERIMENT_FIELDS}
