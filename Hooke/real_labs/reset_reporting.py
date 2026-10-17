"""Preserve observed, absent and uncertain resets across task phases."""


def combine_reset_reports(phases):
    """A confirmed reset wins; missing or uncertain evidence stays unknown."""
    values = [phase.get("reset_performed") for phase in phases]
    status = (
        True
        if any(v is True for v in values)
        else (False if values and all(v is False for v in values) else None)
    )
    return dict(
        reset_performed=status,
        explicit_reset_performed=any(
            p.get("explicit_reset_performed") is True for p in phases
        ),
        automatic_reset_detected=any(
            p.get("automatic_reset_detected") is True for p in phases
        ),
    )
