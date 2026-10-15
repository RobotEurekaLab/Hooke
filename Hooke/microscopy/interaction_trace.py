"""Archive reduced cell interactions separately from sparse image observations."""

import hashlib

import numpy as np


def archive_interaction_trace(mechanics, output):
    snapshot = getattr(mechanics, "interaction_trace", None)
    if snapshot is None:
        return None
    columns = snapshot()
    metadata = mechanics.interaction_trace_metadata()
    rows = metadata["rows"]
    if not columns or "time_s" not in columns:
        raise ValueError("Interaction trace has no time column")
    for values in columns.values():
        if (not isinstance(values, np.ndarray) or values.ndim < 1 or len(values) != rows
                or values.dtype.kind not in "biuf" or not np.isfinite(values).all()):
            raise ValueError("Interaction trace columns must be finite numeric arrays with matching rows")
    times = columns["time_s"]
    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("Interaction trace times must increase strictly")
    if not rows:
        return metadata
    path = output / "interaction.npz"
    temporary = output / "interaction.tmp.npz"
    try:
        np.savez_compressed(temporary, **columns)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(metadata, file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                first_time_s=float(times[0]), last_time_s=float(times[-1]),
                columns=list(columns))
