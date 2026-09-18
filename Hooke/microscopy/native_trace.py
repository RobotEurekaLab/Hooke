"""Archive every observed PhysX step, separately from rendered observations."""

import hashlib

import numpy as np


def archive_native_trace(adapter, runtime, output):
    if not adapter.rows:
        return None
    steps = len(adapter.rows)
    if runtime.get("steps") != steps or runtime.get("physics_events_since_load") != steps:
        raise ValueError("Native trace and actual physics events differ")
    path = output / "trajectory.npz"
    temporary = output / "trajectory.tmp.npz"
    np.savez_compressed(temporary, **{key: np.asarray([row[key] for row in adapter.rows])
                                     for key in adapter.rows[0]})
    temporary.replace(path)
    return dict(schema_version=1, backend="isaac", physics_engine="PhysX",
                file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                steps=steps, physics_events_since_load=runtime["physics_events_since_load"],
                max_fk_position_error_m=adapter.max_fk_position_error,
                max_fk_rotation_error_rad=adapter.max_fk_rotation_error,
                world_origin_m=adapter.loaded["conversion"]["physics_options"].get("world_origin_m", [0., 0., 0.]),
                observation_coordinates="source world, metres",
                source="Observed joints after every actual PhysX control step; no source integration",
                render_observations_are_sparse=True)
