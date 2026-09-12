"""LeverLockSpec instances for the currently-supported centrifuge instruments.

These specs reproduce `mani_centrifuge_5430.py` and `mani_centrifuge_5910.py`
exactly (numeric offsets, recipe ordering) -- see
`archetypes/test_lever_lock_equivalence.py` for the equivalence check.

`mani_centrifuge_mini.py` (the Tiangen T-Gear mini) is deliberately *not*
migrated here yet: its upstream `execute()` never actually finishes the lock
sub-sequence (it computes a lock pose and never calls `move_to` with it,
see the original file), and it was already observed to hit unrecoverable IK
failures on some random seeds during earlier data collection. Faithfully
porting a known-incomplete recipe into the generic archetype wouldn't be a
meaningful equivalence check, so this is tracked as a separate fix rather
than folded into this refactor.
"""
import numpy as np
import mujoco

from instrument import Centrifuge_Eppendorf_5430, Centrifuge_Eppendorf_5910
from archetypes.lever_lock_centrifuge import LeverLockSpec


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


CENTRIFUGE_5430_SPEC = LeverLockSpec(
    name="centrifuge_5430",
    instrument_cls=Centrifuge_Eppendorf_5430,
    instrument_prefix="/centrifuge_eppendorf_5430:",
    scene_file="mani_centrifuge_5430.xml",
    task_name="centrifuge_5430_close_lid",
    prompt_prefix="close the lid of the centrifuge eppendorf 5430",
    time_limit=15.0,
    jntlimit_index=0,
    jntlimit_offset=0.1592037,
    rel_quat=_axis_angle_quat(np.array([1.0, 0.0, 0.0]), np.pi),
    mode_offsets={
        '1/detach': np.array([0.0, 0.008, 0.017]),
        '2/detach': np.array([0.0, 0.008, 0.03]),
        'grip': np.array([0.0, 0.0, -0.005]),
        'lock_pre': np.array([0.0, 0.2, -0.017]),
        'lock': np.array([0.0, -0.045, -0.06]),
    },
    recipe=[
        {"op": "move_to_pose", "mode": "grip", "num_steps": 2},
        {"op": "gripper", "value": 240},
        {"op": "lever_close"},
        {"op": "gripper", "value": 100},
        {"op": "move_to_lever_end", "num_steps": 5},
        {"op": "move_to_pose", "mode": "lock_pre", "num_steps": 5, "quat_override": "lock_quat", "gripper_before": 255},
        {"op": "move_to_pose", "mode": "lock", "num_steps": 5, "quat_override": "lock_quat"},
        {"op": "move_to_pose", "mode": "lock_pre", "num_steps": 5, "quat_override": "lock_quat"},
    ],
    qpos_perturb_lows=(-0.1, 0.0, -0.2, -0.1, 0.0, -0.2),
    qpos_perturb_highs=(0.1, 0.3, 0.2, 0.1, 0.3, 0.2),
)

CENTRIFUGE_5910_SPEC = LeverLockSpec(
    name="centrifuge_5910",
    instrument_cls=Centrifuge_Eppendorf_5910,
    instrument_prefix="/centrifuge_eppendorf_5910:",
    scene_file="mani_centrifuge_5910.xml",
    task_name="centrifuge_5910_lid_close",
    prompt_prefix="close the lid of the centrifuge eppendorf 5910",
    time_limit=30.0,
    jntlimit_index=1,
    jntlimit_offset=-0.2,
    rel_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    mode_offsets={
        '1/detach': np.array([0.0, 0.0, -0.01]),
        '2/detach': np.array([0.0, 0.0, -0.15]),
        'grip': np.array([0.0, 0.0, 0.03]),
        'lock_pre': np.array([0.0, -0.15, 0.1]),
        'lock': np.array([0.02, 0.08, 0.1]),
    },
    recipe=[
        {"op": "move_to_pose", "mode": "1/detach", "num_steps": 2},
        {"op": "gripper", "value": 0},
        {"op": "move_to_pose", "mode": "grip", "num_steps": 2},
        {"op": "gripper", "value": 240},
        {"op": "lever_close"},
        {"op": "gripper", "value": 0},
        {"op": "move_to_lever_end", "num_steps": 5},
        {"op": "move_to_pose", "mode": "lock_pre", "num_steps": 5, "quat_override": "lock_quat", "gripper_before": 255},
        {"op": "move_to_pose", "mode": "lock", "num_steps": 5, "quat_override": "lock_quat"},
        {"op": "force_lock"},
        {"op": "wait", "seconds": 2},
        {"op": "move_to_pose", "mode": "lock_pre", "num_steps": 5, "quat_override": "lock_quat"},
    ],
    qpos_perturb_lows=(-0.2, 0.0, 0.0, -0.05, 0.0, -0.1),
    qpos_perturb_highs=(0.2, 0.3, 0.1, 0.05, 0.3, 0.1),
)

ALL_SPECS = {
    CENTRIFUGE_5430_SPEC.name: CENTRIFUGE_5430_SPEC,
    CENTRIFUGE_5910_SPEC.name: CENTRIFUGE_5910_SPEC,
}
