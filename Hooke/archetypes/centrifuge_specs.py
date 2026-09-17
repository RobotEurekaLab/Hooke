"""Instrument geometry and qualified recipes for the shared lid controller.

5430 keeps its original 15-second limit with explicit 2 rad/s and 1.5 rad/s²
planner limits. The mini now approaches with open fingers, grasps below the
lid site, follows the lever and releases. Historical equivalence reports
refer to their recorded versions, not these repaired controllers.
"""
import numpy as np
import mujoco

from instrument import Centrifuge_Eppendorf_5430, Centrifuge_Eppendorf_5910, Centrifuge_tiangen_tgear_mini
from archetypes.lever_lock_centrifuge import LeverLockSpec
from archetypes.lid_lock import lid_standstill_passes


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
    qc_vel=2.0,
    qc_acc=1.5,
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

CENTRIFUGE_MINI_SPEC = LeverLockSpec(
    name="centrifuge_mini",
    instrument_cls=Centrifuge_tiangen_tgear_mini,
    instrument_prefix="/centrifuge_tiangen_tgear_mini:",
    scene_file="mani_centrifuge_mini.xml",
    task_name="centrifuge_mini_close_lid",
    prompt_prefix="close the lid of the centrifuge mini",
    time_limit=30.0,
    jntlimit_index=1,
    jntlimit_offset=0.0,
    rel_quat=np.array([0.0, -0.7071, 0.7071, 0.0]),
    mode_offsets={
        '1/detach': np.array([0.0, 0.008, 0.017]),
        '2/detach': np.array([0.0, 0.03, 0.03]),
        'grip': np.array([0.0, 0.0, -0.012]),
    },
    recipe=[
        {"op":"home_arm"},
        {"op":"cache_pose", "name":"approach", "mode":"2/detach"},
        {"op":"cache_pose", "name":"grip", "mode":"grip"},
        {"op":"cache_lever_path", "name":"close"},
        {"op":"gripper", "value":0},
        {"op":"move_to_pose", "mode":"2/detach", "num_steps":2,"cached_pose":"approach"},
        {"op":"move_to_pose", "mode":"grip", "num_steps":2,"cached_pose":"grip"},
        {"op":"gripper", "value":240},
        {"op":"lever_close", "cached_path":"close"},
        {"op":"gripper", "value":0},
        {"op":"move_to_lever_end", "num_steps":5},
    ],
    qpos_perturb_lows=(-0.05, 0.0, -0.15, -0.025, 0.0, -0.1),
    qpos_perturb_highs=(0.05, 0.15, -0.05, 0.025, 0.3, 0.1),
    completion_check=lid_standstill_passes,
)


ALL_SPECS = {
    CENTRIFUGE_5430_SPEC.name: CENTRIFUGE_5430_SPEC,
    CENTRIFUGE_5910_SPEC.name: CENTRIFUGE_5910_SPEC,
    CENTRIFUGE_MINI_SPEC.name: CENTRIFUGE_MINI_SPEC,
}
