"""Shared acceptance rules for complete missions and idle control episodes."""

from pathlib import Path

import numpy as np


def expert_passed(result):
    return bool(
        result.get("mode") == "expert"
        and result.get("status") == "TASK_SUCCEEDED"
        and result.get("source_success")
        and result.get("within_declared_time_limit")
        and result.get("surface_mission", {}).get("checks")
        and all(result["surface_mission"]["checks"].values())
    )


def controls_held(folder):
    folder = Path(folder)
    with np.load(folder / "source/model.npz", allow_pickle=False) as model:
        reset = model["reset_ctrl"].copy()
    with np.load(folder / "trajectory.npz", allow_pickle=False) as trajectory:
        controls = trajectory["control"]
        return bool(
            controls.ndim == 2
            and len(controls)
            and controls.shape[1] == len(reset)
            and np.isfinite(controls).all()
            and np.all(controls == reset)
        )


def control_passed(expert, control, folder):
    if not expert_passed(expert):
        return False
    duration_matched = (
        abs(control.get("simulation_s", 0) - expert["simulation_s"]) < 0.0021
    )
    rejection = control.get("control_rejection", {})
    early_rejected = bool(
        control.get("status") == "CONTROL_REJECTED_EARLY"
        and rejection.get("irreversible") is True
        and rejection.get("check") == "humanoid_remained_upright"
        and (
            rejection.get("max_tilt_rad", 0) >= 0.5
            or rejection.get("min_pelvis_height_m", float("inf")) <= 0.55
        )
        and control.get("surface_mission", {})
        .get("checks", {})
        .get("humanoid_remained_upright")
        is False
        and 0 < control.get("simulation_s", 0) <= expert["simulation_s"]
        and abs(control.get("requested_control_horizon_s", 0) - expert["simulation_s"])
        < 0.0021
    )
    return bool(
        expert_passed(expert)
        and control.get("mode") == "no_action"
        and (
            (control.get("status") == "CONTROL_COMPLETE" and duration_matched)
            or early_rejected
        )
        and control.get("source_success") is False
        and controls_held(folder)
    )
