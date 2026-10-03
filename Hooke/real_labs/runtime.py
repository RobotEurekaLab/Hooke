"""Measured, CPU-only equipment controls for reconstructed laboratory scenes.

This runtime exercises mechanical joints. Its optional process output is a
synthetic geometry check, not a chemical, biological or calibrated measurement.
It neither moves samples into instruments nor claims robot task completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import mujoco


@dataclass(frozen=True)
class _Control:
    joint: int
    actuator: int
    lower: float
    upper: float
    units: str
    mode: str


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class InstrumentSession:
    """Control scalar position/velocity actuators and observe their motion.

    Equipment manifest entries use matching semantic keys in ``joints`` and
    ``actuators``. Optional ``process`` metadata specifies ``sample_site`` and
    ``sample_body``, and optionally an ``access_control`` with ``closed_value``
    and ``closed_tolerance``. All distances use metres and angles use radians.
    """

    def __init__(self, model: mujoco.MjModel, manifest: dict[str, Any]):
        self.model = model
        self.manifest = manifest
        self.data = mujoco.MjData(model)
        self.equipment: dict[str, dict[str, Any]] = {}
        self.controls: dict[tuple[str, str], _Control] = {}
        for item in manifest.get("equipment", []):
            equipment_id = item["id"]
            if equipment_id in self.equipment:
                raise ValueError(f"Duplicate equipment id: {equipment_id}")
            self.equipment[equipment_id] = item
            for semantic, actuator_name in item.get("actuators", {}).items():
                joint_name = item.get("joints", {}).get(semantic)
                if joint_name is None:
                    raise ValueError(f"Missing joint for {equipment_id}.{semantic}")
                joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                joint_type = model.jnt_type[joint]
                if joint_type not in (
                    mujoco.mjtJoint.mjJNT_SLIDE,
                    mujoco.mjtJoint.mjJNT_HINGE,
                ):
                    raise ValueError(f"Control requires scalar joint: {joint_name}")
                if (
                    model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or model.actuator_trnid[actuator, 0] != joint
                    or model.actuator_gear[actuator, 0] != 1.0
                ):
                    raise ValueError(
                        f"Control requires direct joint transmission: {actuator_name}"
                    )
                # Verify the declared servo, since motor ctrl uses torque/force.
                mode = item.get("actuator_modes", {}).get(semantic, "position")
                gain = float(model.actuator_gainprm[actuator, 0])
                bias = model.actuator_biasprm[actuator]
                bias_index = {"position": 1, "velocity": 2}.get(mode)
                if (
                    bias_index is None
                    or gain <= 0
                    or bias[0] != 0
                    or not math.isclose(float(bias[bias_index]), -gain)
                    or (mode == "velocity" and bias[1] != 0)
                ):
                    raise ValueError(f"Control requires {mode} servo: {actuator_name}")
                if not model.actuator_ctrllimited[actuator]:
                    raise ValueError(f"Control requires actuator limits: {joint_name}")
                lower, upper = model.actuator_ctrlrange[actuator]
                if mode == "position":
                    if not model.jnt_limited[joint]:
                        raise ValueError(
                            f"Position control requires joint limits: {joint_name}"
                        )
                    lower = max(model.jnt_range[joint, 0], lower)
                    upper = min(model.jnt_range[joint, 1], upper)
                if not math.isfinite(lower + upper) or lower >= upper:
                    raise ValueError(
                        f"Invalid control range: {equipment_id}.{semantic}"
                    )
                if any(
                    c.actuator == actuator or c.joint == joint
                    for c in self.controls.values()
                ):
                    raise ValueError(
                        f"Control is already owned by another entry: {joint_name}"
                    )
                units = "m" if joint_type == mujoco.mjtJoint.mjJNT_SLIDE else "rad"
                self.controls[equipment_id, semantic] = _Control(
                    joint,
                    actuator,
                    float(lower),
                    float(upper),
                    units + ("/s" if mode == "velocity" else ""),
                    mode,
                )
        self.reset()

    @classmethod
    def from_files(cls, scene: str | Path, manifest: str | Path) -> InstrumentSession:
        """Load an exported MJCF scene and its equipment manifest on the CPU."""
        with Path(manifest).open(encoding="utf-8") as source:
            metadata = json.load(source)
        return cls(mujoco.MjModel.from_xml_path(str(scene)), metadata)

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Unknown model object: {name}")
        return result

    def _position(self, control: _Control) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[control.joint]])

    def _measured(self, control: _Control) -> float:
        if control.mode == "velocity":
            return float(self.data.qvel[self.model.jnt_dofadr[control.joint]])
        return self._position(control)

    def reset(self) -> None:
        """Restore authored initial state; this is an episode reset, not a task."""
        mujoco.mj_resetData(self.model, self.data)
        for control in self.controls.values():
            self.data.ctrl[control.actuator] = min(
                control.upper, max(control.lower, self._measured(control))
            )
        mujoco.mj_forward(self.model, self.data)

    def command(
        self, equipment_id: str, control: str, value: float, *, duration: float = 2.0
    ) -> dict[str, Any]:
        """Apply a bounded servo target and report observed position or velocity."""
        spec = self.controls[equipment_id, control]
        target = _finite(value, "Target")
        duration = _finite(duration, "Duration")
        if duration <= 0 or duration > 30:
            raise ValueError("Duration must be in (0, 30] seconds")
        if not spec.lower <= target <= spec.upper:
            raise ValueError(
                f"Target outside [{spec.lower}, {spec.upper}] {spec.units}"
            )
        before = self._measured(spec)
        start_time = float(self.data.time)
        self.data.ctrl[spec.actuator] = target
        steps = max(1, math.ceil(duration / self.model.opt.timestep))
        mujoco.mj_step(self.model, self.data, nstep=steps)
        mujoco.mj_forward(self.model, self.data)
        measured = self._measured(spec)
        if not all(
            math.isfinite(float(q))
            for values in (self.data.qpos, self.data.qvel)
            for q in values
        ):
            raise RuntimeError("Physics produced nonfinite joint state")
        elapsed = float(self.data.time) - start_time
        if not math.isclose(
            elapsed, steps * self.model.opt.timestep, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise RuntimeError(
                "Physics time discontinuity; the simulation may have reset"
            )
        return {
            "equipment": equipment_id,
            "control": control,
            "target": target,
            "before": before,
            "measured": measured,
            "error": measured - target,
            "units": spec.units,
            "mode": spec.mode,
            "joint_position": self._position(spec),
            "simulated_seconds": elapsed,
        }

    def process_readiness(self, equipment_id: str) -> dict[str, Any]:
        """Observe sample proximity and access interlocks without changing state."""
        process = self.equipment[equipment_id].get("process")
        if not process:
            return {"ready": False, "reasons": ["no_process_model"]}
        if not all(
            math.isfinite(float(v))
            for values in (self.data.qpos, self.data.qvel)
            for v in values
        ):
            return {"ready": False, "reasons": ["nonfinite_physics_state"]}
        reasons = []
        if "access_control" in process:
            spec = self.controls[equipment_id, process["access_control"]]
            if spec.mode != "position":
                raise ValueError("Access interlock requires a position control")
            closed = _finite(process.get("closed_value", 0), "Closed value")
            tolerance = _finite(
                process.get("closed_tolerance", 0.01), "Closed tolerance"
            )
            if tolerance < 0:
                raise ValueError("Closed tolerance must be nonnegative")
            if abs(self._position(spec) - closed) > tolerance:
                reasons.append("access_open")
        site = self._id(mujoco.mjtObj.mjOBJ_SITE, process["sample_site"])
        body = self._id(mujoco.mjtObj.mjOBJ_BODY, process["sample_body"])
        joint = int(self.model.body_jntadr[body])
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            reasons.append("sample_is_not_free_body")
        mujoco.mj_forward(self.model, self.data)
        distance = math.dist(self.data.site_xpos[site], self.data.xpos[body])
        load_tolerance = _finite(process.get("load_tolerance", 0.015), "Load tolerance")
        if load_tolerance <= 0:
            raise ValueError("Load tolerance must be positive")
        if distance > load_tolerance:
            reasons.append("sample_not_at_target")
        if joint >= 0 and self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            dof = self.model.jnt_dofadr[joint]
            speed = math.sqrt(sum(float(v) ** 2 for v in self.data.qvel[dof : dof + 3]))
            max_speed = _finite(
                process.get("max_sample_speed", 0.01), "Maximum sample speed"
            )
            if max_speed <= 0:
                raise ValueError("Maximum sample speed must be positive")
            if speed > max_speed:
                reasons.append("sample_moving")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "sample_distance_m": distance,
            "load_tolerance_m": load_tolerance,
            "sample_body": process["sample_body"],
            "sample_site": process["sample_site"],
            "placement_check": "centroid proximity only; containment is not validated",
        }

    def measure(self, equipment_id: str) -> dict[str, Any]:
        """Return an explicitly synthetic placement reading if observed gates pass."""
        readiness = self.process_readiness(equipment_id)
        if not readiness["ready"]:
            raise RuntimeError(
                f"Process unavailable: {', '.join(readiness['reasons'])}"
            )
        return {
            "equipment": equipment_id,
            "measurement_kind": "synthetic_geometry_proxy",
            "scientific_validity": "not a calibrated physical or biological measurement",
            "robot_task_completed": False,
            "sim_time_s": float(self.data.time),
            **readiness,
        }

    def inspect(self) -> dict[str, Any]:
        """Describe authored capabilities alongside current measured state."""
        equipment = []
        for equipment_id, item in self.equipment.items():
            controls = {}
            for (owner, semantic), spec in self.controls.items():
                if owner == equipment_id:
                    controls[semantic] = {
                        "measured": self._measured(spec),
                        "mode": spec.mode,
                        "joint_position": self._position(spec),
                        "target": float(self.data.ctrl[spec.actuator]),
                        "range": [spec.lower, spec.upper],
                        "units": spec.units,
                    }
            equipment.append(
                {
                    "id": equipment_id,
                    "kind": item.get("kind"),
                    "capabilities": item.get("capabilities", []),
                    "controls": controls,
                    "process": self.process_readiness(equipment_id),
                }
            )
        return {
            "id": self.manifest.get("id"),
            "engine": "mujoco",
            "equipment": equipment,
        }

    def smoke_test(self, *, duration: float = 2.0) -> dict[str, Any]:
        """Exercise each control independently, then restore the initial scene.

        A commanded target alone never counts as success: both observed motion
        and final tracking error must pass. Blocked controls therefore fail.
        """
        results = []
        try:
            for (equipment_id, semantic), spec in self.controls.items():
                self.reset()
                span = spec.upper - spec.lower
                target = (spec.lower + spec.upper) / 2
                if abs(target - self._measured(spec)) < span * 0.1:
                    target = spec.lower + span * 0.75
                result = self.command(equipment_id, semantic, target, duration=duration)
                tolerance = max(1e-4, span * 0.05)
                moved = abs(result["measured"] - result["before"]) > span * 0.01
                result.update(
                    {
                        "tracking_tolerance": tolerance,
                        "moved": moved,
                        "passed": moved and abs(result["error"]) <= tolerance,
                    }
                )
                results.append(result)
        finally:
            self.reset()
        return {
            "id": self.manifest.get("id"),
            "control_count": len(results),
            "passed": bool(results) and all(result["passed"] for result in results),
            "scope": "mechanical joint motion only; no robot or scientific task completion",
            "controls": results,
        }
