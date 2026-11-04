"""Explicit dry inspection recipes with measured poses and checked return paths.

These are original bounded positioning tasks, not OEM operating procedures or
scientific process models. Live state changes only through the existing motion
executor. Reference transforms are evaluated in separate read-only FK data.
"""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import mujoco
import numpy as np

from real_labs.motion import _model_signature
from real_labs.inspection_contacts import InspectionAccess
from real_labs.rigid_access import require_binding


@lru_cache(maxsize=1)
def _recipes():
    return json.loads(Path(__file__).with_name("inspection_recipes.json").read_text())


def recipe_for(scene):
    """Return an isolated recipe; callers cannot modify the registered version."""
    return deepcopy(_recipes().get(scene))


def _observation(access, target):
    model, data = access.model, access.data
    reference = mujoco.MjData(model)
    reference.qpos[:] = data.qpos
    reference.qpos[access.motion.qadr] = target
    mujoco.mj_forward(model, reference)
    sites = {}
    for key, name in access.session.equipment[access.motion.equipment]["sites"].items():
        sid = model.site(name).id
        sites[key] = dict(
            actual_position_m=data.site_xpos[sid].tolist(),
            reference_position_m=reference.site_xpos[sid].tolist(),
            translation_error_m=float(np.linalg.norm(data.site_xpos[sid] - reference.site_xpos[sid])),
            actual_rotation=data.site_xmat[sid].tolist(),
            reference_rotation=reference.site_xmat[sid].tolist(),
        )
    error = np.abs(access.motion.measured() - target)
    tolerance = np.array([.0002 if c.units == "m" else .003 for c in access.motion.controls])
    return dict(
        sites=sites, joint_error=error.tolist(), joint_tolerance=tolerance.tolist(),
        registered=bool(np.all(error <= tolerance)),
        measurement="Rigid named-site transforms and joint registration; no synthetic scientific measurement",
    )


def _sequence(device, home):
    current, targets = home.copy(), []
    for action in device["actions"]:
        current = current.copy()
        for key, value in action["targets"].items():
            current[device["controls"].index(key)] = value
        targets.append((action["name"], current.copy()))
    return targets + [("return_" + name, target) for name, target in targets[:-1][::-1]] + [("return_measured_home", home)]


def run_inspection(session, *, on_phase=None):
    """Execute all declared mechanisms and reverse each sequence to measured home.

    ``on_phase`` receives completed phase evidence for incremental persistence.
    It must not mutate the session. Failure preserves the actual stopped state;
    the public operation dispatcher latches failure until an explicit reset.
    """
    scene = session.manifest.get("id")
    recipe = recipe_for(scene)
    if recipe is None:
        raise ValueError(f"No dry inspection recipe for {scene!r}")
    require_binding(session, scene)
    signature = _model_signature(session.model)
    initial = session.data.qpos.copy()
    start_time = float(session.data.time)
    phases, failure, returns = [], None, []
    for device in recipe["devices"]:
        access = InspectionAccess(session, scene, device["equipment"], device["controls"],
                                  minimum_m=device["clearance_m"])
        access.preflight()
        home = access.motion.measured().copy()
        for name, target in _sequence(device, home):
            try:
                report = access.move(target, speed_scale=device["speed_scale"])
                observation = _observation(access, target)
                phase = dict(equipment=device["equipment"], name=name,
                             target=target.tolist(), report=report, observation=observation)
                phases.append(phase)
                if on_phase is not None:
                    on_phase(deepcopy(phase))
                if report["status"] != "succeeded" or not observation["registered"]:
                    failure = dict(phase=name, reason=report.get("failure") or "target_not_registered")
                    break
            except (ValueError, RuntimeError) as error:
                failure = dict(phase=name, reason=f"{type(error).__name__}: {error}")
                break
        returns.append(dict(equipment=device["equipment"], **_observation(access, home)))
        if failure:
            break
    executed = sum(p["report"]["physics_steps"] for p in phases)
    checked = sum(p["report"]["direct_clearance"]["checked_steps"] for p in phases)
    coverage = executed > 0 and executed == checked
    unchanged = _model_signature(session.model) == signature
    returned = len(returns) == len(recipe["devices"]) and all(item["registered"] for item in returns)
    success = failure is None and bool(phases) and coverage and unchanged and returned
    return dict(
        success=success, failure=failure, scene=scene, recipe_version=recipe["version"],
        complete_declared_recipe=success, phases=phases, returns=returns,
        executed_steps=executed, checked_steps=checked, coverage_complete=coverage,
        initial_time_s=start_time, final_time_s=float(session.data.time),
        return_qpos_error=(session.data.qpos-initial).tolist(), model_unchanged=unchanged,
        limits=recipe["limits"] + [
            "Only the explicit positioning sequence is qualified, not the complete control-range Cartesian product.",
            "Discrete per-step checks do not prove continuous-time collision avoidance.",
        ],
    )
