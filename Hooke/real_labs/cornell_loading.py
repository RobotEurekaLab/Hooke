"""Measured hatch/carrier inspection of an authored Cornell loading assembly.

This is a rigid geometric operation, not a vacuum-system operating procedure.
It uses real servo steps, open collision geometry and physical return commands.
Failures preserve the measured state and request hold; reset remains explicit.
"""

from functools import lru_cache
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError, _model_signature


SCENE = "cornell_schlom_mbe"
EQUIPMENT = "oxide_mbe"
CONTROLS = ("load_lock_door", "transfer_rod")


@lru_cache(maxsize=1)
def _authored_binding():
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    session = InstrumentSession(model, manifest)
    return {
        "signature": _model_signature(model),
        "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "controls": session.controls,
        "equipment": session.equipment[EQUIPMENT],
        "camera": manifest["camera"],
        "contype": model.geom_contype.copy(),
        "conaffinity": model.geom_conaffinity.copy(),
        "disableflags": int(model.opt.disableflags),
        "enableflags": int(model.opt.enableflags),
    }


def _require_binding(session):
    expected = _authored_binding()
    if session.manifest.get("id") != SCENE or session.data.model is not session.model:
        raise ValueError("Cornell inspection requires its authored model and state")
    if session.controls != expected["controls"]:
        raise ValueError("Cornell control bindings changed")
    if (
        session.equipment.get(EQUIPMENT) != expected["equipment"]
        or session.manifest.get("camera") != expected["camera"]
    ):
        raise ValueError("Cornell sample or camera metadata binding changed")
    if _model_signature(session.model) != expected["signature"]:
        raise ValueError("Cornell geometry, collision or physics binding changed")
    declared = session.manifest.get("scene_sha256")
    if declared is not None and declared != expected["scene_sha256"]:
        raise ValueError("Cornell manifest does not match the physical model")
    return expected


class LoadingObserver:
    """Direct signed distances include pairs hidden by parent-body filtering."""

    cap_m = 0.002
    minimum_allowed_m = 0.00025

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session = session
        model = session.model
        moving = {
            int(model.jnt_bodyid[session.controls[EQUIPMENT, c].joint])
            for c in CONTROLS
        }
        ids = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith(EQUIPMENT + "__")
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        self.ids = np.array(ids, dtype=int)
        self.pairs = np.array(
            [
                (i, j)
                for i in ids
                for j in ids
                if model.geom_bodyid[i] in moving
                and model.body_weldid[model.geom_bodyid[i]]
                != model.body_weldid[model.geom_bodyid[j]]
                and (model.geom_bodyid[j] not in moving or i < j)
            ],
            dtype=int,
        )
        self.reset()

    def reset(self):
        self.entries = 0
        self.checked = 0
        self.minimum = self.cap_m

    def check(self, model, data):
        self.entries += 1
        binding = self.binding
        if (
            int(model.opt.disableflags) != binding["disableflags"]
            or int(model.opt.enableflags) != binding["enableflags"]
            or not np.array_equal(model.geom_contype, binding["contype"])
            or not np.array_equal(model.geom_conaffinity, binding["conaffinity"])
        ):
            raise RuntimeError("loading_contact_configuration_changed")
        # Exact primitive/mesh AABBs provide a conservative distance lower bound.
        # Far pairs contribute the common cap; every closer candidate is queried.
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum(
            "nij,nj->ni", rotation, model.geom_aabb[:, :3]
        )
        half = np.einsum("nij,nj->ni", np.abs(rotation), model.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        near = np.linalg.norm(separation, axis=1) < self.cap_m
        current = self.cap_m
        for first, second in self.pairs[near]:
            gap = min(
                float(
                    mujoco.mj_geomDistance(
                        model, data, int(first), int(second), self.cap_m, None
                    )
                ),
                self.cap_m,
            )
            current = min(current, gap)
            if gap < self.minimum_allowed_m:
                raise RuntimeError(
                    "loading_clearance_lost:"
                    + model.geom(int(first)).name
                    + ":"
                    + model.geom(int(second)).name
                )
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def result(self, executed_steps):
        complete = executed_steps > 0 and executed_steps == self.checked
        return {
            "executed_steps": executed_steps,
            "observer_entries": self.entries,
            "checked_steps": self.checked,
            "unchecked_steps": executed_steps - self.checked,
            "coverage_complete": complete,
            "checked_steps_clearance_lower_bound_m": (
                self.minimum if self.checked else None
            ),
            "whole_leg_clearance_lower_bound_m": self.minimum if complete else None,
            "clearance_reporting_cap_m": self.cap_m,
        }


class CornellLoading:
    """Bounded, geometry-only inspection with a real supported inert coupon."""

    def __init__(self, session):
        self.session = session
        self.observer = LoadingObserver(session)
        self.motion = JointMotion(session, EQUIPMENT, CONTROLS, clearance_m=0.0003)
        self.motion.resolution[:] = (0.01, 0.002)

    def move(self, target, *, step_observer=None):
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        # Binding failures are preconditions, not simulated task failures. Never
        # apply a hold/reset or mutate state before rejecting an altered model.
        _require_binding(self.session)
        self.observer.reset()
        plan = self.motion.plan_joint_target(target)

        def observe(model, data):
            self.observer.check(model, data)
            if step_observer is not None:
                step_observer(model, data)

        result = self.motion.execute(plan, step_observer=observe)
        result["direct_clearance"] = self.observer.result(result["physics_steps"])
        return result

    def run(self, *, step_observer=None):
        """Open, position, observe, retract and close from the measured home.

        Inspection uses 40 mm of the preserved 200 mm stroke. Other positions
        may be geometrically valid but hidden; this is not a loading/grasp task.
        """
        from real_labs.usability import camera_target_screening

        _require_binding(self.session)
        start = self.motion.measured()
        if np.any(np.abs(start) > (0.003, 0.0002)):
            raise PlanningError(
                "Cornell round trip requires measured closed/retracted home"
            )
        reports, observation = [], None
        for name, target in (
            ("open_hatch", (1.1, start[1])),
            ("position_carrier", (1.1, 0.04)),
            ("retract_carrier", (1.1, start[1])),
            ("close_hatch", start),
        ):
            result = self.move(target, step_observer=step_observer)
            reports.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
            if name == "position_carrier":
                model, data = self.session.model, self.session.data
                screened = camera_target_screening(model, data, self.session.manifest)[
                    0
                ]
                visible = "loading_inspection" in screened["candidate_views"]
                site = model.site(EQUIPMENT + "__site_substrate").id
                observation = {
                    "camera": "loading_inspection",
                    "visible": visible,
                    "camera_screening": screened,
                    "sample_world_m": data.site_xpos[site].tolist(),
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist(),
                    "ctrl": data.ctrl.tolist(),
                    "time_s": float(data.time),
                    "model_signature": _model_signature(model),
                }
                if not visible:
                    result.update(
                        status="failed",
                        failure="sample_not_visible",
                        hold_requested=self.motion._request_hold(),
                    )
                    reports[-1] = dict(phase=name, **result)
                    break
        success = len(reports) == 4 and all(r["status"] == "succeeded" for r in reports)
        return {
            "status": "succeeded" if success else "failed",
            "scene_sha256": self.observer.binding["scene_sha256"],
            **combine_reset_reports(reports),
            "phases": reports,
            "observation": observation,
            "measured_return_error": (self.motion.measured() - start).tolist(),
            "return_error_units": ["rad", "m"],
            "scope": "Measured access and visual inspection of an authored rigid coupon",
            "limitations": [
                "No physical loading, gripping, vacuum seal, pumping or pressure response",
                "No deposition, source heating, RHEED or verified operating interlocks",
                "Finite planning samples and every actual physics-step geometry checks",
                "Position hold on failure is not a certified emergency stop",
            ],
        }


def inspect_cornell(session, *, step_observer=None):
    return CornellLoading(session).run(step_observer=step_observer)
