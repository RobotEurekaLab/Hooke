"""Measured retained-wafer positioning under a visible inspection objective.

This is rigid geometric inspection, not lithographic exposure or optical focus.
The modeled three clamps rigidly bind the wafer to its XY stage. No insertion,
removal, live pose assignment or automatic reset is performed.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError, _model_signature


SCENE = "harvard_cns_cambridge"
CONTROLS = ("wafer_y", "wafer_x", "alignment_head")


@lru_cache(maxsize=1)
def _binding():
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml, {i["file"]: Path(i["source"]).read_bytes() for i in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        equipment=session.equipment,
        controls=session.controls,
        camera=manifest["camera"],
    )


def _require_binding(session):
    if session.data.model is not session.model:
        raise ValueError(
            "Harvard inspection requires data belonging to its exact model"
        )
    reference = _binding()
    if session.manifest.get("id") != SCENE:
        raise ValueError("Harvard inspection requires its authored scene")
    if (
        session.equipment != reference["equipment"]
        or session.controls != reference["controls"]
    ):
        raise ValueError("Harvard control or wafer interface binding changed")
    if session.manifest.get("camera") != reference["camera"]:
        raise ValueError("Harvard review camera binding changed")
    if _model_signature(session.model) not in reference["signatures"]:
        raise ValueError(
            "Harvard geometry, collision, drive or physics binding changed"
        )
    if (
        session.manifest.get("scene_sha256", reference["scene_sha256"])
        != reference["scene_sha256"]
    ):
        raise ValueError("Harvard manifest digest does not match physical source")
    return reference


def full_state(session):
    d = session.data
    return dict(
        time_s=float(d.time),
        qpos=d.qpos.tolist(),
        qvel=d.qvel.tolist(),
        ctrl=d.ctrl.tolist(),
        act=d.act.tolist(),
        qacc_warmstart=d.qacc_warmstart.tolist(),
        qfrc_applied=d.qfrc_applied.tolist(),
        xfrc_applied=d.xfrc_applied.tolist(),
        mocap_pos=d.mocap_pos.tolist(),
        mocap_quat=d.mocap_quat.tolist(),
        eq_active=d.eq_active.tolist(),
        model_signature=_model_signature(session.model),
    )


class WaferInspection:
    """Finite-sampled planning plus direct geometry checks after each real step."""

    cap_m = 0.002
    minimum_allowed_m = 0.00025

    def __init__(self, session, equipment="lithography_aligner"):
        self.binding = _require_binding(session)
        if equipment not in ("lithography_aligner", "inspection_aligner"):
            raise ValueError("Select one of the two authored Harvard fixtures")
        self.session, self.model, self.data = session, session.model, session.data
        self.equipment = equipment
        self.motion = JointMotion(session, equipment, CONTROLS, clearance_m=0.0003)
        self.motion.resolution[:] = 0.002
        m = self.model
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        moving = self.motion.moving_geoms
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (a in moving or b in moving)
            ],
            dtype=int,
        )
        item = session.equipment[equipment]
        self.wafer = m.geom(equipment + "__cns_wafer").id
        self.surface = m.site(item["sites"]["wafer_surface"]).id
        self.axis = m.site(item["sites"]["inspection_axis"]).id
        self.masks = m.geom_contype.copy(), m.geom_conaffinity.copy()
        self.flags = int(m.opt.disableflags), int(m.opt.enableflags)
        self.reset_observations()

    def _identity(self):
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or self.data.model is not self.model
        ):
            raise ValueError("Harvard model or state object was replaced")

    def reset_observations(self):
        """Clear report counters only; physical state is never reset here."""
        self.entries = self.checked = self.queries = 0
        self.minimum = self.cap_m
        self.closest = None

    def _observe(self, m, d):
        self._identity()
        self.entries += 1
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("harvard_collision_configuration_changed")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        near = np.linalg.norm(separation, axis=1) < self.cap_m
        current, closest = self.cap_m, None
        for first, second in self.pairs[near]:
            gap = min(
                float(
                    mujoco.mj_geomDistance(
                        m, d, int(first), int(second), self.cap_m, None
                    )
                ),
                self.cap_m,
            )
            self.queries += 1
            if gap < current:
                current, closest = gap, (int(first), int(second))
            if gap < self.minimum_allowed_m:
                raise RuntimeError(
                    f"harvard_clearance_lost:{m.geom(int(first)).name}:{m.geom(int(second)).name}"
                )
        if current < self.minimum:
            self.minimum = current
            self.closest = [m.geom(i).name for i in closest]
        self.checked += 1

    def move(self, target, *, step_observer=None):
        """Issue a bounded target from measured state; callbacks must be read-only."""
        self._identity()
        _require_binding(self.session)
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        self.reset_observations()
        before = full_state(self.session)
        try:
            plan = self.motion.plan_joint_target(target)
        except PlanningError as error:
            result = dict(
                status="blocked",
                failure=str(error),
                physics_steps=0,
                hold_requested=self.motion._request_hold(),
                reset_performed=False,
            )
        else:

            def observe(m, d):
                self._observe(m, d)
                if step_observer is not None:
                    step_observer(m, d)

            result = self.motion.execute(plan, step_observer=observe)
        complete = (
            result["physics_steps"] > 0 and result["physics_steps"] == self.checked
        )
        result.update(
            before=before,
            after=full_state(self.session),
            direct_checks=dict(
                executed_steps=result["physics_steps"],
                observer_entries=self.entries,
                fully_checked_steps=self.checked,
                coverage_complete=complete,
                checked_steps_clearance_lower_bound_m=(
                    self.minimum if self.checked else None
                ),
                whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
                reporting_cap_m=self.cap_m,
                direct_distance_queries=self.queries,
                closest_checked_pair=self.closest,
            ),
        )
        return result

    def _alignment_target(self):
        """Derive XY coordinates from the actual objective axis and wafer site."""
        m, d = self.model, self.data
        joints = [self.session.controls[self.equipment, c].joint for c in CONTROLS[:2]]
        matrix = np.stack([d.xaxis[j][:2] for j in joints], axis=1)
        displacement = d.site_xpos[self.axis, :2] - d.site_xpos[self.surface, :2]
        target = self.motion.measured().copy()
        target[:2] += np.linalg.solve(matrix, displacement)
        target[2] = 0.015
        if np.any(target < self.motion.lower) or np.any(target > self.motion.upper):
            raise PlanningError(
                "The actual inspection objective cannot be reached within the original XY travel"
            )
        return target

    def _inspection(self):
        from real_labs.usability import camera_target_screening

        m, d = self.model, self.data
        objective = d.site_xpos[self.axis].copy()
        surface = d.site_xpos[self.surface].copy()
        lateral = float(np.linalg.norm(objective[:2] - surface[:2]))
        origin = objective + np.array([0.0, 0.0, -0.0001])
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m, d, origin, np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
        )
        first = None if hit[0] < 0 else m.geom(int(hit[0])).name
        allowed = {self.equipment + "__cns_wafer"} | {
            self.equipment + f"__cns_wafer_cross_{i}" for i in (0, 1)
        }
        clear = first in allowed and distance > 0
        screen = next(
            r
            for r in camera_target_screening(m, d, self.session.manifest)
            if r["equipment"] == self.equipment
        )
        visible = bool(screen["candidate_views"])
        return dict(
            projected_axis_error_m=lateral,
            objective_to_surface_height_m=float(objective[2] - surface[2]),
            objective_world_m=objective.tolist(),
            wafer_surface_world_m=surface.tolist(),
            axis_first_hit=first,
            axis_ray_distance_m=float(distance),
            axis_reaches_wafer=clear,
            visible=visible,
            camera_screening=screen,
            state=full_state(self.session),
            scope="Geometric projected-axis alignment only; distance does not certify focus or lithographic resolution",
        )

    def inspect_and_return(self, *, step_observer=None):
        self._identity()
        _require_binding(self.session)
        start = self.motion.measured().copy()
        if np.max(np.abs(start)) > 0.0002:
            raise PlanningError("Begin at the authored centered, lowered home pose")
        initial = full_state(self.session)
        goal = self._alignment_target()
        legs = []
        observation = None
        for phase, target in (
            ("raise_head", (start[0], start[1], 0.08)),
            ("position_retained_wafer", (goal[0], goal[1], 0.08)),
            ("inspect_axis", goal),
            ("raise_for_return", (goal[0], goal[1], 0.08)),
            ("return_xy", (start[0], start[1], 0.08)),
            ("return_head", start),
        ):
            result = self.move(target, step_observer=step_observer)
            result["phase"] = phase
            legs.append(result)
            if result["status"] != "succeeded":
                break
            if phase == "inspect_axis":
                observation = self._inspection()
                if (
                    observation["projected_axis_error_m"] > 0.0002
                    or not observation["axis_reaches_wafer"]
                    or not observation["visible"]
                ):
                    result.update(
                        status="failed",
                        failure="wafer_axis_inspection_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
                    break
        return_error = np.abs(self.motion.measured() - start)
        success = (
            len(legs) == 6
            and all(r["status"] == "succeeded" for r in legs)
            and np.max(return_error) <= 0.0002
        )
        if len(legs) == 6 and not success and legs[-1]["status"] == "succeeded":
            legs[-1].update(
                status="failed",
                failure="measured_return_not_observed",
                hold_requested=self.motion._request_hold(),
            )
        return dict(
            status="succeeded" if success else "failed",
            equipment=self.equipment,
            scene_sha256=self.binding["scene_sha256"],
            model_signature=self.motion.signature,
            initial_state=initial,
            derived_target_y_x_head_m=goal.tolist(),
            legs=legs,
            observation=observation,
            physics_steps=sum(r["physics_steps"] for r in legs),
            fully_checked_steps=sum(
                r["direct_checks"]["fully_checked_steps"] for r in legs
            ),
            return_error_m=return_error.tolist(),
            final_state=full_state(self.session),
            **combine_reset_reports(legs),
            hold_requested=any(r.get("hold_requested", False) for r in legs),
            scope="Measured XY/head positioning of an already rigidly clamped inert wafer and physical return",
            limitations=[
                "No insertion, extraction, unclamping, lithography, exposure, optical focus or alignment-accuracy model",
                "Original inferred bearing/clamp hardware and actuator parameters are not OEM calibrated",
                "Finite sampled planning and per-step measured clearances are not continuous-time proof",
                "Hold is a servo request, not an instantaneous stop; reset must be explicit",
                "Read-only callbacks only; concurrent model/control/state writers are unsupported",
            ],
        )
