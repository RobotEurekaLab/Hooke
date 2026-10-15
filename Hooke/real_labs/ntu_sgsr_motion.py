"""Measured NTU dry probe alignment and empty-grip inspection.

Only retained, inert specimens and the authored rigid guides are supported.
No AFM feedback, material loading, insertion or automatic reset is performed.
"""

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError, _model_signature


SCENE = "ntu_sgsr_characterization"
CONTROLS = {
    "dimension_icon": ("coarse_x", "coarse_y", "coarse_head"),
    "mechanical_tester": ("crosshead", "upper_grip"),
}


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
        raise ValueError("NTU inspection requires data belonging to its exact model")
    reference = _binding()
    if session.manifest.get("id") != SCENE:
        raise ValueError("NTU inspection requires its authored scene")
    if (
        session.equipment != reference["equipment"]
        or session.controls != reference["controls"]
    ):
        raise ValueError("NTU control or specimen interface binding changed")
    if session.manifest.get("camera") != reference["camera"]:
        raise ValueError("NTU review camera binding changed")
    if _model_signature(session.model) not in reference["signatures"]:
        raise ValueError("NTU geometry, collision, drive or physics binding changed")
    if (
        session.manifest.get("scene_sha256", reference["scene_sha256"])
        != reference["scene_sha256"]
    ):
        raise ValueError("NTU manifest digest does not match physical source")
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


class PrecisionInspection:
    """Finite-sampled planning plus direct geometry checks after each real step."""

    cap_m = 0.002
    minimum_allowed_m = 0.00025

    def __init__(self, session, equipment="dimension_icon"):
        self.binding = _require_binding(session)
        if equipment not in CONTROLS:
            raise ValueError("Select the authored AFM or mechanical test frame")
        self.session, self.model, self.data = session, session.model, session.data
        self.equipment = equipment
        self.motion = JointMotion(
            session, equipment, CONTROLS[equipment], clearance_m=0.0003
        )
        self.motion.resolution[:] = 0.001
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
        self.masks = m.geom_contype.copy(), m.geom_conaffinity.copy()
        self.flags = int(m.opt.disableflags), int(m.opt.enableflags)
        self.reset_observations()

    def _identity(self):
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or self.data.model is not self.model
        ):
            raise ValueError("NTU model or state object was replaced")

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
            raise RuntimeError("ntu_collision_configuration_changed")
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
                    f"ntu_clearance_lost:{m.geom(int(first)).name}:{m.geom(int(second)).name}"
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
            # Existing low-bandwidth servos need short, settled increments.
            # Every increment still runs through the shared measured executor;
            # neither tracking tolerances nor force limits are relaxed.
            increment = 0.8 * 0.025 * (self.motion.upper - self.motion.lower)
            path = [plan.waypoints[0]]
            for first, second in zip(plan.waypoints, plan.waypoints[1:]):
                first, second = np.asarray(first), np.asarray(second)
                count = max(1, int(np.ceil(np.max(np.abs(second - first) / increment))))
                path.extend(
                    tuple(first + (second - first) * i / count)
                    for i in range(1, count + 1)
                )
            plan = replace(plan, waypoints=tuple(path))
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

    def _inspection(self):
        m, d = self.model, self.data
        item = self.session.equipment[self.equipment]
        if self.equipment == "dimension_icon":
            tip = d.site_xpos[m.site(item["sites"]["probe_tip"]).id].copy()
            sample = d.site_xpos[m.site(item["sites"]["sample_coupon"]).id].copy()
            lateral = float(np.linalg.norm(tip[:2] - sample[:2]))
            origin = tip - np.array([0, 0, 0.00001])
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                m, d, origin, np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
            )
            first = m.geom(int(hit[0])).name if hit[0] >= 0 else None
            reached = first == self.equipment + "__ntu_sample" and distance > 0
            gap = float(tip[2] - sample[2])
            return dict(
                projected_probe_error_m=lateral,
                tip_to_sample_m=gap,
                tip_world_m=tip.tolist(),
                sample_surface_world_m=sample.tolist(),
                first_ray_hit=first,
                ray_distance_m=float(distance),
                interface_observed=bool(
                    reached and lateral < 0.0002 and 0.0015 < gap < 0.0025
                ),
                scope="Dry projected probe-axis alignment at millimetre standoff, not AFM contact",
                state=full_state(self.session),
            )
        coupon = m.geom(self.equipment + "__ntu_coupon").id
        grip = m.geom(self.equipment + "__ntu_upper_grip").id
        gap = float(mujoco.mj_geomDistance(m, d, coupon, grip, 1.0, None))
        opening = float(-self.motion.measured()[1])
        return dict(
            upper_grip_to_coupon_distance_m=gap,
            commanded_jaw_opening_m=opening,
            interface_observed=bool(gap > 0.08 and opening > 0.009),
            scope="Open empty upper grip with visible lower-retained coupon; no tensile capture",
            state=full_state(self.session),
        )

    def inspect_and_return(self, *, step_observer=None):
        self._identity()
        _require_binding(self.session)
        start = self.motion.measured().copy()
        if np.max(np.abs(start)) > 0.0002:
            raise PlanningError("Begin at the authored centered home pose")
        initial = full_state(self.session)
        if self.equipment == "dimension_icon":
            item = self.session.equipment[self.equipment]
            tip = self.data.site_xpos[self.model.site(item["sites"]["probe_tip"]).id]
            sample = self.data.site_xpos[
                self.model.site(item["sites"]["sample_coupon"]).id
            ]
            axes = [
                self.data.xaxis[self.session.controls[self.equipment, c].joint][:2]
                for c in CONTROLS[self.equipment][:2]
            ]
            xy = start[:2] + np.linalg.solve(
                np.stack(axes, axis=1), tip[:2] - sample[:2]
            )
            goal = np.array([*xy, -0.0245])
            route = (
                ("raise_head", (start[0], start[1], 0.020)),
                ("survey_retained_disk", (xy[0] + 0.006, xy[1], 0.020)),
                ("align_probe_axis", (*xy, 0.020)),
                ("inspect_interface", goal),
                ("raise_for_return", (*xy, 0.020)),
                ("return_xy", (start[0], start[1], 0.020)),
                ("return_head", start),
            )
        else:
            goal = np.array([-0.18, -0.010])
            route = (
                ("open_empty_grip", (0, -0.010)),
                ("inspect_interface", goal),
                ("return_crosshead", (start[0], -0.010)),
                ("return_grip", start),
            )
        legs, observation = [], None
        for phase, target in route:
            result = self.move(target, step_observer=step_observer)
            result["phase"] = phase
            legs.append(result)
            if result["status"] != "succeeded":
                break
            if phase == "inspect_interface":
                observation = self._inspection()
                if not observation["interface_observed"]:
                    result.update(
                        status="failed",
                        failure="measured_inspection_interface_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
                    break
        error = np.abs(self.motion.measured() - start)
        success = (
            len(legs) == len(route)
            and all(r["status"] == "succeeded" for r in legs)
            and np.max(error) < 0.0002
        )
        if (
            len(legs) == len(route)
            and not success
            and legs[-1]["status"] == "succeeded"
        ):
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
            derived_inspection_target_m=goal.tolist(),
            legs=legs,
            observation=observation,
            physics_steps=sum(r["physics_steps"] for r in legs),
            fully_checked_steps=sum(
                r["direct_checks"]["fully_checked_steps"] for r in legs
            ),
            return_error_m=error.tolist(),
            final_state=full_state(self.session),
            **combine_reset_reports(legs),
            hold_requested=any(r.get("hold_requested", False) for r in legs),
            limitations=[
                "Dry inspection only; no AFM signal, material stress, loading or removal model",
                "Rigid retained samples and ideal authored guides; no sample attachment dynamics",
                "Finite sampled planning and every-step geometry checks are not a continuous-time guarantee",
                "A failed hold is a servo request, not an instantaneous stop; explicit reset is separate",
                "Read-only observers only; concurrent model/control/state writers are unsupported",
            ],
        )
