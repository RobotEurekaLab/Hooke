"""Measured true-scale GSG pad alignment without RF or contact measurement.

Only the authored retained inert specimen and original two-axis fixture are
supported. The noncontact inspection physically returns; it never teleports.
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


SCENE = "yonsei_mics_rf_probe"
CONTROLS = {"rf_probe": ("chuck_x", "right_probe_lift")}


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
        raise ValueError("Yonsei inspection requires data belonging to its exact model")
    reference = _binding()
    if session.manifest.get("id") != SCENE:
        raise ValueError("Yonsei inspection requires its authored scene")
    if (
        session.equipment != reference["equipment"]
        or session.controls != reference["controls"]
    ):
        raise ValueError("Yonsei control or specimen interface binding changed")
    if session.manifest.get("camera") != reference["camera"]:
        raise ValueError("Yonsei review camera binding changed")
    if _model_signature(session.model) not in reference["signatures"]:
        raise ValueError("Yonsei geometry, collision, drive or physics binding changed")
    if (
        session.manifest.get("scene_sha256", reference["scene_sha256"])
        != reference["scene_sha256"]
    ):
        raise ValueError("Yonsei manifest digest does not match physical source")
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


class ProbeInspection:
    """Finite-sampled planning plus direct geometry checks after each real step."""

    cap_m = 0.002
    minimum_allowed_m = 0.00025

    def __init__(self, session, equipment="rf_probe"):
        self.binding = _require_binding(session)
        if equipment not in CONTROLS:
            raise ValueError("Select the authored Yonsei RF fixture")
        self.session, self.model, self.data = session, session.model, session.data
        self.equipment = equipment
        self.motion = JointMotion(
            session, equipment, CONTROLS[equipment], clearance_m=0.0003
        )
        self.motion.resolution[:] = 0.0001
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
            raise ValueError("Yonsei model or state object was replaced")

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
            raise RuntimeError("yonsei_collision_configuration_changed")
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
                    f"yonsei_clearance_lost:{m.geom(int(first)).name}:{m.geom(int(second)).name}"
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
            plan = replace(plan, waypoints=tuple(path) + (tuple(path[-1]),) * 4)
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
        self._identity()
        _require_binding(self.session)
        m, d = self.model, self.data
        interface = self.session.equipment[self.equipment]["inspection_interface"]
        tips = np.array(
            [d.site_xpos[m.site(n).id].copy() for n in interface["tip_sites"]]
        )
        pads = np.array(
            [d.site_xpos[m.site(n).id].copy() for n in interface["pad_sites"]]
        )
        errors = np.linalg.norm(tips[:, :2] - pads[:, :2], axis=1)
        standoffs = tips[:, 2] - pads[:, 2]
        hits = []
        for index, tip in enumerate(tips):
            g = np.array([-1], dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    m,
                    d,
                    tip - [0, 0, 0.00001],
                    np.array([0.0, 0.0, -1.0]),
                    None,
                    1,
                    -1,
                    g,
                )
            )
            name = m.geom(int(g[0])).name if g[0] >= 0 else None
            hits.append(
                dict(
                    first_hit=name,
                    distance_m=distance,
                    pad_reached=name == f"rf_probe__mics_pad_1_{index}",
                )
            )
        valid = bool(
            np.max(errors) < 0.000010
            and np.min(standoffs) > 0.00075
            and np.max(standoffs) < 0.0013
            and all(r["pad_reached"] for r in hits)
        )
        return dict(
            tip_world_m=tips.tolist(),
            pad_world_m=pads.tolist(),
            projected_tip_pad_errors_m=errors.tolist(),
            standoffs_m=standoffs.tolist(),
            axis_rays=hits,
            interface_observed=valid,
            state=full_state(self.session),
            scope="Dry geometric projection of three true-pitch tips over inert pads; no RF/contact or calibrated alignment precision claim",
        )

    def inspect_and_return(self, *, step_observer=None):
        self._identity()
        _require_binding(self.session)
        initial = full_state(self.session)
        start = self.motion.measured().copy()
        if np.max(abs(start)) > 0.00002:
            raise PlanningError(
                "Begin at the authored centered chuck and lowered inspection home"
            )
        m, d = self.model, self.data
        interface = self.session.equipment[self.equipment]["inspection_interface"]
        tip = d.site_xpos[m.site(interface["tip_sites"][1]).id]
        pad = d.site_xpos[m.site(interface["pad_sites"][1]).id]
        axis = d.xaxis[self.session.controls["rf_probe", "chuck_x"].joint]
        aligned_x = float(start[0] + np.dot(tip - pad, axis))
        goal = np.array([aligned_x, 0.0003])
        route = (
            ("retract_probe", (start[0], 0.006)),
            ("align_retained_coupon", (aligned_x, 0.006)),
            ("inspect_true_pitch_pads", goal),
            ("raise_for_return", (aligned_x, 0.006)),
            ("return_chuck", (start[0], 0.006)),
            ("return_probe", start),
        )
        legs, observation = [], None
        for phase, target in route:
            result = self.move(target, step_observer=step_observer)
            result["phase"] = phase
            legs.append(result)
            if result["status"] != "succeeded":
                break
            if phase == "inspect_true_pitch_pads":
                observation = self._inspection()
                if not observation["interface_observed"]:
                    result.update(
                        status="failed",
                        failure="measured_tip_pad_inspection_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
                    break
        error = np.abs(self.motion.measured() - start)
        succeeded = (
            len(legs) == len(route)
            and all(r["status"] == "succeeded" for r in legs)
            and np.max(error) < 0.000010
        )
        if (
            len(legs) == len(route)
            and not succeeded
            and legs[-1]["status"] == "succeeded"
        ):
            legs[-1].update(
                status="failed",
                failure="measured_return_not_observed",
                hold_requested=self.motion._request_hold(),
            )
        return dict(
            status="succeeded" if succeeded else "failed",
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_signature=self.motion.signature,
            initial_state=initial,
            derived_alignment_target_m=goal.tolist(),
            legs=legs,
            observation=observation,
            physics_steps=sum(x["physics_steps"] for x in legs),
            fully_checked_steps=sum(
                x["direct_checks"]["fully_checked_steps"] for x in legs
            ),
            return_error_m=error.tolist(),
            final_state=full_state(self.session),
            **combine_reset_reports(legs),
            hold_requested=any(x.get("hold_requested", False) for x in legs),
            limitations=[
                "All dimensions except source-informed 100 micrometre probe pitch remain authored estimates; no OEM CAD or calibrated mechanics",
                "Rigid retained inert coupon/clip mounting; no sample loading, adhesive, clamp actuation or transfer",
                "Needles remain physically above the pads; no RF signal, impedance, electrical contact, force, scrubbing or semiconductor measurement model",
                "Actual-step direct checks and finite sampled planning do not establish continuous-time collision safety",
                "One exclusive session owner; callbacks must be read-only, arbitrary concurrent writers are unsupported",
                "A failed hold is a servo request, not an instantaneous stop; explicit session reset is separate",
            ],
        )
