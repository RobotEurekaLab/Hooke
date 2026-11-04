"""Measured dry insertion into the authored open UCLA diagnostic segment.

No vacuum state, seal, field, plasma or diagnostic response is evaluated. The
retained inert head returns to the exterior neck rather than being extracted.
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

SCENE = "ucla_bapsf_lapd"
EQUIPMENT = "lapd_segment"
CONTROLS = ("probe_vertical_alignment", "probe_approach")


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
        xml, {r["file"]: Path(r["source"]).read_bytes() for r in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures={_model_signature(raw), _model_signature(portable)},
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        metadata={
            k: manifest.get(k)
            for k in ("camera", "camera_names", "mounted_cameras", "specimens")
        },
    )


def _require_binding(session):
    if session.manifest.get("id") != SCENE or session.data.model is not session.model:
        raise ValueError("UCLA probe access requires its exact model and owned state")
    expected = _binding()
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("Authored UCLA geometry, contacts, physics or drives changed")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or any(session.manifest.get(k) != v for k, v in expected["metadata"].items())
    ):
        raise ValueError("UCLA control, surface or camera metadata changed")
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("UCLA scene hash differs from its authored model")
    return expected


class ProbeObserver:
    """Complete enabled distinct-weld moving-pair set with direct narrow queries."""

    cap_m = 0.010
    minimum_m = 0.00025

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.model, self.data = session, session.model, session.data
        m = self.model
        moving = {
            int(m.body_weldid[m.jnt_bodyid[session.controls[EQUIPMENT, c].joint]])
            for c in CONTROLS
        }
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in moving
                    or m.body_weldid[m.geom_bodyid[b]] in moving
                )
            ],
            dtype=int,
        )
        self.masks = m.geom_contype.copy(), m.geom_conaffinity.copy()
        self.flags = int(m.opt.disableflags), int(m.opt.enableflags)
        self.reset()

    def reset(self):
        self.entries = self.checked = self.queries = 0
        self.minimum = self.cap_m

    def check(self, m, d):
        self.entries += 1
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or m is not self.model
            or d is not self.data
            or d.model is not m
        ):
            raise RuntimeError("probe_model_state_binding_changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("probe_contact_configuration_changed")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap_m
        )
        current = self.cap_m
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap_m, None)),
                self.cap_m,
            )
            self.queries += 1
            current = min(current, gap)
            if gap < self.minimum_m:
                raise RuntimeError(
                    f"probe_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
                )
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def result(self, executed):
        complete = executed > 0 and executed == self.checked
        return dict(
            executed_steps=executed,
            observer_entries=self.entries,
            checked_steps=self.checked,
            unchecked_steps=executed - self.checked,
            coverage_complete=complete,
            pair_count=len(self.pairs),
            direct_near_queries=self.queries,
            clearance_cap_m=self.cap_m,
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
        )


class ProbeAccess:
    """Single-owner model/state access with failure hold and no reset/teleport."""

    def __init__(self, session):
        self.session = session
        self.observer = ProbeObserver(session)
        self.motion = JointMotion(session, EQUIPMENT, CONTROLS, clearance_m=0.0003)
        self.motion.resolution[:] = (0.002, 0.002)

    def _preflight(self):
        _require_binding(self.session)
        if (
            self.session.model is not self.observer.model
            or self.session.data is not self.observer.data
        ):
            raise ValueError("UCLA probe session state was replaced")
        d = self.session.data
        if any(
            not np.isfinite(v).all()
            for v in (
                d.qpos,
                d.qvel,
                d.ctrl,
                d.qfrc_applied,
                d.xfrc_applied,
                np.array([d.time]),
            )
        ):
            raise ValueError("UCLA probe access requires finite state")

    def move(self, target, *, step_observer=None):
        self._preflight()
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        self.observer.reset()
        self.observer.check(self.session.model, self.session.data)
        plan = self.motion.plan_joint_target(target)
        plan = replace(plan, waypoints=plan.waypoints + (plan.waypoints[-1],) * 4)
        self.observer.reset()

        def observe(m, d):
            self.observer.check(m, d)
            if step_observer is not None:
                step_observer(m, d)
                if _model_signature(m) not in self.observer.binding["signatures"]:
                    raise RuntimeError("probe_readonly_observer_changed_model")

        result = self.motion.execute(plan, step_observer=observe)
        result["direct_clearance"] = self.observer.result(result["physics_steps"])
        return result

    def observe(self):
        from real_labs.usability import camera_target_screening
        from real_labs.ucla_probe_geometry import VESSEL_CENTER_Z, VESSEL_INNER_RADIUS

        self._preflight()
        m, d = self.session.model, self.session.data
        equipment = m.body(EQUIPMENT).id
        origin, rotation = d.xpos[equipment], d.xmat[equipment].reshape(3, 3)

        def local(site):
            return rotation.T @ (
                d.site_xpos[m.site(EQUIPMENT + "__site_" + site).id] - origin
            )

        tip, center, axis = (
            local("probe_tip"),
            local("probe_center"),
            local("access_axis"),
        )
        head = m.geom(EQUIPMENT + "__lapd_inert_probe_head").id
        corners = (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * m.geom_size[head]
        )
        corners = corners @ d.geom_xmat[head].reshape(3, 3).T + d.geom_xpos[head]
        corners = (corners - origin) @ rotation
        maximum_radial = float(
            np.max(np.hypot(corners[:, 0], corners[:, 2] - VESSEL_CENTER_Z))
        )
        inside_boundary = np.sqrt(
            VESSEL_INNER_RADIUS**2 - (tip[2] - VESSEL_CENTER_Z) ** 2
        )
        screen = next(
            r
            for r in camera_target_screening(m, d, self.session.manifest)
            if r["equipment"] == EQUIPMENT
        )
        view = next(c for c in screen["cameras"] if c["camera"] == "port_inspection")
        visible = (
            view.get("opaque_line_of_sight", False)
            and (view.get("first_hit") or {}).get("name") == m.geom(head).name
        )
        return dict(
            tip_local_m=tip.tolist(),
            head_center_local_m=center.tolist(),
            port_axis_local_m=axis.tolist(),
            axis_alignment_error_m=(center[1:] - axis[1:]).tolist(),
            tip_depth_inside_inner_cylindrical_envelope_m=float(
                inside_boundary - tip[0]
            ),
            head_maximum_radius_m=maximum_radial,
            head_radial_margin_inside_chamber_m=VESSEL_INNER_RADIUS - maximum_radial,
            head_inside_empty_chamber=bool(
                maximum_radial < VESSEL_INNER_RADIUS
                and np.max(np.abs(corners[:, 1])) < 3.9
            ),
            visible=bool(visible),
            camera="port_inspection",
            camera_screening=screen,
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            model_signature=_model_signature(m),
            scope="Actual geometric head/corner positions relative to an authored empty chamber, not a plasma measurement",
        )

    def run(self, *, step_observer=None):
        self._preflight()
        start = self.motion.measured()
        if np.any(np.abs(start) > 0.0002):
            raise PlanningError("UCLA round trip requires measured diagnostic home")
        initial = self.observe()
        phases, checkpoints = [], {}
        for name, target in (
            ("offset_check", (0.025, 0)),
            ("align_port", (0, 0)),
            ("insert_probe", (0, -0.19)),
            ("retract_probe", start),
        ):
            result = self.move(target, step_observer=step_observer)
            if result["status"] == "succeeded":
                observed = self.observe()
                checkpoints[name] = observed
                aligned = np.linalg.norm(observed["axis_alignment_error_m"]) <= 0.0002
                if name == "align_port" and not aligned:
                    result.update(
                        status="failed",
                        failure="probe_axis_alignment_not_reached",
                        hold_requested=self.motion._request_hold(),
                    )
                if name == "insert_probe" and (
                    not aligned
                    or not observed["head_inside_empty_chamber"]
                    or not observed["visible"]
                    or observed["tip_depth_inside_inner_cylindrical_envelope_m"] < 0.025
                ):
                    result.update(
                        status="failed",
                        failure="physical_probe_insertion_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
        return dict(
            status=(
                "succeeded"
                if len(phases) == 4 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.observer.binding["scene_sha256"],
            **combine_reset_reports(phases),
            phases=phases,
            initial_observation=initial,
            checkpoints=checkpoints,
            measured_return_error_m=(self.motion.measured() - start).tolist(),
            limitations=[
                "Retained inert head returns to the external port neck; no complete extraction or loading",
                "Dimensions, guides, port and control fixture are authored estimates",
                "Ideal joints and uncalibrated surrogate masses/gains do not validate bearing forces or real hardware",
                "Finite planning samples and checks after completed physics steps are not continuous-time guarantees",
                "One exclusive state owner; arbitrary concurrent mutation is unsupported",
                "Position hold is not a certified emergency stop",
                "No vacuum, seal, coolant flow, plasma, field, radiation or diagnostic response model",
            ],
        )


def inspect_ucla_probe(session, *, step_observer=None):
    return ProbeAccess(session).run(step_observer=step_observer)
