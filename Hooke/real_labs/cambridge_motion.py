"""Measured access to an authored Turner-inspired retained inert specimen.

Speed bounds apply to this local inspection task, not raw servo commands or a
real centrifuge. Geometry and calibrated-in-model positions do not validate
rotor balance, geotechnical scaling, safety interlocks or operating procedures.
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

SCENE = "cambridge_schofield_turner"
EQUIPMENT = "turner_rotor"
CONTROLS = ("maintenance_index", "specimen_carriage")
SPEED_LIMITS = np.array([0.025, 0.035])
RADIUS_BOUND_M = 4.8


@lru_cache(maxsize=1)
def _authored_binding():
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    session = InstrumentSession(model, manifest)
    resources = _portable_resources(root)
    portable = ET.tostring(root, encoding="unicode")
    exported = mujoco.MjModel.from_xml_string(
        portable, {r["file"]: Path(r["source"]).read_bytes() for r in resources}
    )
    return dict(
        signatures={_model_signature(model), _model_signature(exported)},
        scene_sha256=hashlib.sha256(portable.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        camera=manifest["camera"],
        specimens=manifest["specimens"],
    )


def _require_binding(session):
    if session.data.model is not session.model or session.manifest.get("id") != SCENE:
        raise ValueError("Cambridge access requires its authored model and state")
    expected = _authored_binding()
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("Cambridge geometry, contacts, physics or drives changed")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or session.manifest.get("camera") != expected["camera"]
        or session.manifest.get("specimens") != expected["specimens"]
    ):
        raise ValueError("Cambridge control, sample or camera binding changed")
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("Cambridge scene digest differs from the authored model")
    return expected


class AccessObserver:
    """Every completed step checks direct pairs, including parent-filtered ones."""

    cap_m = 0.002
    clearance_m = 0.00025

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.model, self.data = session, session.model, session.data
        m = self.model
        self.dadr = np.array(
            [m.jnt_dofadr[session.controls[EQUIPMENT, c].joint] for c in CONTROLS]
        )
        moving = {
            int(m.jnt_bodyid[session.controls[EQUIPMENT, c].joint]) for c in CONTROLS
        }
        ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.pairs = np.array(
            [
                (a, b)
                for index, a in enumerate(ids)
                for b in ids[index + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in moving
                    or m.body_weldid[m.geom_bodyid[b]] in moving
                )
            ],
            dtype=int,
        )
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.reset()

    def reset(self):
        self.entries = self.checked = self.queries = 0
        self.minimum = self.cap_m
        self.peak = np.zeros(2)
        self.peak_coupled = 0.0

    def check(self, model, data):
        self.entries += 1
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or model is not self.model
            or data is not self.data
            or data.model is not model
        ):
            raise RuntimeError("turner_state_binding_changed")
        if (
            (int(model.opt.disableflags), int(model.opt.enableflags)) != self.flags
            or not np.array_equal(model.geom_contype, self.masks[0])
            or not np.array_equal(model.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("turner_contact_configuration_changed")
        speed = np.abs(data.qvel[self.dadr])
        self.peak = np.maximum(self.peak, speed)
        self.peak_coupled = max(
            self.peak_coupled, float(RADIUS_BOUND_M * speed[0] + speed[1])
        )
        if np.any(speed > SPEED_LIMITS):
            raise RuntimeError("turner_authored_inspection_speed_exceeded")
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
            self.queries += 1
            current = min(current, gap)
            if gap < self.clearance_m:
                raise RuntimeError(
                    "turner_clearance_lost:"
                    + model.geom(int(first)).name
                    + ":"
                    + model.geom(int(second)).name
                )
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def result(self, executed_steps):
        complete = executed_steps > 0 and executed_steps == self.checked
        return dict(
            executed_steps=executed_steps,
            observer_entries=self.entries,
            checked_steps=self.checked,
            unchecked_steps=executed_steps - self.checked,
            coverage_complete=complete,
            pair_count=len(self.pairs),
            direct_near_queries=self.queries,
            clearance_reporting_cap_m=self.cap_m,
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
            maximum_observed_speed=self.peak.tolist(),
            speed_units=["rad/s", "m/s"],
            authored_speed_limits=SPEED_LIMITS.tolist(),
            maximum_observed_coupled_point_speed_bound_m_s=self.peak_coupled,
            radius_bound_m=RADIUS_BOUND_M,
            authored_coupled_point_speed_bound_m_s=float(
                RADIUS_BOUND_M * SPEED_LIMITS[0] + SPEED_LIMITS[1]
            ),
        )


class CambridgeAccess:
    """Index/expose/observe/return, with contact-aware physics and no reset."""

    def __init__(self, session):
        self.session = session
        self.observer = AccessObserver(session)
        self.motion = JointMotion(session, EQUIPMENT, CONTROLS, clearance_m=0.0003)
        self.motion.resolution[:] = (0.005, 0.002)

    def _preflight(self):
        _require_binding(self.session)
        if (
            self.session.model is not self.observer.model
            or self.session.data is not self.observer.data
        ):
            raise ValueError("Cambridge access session state was replaced")
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
            raise ValueError("Cambridge access requires finite measured state")

    def move(self, target, *, step_observer=None):
        self._preflight()
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        # Direct preflight catches physical pairs omitted by native parent filtering.
        self.observer.reset()
        self.observer.check(self.session.model, self.session.data)
        plan = self.motion.plan_joint_target(target)
        # JointMotion uses >=0.2 s quintic segments. Splitting only angular
        # travel to <=2 mrad bounds requested angular speed at 0.01875 rad/s.
        # Measured speed is checked independently, including settling/held axes.
        path = [plan.waypoints[0]]
        for first, second in zip(plan.waypoints, plan.waypoints[1:]):
            first, second = np.asarray(first), np.asarray(second)
            count = max(1, int(np.ceil(abs(second[0] - first[0]) / 0.002)))
            path.extend(
                tuple(first + t * (second - first))
                for t in np.linspace(0, 1, count + 1)[1:]
            )
        # Repeated final targets are real >=0.2 s hold segments. The shared
        # executor's generic angular settling tolerance is too coarse at this
        # beam radius; an additional 1.6 s observed dwell reduces residual lag.
        # No qpos/qvel assignment or reset is used to achieve registration.
        path.extend([path[-1]] * 8)
        plan = replace(
            plan,
            waypoints=tuple(path),
            description="Authored bounded-speed Turner inspection",
        )
        self.observer.reset()

        def observe(model, data):
            self.observer.check(model, data)
            if step_observer is not None:
                step_observer(model, data)
                # Callback is read-only; never label altered-model output canonical.
                if _model_signature(model) not in self.observer.binding["signatures"]:
                    raise RuntimeError("turner_observer_changed_model")

        result = self.motion.execute(plan, step_observer=observe)
        result["direct_clearance"] = self.observer.result(result["physics_steps"])
        result["registration_tolerance"] = [0.0002, 0.0002]
        result["registration_units"] = ["rad", "m"]
        result["observed_endpoint_dwell_s"] = 1.6
        if result["status"] == "succeeded" and (
            np.any(np.abs(self.motion.measured() - np.asarray(target)) > 0.0002)
            or np.any(np.abs(self.session.data.qvel[self.motion.dadr]) > 0.0005)
        ):
            result.update(
                status="failed",
                failure="turner_registration_not_settled",
                hold_requested=self.motion._request_hold(),
            )
        return result

    def _observe_surface(self):
        from real_labs.usability import camera_target_screening

        m, d = self.session.model, self.session.data
        screen = next(
            row
            for row in camera_target_screening(m, d, self.session.manifest)
            if row["equipment"] == EQUIPMENT
        )
        camera = next(c for c in screen["cameras"] if c["camera"] == "specimen_access")
        visible = (
            "specimen_access" in screen["candidate_views"]
            and camera.get("first_hit", {}).get("name")
            == EQUIPMENT + "__turner_inert_sample"
        )
        site = m.site(EQUIPMENT + "__site_inert_soil_model").id
        top = d.site_xpos[site].copy()
        hits = []
        for x in np.linspace(-0.015, 0.015, 5):
            for y in np.linspace(-0.015, 0.015, 5):
                gid = np.array([-1], dtype=np.int32)
                origin = top + np.array([x, y, 0.00001])
                gap = mujoco.mj_ray(
                    m, d, origin, np.array([0.0, 0.0, 1.0]), None, 1, -1, gid
                )
                if 0 <= gap < 0.10:
                    hits.append(dict(geom=m.geom(int(gid[0])).name, gap_m=float(gap)))
        return dict(
            visible=visible,
            camera="specimen_access",
            camera_screening=screen,
            surface_world_m=top.tolist(),
            clear_vertical_access_rays=25 - len(hits),
            access_ray_length_m=0.10,
            access_ray_hits=hits,
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            model_signature=_model_signature(m),
        )

    def run(self, *, step_observer=None):
        self._preflight()
        start = self.motion.measured()
        if np.any(np.abs(start) > (0.003, 0.0002)):
            raise PlanningError(
                "Cambridge round trip requires measured indexed/retracted home"
            )
        reports = []
        observation = None
        for name, target in (
            ("index_for_access", (-0.03, start[1])),
            ("expose_carriage", (-0.03, 0.12)),
            ("return_carriage", (-0.03, start[1])),
            ("return_index", start),
        ):
            result = self.move(target, step_observer=step_observer)
            reports.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
            if name == "expose_carriage":
                observation = self._observe_surface()
                if not observation["visible"] or observation["access_ray_hits"]:
                    result.update(
                        status="failed",
                        failure="retained_surface_not_accessible",
                        hold_requested=self.motion._request_hold(),
                    )
                    reports[-1] = dict(phase=name, **result)
                    break
        success = len(reports) == 4 and all(r["status"] == "succeeded" for r in reports)
        return dict(
            status="succeeded" if success else "failed",
            scene_sha256=self.observer.binding["scene_sha256"],
            **combine_reset_reports(reports),
            phases=reports,
            observation=observation,
            measured_return_error=(self.motion.measured() - start).tolist(),
            return_error_units=["rad", "m"],
            scope="Rigid retained specimen geometric access at authored inspection speeds",
            limitations=[
                "No sample pickup or soil response; the specimen is fixed to the carrier",
                "Surrogate mass/inertia/gravity compensation, not a balanced physical centrifuge",
                "Speed limits apply to this task only; raw servo commands are unrestricted in speed",
                "10 cm vertical ray access is not a robot/tool swept-volume proof",
                "Finite planning samples and every completed physics-step direct checks",
                "Hold on failure is a servo request, not a safety-certified stop",
                "No real Cambridge operating procedure, interlock or scientific validation",
            ],
        )


def inspect_cambridge(session, *, step_observer=None):
    return CambridgeAccess(session).run(step_observer=step_observer)
