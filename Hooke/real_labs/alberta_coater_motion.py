"""Measured dry nozzle-axis registration and supported hinged access.

The sample remains fixed. No spray, aerosol, coating, handling or hardware-safety
claim follows from geometric alignment, an open door or point-ray access tests.
"""

from copy import copy
from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError, _model_signature

SCENE = "alberta_nanofab_exactacoat"
EQUIPMENT = "coater_alignment"
CONTROLS = ("dry_head_x", "dry_head_y", "access_door")


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
        raise ValueError("Alberta access requires its authored model and owned state")
    expected = _binding()
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError(
            "Alberta authored geometry, contacts, physics or drive changed"
        )
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or any(session.manifest.get(k) != v for k, v in expected["metadata"].items())
    ):
        raise ValueError("Alberta control, surface or camera metadata changed")
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("Alberta scene hash differs from its authored model")
    return expected


class CoaterObserver:
    """Direct cross-body clearance checks retain parent-filtered geometry."""

    cap_m = 0.002
    minimum_m = 0.00025

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.model, self.data = session, session.model, session.data
        m = self.model
        moving = {
            int(m.jnt_bodyid[session.controls[EQUIPMENT, c].joint]) for c in CONTROLS
        }
        ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(ids)
                for b in ids[i + 1 :]
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

    def check(self, m, d):
        self.entries += 1
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or m is not self.model
            or d is not self.data
            or d.model is not m
        ):
            raise RuntimeError("coater_model_state_binding_changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("coater_contact_configuration_changed")
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
                    "coater_clearance_lost:"
                    + m.geom(int(a)).name
                    + ":"
                    + m.geom(int(b)).name
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


class CoaterAccess:
    def __init__(self, session):
        self.session = session
        self.observer = CoaterObserver(session)
        self.motion = JointMotion(session, EQUIPMENT, CONTROLS, clearance_m=0.0003)
        self.motion.resolution[:] = (0.002, 0.002, 0.01)

    def _preflight(self):
        _require_binding(self.session)
        if (
            self.session.model is not self.observer.model
            or self.session.data is not self.observer.data
        ):
            raise ValueError("Alberta access session state was replaced")
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
            raise ValueError("Alberta access requires finite state")

    def move(self, target, *, step_observer=None):
        self._preflight()
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        self.observer.reset()
        self.observer.check(self.session.model, self.session.data)
        plan = self.motion.plan_joint_target(target)
        # Actual observed endpoint holding segments improve final registration.
        plan = replace(plan, waypoints=plan.waypoints + (plan.waypoints[-1],) * 4)
        self.observer.reset()

        def observe(m, d):
            self.observer.check(m, d)
            if step_observer is not None:
                step_observer(m, d)
                if _model_signature(m) not in self.observer.binding["signatures"]:
                    raise RuntimeError("coater_readonly_observer_changed_model")

        result = self.motion.execute(plan, step_observer=observe)
        result["direct_clearance"] = self.observer.result(result["physics_steps"])
        return result

    def _state(self):
        d = self.session.data
        return dict(
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            model_signature=_model_signature(self.session.model),
        )

    def registration(self):
        m, d = self.session.model, self.session.data
        surface = d.site_xpos[m.site(EQUIPMENT + "__site_coater_wafer").id]
        tip = d.site_xpos[m.site(EQUIPMENT + "__site_dry_nozzle_tip").id]
        return dict(
            surface_world_m=surface.tolist(),
            nozzle_tip_world_m=tip.tolist(),
            xy_error_m=(tip[:2] - surface[:2]).tolist(),
            marker_standoff_m=float(tip[2] - surface[2]),
            authored_marker_standoff_m=0.048,
            authored_wafer_standoff_m=0.05,
            standoff_basis="Uncalibrated dry geometric clearance, not a spray working-height specification",
            **self._state(),
        )

    def access_observation(self):
        from real_labs.usability import camera_target_screening

        m, d = self.session.model, self.session.data
        # A private model selects only configured contact geometry for rays.
        # The live collision masks/groups remain untouched.
        ray_model = copy(m)
        ray_model.geom_group[:] = np.where(
            (m.geom_contype != 0) | (m.geom_conaffinity != 0), 0, 1
        )
        ray_data = mujoco.MjData(ray_model)
        ray_data.qpos[:] = d.qpos
        mujoco.mj_forward(ray_model, ray_data)
        surface = d.site_xpos[m.site(EQUIPMENT + "__site_coater_wafer").id]
        rays = []
        for x in np.linspace(-0.04, 0.04, 5):
            for z in np.linspace(0.02, 0.04, 5):
                endpoint = surface + np.array([x, 0, z])
                origin = endpoint.copy()
                origin[1] = -0.15
                vec = endpoint - origin
                length = float(np.linalg.norm(vec))
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    ray_model,
                    ray_data,
                    origin,
                    vec / length,
                    np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                    1,
                    -1,
                    hit,
                )
                rays.append(
                    dict(
                        clear=bool(distance < 0 or distance >= length),
                        distance_m=float(distance),
                        length_m=length,
                        first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
                    )
                )
        screen = next(
            r
            for r in camera_target_screening(m, d, self.session.manifest)
            if r["equipment"] == EQUIPMENT
        )
        view = next(c for c in screen["cameras"] if c["camera"] == "coater_inspection")
        visible = (
            view.get("opaque_line_of_sight", False)
            and (view.get("first_hit") or {}).get("name")
            == EQUIPMENT + "__coater_physical_marker"
        )
        return dict(
            visible=visible,
            camera="coater_inspection",
            camera_screening=screen,
            clear_front_access_rays=sum(r["clear"] for r in rays),
            front_access_rays=rays,
            ray_scope="25 points above the marker through the physical doorway; not a robot/gripper swept-volume or wafer removal proof",
            **self._state(),
        )

    def run(self, *, step_observer=None):
        self._preflight()
        start = self.motion.measured()
        if np.any(np.abs(start) > (0.0002, 0.0002, 0.003)):
            raise PlanningError(
                "Alberta round trip requires measured XY home and closed door"
            )
        closed = self.access_observation()
        reports = []
        registration = None
        opened = None
        for name, target in (
            ("register_dry_nozzle", (0.07, -0.085, 0)),
            ("park_head", (-0.18, 0.10, 0)),
            ("open_door", (-0.18, 0.10, 1.15)),
            ("close_door", (-0.18, 0.10, 0)),
            ("return_head", start),
        ):
            result = self.move(target, step_observer=step_observer)
            reports.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
            if name == "register_dry_nozzle":
                registration = self.registration()
                if (
                    np.linalg.norm(registration["xy_error_m"]) > 0.0002
                    or abs(registration["marker_standoff_m"] - 0.048) > 1e-7
                ):
                    result.update(
                        status="failed",
                        failure="dry_registration_not_reached",
                        hold_requested=self.motion._request_hold(),
                    )
                    reports[-1] = dict(phase=name, **result)
                    break
            if name == "open_door":
                opened = self.access_observation()
                if not opened["visible"] or opened["clear_front_access_rays"] != 25:
                    result.update(
                        status="failed",
                        failure="physical_marker_access_not_observed",
                        hold_requested=self.motion._request_hold(),
                    )
                    reports[-1] = dict(phase=name, **result)
                    break
        return dict(
            status=(
                "succeeded"
                if len(reports) == 5
                and all(r["status"] == "succeeded" for r in reports)
                else "failed"
            ),
            scene_sha256=self.observer.binding["scene_sha256"],
            **combine_reset_reports(reports),
            phases=reports,
            registration=registration,
            closed_access_observation=closed,
            open_access_observation=opened,
            measured_return_error=(self.motion.measured() - start).tolist(),
            return_error_units=["m", "m", "rad"],
            scope="Measured dry XY surface registration and physical hinged-door access",
            limitations=[
                "Wafer remains fixed; no sample loading, pickup or removal occurs",
                "Fifty millimetre wafer standoff is authored geometry, not a validated spray setting",
                "Running clearances are carried by ideal joints; no bearing preload/friction calibration",
                "Parked fixed service loop and capped head lines carry no moving fluid feed",
                "Finite planning samples and every completed physics-step direct checks",
                "Point-ray access is not a robot gripper-volume guarantee",
                "Position hold on failure is not a certified emergency stop",
                "No aerosol, ultrasonic atomization, deposition, chemistry or hardware interlock simulation",
            ],
        )


def inspect_alberta_coater(session, *, step_observer=None):
    return CoaterAccess(session).run(step_observer=step_observer)
