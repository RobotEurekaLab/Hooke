"""Measured dry cassette loading by contact; no state placement during motion.

Only the two powered prismatic axes are commanded. The cassette remains a
six-DOF free body throughout loading, closing and return to external staging.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports

SCENE = "penn_pathology"
OWNER = "sample_processor"
CONTROLS = ("access_door", "carrier_translation")


@lru_cache(maxsize=1)
def _authored_binding():
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    session = InstrumentSession(model, manifest)
    resources = _portable_resources(root)
    canonical_xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        canonical_xml,
        {r["file"]: Path(r["source"]).read_bytes() for r in resources},
    )
    return dict(
        signatures={_model_signature(model), _model_signature(portable)},
        scene_sha256=hashlib.sha256(canonical_xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment[OWNER],
        cameras=manifest["camera"],
        contype=model.geom_contype.copy(),
        conaffinity=model.geom_conaffinity.copy(),
        disableflags=int(model.opt.disableflags),
        enableflags=int(model.opt.enableflags),
    )


def _require_binding(session):
    expected = _authored_binding()
    if session.manifest.get("id") != SCENE or session.data.model is not session.model:
        raise ValueError("Penn route requires the authored scene and its own state")
    if (
        session.controls != expected["controls"]
        or session.equipment.get(OWNER) != expected["equipment"]
        or session.manifest.get("camera") != expected["cameras"]
    ):
        raise ValueError("Penn control/specimen/camera binding changed")
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("Penn geometry, collision or physics binding changed")
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("Penn manifest scene digest changed")
    return expected


class PennProcessor:
    """Guarded, bounded dry loading route with per-step native/direct checks.

    Failure requests measured axis hold; it never resets or relocates a sample.
    The tray is the declared manual-loading surrogate. Off-tray pickup, tissue
    processing and arbitrary raw actuator sequences are outside this contract.
    """

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session = session
        model, data = session.model, session.data
        self.axes = [session.controls[OWNER, key] for key in CONTROLS]
        self.qadr = np.array([model.jnt_qposadr[c.joint] for c in self.axes])
        self.dadr = np.array([model.jnt_dofadr[c.joint] for c in self.axes])
        self.base = model.body(OWNER).id
        self.cassette = model.body(OWNER + "__cassette").id
        self.geom = model.geom(OWNER + "__cassette__collision").id
        self.seat = model.site(OWNER + "__site_specimen_seat").id
        self.support = model.geom(OWNER + "__carrier_floor").id
        self.free = model.joint(OWNER + "__cassette__free").id
        self.allowed = {self.support} | {
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith(OWNER + "__pocket_")
        }
        affected = {int(model.jnt_bodyid[c.joint]) for c in self.axes} | {self.cassette}
        enabled = np.flatnonzero(
            (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        moving = [
            i
            for i in enabled
            if int(model.body_weldid[model.geom_bodyid[i]]) in affected
        ]
        self.pairs = np.array(
            sorted(
                {
                    tuple(sorted((int(i), int(j))))
                    for i in moving
                    for j in enabled
                    if model.body_weldid[model.geom_bodyid[i]]
                    != model.body_weldid[model.geom_bodyid[j]]
                }
            ),
            dtype=int,
        )
        self.allowed_pairs = np.array(
            [
                self.geom in pair
                and next(i for i in pair if i != self.geom) in self.allowed
                for pair in self.pairs
            ]
        )
        self.held_controls = {
            key: float(data.qpos[model.jnt_qposadr[c.joint]])
            for key, c in session.controls.items()
            if key[0] != OWNER
        }

    def measured(self):
        return self.session.data.qpos[self.qadr].copy()

    def specimen(self):
        model, data = self.session.model, self.session.data
        rotation = data.xmat[self.base].reshape(3, 3)
        local_center = rotation.T @ (data.xpos[self.cassette] - data.xpos[self.base])
        local_rotation = rotation.T @ data.geom_xmat[self.geom].reshape(3, 3)
        half = np.abs(local_rotation) @ model.geom_size[self.geom]
        lower, upper = local_center - half, local_center + half
        support_contacts = [
            c
            for c in data.contact
            if {int(c.geom1), int(c.geom2)} == {self.geom, self.support}
        ]
        velocity = data.qvel[
            model.jnt_dofadr[self.free] : model.jnt_dofadr[self.free] + 6
        ]
        return dict(
            local_center_m=local_center.tolist(),
            local_bounds_m=[lower.tolist(), upper.tolist()],
            seat_error_m=float(
                np.linalg.norm(data.xpos[self.cassette] - data.site_xpos[self.seat])
            ),
            speed_m_s=float(np.linalg.norm(velocity[:3])),
            angular_speed_rad_s=float(np.linalg.norm(velocity[3:])),
            upright_cosine=float(local_rotation[2, 2]),
            support_contacts=len(support_contacts),
            inside_chamber=bool(
                lower[0] > -0.166
                and upper[0] < 0.166
                and lower[1] > -0.123
                and upper[1] < 0.213
                and lower[2] > 0.065
                and upper[2] < 0.361
            ),
            outside_chamber=bool(upper[1] < -0.145),
            access_closed=bool(abs(self.measured()[0]) < 0.0001),
        )

    def _check(self):
        model, data = self.session.model, self.session.data
        binding = self.binding
        if (
            int(model.opt.disableflags) != binding["disableflags"]
            or int(model.opt.enableflags) != binding["enableflags"]
            or not np.array_equal(model.geom_contype, binding["contype"])
            or not np.array_equal(model.geom_conaffinity, binding["conaffinity"])
        ):
            raise RuntimeError("contact_configuration_changed")
        if not all(
            np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.ctrl)
        ):
            raise RuntimeError("nonfinite_physics")
        for key, initial in self.held_controls.items():
            c = self.session.controls[key]
            if abs(data.qpos[model.jnt_qposadr[c.joint]] - initial) > 0.001:
                raise RuntimeError("unrelated_control_moved:" + ".".join(key))
        sample = self.specimen()
        if sample["seat_error_m"] > 0.002 or sample["upright_cosine"] < 0.995:
            raise RuntimeError("cassette_left_physical_pocket")
        # World AABBs are conservative bounds for every primitive/mesh. All
        # potentially close cross-weld pairs involving door/tray/free cassette
        # receive a direct query, including native parent-filtered surfaces.
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum(
            "nij,nj->ni", rotation, model.geom_aabb[:, :3]
        )
        half = np.einsum("nij,nj->ni", np.abs(rotation), model.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        near = np.linalg.norm(separation, axis=1) < 0.002
        minimum = 0.002
        support_min = 0.002
        queries = 0
        for index in np.flatnonzero(near):
            i, j = map(int, self.pairs[index])
            queries += 1
            gap = float(mujoco.mj_geomDistance(model, data, i, j, 0.002, None))
            if self.allowed_pairs[index]:
                support_min = min(support_min, gap)
                if gap < -0.0002:
                    raise RuntimeError("excessive_cassette_contact_penetration")
            else:
                minimum = min(minimum, gap)
                if gap < 0.00025:
                    raise RuntimeError(
                        "clearance_lost:"
                        + model.geom(i).name
                        + ":"
                        + model.geom(j).name
                    )
        for contact in data.contact:
            if contact.dist < -0.0002:
                raise RuntimeError(
                    "native_penetration:"
                    + model.geom(int(contact.geom1)).name
                    + ":"
                    + model.geom(int(contact.geom2)).name
                )
        return minimum, support_min, queries

    def move(self, control, target, *, step_observer=None):
        """Move one endpoint through a 2 s smooth ramp and 1 s settling period."""
        _require_binding(self.session)
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        if (
            control not in CONTROLS
            or not np.isfinite(target)
            or target not in ((0.0, 0.255) if control == "access_door" else (0.0, 0.38))
        ):
            raise ValueError("Penn route supports only authored axis endpoints")
        q = self.measured()
        if np.max(np.abs(self.session.data.qvel[self.dadr])) > 0.005:
            raise ValueError("Penn move requires settled powered axes")
        if control == "carrier_translation" and abs(q[0] - 0.255) > 0.0001:
            raise ValueError("Open access door before moving the carrier")
        if (
            control == "access_door"
            and target == 0
            and min(abs(q[1]), abs(q[1] - 0.38)) > 0.0001
        ):
            raise ValueError("Carrier must be fully inside or outside before closing")
        self._check()  # Reject bad initial state before any input mutation.
        model, data = self.session.model, self.session.data
        c = self.session.controls[OWNER, control]
        start = float(data.qpos[model.jnt_qposadr[c.joint]])
        start_time = float(data.time)
        steps = checked = queries = 0
        advance_pending = False
        minimum = 0.002
        support_min = 0.002
        events = []
        failure = None
        warning = _warning_state(data)
        if np.any(warning[:, 0]):
            raise ValueError("Explicit reset required after native warning")
        try:
            count = int(round(3 / model.opt.timestep))
            for n in range(count):
                t = min(1.0, (n + 1) * model.opt.timestep / 2)
                ramp = t * t * t * (10 + t * (-15 + 6 * t))
                data.ctrl[c.actuator] = start + (target - start) * ramp
                before = float(data.time)
                reset_inputs = {
                    name: getattr(data, name).copy()
                    for name in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                advance_pending = True
                self.session.advance(1)
                steps += 1
                advance_pending = False
                event, warning = _physics_event(
                    model, data, before, warning, reset_inputs
                )
                if event is not None:
                    events.append(event)
                    raise RuntimeError("physics_event")
                mujoco.mj_forward(model, data)
                gap, support_gap, calls = self._check()
                minimum = min(minimum, gap)
                support_min = min(support_min, support_gap)
                queries += calls
                checked += 1
                if step_observer is not None:
                    saved = {
                        name: getattr(data, name).copy()
                        for name in (
                            "qpos",
                            "qvel",
                            "ctrl",
                            "qfrc_applied",
                            "xfrc_applied",
                            "mocap_pos",
                            "mocap_quat",
                        )
                    }
                    time = float(data.time)
                    step_observer(model, data)
                    if time != data.time or any(
                        not np.array_equal(value, getattr(data, name))
                        for name, value in saved.items()
                    ):
                        raise RuntimeError("step_observer_changed_physical_state")
            _require_binding(self.session)
            if (
                abs(data.qpos[model.jnt_qposadr[c.joint]] - target) > 0.0001
                or np.max(np.abs(data.qvel[self.dadr])) > 0.005
            ):
                raise RuntimeError("axis_did_not_reach_and_settle")
            sample = self.specimen()
            if sample["support_contacts"] < 1 or sample["speed_m_s"] > 0.005:
                raise RuntimeError("cassette_not_settled_on_physical_support")
        except Exception as error:
            # An advance wrapper may raise after advancing. Preserve native
            # warning/clock evidence instead of inferring that no reset ran.
            if advance_pending:
                event, warning = _physics_event(
                    model, data, before, warning, reset_inputs
                )
                if event is not None:
                    events.append(event)
                if data.time != before or event is not None:
                    steps += 1
            failure = str(error)
            for axis in self.axes:
                data.ctrl[axis.actuator] = float(
                    np.clip(
                        data.qpos[model.jnt_qposadr[axis.joint]], axis.lower, axis.upper
                    )
                )
        reset = (
            combine_reset_reports(events)
            if events
            else dict(
                reset_performed=False,
                automatic_reset_detected=False,
                explicit_reset_performed=False,
            )
        )
        return dict(
            status="failed" if failure else "succeeded",
            failure=failure,
            hold_requested=bool(failure),
            control=control,
            target_m=target,
            measured_m=float(data.qpos[model.jnt_qposadr[c.joint]]),
            physics_steps=steps,
            elapsed_s=float(data.time) - start_time,
            **reset,
            physics_events=events,
            direct_clearance=dict(
                cross_weld_pairs=len(self.pairs),
                checked_steps=checked,
                unchecked_steps=steps - checked,
                coverage_complete=steps > 0 and steps == checked and failure is None,
                checked_steps_minimum_m=minimum if checked else None,
                whole_leg_minimum_m=(
                    minimum
                    if steps > 0 and steps == checked and failure is None
                    else None
                ),
                intended_cassette_contact_minimum_m=support_min if checked else None,
                direct_queries=queries,
                reporting_cap_m=0.002,
            ),
            specimen=self.specimen(),
            state=dict(
                qpos=data.qpos.tolist(),
                qvel=data.qvel.tolist(),
                ctrl=data.ctrl.tolist(),
                time_s=float(data.time),
            )
        )

    def run(self, *, step_observer=None):
        """Load/close/reopen/unload/close; sample placement and reset are absent."""
        _require_binding(self.session)
        if (
            np.max(np.abs(self.measured())) > 0.0001
            or not self.specimen()["outside_chamber"]
        ):
            raise ValueError(
                "Round trip requires closed door and external staged cassette"
            )
        start = self.measured()
        sample_start = self.session.data.xpos[self.cassette].copy()
        phases = []
        for phase, control, target in (
            ("open_for_loading", "access_door", 0.255),
            ("load_cassette", "carrier_translation", 0.38),
            ("close_loaded_chamber", "access_door", 0.0),
            ("open_for_unloading", "access_door", 0.255),
            ("unload_to_external_staging", "carrier_translation", 0.0),
            ("close_empty_chamber", "access_door", 0.0),
        ):
            result = self.move(control, target, step_observer=step_observer)
            if (
                result["status"] == "succeeded"
                and phase in ("load_cassette", "close_loaded_chamber")
                and not result["specimen"]["inside_chamber"]
            ):
                result.update(status="failed", failure="cassette_not_physically_inside")
            if (
                result["status"] == "succeeded"
                and phase in ("unload_to_external_staging", "close_empty_chamber")
                and not result["specimen"]["outside_chamber"]
            ):
                result.update(
                    status="failed", failure="cassette_not_physically_outside"
                )
            phases.append(dict(phase=phase, **result))
            if result["status"] != "succeeded":
                break
        success = len(phases) == 6 and all(p["status"] == "succeeded" for p in phases)
        return dict(
            status="succeeded" if success else "failed",
            scene_sha256=self.binding["scene_sha256"],
            **combine_reset_reports(phases),
            phases=phases,
            physics_steps=sum(p["physics_steps"] for p in phases),
            measured_axis_return_error_m=(self.measured() - start).tolist(),
            measured_cassette_return_delta_m=(
                self.session.data.xpos[self.cassette] - sample_start
            ).tolist(),
            scope="Free inert cassette physically conveyed from external staging into dry enclosure, closed access, then physically returned to external staging",
            limitations=[
                "No off-tray pickup, gripping, clinical routing database or tissue processing",
                "Authored representative processor, not identified Penn/OEM hardware",
                "Finite actual CPU trajectory; arbitrary door/carriage combinations include physical collisions",
                "No native Isaac validation; position hold on failure is not a certified stop",
            ]
        )
