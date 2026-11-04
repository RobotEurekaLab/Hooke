"""Measured dry access to a retained, freely supported UVA surrogate coupon.

Lid, loaded platen and handle positions are commanded through existing drives.
The coupon remains physically free and contact-supported. There is no specimen transfer,
pumping, pressure/seal verification or deposition operation.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports


SCENE = "uva_deposition"
EQUIPMENT = "deposition_chamber"
CONTROLS = ("chamber_lid", "platen", "vacuum_valve")


@lru_cache(maxsize=1)
def _authored_binding():
    """Accept the exact source or portable asset paths, never changed geometry."""
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(SCENE)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml,
        assets={item["file"]: Path(item["source"]).read_bytes() for item in resources},
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        camera=manifest["camera"],
        specimens=manifest["specimens"],
    )


def _require_binding(session):
    # Reject before control, hold, counters or simulation state can be changed.
    if session.data.model is not session.model:
        raise ValueError("UVA access requires data belonging to its physical model")
    expected = _authored_binding()
    if session.manifest.get("id") != SCENE:
        raise ValueError("UVA access requires its authored scene")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
    ):
        raise ValueError("UVA control or specimen metadata binding changed")
    if (
        session.manifest.get("camera") != expected["camera"]
        or session.manifest.get("specimens") != expected["specimens"]
    ):
        raise ValueError("UVA camera metadata binding changed")
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("UVA geometry, masks, constraints or physics binding changed")
    declared = session.manifest.get("scene_sha256")
    if declared is not None and declared != expected["scene_sha256"]:
        raise ValueError("UVA manifest does not match the physical model")
    return expected


def full_state(session):
    data = session.data
    return dict(
        time_s=float(data.time),
        qpos=data.qpos.tolist(),
        qvel=data.qvel.tolist(),
        ctrl=data.ctrl.tolist(),
        act=data.act.tolist(),
        qacc_warmstart=data.qacc_warmstart.tolist(),
        qfrc_applied=data.qfrc_applied.tolist(),
        xfrc_applied=data.xfrc_applied.tolist(),
        mocap_pos=data.mocap_pos.tolist(),
        mocap_quat=data.mocap_quat.tolist(),
        eq_active=data.eq_active.tolist(),
        model_signature=_model_signature(session.model),
    )


class ChamberAccess:
    """Local measured execution allows only the intended supported free carrier.

    Every step checks rigid direct distances, the free coupon's physical support,
    retained footprint and tilt, command tracking and all nonselected controls.
    Paths are fixed authored inspection trajectories, not a general planner.
    """

    cap_m = 0.002
    clearance_m = 0.00025

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.model, self.data = session, session.model, session.data
        m, d = self.model, self.data
        if not np.isfinite(d.time):
            raise ValueError("UVA access requires a finite simulation clock")
        self.signature = _model_signature(m)
        self.controls = [session.controls[EQUIPMENT, key] for key in CONTROLS]
        self.actuators = np.array([c.actuator for c in self.controls])
        self.qadr = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.dadr = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.sample = m.body(session.equipment[EQUIPMENT]["process"]["sample_body"]).id
        self.specimen = m.geom(EQUIPMENT + "__specimen__collision").id
        self.pad = m.geom(EQUIPMENT + "__carrier_bearing_pad").id
        self.platen = int(m.jnt_bodyid[self.controls[1].joint])
        supports = [self.pad] + [
            m.geom(f"{EQUIPMENT}__carrier_guide_{axis}_{sign}").id
            for axis in (0, 1)
            for sign in (-1, 1)
        ]
        self.allowed = {frozenset((self.specimen, g)) for g in supports}
        roots = {int(m.jnt_bodyid[c.joint]) for c in self.controls} | {self.sample}
        self.relevant = {
            i for i in range(m.ngeom) if m.body_weldid[m.geom_bodyid[i]] in roots
        }
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (a in self.relevant or b in self.relevant)
            ],
            dtype=int,
        )
        # Conservative whole-rotation spheres prune only pairs that cannot
        # approach within the reporting cap at ANY angle. Each selected hinge
        # has a fixed world anchor in the bound model. Free carriers and other
        # moving mechanisms are deliberately left unpruned.
        unpruned = len(self.pairs)
        anchors = {
            int(m.jnt_bodyid[c.joint]): d.xanchor[c.joint].copy() for c in self.controls
        }

        def swept_sphere(geom):
            weld = int(m.body_weldid[m.geom_bodyid[geom]])
            if weld == 0:
                return d.geom_xpos[geom], float(m.geom_rbound[geom])
            if weld in anchors:
                center = anchors[weld]
                return center, float(
                    np.linalg.norm(d.geom_xpos[geom] - center) + m.geom_rbound[geom]
                )
            return None

        spheres = [swept_sphere(i) for i in range(m.ngeom)]
        self.pairs = np.array(
            [
                (a, b)
                for a, b in self.pairs
                if spheres[a] is None
                or spheres[b] is None
                or np.linalg.norm(spheres[a][0] - spheres[b][0])
                <= spheres[a][1] + spheres[b][1] + self.cap_m + 1e-9
            ],
            dtype=int,
        )
        self.swept_pair_count = len(self.pairs)
        self.proven_distant_pairs = unpruned - len(self.pairs)
        self.held = {
            key: (c, float(d.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator not in self.actuators
        }
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.initial_eq = d.eq_active.copy()
        self.warnings = d.warning.number.copy()
        self.dt = float(m.opt.timestep)
        self.start_time = float(d.time)
        self.home = d.qpos[self.qadr].copy()
        self.report = dict(
            status="pending",
            failure=None,
            physics_steps=0,
            observer_entries=0,
            geometry_checks=0,
            fully_checked_steps=0,
            direct_pair_evaluations=0,
            minimum_noncontact_clearance_lower_bound_m=self.cap_m,
            maximum_carrier_lateral_offset_m=0.0,
            maximum_carrier_tilt_rad=0.0,
            maximum_carrier_yaw_error_rad=0.0,
            maximum_support_penetration_m=0.0,
            minimum_motion_support_contacts=None,
            held_control_maximum_error={},
            maximum_tracking_error_rad=[0.0, 0.0, 0.0],
            legs=[],
            original_pair_count=unpruned,
            swept_pair_count=self.swept_pair_count,
            whole_rotation_proven_distant_pairs=self.proven_distant_pairs,
            initial_state=full_state(session),
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            physics_events=[],
            hold_requested=False,
            scene_sha256=self.binding["scene_sha256"],
            model_signature=self.signature,
            scope="Measured open-top access, contact-supported free-coupon angular positioning and return",
        )

    def _identity(self):
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or self.data.model is not self.model
        ):
            raise ValueError("UVA physical model or data object was replaced")

    def _hold(self):
        if (
            self.session.model is self.model
            and self.session.data is self.data
            and np.isfinite(self.data.qpos[self.qadr]).all()
        ):
            self.data.ctrl[self.actuators] = np.clip(
                self.data.qpos[self.qadr], self.lower, self.upper
            )
            return True
        return False

    def _observe(self, *, settling=False):
        self._identity()
        m, d, r = self.model, self.data, self.report
        r["observer_entries"] += 1
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("uva_contact_configuration_changed")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_physics_state")
        if not np.array_equal(d.warning.number, self.warnings):
            raise RuntimeError("physics_warning")
        if not np.array_equal(d.eq_active, self.initial_eq):
            raise RuntimeError("equality_activation_changed")
        if abs(d.time - self.start_time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        for key, (control, target) in self.held.items():
            error = abs(self.session._measured(control) - target)
            label = ".".join(key)
            r["held_control_maximum_error"][label] = max(
                error, r["held_control_maximum_error"].get(label, 0)
            )
            if d.ctrl[control.actuator] != target or error > (
                0.0005 if control.units == "m" else 0.003
            ):
                raise RuntimeError("held_control_changed_or_drifted")
        if np.any(d.qpos[self.qadr] < self.lower - 0.0001) or np.any(
            d.qpos[self.qadr] > self.upper + 0.0001
        ):
            raise RuntimeError("mechanism_limit_violation")
        tracking = np.abs(d.qpos[self.qadr] - d.ctrl[self.actuators])
        r["maximum_tracking_error_rad"] = np.maximum(
            tracking, r["maximum_tracking_error_rad"]
        ).tolist()
        if np.any(tracking > np.array([0.075, 0.035, 0.025])):
            raise RuntimeError("trajectory_tracking_limit")
        if np.any(np.abs(d.qvel[self.dadr]) > np.array([0.5, 0.35, 0.4])):
            raise RuntimeError("mechanism_velocity_limit")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rotation), m.geom_aabb[:, 3:])
        first, second = self.pairs.T
        separation = np.maximum(
            np.abs(center[first] - center[second]) - half[first] - half[second], 0
        )
        near = np.linalg.norm(separation, axis=1) < self.cap_m
        minimum, penetration = self.cap_m, 0.0
        violation = None
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap_m, None)),
                self.cap_m,
            )
            r["direct_pair_evaluations"] += 1
            if frozenset((int(a), int(b))) in self.allowed:
                penetration = max(penetration, -gap)
            else:
                minimum = min(minimum, gap)
                if gap < self.clearance_m:
                    violation = f"uva_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
        r["geometry_checks"] += 1
        if violation:
            raise RuntimeError(violation)
        relative = rotation[self.pad].T @ (d.xpos[self.sample] - d.geom_xpos[self.pad])
        lateral = float(np.max(np.abs(relative[:2])))
        attitude = rotation[self.pad].T @ d.xmat[self.sample].reshape(3, 3)
        tilt = float(np.arccos(np.clip(attitude[2, 2], -1, 1)))
        yaw_error = float(np.arctan2(attitude[1, 0], attitude[0, 0]))
        r["maximum_carrier_lateral_offset_m"] = max(
            lateral, r["maximum_carrier_lateral_offset_m"]
        )
        r["maximum_carrier_tilt_rad"] = max(tilt, r["maximum_carrier_tilt_rad"])
        r["maximum_carrier_yaw_error_rad"] = max(
            abs(yaw_error), r["maximum_carrier_yaw_error_rad"]
        )
        r["maximum_support_penetration_m"] = max(
            penetration, r["maximum_support_penetration_m"]
        )
        # A 40 mm square must actually fit in the 40.4 mm pocket. Checking its
        # rotated corners detects angular escape that a centroid test misses.
        corners = [
            relative + attitude @ np.array([x, y, 0.0])
            for x in (-0.02, 0.02)
            for y in (-0.02, 0.02)
        ]
        if (
            max(float(np.max(np.abs(c[:2]))) for c in corners) > 0.0203
            or abs(relative[2] - 0.0035) > 0.0001
            or tilt > 0.01
        ):
            raise RuntimeError("uva_free_carrier_left_support")
        if penetration > 0.0001:
            raise RuntimeError("uva_free_carrier_support_penetration")
        support = [
            c
            for c in d.contact
            if frozenset((int(c.geom1), int(c.geom2)))
            == frozenset((self.specimen, self.pad))
        ]
        if not settling:
            if not support:
                raise RuntimeError("uva_free_carrier_has_no_support_contact")
            old = r["minimum_motion_support_contacts"]
            r["minimum_motion_support_contacts"] = min(
                old if old is not None else len(support), len(support)
            )
        r["minimum_noncontact_clearance_lower_bound_m"] = min(
            minimum, r["minimum_noncontact_clearance_lower_bound_m"]
        )
        r["fully_checked_steps"] += 1

    def _phase(self, name, target, ramp_s, settle_s, observer=None):
        self._identity()
        _require_binding(self.session)
        m, d, r = self.model, self.data, self.report
        if not np.isfinite(d.time):
            raise ValueError("UVA access requires a finite simulation clock")
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Target must remain within the three original ranges")
        start = d.qpos[self.qadr].copy()
        n = max(1, round(ramp_s / self.dt))
        leg = dict(
            phase=name,
            target_rad=target.tolist(),
            status="running",
            physics_steps=0,
            fully_checked_steps=0,
            geometry_checks=0,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            before=full_state(self.session),
        )
        r["legs"].append(leg)
        before = r["fully_checked_steps"], r["geometry_checks"]
        try:
            for i in range(n + round(settle_s / self.dt)):
                self._identity()
                x = min(1.0, (i + 1) / n)
                blend = 10 * x**3 - 15 * x**4 + 6 * x**5
                d.ctrl[self.actuators] = start + blend * (target - start)
                before_time = float(d.time)
                before_warnings = _warning_state(d)
                reset_inputs = {
                    key: getattr(d, key).copy()
                    for key in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                r["physics_steps"] += 1
                leg["physics_steps"] += 1
                mujoco.mj_forward(m, d)
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, reset_inputs
                )
                if event is not None:
                    r["physics_events"].append(
                        dict(executed_step=r["physics_steps"], **event)
                    )
                    r.update(combine_reset_reports(r["physics_events"]))
                    leg.update(combine_reset_reports([event]), physics_event=event)
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                self._observe(settling=name == "settle_loaded_coupon")
                if observer is not None:
                    observer(m, d)  # Read-only callback, as in JointMotion.
                    if _model_signature(m) != self.signature:
                        raise RuntimeError("observer_modified_model")
            if (
                np.any(np.abs(d.qpos[self.qadr] - target) > 0.003)
                or np.max(np.abs(d.qvel[self.dadr])) > 0.002
            ):
                raise RuntimeError("phase_endpoint_not_settled")
            leg["status"] = "succeeded"
        except (RuntimeError, ValueError) as exc:
            leg.update(status="failed", failure=str(exc))
            raise
        finally:
            leg.update(
                fully_checked_steps=r["fully_checked_steps"] - before[0],
                geometry_checks=r["geometry_checks"] - before[1],
                after=full_state(self.session),
            )
            leg["coverage_complete"] = (
                leg["physics_steps"] > 0
                and leg["physics_steps"] == leg["fully_checked_steps"]
            )

    def _observation(self):
        from real_labs.usability import camera_target_screening

        m, d = self.model, self.data
        screen = next(
            r
            for r in camera_target_screening(m, d, self.session.manifest)
            if r["equipment"] == EQUIPMENT
        )
        view = next(c for c in screen["cameras"] if c["camera"] == "chamber_loading")
        visible = (
            view.get("opaque_line_of_sight", False)
            and view.get("first_hit", {}).get("name") == m.geom(self.specimen).name
            and view.get("estimated_target_diameter_px", 0) >= 40
        )
        hits = []
        rotation = d.geom_xmat[self.specimen].reshape(3, 3)
        for x in np.linspace(-0.018, 0.018, 5):
            for y in np.linspace(-0.018, 0.018, 5):
                origin = d.geom_xpos[self.specimen] + rotation @ np.array(
                    [x, y, 0.0032]
                )
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(
                    m, d, origin, np.array([0.0, 0.0, 1.0]), None, 1, self.sample, hit
                )
                hits.append(
                    dict(
                        distance_m=float(distance),
                        geom=None if hit[0] < 0 else m.geom(int(hit[0])).name,
                    )
                )
        clear = all(h["distance_m"] < 0 or h["distance_m"] >= 0.30 for h in hits)
        return dict(
            visible=visible,
            camera="chamber_loading",
            camera_screening=screen,
            upward_300mm_rays_clear=clear,
            upward_rays=hits,
            measured_platen_rad=float(d.qpos[self.qadr[1]]),
            measured_coupon_rotation=d.xmat[self.sample].tolist(),
            state=full_state(self.session),
            scope="Sampled vertical geometric access only; no gripper insertion or loading performed",
        )

    def open_inspect_return(self, *, step_observer=None):
        self._identity()
        _require_binding(self.session)
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        m, d, r = self.model, self.data, self.report
        if (
            r["physics_steps"]
            or np.max(np.abs(d.qpos[self.qadr])) > 0.003
            or np.max(np.abs(d.qvel)) > 0.01
        ):
            raise ValueError(
                "Begin a fresh operation at measured closed, centered home"
            )
        try:
            self._phase("settle_loaded_coupon", self.home, 0.002, 0.25, step_observer)
            self._phase(
                "open_lid", (-1.5, self.home[1], self.home[2]), 10, 1.0, step_observer
            )
            self._phase(
                "position_loaded_platen", (-1.5, 0.5, self.home[2]), 4, 1, step_observer
            )
            self._phase("inspect_dry_handle", (-1.5, 0.5, 0.5), 3, 0.6, step_observer)
            r["observation"] = self._observation()
            if (
                not r["observation"]["visible"]
                or not r["observation"]["upward_300mm_rays_clear"]
            ):
                raise RuntimeError("loaded_coupon_access_not_observed")
            self._phase(
                "return_handle", (-1.5, 0.5, self.home[2]), 3, 0.6, step_observer
            )
            self._phase(
                "return_loaded_platen",
                (-1.5, self.home[1], self.home[2]),
                4,
                1,
                step_observer,
            )
            self._phase("return_lid", self.home, 10, 1.0, step_observer)
            _require_binding(self.session)
            r["status"] = "succeeded"
        except (RuntimeError, ValueError) as error:
            r.update(status="failed", failure=str(error), hold_requested=self._hold())
        complete = (
            r["physics_steps"] > 0 and r["physics_steps"] == r["fully_checked_steps"]
        )
        r.update(
            coverage_complete=complete,
            whole_run_clearance_lower_bound_m=(
                r["minimum_noncontact_clearance_lower_bound_m"] if complete else None
            ),
            unchecked_steps=r["physics_steps"] - r["fully_checked_steps"],
            final_state=full_state(self.session),
            return_error_rad=np.abs(d.qpos[self.qadr] - self.home).tolist(),
            model_unchanged=_model_signature(m) == self.signature,
            limitations=[
                "No specimen transfer, vacuum, pumping, pressure, sealing or deposition process",
                "Fixed authored trajectories with per-physics-step checks, not a general or continuous-time motion planner",
                "Ideal bearings and drive parameters are authored, not manufacturer calibrated",
                "Failure hold is a servo request, not an instantaneous stop; reset remains explicit",
                "Callbacks must be read-only; concurrent external state/control/model writers are unsupported",
            ],
        )
        return r
