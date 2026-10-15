"""Measured access to a freely supported inert filter carrier.

Only the existing finite-force drawer servo is commanded. No carrier state is
assigned or explicitly reset; native resets are observed and reported. No
flow/filtration/scientific response is modeled.
The local executor observes the moving free carrier instead of planning against
its frozen reset pose as a stationary obstacle.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports

SCENE = "eth_air_quality"
EQUIPMENT = "aerosol_sampler"


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
    portable = ET.tostring(root, encoding="unicode")
    exported_model = mujoco.MjModel.from_xml_string(
        portable,
        {r["file"]: Path(r["source"]).read_bytes() for r in resources},
    )
    return dict(
        signatures={_model_signature(model), _model_signature(exported_model)},
        scene_sha256=hashlib.sha256(portable.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        camera=manifest["camera"],
        specimens=manifest["specimens"],
        carrier_offset=(
            session.data.xpos[model.body(EQUIPMENT + "__specimen").id]
            - session.data.site_xpos[model.site(EQUIPMENT + "__site_filter").id]
        ),
        carrier_rotation=session.data.xmat[
            model.body(EQUIPMENT + "__specimen").id
        ].copy(),
    )


def _require_binding(session):
    if session.data.model is not session.model or session.manifest.get("id") != SCENE:
        raise ValueError(
            "ETH filter access requires its exact authored model and state"
        )
    expected = _authored_binding()
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("ETH authored geometry, contacts, physics or drive changed")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or session.manifest.get("camera") != expected["camera"]
        or session.manifest.get("specimens") != expected["specimens"]
    ):
        raise ValueError("ETH control, carrier or camera metadata binding changed")
    if (
        session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError("ETH scene digest does not match the authored model")
    return expected


class FilterAccess:
    """Finite low-speed exposure/return with contact and held-axis observations."""

    cap_m = 0.002
    carrier_error_limit_m = 0.0008
    penetration_limit_m = 0.00015

    def __init__(self, session):
        self.binding = _require_binding(session)  # No mutation before rejection.
        self.session, self.model, self.data = session, session.model, session.data
        m, d = self.model, self.data
        if any(
            not np.isfinite(values).all()
            for values in (
                d.qpos,
                d.qvel,
                d.ctrl,
                d.qfrc_applied,
                d.xfrc_applied,
                np.array([d.time]),
            )
        ):
            raise ValueError(
                "Filter access requires finite measured state and applied forces"
            )
        self.signature = _model_signature(m)
        self.control = session.controls[EQUIPMENT, "filter_drawer"]
        self.qadr = int(m.jnt_qposadr[self.control.joint])
        self.dadr = int(m.jnt_dofadr[self.control.joint])
        self.actuator = self.control.actuator
        self.carrier = m.body(EQUIPMENT + "__specimen").id
        self.specimen = m.geom(EQUIPMENT + "__specimen__collision").id
        self.site = m.site(EQUIPMENT + "__site_filter").id
        self.drawer = int(m.jnt_bodyid[self.control.joint])
        valve = int(m.jnt_bodyid[session.controls[EQUIPMENT, "flow_valve"].joint])
        moving_welds = {self.drawer, valve, self.carrier}
        ids = [i for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
        self.pairs = np.array(
            [
                (a, b)
                for index, a in enumerate(ids)
                for b in ids[index + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in moving_welds
                    or m.body_weldid[m.geom_bodyid[b]] in moving_welds
                )
            ],
            dtype=int,
        )
        supports = [m.geom(EQUIPMENT + "__carrier_bearing_pad").id] + [
            m.geom(f"{EQUIPMENT}__carrier_guide_{axis}_{sign}").id
            for axis in (0, 1)
            for sign in (-1, 1)
        ]
        self.allowed = {frozenset((self.specimen, g)) for g in supports}
        self.held = {
            key: (c, float(d.ctrl[c.actuator]))
            for key, c in session.controls.items()
            if c.actuator != self.actuator
        }
        self.initial_offset = d.xpos[self.carrier] - d.site_xpos[self.site]
        if np.linalg.norm(
            self.initial_offset - self.binding["carrier_offset"]
        ) > 0.0005 or not np.allclose(
            d.xmat[self.carrier], self.binding["carrier_rotation"], atol=0.01
        ):
            raise ValueError(
                "Begin with the inert carrier physically loaded in its holder"
            )
        self.initial_rotation = d.xmat[self.carrier].copy()
        self.initial_eq = d.eq_active.copy()
        self.initial_masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.warnings = d.warning.number.copy()
        self.start_time, self.dt = float(d.time), float(m.opt.timestep)
        self.home = float(d.qpos[self.qadr])
        self.report = dict(
            success=False,
            failure=None,
            physics_steps=0,
            observer_entries=0,
            checked_steps=0,
            geometry_checks=0,
            unchecked_steps=0,
            preflight_checks=0,
            phases=[],
            direct_pair_count=len(self.pairs),
            direct_distance_queries=0,
            capped_pair_evaluations=0,
            minimum_noncontact_clearance_lower_bound_m=self.cap_m,
            maximum_carrier_tracking_error_m=0.0,
            maximum_carrier_penetration_m=0.0,
            maximum_carrier_tilt_rad=0.0,
            held_control_maximum_error={},
            clearance_reporting_cap_m=self.cap_m,
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            time_discontinuity_detected=False,
            physics_events=[],
            model_signature=self.signature,
            scene_sha256=self.binding["scene_sha256"],
            interpretation="Already loaded, physically free inert-carrier access and return. No placement, extraction, weighing, flow or filtration process.",
        )

    def _identity(self):
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or self.data.model is not self.model
        ):
            raise ValueError("ETH session model or state object was replaced")

    def observe(self):
        """A completed check includes actual direct geometry and carrier limits."""
        self._identity()
        m, d, r = self.model, self.data, self.report
        r["observer_entries"] += 1
        if (
            self.flags != (int(m.opt.disableflags), int(m.opt.enableflags))
            or not np.array_equal(m.geom_contype, self.initial_masks[0])
            or not np.array_equal(m.geom_conaffinity, self.initial_masks[1])
        ):
            raise RuntimeError("contact_or_physics_configuration_changed")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_state")
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
                error, r["held_control_maximum_error"].get(label, 0.0)
            )
            if d.ctrl[control.actuator] != target or error > (
                0.0005 if control.units == "m" else 0.003
            ):
                raise RuntimeError("held_control_changed_or_drifted")
        if (
            not self.control.lower - 0.0001
            <= d.qpos[self.qadr]
            <= self.control.upper + 0.0001
        ):
            raise RuntimeError("drawer_limit_violation")
        error = float(
            np.linalg.norm(
                d.xpos[self.carrier] - d.site_xpos[self.site] - self.initial_offset
            )
        )
        rotation = d.xmat[self.carrier].reshape(3, 3).T @ self.initial_rotation.reshape(
            3, 3
        )
        tilt = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        r["maximum_carrier_tracking_error_m"] = max(
            error, r["maximum_carrier_tracking_error_m"]
        )
        r["maximum_carrier_tilt_rad"] = max(tilt, r["maximum_carrier_tilt_rad"])
        # Conservative exact-geom AABBs discard only pairs whose distance is
        # provably >= the same 2 mm reporting cap. Parent filtering is irrelevant.
        rot = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rot, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(rot), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        sep = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        near = np.linalg.norm(sep, axis=1) < self.cap_m
        r["capped_pair_evaluations"] += len(self.pairs)
        closest, penetration = self.cap_m, 0.0
        violation = None
        for first, second in self.pairs[near]:
            gap = min(
                float(
                    mujoco.mj_geomDistance(
                        m, d, int(first), int(second), self.cap_m, None
                    )
                ),
                self.cap_m,
            )
            r["direct_distance_queries"] += 1
            if frozenset((int(first), int(second))) in self.allowed:
                penetration = max(penetration, -gap)
            else:
                closest = min(closest, gap)
                if gap < 0.0001:
                    violation = f"rigid_clearance_lost:{m.geom(first).name}:{m.geom(second).name}"
        r["minimum_noncontact_clearance_lower_bound_m"] = min(
            closest, r["minimum_noncontact_clearance_lower_bound_m"]
        )
        r["maximum_carrier_penetration_m"] = max(
            penetration, r["maximum_carrier_penetration_m"]
        )
        r["geometry_checks"] += 1
        if violation:
            raise RuntimeError(violation)
        if penetration > self.penetration_limit_m:
            raise RuntimeError("carrier_contact_penetration_limit")
        if error > self.carrier_error_limit_m or tilt > 0.02:
            raise RuntimeError("free_carrier_not_retained")
        if abs(d.qvel[self.dadr]) > 0.09:
            raise RuntimeError("drawer_velocity_limit")
        if abs(d.qpos[self.qadr] - d.ctrl[self.actuator]) > 0.012:
            raise RuntimeError("drawer_tracking_limit")
        r["checked_steps"] += 1

    def phase(self, name, target, ramp_s, settle_s, observer=None):
        self._identity()
        _require_binding(self.session)
        m, d, r = self.model, self.data, self.report
        if not self.control.lower <= target <= self.control.upper:
            raise ValueError("Drawer target is outside its preserved range")
        if not np.isfinite(float(d.time)):
            raise ValueError("Phase requires finite simulation time before commands")
        start, n = float(d.ctrl[self.actuator]), max(1, round(ramp_s / self.dt))
        record = dict(
            name=name,
            target_m=target,
            executed_steps=0,
            checked_steps=0,
            geometry_checks=0,
            success=False,
        )
        r["phases"].append(record)
        before = (r["checked_steps"], r["geometry_checks"])
        for i in range(n + round(settle_s / self.dt)):
            self._identity()
            blend = min(1.0, (i + 1) / n)
            blend = blend * blend * (3 - 2 * blend)
            d.ctrl[self.actuator] = start + blend * (target - start)
            before_time = float(d.time)
            before_warnings = _warning_state(d)
            reset_inputs = {
                key: getattr(d, key).copy()
                for key in ("ctrl", "qfrc_applied", "xfrc_applied")
            }
            mujoco.mj_step(m, d)
            r["physics_steps"] += 1
            record["executed_steps"] += 1
            mujoco.mj_forward(m, d)
            try:
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, reset_inputs
                )
                if event is not None:
                    r["physics_events"].append(
                        dict(executed_step=r["physics_steps"], **event)
                    )
                    r.update(combine_reset_reports(r["physics_events"]))
                    r["time_discontinuity_detected"] = any(
                        not e["time_continuous"] for e in r["physics_events"]
                    )
                    record.update(combine_reset_reports([event]))
                    record["physics_event"] = event
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                self.observe()
                if observer is not None:
                    observer(m, d)
                    # Callbacks are read-only; reject any compiled-model mutation.
                    if _model_signature(m) != self.signature:
                        raise RuntimeError("observer_modified_model")
            finally:
                record["checked_steps"] = r["checked_steps"] - before[0]
                record["geometry_checks"] = r["geometry_checks"] - before[1]
        record.update(measured_m=float(d.qpos[self.qadr]), time_s=float(d.time))
        if abs(d.qpos[self.qadr] - target) > 0.0002 or abs(d.qvel[self.dadr]) > 0.001:
            raise RuntimeError("drawer_endpoint_not_settled")
        record["success"] = True

    def access_observation(self):
        from real_labs.usability import camera_target_screening

        m, d = self.model, self.data
        screening = next(
            row
            for row in camera_target_screening(m, d, self.session.manifest)
            if row["equipment"] == EQUIPMENT
        )
        view = next(
            c for c in screening["cameras"] if c["camera"] == "filter_inspection"
        )
        visible = (
            view.get("opaque_line_of_sight", False)
            and view.get("first_hit", {}).get("name") == m.geom(self.specimen).name
            and view.get("estimated_target_diameter_px", 0) >= 40
        )
        # Twenty-five upward rays inside the actual carrier footprint test a
        # 100 mm lifting approach; they do not certify a gripper's swept volume.
        hits, directions = [], np.array([0.0, 0.0, 1.0])
        rotation = d.geom_xmat[self.specimen].reshape(3, 3)
        for x in np.linspace(-0.016, 0.016, 5):
            for y in np.linspace(-0.016, 0.016, 5):
                origin = d.geom_xpos[self.specimen] + rotation @ np.array(
                    [x, y, m.geom_size[self.specimen, 2] + 0.0002]
                )
                hit = np.array([-1], dtype=np.int32)
                gap = mujoco.mj_ray(
                    m, d, origin, directions, None, 1, self.carrier, hit
                )
                hits.append(
                    dict(
                        distance_m=float(gap),
                        geom=None if hit[0] < 0 else m.geom(int(hit[0])).name,
                    )
                )
        clear = all(row["distance_m"] < 0 or row["distance_m"] >= 0.10 for row in hits)
        return dict(
            visible=visible,
            upward_100mm_rays_clear=clear,
            upward_rays=hits,
            camera_screening=screening,
            sample_world_m=d.xpos[self.carrier].tolist(),
            qpos=d.qpos.tolist(),
            qvel=d.qvel.tolist(),
            ctrl=d.ctrl.tolist(),
            time_s=float(d.time),
            model_signature=self.signature,
            scope="Visible access and unobstructed sampled vertical approach; no grasp or removal performed",
        )

    def run(self, *, step_observer=None):
        # All preconditions are checked before commands or failure holds.
        self._identity()
        _require_binding(self.session)
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable and read-only")
        m, d, r = self.model, self.data, self.report
        if not np.isfinite(float(d.time)):
            raise ValueError(
                "Filter access requires finite simulation time before commands"
            )
        if (
            r["physics_steps"]
            or abs(d.qpos[self.qadr]) > 0.0002
            or np.max(np.abs(d.qvel)) > 0.01
        ):
            raise ValueError("Begin a fresh operation with the closed drawer at rest")
        try:
            try:
                self.observe()
            finally:
                r["preflight_checks"] = r["checked_steps"]
                r["observer_entries"] = r["checked_steps"] = r["geometry_checks"] = 0
            self.phase("settle_loaded_carrier", self.home, 0.002, 0.4, step_observer)
            self.phase("expose_loaded_filter", 0.14, 3.5, 1.0, step_observer)
            self.phase("observe_filter_access", 0.14, 0.002, 0.4, step_observer)
            r["observation"] = self.access_observation()
            if (
                not r["observation"]["visible"]
                or not r["observation"]["upward_100mm_rays_clear"]
            ):
                raise RuntimeError("filter_access_not_observed")
            self.phase("return_loaded_filter", self.home, 3.5, 1.0, step_observer)
            _require_binding(self.session)
            r["success"] = True
        except (RuntimeError, ValueError) as exc:
            r["failure"] = str(exc)
            if self.session.data is d and self.session.model is m:
                d.ctrl[self.actuator] = np.clip(
                    d.qpos[self.qadr], self.control.lower, self.control.upper
                )
                r["hold_requested"] = True
        r.update(
            coverage_complete=r["physics_steps"] > 0
            and r["checked_steps"] == r["physics_steps"],
            unchecked_steps=r["physics_steps"] - r["checked_steps"],
            whole_run_clearance_lower_bound_m=(
                r["minimum_noncontact_clearance_lower_bound_m"]
                if r["physics_steps"] > 0 and r["checked_steps"] == r["physics_steps"]
                else None
            ),
            final_qpos=d.qpos.tolist(),
            final_qvel=d.qvel.tolist(),
            final_ctrl=d.ctrl.tolist(),
            measured_return_error_m=float(d.qpos[self.qadr] - self.home),
            elapsed_simulation_s=float(d.time) - self.start_time,
            model_unchanged=_model_signature(m) == self.signature,
        )
        return r


def inspect_eth_filter(session, *, step_observer=None):
    return FilterAccess(session).run(step_observer=step_observer)
