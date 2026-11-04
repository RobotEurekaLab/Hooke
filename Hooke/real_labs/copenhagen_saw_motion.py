"""Measured, force-limited contact with an unpowered saw and retained dummy.

This is a version-2 mechanical registration operation. The original separated
carriage has no coordinate migration. Nothing here models cutting or ice.
"""

from copy import copy

import mujoco
import numpy as np

from real_labs.copenhagen_saw_geometry import PREFIX
from real_labs.rigid_access import require_binding
from real_labs.motion import _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports

SCENE = "copenhagen_pice_ice_workroom"
EQUIPMENT = "ice_station"
CONTROLS = ("wheel_guard", "core_feed_v2")


def core_blade_gap(model, data, core, blade):
    """Analytic separation for the finite aligned cylinder and blade box.

    At tangency the native cylinder/box query can return a false deep overlap.
    These particular shapes are Cartesian products of axial intervals and a
    circle/rectangle cross-section. Negative values certify intersection; deep
    containment magnitude is not an exact penetration depth. Serialized angle
    error is bounded by the displacement of the cylinder's enclosing sphere.
    """
    rotation = data.geom_xmat[blade].reshape(3, 3)
    axis = rotation.T @ data.geom_xmat[core].reshape(3, 3)[:, 2]
    aligned = np.array([np.sign(axis[0]), 0, 0])
    error = np.linalg.norm(axis - aligned)
    if error > 1e-6:
        raise RuntimeError("saw_core_blade_alignment_changed")
    center = rotation.T @ (data.geom_xpos[core] - data.geom_xpos[blade])
    radius, half_length = model.geom_size[core, :2]
    half = model.geom_size[blade]
    axial = abs(center[0]) - half[0] - half_length
    radial = np.linalg.norm(np.maximum(abs(center[1:]) - half[1:], 0)) - radius
    gap = (
        np.linalg.norm(np.maximum([axial, radial], 0))
        if max(axial, radial) > 0
        else max(axial, radial)
    )
    return float(gap - np.hypot(radius, half_length) * error)


class SawClearance:
    """Every enabled cross-weld pair, including parent-filtered geometry."""

    cap = 0.002
    minimum = 0.00025
    contact_penetration = 0.0001

    def __init__(self, model):
        self.model = model
        self.core = model.geom(PREFIX + "inert_core").id
        self.blade = model.geom(PREFIX + "stationary_blade").id
        self.allowed = frozenset((self.core, self.blade))
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        enabled[np.r_[model.pair_geom1, model.pair_geom2]] = True
        ids = np.flatnonzero(enabled)
        a, b = np.triu_indices(len(ids), 1)
        a, b = ids[a], ids[b]
        weld = model.body_weldid[model.geom_bodyid]
        keep = weld[a] != weld[b]
        self.pairs = np.column_stack((a[keep], b[keep]))
        self.queries = self.native_disagreements = 0
        self.minimum_seen = self.cap
        self.maximum_penetration = self.maximum_force = 0.0
        self.last = {}

    def check(self, data):
        m = self.model
        if data.model is not m:
            raise RuntimeError("saw_contact_model_state_changed")
        gap = core_blade_gap(m, data, self.core, self.blade)
        if gap < -self.contact_penetration:
            raise RuntimeError("saw_core_blade_penetration_limit")
        normal = tangential = 0.0
        on_core = np.zeros(3)
        for index, contact in enumerate(data.contact):
            pair = frozenset((int(contact.geom1), int(contact.geom2)))
            if pair == self.allowed:
                if contact.dist < -self.contact_penetration:
                    raise RuntimeError("saw_native_contact_penetration_limit")
                force = np.zeros(6)
                mujoco.mj_contactForce(m, data, index, force)
                normal += force[0]
                tangential += np.linalg.norm(force[1:3])
                world = contact.frame.reshape(3, 3).T @ force[:3]
                on_core += world if contact.geom2 == self.core else -world
            elif contact.dist < -1e-8:
                raise RuntimeError(
                    f"saw_forbidden_contact:{m.geom(contact.geom1).name}:"
                    f"{m.geom(contact.geom2).name}"
                )
        if not np.isfinite(np.r_[normal, tangential, on_core]).all():
            raise RuntimeError("saw_nonfinite_contact_response")
        if normal > 1.0:
            raise RuntimeError("saw_contact_force_limit")
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        near = self.pairs
        for coordinate in range(3):
            a, b = near.T
            near = near[
                abs(center[a, coordinate] - center[b, coordinate])
                <= half[a, coordinate] + half[b, coordinate] + self.cap
            ]
        current = self.cap
        for a, b in near:
            a, b = int(a), int(b)
            raw = float(mujoco.mj_geomDistance(m, data, a, b, self.cap, None))
            self.queries += 1
            if frozenset((a, b)) == self.allowed:
                self.native_disagreements += (
                    abs(min(raw, self.cap) - min(gap, self.cap)) > 1e-6
                )
                continue
            current = min(current, raw)
            if raw < self.minimum:
                raise RuntimeError(
                    f"saw_direct_clearance:{m.geom(a).name}:{m.geom(b).name}"
                )
        self.minimum_seen = min(self.minimum_seen, current)
        self.maximum_penetration = max(self.maximum_penetration, -gap)
        self.maximum_force = max(self.maximum_force, normal)
        self.last = dict(
            core_blade_gap_m=gap,
            normal_force_N=float(normal),
            summed_tangential_force_magnitudes_N=float(tangential),
            force_on_core_world_N=on_core.tolist(),
        )
        return self.last


class SawRegistration:
    """Actual physics stepping; no qpos writes, reset, weld or collision bypass."""

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session, self.model, self.data = session, session.model, session.data
        m, d = self.model, self.data
        self.controls = [session.controls[EQUIPMENT, key] for key in CONTROLS]
        self.qadr = [m.jnt_qposadr[c.joint] for c in self.controls]
        self.dadr = [m.jnt_dofadr[c.joint] for c in self.controls]
        self.actuators = [c.actuator for c in self.controls]
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.initial_ctrl, self.initial_eq = d.ctrl.copy(), d.eq_active.copy()
        self.held = {
            key: (c, session._measured(c))
            for key, c in session.controls.items()
            if c.actuator not in self.actuators
        }
        self.contract = {
            key: getattr(m, key).copy()
            for key in (
                "geom_contype",
                "geom_conaffinity",
                "geom_pos",
                "geom_quat",
                "geom_size",
                "body_pos",
                "body_quat",
                "body_mass",
                "body_inertia",
                "body_gravcomp",
                "dof_damping",
                "dof_armature",
                "geom_friction",
                "geom_solref",
                "geom_solimp",
                "jnt_pos",
                "jnt_axis",
                "jnt_range",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
            )
        }
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.executed = self.entries = self.checked = 0
        self.initial_time = float(d.time)
        self.phases, self.telemetry = [], []
        self.physics_events = []
        self._preflight()
        probe = copy(d)
        mujoco.mj_forward(m, probe)
        self.clearance = SawClearance(m)
        self.clearance.check(probe)

    def _preflight(self):
        if self.session.model is not self.model or self.session.data is not self.data:
            raise ValueError("Saw session model or state was replaced")
        require_binding(self.session, SCENE)
        if not np.array_equal(self.data.eq_active, self.model.eq_active0):
            raise ValueError("Saw equality configuration changed")
        if not all(
            np.isfinite(v).all()
            for v in (
                self.data.qpos,
                self.data.qvel,
                self.data.ctrl,
                self.data.qfrc_applied,
                self.data.xfrc_applied,
                [self.data.time],
            )
        ):
            raise ValueError("Saw operation requires finite state")

    def measured(self):
        return self.data.qpos[self.qadr].copy()

    def state(self):
        return dict(
            **{
                key: getattr(self.data, key).tolist()
                for key in (
                    "qpos",
                    "qvel",
                    "ctrl",
                    "act",
                    "qacc_warmstart",
                    "qfrc_applied",
                    "xfrc_applied",
                    "eq_active",
                    "mocap_pos",
                    "mocap_quat",
                )
            },
            time_s=float(self.data.time),
        )

    def _step(self, command):
        m, d = self.model, self.data
        d.ctrl[self.actuators] = command
        before_time = float(d.time)
        before_warnings = _warning_state(d)
        reset_inputs = {
            key: getattr(d, key).copy()
            for key in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        mujoco.mj_step(m, d)
        self.executed += 1
        mujoco.mj_forward(m, d)
        self.entries += 1
        event, _ = _physics_event(m, d, before_time, before_warnings, reset_inputs)
        if event is not None:
            self.physics_events.append(dict(executed_step=self.executed, **event))
            raise RuntimeError("saw_native_physics_warning_or_time_discontinuity")
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("saw_model_state_changed")
        if (
            any(w.number for w in d.warning)
            or not np.isfinite(
                np.r_[d.qpos, d.qvel, d.qfrc_applied, d.xfrc_applied.ravel()]
            ).all()
            or abs(d.time - self.initial_time - self.executed * m.opt.timestep) > 1e-8
        ):
            raise RuntimeError("saw_invalid_physics_or_time")
        if (
            self.flags != (int(m.opt.disableflags), int(m.opt.enableflags))
            or not np.array_equal(d.eq_active, self.initial_eq)
            or any(
                not np.array_equal(getattr(m, key), value)
                for key, value in self.contract.items()
            )
        ):
            raise RuntimeError("saw_geometry_or_contacts_changed")
        expected = self.initial_ctrl.copy()
        expected[self.actuators] = command
        if not np.array_equal(d.ctrl, expected):
            raise RuntimeError("saw_control_changed")
        for key, (control, start) in self.held.items():
            tolerance = 0.0005 if control.units == "m" else 0.005
            if abs(self.session._measured(control) - start) > tolerance:
                raise RuntimeError("saw_held_axis_drift:" + ".".join(key))
        if np.any(self.measured() < self.lower - 1e-5) or np.any(
            self.measured() > self.upper + 1e-5
        ):
            raise RuntimeError("saw_measured_joint_limit")
        if np.any(abs(self.measured() - command) > [0.025, 0.003]):
            raise RuntimeError("saw_tracking_limit")
        contact = self.clearance.check(d)
        if contact["normal_force_N"] > 0.01 and abs(self.measured()[0]) > 0.005:
            raise RuntimeError("saw_contact_requires_closed_guard")
        self.checked += 1

    def _phase(self, name, target, duration, *, contact=False):
        self._preflight()
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Saw target lies outside the versioned controls")
        start = self.measured()
        begin, checked, entries = self.executed, self.checked, self.entries
        record = dict(phase=name, command_rad_m=target.tolist(), success=False)
        self.phases.append(record)
        try:
            count = round(duration / self.model.opt.timestep)
            for index in range(1, count + round(1 / self.model.opt.timestep) + 1):
                fraction = min(index / count, 1)
                blend = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
                self._step(start + (target - start) * blend)
                if self.executed % 10 == 0:
                    self.telemetry.append(
                        dict(
                            phase=name,
                            time_s=float(self.data.time),
                            qpos=self.data.qpos.tolist(),
                            qvel=self.data.qvel.tolist(),
                            command_rad_m=self.data.ctrl[self.actuators].tolist(),
                            **self.clearance.last,
                        )
                    )
            endpoint = self.clearance.last
            if np.max(abs(self.data.qvel[self.dadr])) > 0.0001:
                raise RuntimeError("saw_endpoint_not_settled")
            if abs(self.measured()[0] - target[0]) > 0.001:
                raise RuntimeError("saw_guard_endpoint_error")
            if contact:
                # Success is a measured opposing response, not the servo target.
                feed_axis = self.data.xaxis[self.controls[1].joint]
                opposing = -float(np.dot(endpoint["force_on_core_world_N"], feed_axis))
                if not (
                    0.1 <= endpoint["normal_force_N"] <= 0.3
                    and opposing >= 0.1
                    and -self.clearance.contact_penetration
                    <= endpoint["core_blade_gap_m"]
                    <= 1e-7
                ):
                    raise RuntimeError("saw_registration_contact_not_established")
            elif (
                abs(self.measured()[1] - target[1]) > 0.0001
                or endpoint["normal_force_N"] > 0.01
            ):
                raise RuntimeError("saw_free_endpoint_error_or_contact")
            self._preflight()
            record.update(success=True, **endpoint)
        finally:
            n, c = self.executed - begin, self.checked - checked
            record.update(
                executed_steps=n,
                observer_entries=self.entries - entries,
                completed_checks=c,
                coverage_complete=n > 0 and n == c,
                state=self.state(),
            )

    def run(self):
        self._preflight()
        home = self.measured()
        if (
            np.any(home < self.lower - 1e-6)
            or np.any(home > self.upper + 1e-6)
            or abs(home[1]) > 0.0001
            or np.max(abs(self.data.qvel[self.dadr])) > 0.0001
        ):
            raise ValueError(
                "Saw registration begins with the stationary retracted feed"
            )
        initial = self.state()
        failure, hold = None, False
        try:
            self._phase("close_guard", [0, 0], 5)
            self._phase("approach_standoff", [0, 0.150], 10)
            self._phase("register_stationary_blade", [0, 0.152], 2, contact=True)
            self._phase("retract_core", [0, 0], 10)
            self._phase("return_guard", home, 5)
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            if (
                self.session.model is self.model
                and self.session.data is self.data
                and self.data.model is self.model
            ):
                self.data.ctrl[self.actuators] = np.clip(
                    self.measured(), self.lower, self.upper
                )
                hold = True
            if self.phases:
                self.phases[-1].update(success=False, failure=failure)
        complete = self.executed > 0 and self.checked == self.executed
        reset = combine_reset_reports(
            self.physics_events or [{"reset_performed": False}]
        )
        return dict(
            status="succeeded" if failure is None else "failed",
            failure=failure,
            scene_sha256=self.binding["scene_sha256"],
            interface_revision=2,
            initial=initial,
            final=self.state(),
            phases=self.phases,
            telemetry=self.telemetry,
            executed_steps=self.executed,
            observer_entries=self.entries,
            completed_checks=self.checked,
            coverage_complete=complete,
            direct_pair_count=len(self.clearance.pairs),
            direct_queries=self.clearance.queries,
            whole_route_unintended_clearance_lower_bound_m=(
                self.clearance.minimum_seen if complete else None
            ),
            checked_subset_unintended_clearance_lower_bound_m=(
                self.clearance.minimum_seen if self.checked else None
            ),
            maximum_core_blade_penetration_m=self.clearance.maximum_penetration,
            maximum_contact_normal_force_N=self.clearance.maximum_force,
            native_cylinder_query_disagreements_over_1um=self.clearance.native_disagreements,
            failure_hold_requested=hold,
            state_reset=reset["reset_performed"],
            **reset,
            physics_events=self.physics_events,
            time_discontinuity_detected=any(
                not event["time_continuous"] for event in self.physics_events
            ),
            limits=dict(
                unintended_clearance_m=self.clearance.minimum,
                allowed_core_blade_penetration_m=self.clearance.contact_penetration,
                maximum_contact_force_N=1.0,
            ),
            scope="Rigid retained dummy registration against a stationary unpowered blade; no cutting, fracture, blade drive, ice properties or scientific calibration. Servo overtravel commands compliance, not blade penetration success.",
        )


def register_copenhagen_core(session):
    """Close guard, approach, measure gentle contact, retract and return guard."""
    return SawRegistration(session).run()
