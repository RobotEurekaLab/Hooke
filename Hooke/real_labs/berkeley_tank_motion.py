"""Source-bound carriage transit and geometric immersion of an inert model.

An exact capsule/box intersection check supplements MuJoCo's distance query:
the original long sting through a solid platform produced a false positive
separation. Water remains visual context; this operation has no fluid forces.
"""

from copy import copy

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding


SCENE = "berkeley_richmond_towing_tank"
EQUIPMENT = "towing_facility"
CONTROLS = ("carriage_travel", "model_immersion")


def capsule_box_gap(model, data, capsule, box):
    """Exact segment/OBB distance minus capsule radius.

    Negative means genuine overlap even when the segment runs through the box.
    Its magnitude is not an exact penetration depth for containment cases.
    The squared segment/AABB distance is quadratic between face crossings.
    """
    if (
        model.geom_type[capsule] != mujoco.mjtGeom.mjGEOM_CAPSULE
        or model.geom_type[box] != mujoco.mjtGeom.mjGEOM_BOX
    ):
        raise ValueError("Expected a capsule and a box")
    rotation = data.geom_xmat[box].reshape(3, 3)
    direction = data.geom_xmat[capsule].reshape(3, 3)[:, 2]
    half_line = model.geom_size[capsule, 1]
    start = rotation.T @ (
        data.geom_xpos[capsule] - direction * half_line - data.geom_xpos[box]
    )
    delta = rotation.T @ (direction * 2 * half_line)
    half = model.geom_size[box]
    knots = [0.0, 1.0]
    for axis in range(3):
        if abs(delta[axis]) > 1e-15:
            for face in (-half[axis], half[axis]):
                fraction = (face - start[axis]) / delta[axis]
                if 0 < fraction < 1:
                    knots.append(float(fraction))
    knots = sorted(set(knots))
    candidates = list(knots)
    for low, high in zip(knots, knots[1:]):
        middle = start + delta * ((low + high) / 2)
        active = abs(middle) > half
        denominator = float(delta[active] @ delta[active])
        if denominator > 0:
            face = np.sign(middle[active]) * half[active]
            fraction = -float(delta[active] @ (start[active] - face)) / denominator
            candidates.append(float(np.clip(fraction, low, high)))
    points = start + np.asarray(candidates)[:, None] * delta
    distance = np.linalg.norm(np.maximum(abs(points) - half, 0), axis=1).min()
    return float(distance - model.geom_size[capsule, 0])


def cylinder_box_gap(model, data, cylinder, box):
    """Separation bound for a cylinder parallel to one box axis.

    The two shapes are Cartesian products of an axial interval and planar
    cross sections. Negative values certify intersection; deep containment
    magnitude is bounded, not an exact penetration depth. Other orientations
    return None instead of claiming an analytic result. Tiny serialized-angle
    errors subtract the rigorous enclosing-sphere rotation displacement bound.
    """
    if (
        model.geom_type[cylinder] != mujoco.mjtGeom.mjGEOM_CYLINDER
        or model.geom_type[box] != mujoco.mjtGeom.mjGEOM_BOX
    ):
        raise ValueError("Expected a cylinder and a box")
    rotation = data.geom_xmat[box].reshape(3, 3)
    axis = rotation.T @ data.geom_xmat[cylinder].reshape(3, 3)[:, 2]
    index = int(np.argmax(abs(axis)))
    aligned = np.zeros(3)
    aligned[index] = np.sign(axis[index])
    axis_error = float(np.linalg.norm(axis - aligned))
    if axis_error > 1e-6:
        return None
    center = rotation.T @ (data.geom_xpos[cylinder] - data.geom_xpos[box])
    half = model.geom_size[box]
    radius, length = model.geom_size[cylinder, :2]
    rotation_bound = float(np.hypot(radius, length) * axis_error)
    radial = (
        np.linalg.norm(np.maximum(np.delete(abs(center) - half, index), 0)) - radius
    )
    axial = abs(center[index]) - half[index] - length
    if max(radial, axial) > 0:
        return float(np.linalg.norm(np.maximum([radial, axial], 0)) - rotation_bound)
    return float(max(radial, axial) - rotation_bound)


class TankClearance:
    """All enabled cross-weld moving pairs, with explicit rail-bearing contact."""

    cap = 0.002
    clearance = 0.00025
    bearing_penetration = 1e-5

    def __init__(self, model, data):
        self.model = model
        bodies = [model.body(f"{EQUIPMENT}__body_{name}").id for name in CONTROLS]
        descendants = set(bodies)
        for body in range(1, model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        groups = model.body_weldid[list(descendants)]
        self.moving = np.isin(model.body_weldid[model.geom_bodyid], groups)
        enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        enabled[np.r_[model.pair_geom1, model.pair_geom2]] = True
        ids = np.flatnonzero(enabled)
        a, b = np.triu_indices(len(ids), 1)
        a, b = ids[a], ids[b]
        keep = (self.moving[a] | self.moving[b]) & (
            model.body_weldid[model.geom_bodyid[a]]
            != model.body_weldid[model.geom_bodyid[b]]
        )
        self.pairs = np.column_stack((a[keep], b[keep]))
        wheels = [i for i in ids if "__carriage_wheel_" in model.geom(i).name]
        rails = [i for i in ids if "__carriage_rail_" in model.geom(i).name]
        if len(wheels) != 4 or len(rails) != 2:
            raise ValueError(
                "The authored carriage requires four enabled wheels and two rails"
            )
        self.bearings = {
            frozenset(
                (
                    wheel,
                    min(
                        rails,
                        key=lambda rail: abs(
                            data.geom_xpos[rail, 0] - data.geom_xpos[wheel, 0]
                        ),
                    ),
                )
            )
            for wheel in wheels
        }
        self.wheels = wheels
        self.queries = self.analytic_queries = 0
        self.cylinder_queries = self.query_disagreements = 0
        self.maximum_query_disagreement = 0.0
        self.minimum_clearance = self.cap
        self.maximum_bearing_penetration = 0.0

    def limit(self, a, b):
        return (
            -self.bearing_penetration
            if frozenset((a, b)) in self.bearings
            else self.clearance
        )

    def check(self, data):
        m = self.model
        if data.model is not m:
            raise RuntimeError("tank_contact_model_state_changed")
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (self.moving[a] or self.moving[b]) and contact.dist < min(
                0, self.limit(a, b)
            ):
                raise RuntimeError(
                    f"tank_native_overlap:{m.geom(a).name}:{m.geom(b).name}"
                )
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        candidates = self.pairs
        for coordinate in range(3):
            a, b = candidates.T
            candidates = candidates[
                abs(center[a, coordinate] - center[b, coordinate])
                <= half[a, coordinate] + half[b, coordinate] + self.cap
            ]
        for a, b in candidates:
            a, b = int(a), int(b)
            gap = float(mujoco.mj_geomDistance(m, data, a, b, self.cap, None))
            raw_gap = gap
            self.queries += 1
            types = (m.geom_type[a], m.geom_type[b])
            if types == (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX):
                gap = min(gap, capsule_box_gap(m, data, a, b))
                self.analytic_queries += 1
            elif types == (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE):
                gap = min(gap, capsule_box_gap(m, data, b, a))
                self.analytic_queries += 1
            cylinder_gap = None
            if types == (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX):
                cylinder_gap = cylinder_box_gap(m, data, a, b)
            elif types == (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
                cylinder_gap = cylinder_box_gap(m, data, b, a)
            if cylinder_gap is not None:
                self.cylinder_queries += 1
                gap = min(self.cap, cylinder_gap)
                difference = abs(gap - raw_gap)
                self.maximum_query_disagreement = max(
                    self.maximum_query_disagreement, difference
                )
                self.query_disagreements += difference > 1e-6
            if frozenset((a, b)) in self.bearings:
                self.maximum_bearing_penetration = max(
                    self.maximum_bearing_penetration, -gap
                )
            else:
                self.minimum_clearance = min(self.minimum_clearance, gap)
            if gap < self.limit(a, b):
                raise RuntimeError(
                    f"tank_direct_clearance:{m.geom(a).name}:{m.geom(b).name}"
                )


class TankTransit:
    """Execute coupled full-stroke movement without state assignment or explicit resets."""

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session, self.model, self.data = session, session.model, session.data
        m, d = self.model, self.data
        self.controls = [session.controls[EQUIPMENT, name] for name in CONTROLS]
        self.qadr = [m.jnt_qposadr[c.joint] for c in self.controls]
        self.dadr = [m.jnt_dofadr[c.joint] for c in self.controls]
        self.actuators = [c.actuator for c in self.controls]
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.site = m.site(EQUIPMENT + "__site_tow_model").id
        self.hull = m.geom(EQUIPMENT + "__berkeley_inert_hull").id
        wheel_joints = [
            m.joint(f"{EQUIPMENT}__berkeley_wheel_{side}_{end}_spin").id
            for side in range(2)
            for end in range(2)
        ]
        self.wheel_qadr = m.jnt_qposadr[wheel_joints]
        self.wheel_dadr = m.jnt_dofadr[wheel_joints]
        self.play_qadr = [
            m.joint(f"{EQUIPMENT}__berkeley_wheel_{side}_{end}_radial_play").qposadr[0]
            for side in range(2)
            for end in range(2)
        ]
        self._preflight()
        probe = copy(d)
        mujoco.mj_forward(m, probe)
        self.clearances = TankClearance(m, probe)
        self.clearances.check(probe)
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
                "jnt_pos",
                "jnt_axis",
                "jnt_range",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
            )
        }
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.initial_ctrl = d.ctrl.copy()
        self.initial_eq = d.eq_active.copy()
        self.held = {
            key: (control, session._measured(control))
            for key, control in session.controls.items()
            if control.actuator not in self.actuators
        }
        self.executed = self.entries = self.checked = 0
        self.physics_events = []
        self.initial_time = float(d.time)
        self.phases, self.telemetry = [], []
        self.wheel_start = d.qpos[self.wheel_qadr].copy()
        self.carriage_start = float(d.qpos[self.qadr[0]])
        self.maximum_slip_speed = 0.0
        self.maximum_slip_distance = 0.0
        self.slip_squared_sum = 0.0
        self.minimum_wheel_normal_load = None
        self.maximum_wheel_normal_load = 0.0
        self.last_wheel_loads = np.zeros(4)

    def _preflight(self):
        if self.session.model is not self.model or self.session.data is not self.data:
            raise ValueError("Tank session model or state was replaced")
        require_binding(self.session, SCENE)
        if not np.array_equal(self.data.eq_active, self.model.eq_active0):
            raise ValueError("Tank equality configuration changed")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.data.qpos,
                self.data.qvel,
                self.data.ctrl,
                self.data.qfrc_applied,
                self.data.xfrc_applied,
                [self.data.time],
            )
        ):
            raise ValueError("Tank inspection requires finite state")

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

    def measured(self):
        return self.data.qpos[self.qadr].copy()

    def _hold(self):
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or self.data.model is not self.model
        ):
            return False
        self.data.ctrl[self.actuators] = np.clip(
            self.measured(), self.lower, self.upper
        )
        return True

    def _step(self, command):
        m, d = self.model, self.data
        d.ctrl[self.actuators] = command
        before_time, before_warnings = float(d.time), _warning_state(d)
        reset_inputs = {
            key: getattr(d, key).copy()
            for key in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        mujoco.mj_step(m, d)
        self.executed += 1
        mujoco.mj_forward(m, d)
        self.entries += 1
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("tank_model_state_changed")
        event, _ = _physics_event(m, d, before_time, before_warnings, reset_inputs)
        if event is not None:
            self.physics_events.append(dict(executed_step=self.executed, **event))
            raise RuntimeError("native_physics_warning_or_time_discontinuity")
        if (
            any(w.number for w in d.warning)
            or not np.isfinite(np.r_[d.qpos, d.qvel]).all()
            or abs(d.time - self.initial_time - self.executed * m.opt.timestep) > 1e-8
        ):
            raise RuntimeError("tank_invalid_physics_or_time")
        if (
            self.flags != (int(m.opt.disableflags), int(m.opt.enableflags))
            or not np.array_equal(d.eq_active, self.initial_eq)
            or any(
                not np.array_equal(getattr(m, key), value)
                for key, value in self.contract.items()
            )
        ):
            raise RuntimeError("tank_geometry_or_contacts_changed")
        expected = self.initial_ctrl.copy()
        expected[self.actuators] = command
        if not np.array_equal(d.ctrl, expected):
            raise RuntimeError("tank_control_changed")
        for key, (control, start) in self.held.items():
            if abs(self.session._measured(control) - start) > (
                0.0005 if control.units == "m" else 0.005
            ):
                raise RuntimeError("tank_held_axis_drift:" + ".".join(key))
        if np.any(abs(self.measured() - command) > [0.08, 0.006]):
            raise RuntimeError("tank_tracking_limit")
        self.clearances.check(d)
        slip_speed = d.qvel[self.dadr[0]] + 0.1 * d.qvel[self.wheel_dadr]
        slip_distance = (
            d.qpos[self.qadr[0]]
            - self.carriage_start
            + 0.1 * (d.qpos[self.wheel_qadr] - self.wheel_start)
        )
        if max(abs(slip_speed)) > 0.001 or max(abs(slip_distance)) > 0.001:
            raise RuntimeError("tank_wheel_rolling_slip_limit")
        if max(abs(d.qpos[self.play_qadr])) > 0.0003:
            raise RuntimeError("tank_wheel_journal_travel_limit")
        loads = np.zeros(4)
        force = np.zeros(6)
        wheel_indices = {g: i for i, g in enumerate(self.clearances.wheels)}
        for index, contact in enumerate(d.contact):
            pair = frozenset((int(contact.geom1), int(contact.geom2)))
            if pair in self.clearances.bearings:
                mujoco.mj_contactForce(m, d, index, force)
                wheel = next(g for g in pair if g in wheel_indices)
                loads[wheel_indices[wheel]] += force[0]
        if min(loads) < 0.1:
            raise RuntimeError("tank_wheel_lost_loaded_rail_contact")
        self.last_wheel_loads = loads
        self.minimum_wheel_normal_load = (
            float(min(loads))
            if self.minimum_wheel_normal_load is None
            else min(self.minimum_wheel_normal_load, float(min(loads)))
        )
        self.maximum_wheel_normal_load = max(
            self.maximum_wheel_normal_load, float(max(loads))
        )
        self.maximum_slip_speed = max(
            self.maximum_slip_speed, float(max(abs(slip_speed)))
        )
        self.maximum_slip_distance = max(
            self.maximum_slip_distance, float(max(abs(slip_distance)))
        )
        self.slip_squared_sum += float(slip_speed @ slip_speed)
        self.checked += 1

    def _phase(self, name, target, duration):
        self._preflight()
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Tank target lies outside the original controls")
        start = self.measured()
        begin, checked = self.executed, self.checked
        event_start = len(self.physics_events)
        record = dict(phase=name, target_m=target.tolist(), success=False)
        self.phases.append(record)
        try:
            count = round(duration / self.model.opt.timestep)
            for index in range(1, count + 1):
                fraction = index / count
                blend = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
                self._step(start + (target - start) * blend)
                if self.executed % 10 == 0:
                    self.telemetry.append(
                        dict(
                            phase=name,
                            time_s=float(self.data.time),
                            command_m=self.data.ctrl[self.actuators].tolist(),
                            measured_m=self.measured().tolist(),
                            qpos=self.data.qpos.tolist(),
                            qvel=self.data.qvel.tolist(),
                            wheel_slip_speed_m_s=(
                                self.data.qvel[self.dadr[0]]
                                + 0.1 * self.data.qvel[self.wheel_dadr]
                            ).tolist(),
                            wheel_normal_load_N=self.last_wheel_loads.tolist(),
                            tow_model_world_m=self.data.site_xpos[self.site].tolist(),
                        )
                    )
            for _ in range(round(2 / self.model.opt.timestep)):
                self._step(target)
                if (
                    np.all(abs(self.measured() - target) <= 0.0002)
                    and np.max(abs(self.data.qvel[self.dadr])) < 0.001
                ):
                    break
            else:
                raise RuntimeError("tank_endpoint_did_not_settle")
            self._preflight()
            record.update(
                success=True,
                endpoint_error_m=(self.measured() - target).tolist(),
                wheel_angle_rad=self.data.qpos[self.wheel_qadr].tolist(),
                tow_model_world_m=self.data.site_xpos[self.site].tolist(),
                hull_top_relative_to_visual_water_m=float(
                    self.data.geom_xpos[self.hull, 2] + 0.12 - 1.67
                ),
            )
        finally:
            record.update(
                executed_steps=self.executed - begin,
                completed_checks=self.checked - checked,
                state=self.state(),
                **combine_reset_reports(
                    self.physics_events[event_start:] or [{"reset_performed": False}]
                ),
            )

    def run(self):
        self._preflight()
        if (
            np.max(abs(self.measured())) > 0.0002
            or np.max(abs(self.data.qvel[self.dadr])) > 0.001
        ):
            raise ValueError("Tank route begins at the measured, stationary home pose")
        mujoco.mj_forward(self.model, self.data)
        initial = self.state()
        failure, hold = None, False
        try:
            for name, target, duration in (
                ("immerse_at_start", (0, 0.35), 12),
                ("full_transit_with_lift", (8, 0.175), 40),
                ("immerse_at_far_end", (8, 0.35), 8),
                ("full_return_with_retraction", (0, 0), 40),
            ):
                self._phase(name, target, duration)
        except (RuntimeError, ValueError) as error:
            failure, hold = str(error), self._hold()
            if self.phases:
                self.phases[-1].update(success=False, failure=failure)
        return dict(
            status="succeeded" if failure is None else "failed",
            failure=failure,
            scene_sha256=self.binding["scene_sha256"],
            initial=initial,
            final=self.state(),
            phases=self.phases,
            telemetry=self.telemetry,
            executed_steps=self.executed,
            observer_entries=self.entries,
            completed_checks=self.checked,
            coverage_complete=self.executed > 0 and self.executed == self.checked,
            direct_pair_count=len(self.clearances.pairs),
            direct_queries=self.clearances.queries,
            analytic_capsule_box_queries=self.clearances.analytic_queries,
            analytic_parallel_cylinder_box_queries=self.clearances.cylinder_queries,
            direct_query_disagreements_over_1um=self.clearances.query_disagreements,
            maximum_direct_query_disagreement_m=self.clearances.maximum_query_disagreement,
            checked_clearance_lower_bound_m=(
                self.clearances.minimum_clearance if self.checked else None
            ),
            whole_route_clearance_lower_bound_m=(
                self.clearances.minimum_clearance
                if self.executed > 0 and self.executed == self.checked
                else None
            ),
            maximum_permitted_wheel_contact_penetration_m=self.clearances.maximum_bearing_penetration,
            passive_wheel_rotation=dict(
                qpos_indices=self.wheel_qadr.tolist(),
                dof_indices=self.wheel_dadr.tolist(),
                radius_m=0.1,
                maximum_checked_slip_speed_m_s=self.maximum_slip_speed,
                rms_checked_slip_speed_m_s=(
                    float(np.sqrt(self.slip_squared_sum / (4 * self.checked)))
                    if self.checked
                    else None
                ),
                maximum_checked_accumulated_slip_m=self.maximum_slip_distance,
                minimum_checked_normal_load_N=self.minimum_wheel_normal_load,
                maximum_checked_normal_load_N=self.maximum_wheel_normal_load,
                maximum_allowed_slip_speed_m_s=0.001,
                maximum_allowed_accumulated_slip_m=0.001,
                minimum_required_normal_load_N=0.1,
                meaning="Slip is carriage velocity plus radius times passive wheel angular velocity; no wheel actuation or kinematic rolling constraint",
            ),
            hold_requested=hold,
            **combine_reset_reports(
                self.physics_events or [{"reset_performed": False}]
            ),
            physics_events=self.physics_events,
            time_discontinuity_detected=any(
                not event["time_continuous"] for event in self.physics_events
            ),
            model_signature_unchanged=_model_signature(self.model)
            in self.binding["signatures"],
            scope="Measured 8 m carriage transit and full 350 mm immersion/retraction of a rigid inert mounted model",
            limitations=[
                "Water is visual only; no buoyancy, drag, waves or hydrodynamic measurements",
                "Carriage, open guide bearings and sting clamp are authored estimates; source photograph does not resolve them",
                "Original controlled-body inertia/drives retained; four passive wheel/shaft bodies add0.4 kg with authored analytic cylinder inertia",
                "Four passive vertical journal-play coordinates let each wheel bear its own gravity load; the ideal carriage guide supports the main carriage",
                "Wheel-local 4 ms contact response and 5 micrometre activation margin stabilize loaded contact at the original 2 ms timestep; shared physics is unchanged",
                "Ideal translation and wheel-hinge constraints remain; rail-load distribution and rolling resistance are not calibrated",
                "Only four identified wheel/rail interfaces permit up to10 micrometres numerical penetration; other moving pairs require0.25 mm separation",
                "A failure requests a measured hold without explicitly resetting or guaranteeing a settled stop",
            ],
        )


def inspect_berkeley_tank(session):
    return TankTransit(session).run()
