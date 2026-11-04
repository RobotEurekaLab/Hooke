"""Collision-aware, measured joint motion for rigid laboratory surrogates.

Planning uses only enabled MuJoCo collision geometry and assumes a stationary
world. A private model expands contact margins for clearance checks; the live
model, joint limits and collision masks stay unchanged. Cartesian goals denote
empty-tool inspection positions, not a grasp or a scientific operation.
"""

from copy import copy
from dataclasses import dataclass, replace
import hashlib
import math

import mujoco
import numpy as np


class PlanningError(ValueError):
    """No admissible path was found within the declared planning budget."""


@dataclass(frozen=True)
class MotionPlan:
    equipment: str
    controls: tuple[str, ...]
    waypoints: tuple[tuple[float, ...], ...]
    model_signature: str
    environment_signature: str
    method: str
    clearance_m: float
    interpolation_steps: tuple[float, ...]
    target_site: str | None = None
    target_position_m: tuple[float, ...] | None = None
    description: str = "Measured joint inspection"


def _model_signature(model):
    """Hash the complete compiled model, including meshes/options/pair filters."""
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return hashlib.sha256(binary).hexdigest()


def _warning_state(data):
    return np.array([(w.number, w.lastinfo) for w in data.warning], dtype=int)


def _physics_event(model, data, before_time, before_warnings, reset_inputs):
    """Observe native reset evidence without intercepting or altering physics.

    A native reset can restore a previous warning count instead of increasing
    it. At time zero it can also leave the expected clock increment. A bad-state
    warning at that clock is ambiguous unless a changed warning or reset-cleared
    input distinguishes it; never turn that ambiguity into a no-reset claim.
    """
    after_time = float(data.time)
    after_warnings = _warning_state(data)
    continuous = bool(
        np.isclose(after_time - before_time, model.opt.timestep, rtol=1e-8, atol=1e-10)
    )
    changed = np.flatnonzero(np.any(after_warnings != before_warnings, axis=1))
    bad = np.array(
        [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
    )
    reset_clock = bool(np.isclose(after_time, model.opt.timestep, rtol=0, atol=1e-10))
    autoreset = not int(model.opt.disableflags) & int(
        mujoco.mjtDisableBit.mjDSBL_AUTORESET
    )
    bad_warning = bool(np.any(after_warnings[bad, 0] > 0))
    cleared = []
    if autoreset and reset_clock and bad_warning:
        cleared = [
            name
            for name, expected in reset_inputs.items()
            if np.any(expected != 0) and not np.any(getattr(data, name) != 0)
        ]
    native_reset = bool(
        autoreset
        and reset_clock
        and bad_warning
        and (
            not continuous
            or np.any(after_warnings[bad] != before_warnings[bad])
            or np.any(after_warnings[:, 0] < before_warnings[:, 0])
            or cleared
        )
    )
    ambiguous = bool(autoreset and reset_clock and bad_warning and not native_reset)
    if continuous and not len(changed) and not native_reset and not ambiguous:
        return None, after_warnings
    reset = True if native_reset else (None if not continuous or ambiguous else False)
    return (
        dict(
            before_time_s=before_time,
            after_time_s=after_time,
            expected_timestep_s=float(model.opt.timestep),
            time_continuous=continuous,
            warning_indices=changed.tolist(),
            warnings_before=before_warnings.tolist(),
            warnings_after=after_warnings.tolist(),
            reset_cleared_inputs=cleared,
            automatic_reset_detected=native_reset,
            reset_performed=reset,
        ),
        after_warnings,
    )


class JointMotion:
    """Plan from measured joints; execution never teleports or silently resets.

    A failed execution requests a position hold at the last measured pose and
    leaves its physical state available for inspection. This is not a certified
    emergency stop. Call the existing session.reset() explicitly when desired.
    """

    def __init__(self, session, equipment, controls, *, clearance_m=0.015):
        self.session = session
        if session.model.nflex or session.model.nplugin:
            raise ValueError("Only rigid, plugin-free mechanisms are supported")
        self.equipment = equipment
        self.names = tuple(controls)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Select distinct position controls")
        self.controls = tuple(session.controls[equipment, name] for name in self.names)
        if len({c.joint for c in self.controls}) != len(self.controls):
            raise ValueError("Selected controls must address distinct joints")
        if any(c.mode != "position" for c in self.controls):
            raise ValueError("Joint motion requires bounded position controls")
        self.clearance = float(clearance_m)
        if not math.isfinite(self.clearance) or self.clearance <= 0:
            raise ValueError("Clearance must be finite and positive")
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.qadr = np.array(
            [session.model.jnt_qposadr[c.joint] for c in self.controls]
        )
        self.dadr = np.array([session.model.jnt_dofadr[c.joint] for c in self.controls])
        self.actuators = np.array([c.actuator for c in self.controls])
        self.resolution = np.array(
            [0.003 if c.units == "m" else 0.02 for c in self.controls]
        )
        self.extension = np.array(
            [0.06 if c.units == "m" else 0.3 for c in self.controls]
        )
        roots = {int(session.model.jnt_bodyid[c.joint]) for c in self.controls}
        self.moving_bodies = set()
        for body in range(session.model.nbody):
            ancestor = body
            while ancestor and ancestor not in roots:
                ancestor = int(session.model.body_parentid[ancestor])
            if ancestor in roots:
                self.moving_bodies.add(body)
        explicit_pair_geoms = set(session.model.pair_geom1) | set(
            session.model.pair_geom2
        )
        self.moving_geoms = {
            geom
            for geom, body in enumerate(session.model.geom_bodyid)
            if body in self.moving_bodies
            and (
                session.model.geom_contype[geom]
                or session.model.geom_conaffinity[geom]
                or geom in explicit_pair_geoms
            )
        }
        self.stationary_geoms = np.array(
            [
                geom
                for geom, body in enumerate(session.model.geom_bodyid)
                if body not in self.moving_bodies
            ],
            dtype=int,
        )
        self.stationary_dofs = np.setdiff1d(np.arange(session.model.nv), self.dadr)
        if np.any(session.model.actuator_actadr[self.actuators] >= 0):
            raise ValueError("Stateful selected actuator dynamics are unsupported")
        if not self.moving_geoms:
            raise ValueError("Selected mechanisms have no enabled collision geometry")
        self._refresh_checker()

    def _environment_signature(self):
        """Bind all nonselected joints and externally supplied simulation inputs."""
        data = self.session.data
        digest = hashlib.sha256()
        arrays = (
            np.delete(data.qpos, self.qadr),
            np.delete(data.qvel, self.dadr),
            np.delete(data.ctrl, self.actuators),
            data.act,
            data.mocap_pos,
            data.mocap_quat,
            data.eq_active,
            data.qfrc_applied,
            data.xfrc_applied,
        )
        for values in arrays:
            digest.update(np.asarray(values).tobytes())
        return digest.hexdigest()

    def _require_stationary_world(self):
        if not math.isfinite(float(self.session.data.time)) or any(
            not np.isfinite(values).all()
            for values in (
                self.session.data.qpos,
                self.session.data.qvel,
                self.session.data.ctrl,
            )
        ):
            raise PlanningError("Planning requires finite simulation state")
        if np.any(np.abs(self.session.data.qvel[self.stationary_dofs]) > 0.01):
            raise PlanningError("Nonselected mechanisms must be stationary")
        for key, control in self.session.controls.items():
            if control.actuator in self.actuators:
                continue
            target = self.session.data.ctrl[control.actuator]
            measured = self.session._measured(control)
            tolerance = 0.0002 if control.units == "m" else 0.003
            if (control.mode == "velocity" and abs(target) > 1e-9) or (
                control.mode == "position" and abs(measured - target) > tolerance
            ):
                raise PlanningError(f"Nonselected control is not holding: {key}")

    def _refresh_checker(self):
        disabled = int(self.session.model.opt.disableflags)
        required = (
            int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
            | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        )
        if disabled & required:
            raise PlanningError(
                "Contacts, constraints, joint limits and actuation must remain enabled"
            )
        signature = _model_signature(self.session.model)
        if hasattr(self, "signature") and signature != self.signature:
            raise PlanningError("Model changed; create a new JointMotion instance")
        self.model = copy(self.session.model)
        self.data = copy(self.session.data)
        # MuJoCo's contact override takes precedence over geom and pair margins.
        # Apply the same clearance in the private override; never alter the live one.
        if int(self.model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_OVERRIDE):
            self.model.opt.o_margin = max(self.model.opt.o_margin, self.clearance)
        # Private positive margins expose near contacts using MuJoCo's own
        # pair filtering, including its parent-body and explicit exclusions.
        indices = list(self.moving_geoms)
        self.model.geom_margin[indices] = np.maximum(
            self.model.geom_margin[indices], self.clearance
        )
        # MuJoCo caches each body's maximum geom margin during compilation.
        # Updating geom_margin alone leaves small separated shapes invisible to
        # broadphase clearance queries. Use actual geom owners, including fixed
        # child bodies welded into an articulated link, and preserve larger margins.
        np.maximum.at(
            self.model.body_margin,
            self.model.geom_bodyid[indices],
            self.model.geom_margin[indices],
        )
        for pair in range(self.model.npair):
            if (
                self.model.pair_geom1[pair] in self.moving_geoms
                or self.model.pair_geom2[pair] in self.moving_geoms
            ):
                self.model.pair_margin[pair] = max(
                    self.model.pair_margin[pair], self.clearance
                )
        self.reference_qpos = self.session.data.qpos.copy()
        self.signature = signature

    def _configuration(self, values):
        q = np.asarray(values, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError("Configuration must contain one finite value per control")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Configuration exceeds the original control limits")
        return q

    def measured(self):
        return self.session.data.qpos[self.qadr].copy()

    def _planning_start(self):
        """Project only floating-point boundary roundoff in the planning copy.

        Live measured joints and explicit requested targets remain untouched.
        A physical excursion outside the original limits still fails preflight.
        """
        measured = self.measured()
        bounded = np.clip(measured, self.lower, self.upper)
        slack = 16 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(self.lower), abs(self.upper)))
        if not np.isfinite(measured).all() or np.any(abs(bounded - measured) > slack):
            raise PlanningError("Measured start exceeds the original control limits")
        return bounded

    def _pose(self, q):
        self.data.qpos[:] = self.reference_qpos
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def configuration_contacts(self, values):
        """Return enabled-geometry clearance violations in the planning copy."""
        q = self._configuration(values)
        self._pose(q)
        return [
            dict(
                geoms=[self.model.geom(c.geom1).name, self.model.geom(c.geom2).name],
                distance_m=float(c.dist),
            )
            for c in self.data.contact
            if c.dist < self.clearance
            and (c.geom1 in self.moving_geoms or c.geom2 in self.moving_geoms)
        ]

    def _edge_clear(self, first, second):
        count = max(1, int(np.ceil(np.max(np.abs(second - first) / self.resolution))))
        return all(
            not self.configuration_contacts(
                first
                if fraction == 0
                else second if fraction == 1 else first + fraction * (second - first)
            )
            for fraction in np.linspace(0, 1, count + 1)
        )

    def plan_joint_target(self, target, *, seed=0, max_nodes=1200):
        """Check direct interpolation, then bounded deterministic RRT if needed."""
        if not 1 <= max_nodes <= 10000:
            raise ValueError("Planning budget must be between 1 and 10000 nodes")
        self._refresh_checker()
        self._require_stationary_world()
        start, goal = self._planning_start(), self._configuration(target)
        if np.max(np.abs(self.session.data.qvel[self.dadr])) > 0.01:
            raise PlanningError("Start joints must be stationary before planning")
        for label, q in (("start", start), ("goal", goal)):
            contacts = self.configuration_contacts(q)
            if contacts:
                raise PlanningError(
                    f"{label} violates collision clearance: {contacts[0]}"
                )
        if self._edge_clear(start, goal):
            path, method = [start, goal], "checked_joint_interpolation"
        else:
            rng = np.random.default_rng(seed)
            nodes, parents = [start], [-1]
            path = None
            for _ in range(max_nodes):
                sample = (
                    goal if rng.random() < 0.2 else rng.uniform(self.lower, self.upper)
                )
                near = int(
                    np.argmin(
                        [
                            np.linalg.norm((q - sample) / (self.upper - self.lower))
                            for q in nodes
                        ]
                    )
                )
                delta = sample - nodes[near]
                fraction = min(
                    1.0, 1 / max(float(np.max(np.abs(delta) / self.extension)), 1e-12)
                )
                candidate = nodes[near] + fraction * delta
                if not self._edge_clear(nodes[near], candidate):
                    continue
                nodes.append(candidate)
                parents.append(near)
                if self._edge_clear(candidate, goal):
                    path = [goal]
                    index = len(nodes) - 1
                    while index >= 0:
                        path.append(nodes[index])
                        index = parents[index]
                    path.reverse()
                    break
            if path is None:
                raise PlanningError("No path found within the collision-search budget")
            # Keep only shortcuts that pass the same full interpolation check.
            reduced = [path[0]]
            index = 0
            while index < len(path) - 1:
                following = next(
                    j
                    for j in range(len(path) - 1, index, -1)
                    if self._edge_clear(path[index], path[j])
                )
                reduced.append(path[following])
                index = following
            path, method = reduced, "rrt_with_checked_shortcuts"
        return MotionPlan(
            self.equipment,
            self.names,
            tuple(tuple(map(float, q)) for q in path),
            self.signature,
            self._environment_signature(),
            method,
            self.clearance,
            tuple(self.resolution),
        )

    def plan_to_point(
        self, site, target, *, fixed_goal_joints=None, seed=0, attempts=16
    ):
        """Solve Cartesian position and optional endpoint joints; orientation is free."""
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target must be a finite world position in metres")
        if not 1 <= attempts <= 100:
            raise ValueError("IK attempts must be between 1 and 100")
        self._refresh_checker()
        site_id = self.model.site(site).id
        if self.model.site_bodyid[site_id] not in self.moving_bodies:
            raise ValueError("Inspection site must belong to the selected mechanism")
        fixed = {}
        for name, value in (fixed_goal_joints or {}).items():
            index = self.names.index(name)
            value = float(value)
            if (
                not math.isfinite(value)
                or not self.lower[index] <= value <= self.upper[index]
            ):
                raise ValueError(
                    "Fixed endpoint joint lies outside the original limits"
                )
            fixed[index] = value
        rng = np.random.default_rng(seed)
        jacobian = np.zeros((3, self.model.nv))
        for attempt in range(attempts):
            q = self.measured() if attempt == 0 else rng.uniform(self.lower, self.upper)
            for index, value in fixed.items():
                q[index] = value
            for _ in range(220):
                self._pose(q)
                error = target - self.data.site_xpos[site_id]
                if np.linalg.norm(error) <= 0.0005:
                    break
                mujoco.mj_jacSite(self.model, self.data, jacobian, None, site_id)
                selected = jacobian[:, self.dadr].copy()
                selected[:, list(fixed)] = 0
                increment = selected.T @ np.linalg.solve(
                    selected @ selected.T + 4e-6 * np.eye(3), error
                )
                increment *= min(
                    1.0, 0.15 / max(float(np.max(np.abs(increment))), 1e-12)
                )
                q = np.clip(q + increment, self.lower, self.upper)
            self._pose(q)
            if np.linalg.norm(target - self.data.site_xpos[site_id]) > 0.0005:
                continue
            try:
                path = self.plan_joint_target(q, seed=seed + attempt)
            except PlanningError:
                continue
            return MotionPlan(
                **{
                    **path.__dict__,
                    "target_site": site,
                    "target_position_m": tuple(map(float, target)),
                    "description": "Collision-checked empty-tool Cartesian inspection approach",
                }
            )
        raise PlanningError(
            "No reachable collision-clear IK goal/path found within budget"
        )

    def _request_hold(self):
        measured = self.measured()
        if not np.isfinite(measured).all():
            return False
        self.session.data.ctrl[self.actuators] = np.clip(
            measured, self.lower, self.upper
        )
        return True

    def execute(self, plan, *, position_tolerance_m=0.005, step_observer=None, speed_scale=1.0):
        """Time-scale a checked path and observe every actual physics step.

        Invalid/stale plans raise before applying controls. Dynamic failure
        returns a failed report and requests a hold without changing qpos/qvel.
        An optional read-only ``step_observer(model, data)`` runs after core
        safety checks on every physics step, including settling. RuntimeError
        from that observer uses the same failure/hold report; it must not edit
        model, state or controls and must not step the simulation.
        ``speed_scale`` reduces command speed and acceleration without changing
        the actuator model, tracking tolerances or collision contract.
        """
        if plan.equipment != self.equipment or plan.controls != self.names:
            raise ValueError("Plan belongs to a different mechanism")
        if plan.clearance_m != self.clearance or tuple(
            plan.interpolation_steps
        ) != tuple(self.resolution):
            raise PlanningError("Plan clearance/interpolation settings differ")
        if plan.model_signature != _model_signature(self.session.model):
            raise PlanningError("Model changed after planning; create a fresh plan")
        if plan.environment_signature != self._environment_signature():
            raise PlanningError("Nonselected world state changed after planning")
        self._require_stationary_world()
        path = [self._configuration(q) for q in plan.waypoints]
        if len(path) < 2:
            raise ValueError("A path needs a start and a goal")
        tolerance = np.array(
            [0.0002 if c.units == "m" else 0.003 for c in self.controls]
        )
        measured_start = self.measured()
        if np.any(np.abs(measured_start - path[0]) > tolerance):
            raise PlanningError("Measured start differs from planned start")
        if np.max(np.abs(self.session.data.qvel[self.dadr])) > 0.01:
            raise PlanningError("Start joints are moving; wait or replan")
        self._refresh_checker()
        if any(not self._edge_clear(a, b) for a, b in zip(path, path[1:])):
            raise PlanningError("Current scene obstructs the previously planned path")
        if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
            raise ValueError("Position tolerance must be finite and positive")
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable")
        speed_scale = float(speed_scale)
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("Speed scale must be finite and in (0, 1]")
        model, data = self.session.model, self.session.data
        site_name = plan.target_site or self.session.equipment[self.equipment].get(
            "sites", {}
        ).get("tcp")
        site_name = site_name or self.session.equipment[self.equipment].get(
            "sites", {}
        ).get("arm_tcp")
        site = model.site(site_name).id if site_name else None
        if site is not None and model.site_bodyid[site] not in self.moving_bodies:
            raise ValueError("Inspection site must belong to the selected mechanism")
        target_position = None
        if plan.target_position_m is not None:
            target_position = np.asarray(plan.target_position_m, dtype=float)
            if (
                site is None
                or target_position.shape != (3,)
                or not np.isfinite(target_position).all()
            ):
                raise ValueError(
                    "Cartesian goal requires a valid tool site and position"
                )
        start_time = float(data.time)
        warnings_before = _warning_state(data)
        physics_events = []
        reset_inputs = {
            name: getattr(data, name).copy()
            for name in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        stationary_positions = data.geom_xpos[self.stationary_geoms].copy()
        stationary_rotations = data.geom_xmat[self.stationary_geoms].copy()
        environment_limit = min(0.001, self.clearance / 4)
        maximum_environment_displacement = 0.0
        telemetry, collisions, checkpoints = [], [], []
        maximum_error = np.zeros(len(self.names))
        tracking_limits = 0.025 * (self.upper - self.lower)
        maximum_speed = np.array(
            [0.03 if c.units == "m" else 0.3 for c in self.controls]
        )
        maximum_acceleration = np.array(
            [0.08 if c.units == "m" else 0.8 for c in self.controls]
        )
        maximum_speed *= speed_scale
        maximum_acceleration *= speed_scale**2
        held = {
            key: c
            for key, c in self.session.controls.items()
            if c.actuator not in self.actuators
        }
        held_targets = {key: float(data.ctrl[c.actuator]) for key, c in held.items()}
        held_drift = {}
        step_count = 0
        checked_steps = 0
        next_record = start_time

        def step(command):
            nonlocal step_count, next_record, maximum_environment_displacement
            nonlocal warnings_before, checked_steps
            before = float(data.time)
            data.ctrl[self.actuators] = command
            reset_inputs["ctrl"][self.actuators] = command
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            step_count += 1
            event, warnings_before = _physics_event(
                model, data, before, warnings_before, reset_inputs
            )
            if event is not None:
                physics_events.append(dict(executed_step=step_count, **event))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite_physics_state")
            if event is not None and not event["time_continuous"]:
                raise RuntimeError("physics_time_discontinuity")
            if event is not None and (
                event["warning_indices"] or event["automatic_reset_detected"]
            ):
                raise RuntimeError("physics_warning")
            if event is not None:
                raise RuntimeError("physics_reset_ambiguous")
            # Bound stationary geometry surface motion, including free carriers.
            # The rotation term bounds any point inside the geom bounding sphere.
            displacement = np.linalg.norm(
                data.geom_xpos[self.stationary_geoms] - stationary_positions, axis=1
            )
            rotation_change = np.linalg.norm(
                data.geom_xmat[self.stationary_geoms] - stationary_rotations, axis=1
            )
            displacement += model.geom_rbound[self.stationary_geoms] * rotation_change
            maximum_environment_displacement = max(
                maximum_environment_displacement, float(np.max(displacement, initial=0))
            )
            if maximum_environment_displacement > environment_limit:
                raise RuntimeError("stationary_world_moved")
            for contact in data.contact:
                if contact.dist < -1e-4 and (
                    contact.geom1 in self.moving_geoms
                    or contact.geom2 in self.moving_geoms
                ):
                    collisions.append(
                        dict(
                            time_s=float(data.time),
                            distance_m=float(contact.dist),
                            geoms=[
                                model.geom(contact.geom1).name,
                                model.geom(contact.geom2).name,
                            ],
                        )
                    )
            if collisions:
                raise RuntimeError("measured_collision")
            error = np.abs(self.measured() - command)
            maximum_error[:] = np.maximum(maximum_error, error)
            if np.any(error > tracking_limits):
                raise RuntimeError("trajectory_tracking_limit")
            for key, control in held.items():
                drift = abs(self.session._measured(control) - held_targets[key])
                if drift > max(1e-9, (control.upper - control.lower) * 0.025):
                    held_drift[f"{key[0]}.{key[1]}"] = drift
                    raise RuntimeError("uncommanded_control_drift")
            if step_observer is not None:
                step_observer(model, data)
            checked_steps += 1
            if data.time >= next_record:
                telemetry.append(
                    dict(
                        time_s=float(data.time),
                        commanded=command.tolist(),
                        measured=self.measured().tolist(),
                        tool_position_m=(
                            None if site is None else data.site_xpos[site].tolist()
                        ),
                    )
                )
                next_record = float(data.time) + 0.02

        failure = None
        hold_requested = False
        try:
            for first, second in zip(path, path[1:]):
                distance = np.abs(second - first)
                duration = max(
                    0.2,
                    float(np.max(1.875 * distance / maximum_speed)),
                    float(np.max(np.sqrt(5.774 * distance / maximum_acceleration))),
                )
                count = max(1, math.ceil(duration / model.opt.timestep))
                for index in range(1, count + 1):
                    fraction = index / count
                    blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
                    step(first + blend * (second - first))
                settled = False
                for _ in range(math.ceil(2 / model.opt.timestep)):
                    step(second)
                    if (
                        np.all(np.abs(self.measured() - second) <= tolerance)
                        and np.max(np.abs(data.qvel[self.dadr])) < 0.01
                    ):
                        settled = True
                        break
                if not settled:
                    raise RuntimeError("waypoint_did_not_settle")
                checkpoints.append(
                    dict(
                        commanded=second.tolist(),
                        measured=self.measured().tolist(),
                        tool_position_m=(
                            None if site is None else data.site_xpos[site].tolist()
                        ),
                    )
                )
            if (
                target_position is not None
                and np.linalg.norm(data.site_xpos[site] - target_position)
                > position_tolerance_m
            ):
                raise RuntimeError("cartesian_goal_not_reached")
        except RuntimeError as error:
            failure = str(error)
            # Preserve the failure pose and velocity. A position hold is a
            # request to the existing servos, not a state reset or stop guarantee.
            hold_requested = self._request_hold()
        final_position = None if site is None else data.site_xpos[site].copy()
        native_reset = any(e["automatic_reset_detected"] for e in physics_events)
        reset_unknown = any(e["reset_performed"] is None for e in physics_events)
        return dict(
            physics_monitor_version=2,
            status="failed" if failure else "succeeded",
            failure=failure,
            hold_requested=hold_requested,
            reset_performed=(
                True if native_reset else (None if reset_unknown else False)
            ),
            explicit_reset_performed=False,
            automatic_reset_detected=native_reset,
            time_discontinuity_detected=any(
                not e["time_continuous"] for e in physics_events
            ),
            physics_events=physics_events,
            equipment=self.equipment,
            controls=list(self.names),
            tool_site=site_name,
            model_signature=plan.model_signature,
            environment_signature=plan.environment_signature,
            method=plan.method,
            interpolation_steps=list(plan.interpolation_steps),
            planned_waypoints=[list(q) for q in plan.waypoints],
            measured_start_joint_positions=measured_start.tolist(),
            planned_start_delta=(path[0] - measured_start).tolist(),
            maximum_stationary_geometry_displacement_m=maximum_environment_displacement,
            stationary_geometry_displacement_limit_m=environment_limit,
            simulated_seconds=float(data.time) - start_time,
            physics_steps=step_count,
            checked_steps=checked_steps,
            coverage_complete=step_count > 0 and checked_steps == step_count,
            collision_sample_period_s=float(model.opt.timestep),
            speed_scale=speed_scale,
            clearance_planning_m=self.clearance,
            contact_penetration_threshold_m=1e-4,
            collisions=collisions,
            held_control_drift=held_drift,
            maximum_joint_tracking_error=maximum_error.tolist(),
            tracking_limits=tracking_limits.tolist(),
            checkpoints=checkpoints,
            telemetry=telemetry,
            final_joint_positions=self.measured().tolist(),
            final_tool_position_m=(
                None if final_position is None else final_position.tolist()
            ),
            target_position_m=(
                None if target_position is None else target_position.tolist()
            ),
            cartesian_error_m=(
                None
                if target_position is None
                else float(np.linalg.norm(final_position - target_position))
            ),
            limitations=[
                "Enabled collision proxies only; visual-only surfaces are not certified",
                "Discrete collision sampling, stationary-world planning and joint PD execution",
                "No concurrent model edits or external control/state writers are supported",
                "Sub-millimetre stationary geometry settling is measured; moving obstacles are unsupported",
                "No grasp, object transfer, sensor-quality or scientific-process validation",
            ],
        )


_ARM_CONTROLS = (
    "base_yaw",
    "shoulder",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "tool_yaw",
)


def inspect_point_and_return(motion, site, target, *, fixed_goal_joints=None):
    """Measure an empty-tool approach, then freshly plan back to its starting pose.

    An invalid outward plan raises before motion. A failed return plan is
    reported in place and requests a measured hold, without an implicit reset.
    The caller exclusively owns the session throughout planning/execution.
    """
    session = motion.session
    initial = motion.measured()
    site_id = session.model.site(site).id
    initial_tool = session.data.site_xpos[site_id].copy()
    outward_plan = motion.plan_to_point(
        site, target, fixed_goal_joints=fixed_goal_joints
    )
    outward = motion.execute(outward_plan)
    returning = None
    if outward["status"] == "succeeded":
        try:
            return_plan = replace(
                motion.plan_joint_target(initial),
                target_site=site,
                target_position_m=tuple(map(float, initial_tool)),
                description="Return the requested inspection site to its measured start",
            )
            returning = motion.execute(return_plan)
        except PlanningError as error:
            returning = dict(
                status="failed",
                failure=f"return_planning: {error}",
                hold_requested=motion._request_hold(),
                reset_performed=False,
                final_tool_position_m=session.data.site_xpos[site_id].tolist(),
            )
    returned = returning is not None and returning["status"] == "succeeded"
    return dict(
        status="succeeded" if returned else "failed",
        target_frame="world",
        target_position_m=list(map(float, target)),
        initial_joint_positions=initial.tolist(),
        initial_tool_position_m=initial_tool.tolist(),
        outward=outward,
        returning=returning,
        returned_to_start=returned,
        final_tool_return_error_m=(
            None
            if returning is None
            else float(
                np.linalg.norm(
                    np.asarray(returning["final_tool_position_m"]) - initial_tool
                )
            )
        ),
        grasp_attempted=False,
        object_transfer_validated=False,
        scientific_process_validated=False,
        native_runtime_validated=False,
    )


def inspect_waterloo_handoff(session):
    """Run the source-bound complete-arm handoff inspection and measured return."""
    from real_labs.robot_arm_motion import inspect_waterloo_handoff as inspect

    return inspect(session)


def inspect_nasa_cache(session, *, cache=3):
    """Legacy fixed-base inspection; grounded scenes require the rover operation."""
    from real_labs.robot_arm_motion import inspect_nasa_cache as inspect

    return inspect(session, cache=cache)
