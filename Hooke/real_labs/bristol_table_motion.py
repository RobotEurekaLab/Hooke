"""Source-bound coordinated displacement of the inert EQUALS surrogate.

Eight real leg servos act through articulated links on a free platform. This
module commands only actuator lengths; it never assigns a live platform pose.
The measured operation is low-speed positioning, not earthquake reproduction.
"""

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.bristol_table_geometry import (
    CONTROLS,
    ID,
    LEGS,
    PLATFORM_ORIGIN,
    PREFIX,
    SCENE,
    leg_lengths,
    orientation,
)
from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.rigid_access import require_binding

STATE_FIELDS = (
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


def rotation_vector(vector):
    angle = float(np.linalg.norm(vector))
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(
        quat, np.asarray(vector) / angle if angle else np.array([0.0, 0.0, 1.0]), angle
    )
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3), quat


def commanded_pose(lengths, measured_pose):
    """Recover a compatible command pose without confusing servo error with it.

    Finite compliance can place the actual table slightly beyond a commanded
    endpoint. The next leg starts from the existing *command*, not an illegal
    inverse-kinematic request reconstructed from that displaced measurement.
    A redundant, incompatible eight-value vector is rejected.
    """
    lengths = np.asarray(lengths, dtype=float)
    if (
        lengths.shape != (8,)
        or not np.isfinite(lengths).all()
        or np.max(abs(lengths)) > 0.15
    ):
        raise ValueError("Existing commands must be eight bounded leg lengths")
    pose = np.array(measured_pose, dtype=float)
    for _ in range(12):
        rotation, _ = rotation_vector(pose[3:])
        current = leg_lengths(pose[:3], rotation)
        error = current - lengths
        if np.max(abs(error)) < 1e-13:
            break
        columns = []
        for axis in range(6):
            probe = pose.copy()
            probe[axis] += 1e-6
            rotation, _ = rotation_vector(probe[3:])
            columns.append((leg_lengths(probe[:3], rotation) - current) / 1e-6)
        pose -= np.linalg.lstsq(np.array(columns).T, error, rcond=None)[0]
    # Only numerical solution residue is normalized in this private array.
    # Public requested coordinates, physical joint limits and live state stay
    # strict and unchanged.
    for axis in range(3):
        if abs(abs(pose[axis]) - 0.15) <= 1e-12:
            pose[axis] = math.copysign(0.15, pose[axis])
    rotation, _ = rotation_vector(pose[3:])
    if (
        np.max(abs(leg_lengths(pose[:3], rotation) - lengths)) > 1e-8
        or np.max(abs(pose[:3])) > 0.15
        or np.linalg.norm(pose[3:]) > 0.04
    ):
        raise ValueError(
            "Existing leg commands do not describe a compatible bounded platform pose"
        )
    return pose


def diagnostic_pose(model, data, pose):
    """Set an isolated scratch configuration for geometry screening only.

    Never called with the live session state. The free platform and passive
    bearings are solved analytically, so every spherical constraint closes.
    """
    if data.model is not model:
        raise ValueError("Diagnostic state does not belong to this model")
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Six finite platform-pose values are required")
    if np.max(abs(pose[:3])) > 0.15 or np.linalg.norm(pose[3:]) > 0.04:
        raise ValueError("Platform command is outside the authored inspection envelope")
    rotation, quat = rotation_vector(pose[3:])
    lengths = leg_lengths(pose[:3], rotation)
    if np.max(abs(lengths)) > 0.15 + 1e-12:
        raise ValueError("Coupled pose exceeds at least one genuine 300 mm leg stroke")
    free = model.joint(PREFIX + "platform_free").qposadr[0]
    data.qpos[free : free + 3] = PLATFORM_ORIGIN + pose[:3]
    data.qpos[free + 3 : free + 7] = quat
    for i, (name, anchor, tip) in enumerate(LEGS):
        origin = np.asarray(anchor)
        desired = (
            PLATFORM_ORIGIN
            + pose[:3]
            + rotation @ (np.asarray(tip) - PLATFORM_ORIGIN)
            - origin
        )
        neutral = np.asarray(tip) - origin
        base_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(base_rotation, orientation(neutral))
        local = base_rotation.reshape(3, 3).T @ (desired / np.linalg.norm(desired))
        angles = (
            math.atan2(-local[1], local[2]),
            math.asin(float(np.clip(local[0], -1, 1))),
        )
        if max(map(abs, angles)) > 0.25:
            raise ValueError("Passive angular-bearing travel exceeded")
        for key, value in zip(("pitch", "yaw"), angles):
            data.qpos[model.joint(PREFIX + name + "_" + key).qposadr[0]] = value
        data.qpos[model.joint(ID + "__joint_" + CONTROLS[i]).qposadr[0]] = lengths[i]
    mujoco.mj_forward(model, data)
    return lengths


class TableInspection:
    """Exclusive-session platform positioning with complete moving-pair checks."""

    def __init__(self, session):
        if not math.isfinite(float(session.data.time)):
            raise ValueError("Platform time must be finite")
        self.binding = require_binding(session, SCENE)
        self.session, self.m, self.d = session, session.model, session.data
        if not np.array_equal(self.d.eq_active, self.m.eq_active0):
            raise ValueError("All authored spherical connections must remain active")
        self.signature = _model_signature(self.m)
        self.task_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.controls = [session.controls[ID, key] for key in CONTROLS]
        self.actuators = np.array([c.actuator for c in self.controls])
        self.addresses = np.array([self.m.jnt_qposadr[c.joint] for c in self.controls])
        self.platform = self.m.body(PREFIX + "platform_body").id
        self.masks = self.m.geom_contype.copy(), self.m.geom_conaffinity.copy()
        self.flags = int(self.m.opt.disableflags), int(self.m.opt.enableflags)
        self.eq_active = self.d.eq_active.copy()
        self.cap = 0.002
        self.minimum_required = 0.00025
        self._build_pairs()
        self.reset_counts()

    def _build_pairs(self):
        m = self.m
        weld = m.body_weldid[m.geom_bodyid]
        enabled = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        # Every moving weld group is included, not only the eight driven rods.
        # This includes the free platform, passive barrels and foreign bodies.
        self.all_pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if weld[a] != weld[b] and (weld[a] or weld[b])
            ],
            dtype=int,
        ).reshape(-1, 2)
        reset = mujoco.MjData(m)
        mujoco.mj_forward(m, reset)
        matrices = reset.geom_xmat.reshape(-1, 3, 3)
        center = reset.geom_xpos + np.einsum("nij,nj->ni", matrices, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(matrices), m.geom_aabb[:, 3:])
        lower, upper = center - half, center + half
        known = {0}
        platform_group = int(m.body_weldid[self.platform])
        known.add(platform_group)
        ids = np.flatnonzero(weld == platform_group)
        radius = np.linalg.norm(abs(center[ids] - PLATFORM_ORIGIN) + half[ids], axis=1)
        inflation = 0.17 + 2 * np.sin(0.06 / 2) * radius
        lower[ids] -= inflation[:, None]
        upper[ids] += inflation[:, None]
        self.passive_addresses = []
        for name, anchor, _ in LEGS:
            for suffix in ("barrel", "rod"):
                group = int(m.body_weldid[m.body(PREFIX + name + "_" + suffix).id])
                known.add(group)
                ids = np.flatnonzero(weld == group)
                slide = 0.151 if suffix == "rod" else 0.0
                radius = (
                    np.linalg.norm(abs(center[ids] - anchor) + half[ids], axis=1)
                    + slide
                )
                # Two rotations <=.251 rad have geodesic angle <=.502 rad.
                # A point at radius r moves <=2*r*sin(.502/2), plus slide.
                inflation = slide + 2 * np.sin(0.251) * radius
                lower[ids] -= inflation[:, None]
                upper[ids] += inflation[:, None]
            self.passive_addresses.extend(
                m.joint(PREFIX + name + "_" + key).qposadr[0]
                for key in ("pitch", "yaw")
            )
        unknown = ~np.isin(weld, list(known))
        a, b = self.all_pairs.T
        separation = np.linalg.norm(
            np.maximum(np.maximum(lower[a] - upper[b], lower[b] - upper[a]), 0), axis=1
        )
        self.pairs = self.all_pairs[
            (separation <= self.cap + 1e-9) | unknown[a] | unknown[b]
        ]

    def reset_counts(self):
        self.executed = self.entries = self.checked = self.queries = 0
        self.minimum = self.cap
        self.max_force = self.max_constraint_error = 0.0

    def state(self):
        return {
            **{name: getattr(self.d, name).tolist() for name in STATE_FIELDS},
            "time_s": float(self.d.time),
            "model_signature": self.signature,
        }

    def preflight(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Platform model/state ownership changed")
        require_binding(self.session, SCENE)
        if (
            _model_signature(self.m) != self.signature
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.task_sha
        ):
            raise ValueError("Platform source/model binding changed")
        if not math.isfinite(float(self.d.time)) or not all(
            np.isfinite(getattr(self.d, name)).all() for name in STATE_FIELDS
        ):
            raise ValueError("Platform state must be finite")
        if not np.array_equal(self.d.eq_active, self.eq_active):
            raise ValueError("Authored spherical connections changed")

    def direct_clearance(self, data):
        m = self.m
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap
        )
        minimum, queries = self.cap, 0
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, data, int(a), int(b), self.cap, None)),
                self.cap,
            )
            queries += 1
            minimum = min(minimum, gap)
            if gap < self.minimum_required:
                raise RuntimeError(
                    f"Moving clearance lost: {m.geom(a).name} / {m.geom(b).name} ({gap:.9g} m)"
                )
        return minimum, queries

    def check(self):
        self.entries += 1
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("Platform state binding changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(d.eq_active, self.eq_active)
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("Platform contacts/constraints changed")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("Nonfinite platform motion")
        if np.max(abs(d.qpos[self.addresses])) > 0.151 or np.max(abs(d.qvel)) > 2:
            raise RuntimeError("Platform motion left the bounded physical envelope")
        if (
            np.max(abs(d.qpos[self.passive_addresses])) > 0.251
            or np.max(abs(d.xpos[self.platform] - PLATFORM_ORIGIN)) > 0.17
            or 2 * math.acos(float(np.clip(abs(d.xquat[self.platform, 0]), 0, 1)))
            > 0.06
        ):
            raise RuntimeError(
                "Platform motion left the conservative collision-pruning domain"
            )
        # Connect constraints transfer load through real spherical interfaces.
        # Their finite numerical compliance is measured, never called rigidity.
        error = float(np.max(abs(d.efc_pos[: 3 * m.neq])))
        if error > 0.00025:
            raise RuntimeError("Spherical constraint closure exceeded 250 micrometres")
        minimum, queries = self.direct_clearance(d)
        self.queries += queries
        self.minimum = min(self.minimum, minimum)
        self.max_force = max(
            self.max_force, float(np.max(abs(d.actuator_force[self.actuators])))
        )
        self.max_constraint_error = max(self.max_constraint_error, error)
        self.checked += 1

    def observe(self):
        self.preflight()
        m, d = self.m, self.d
        body = self.platform
        witness = m.site(ID + "__site_physical_witness").id
        # The original workstation view lies outside a cut-away front wall.
        # Use the existing interior camera, whose physical ray actually reaches
        # the target without applying renderer-only visibility exclusions.
        camera = m.camera("interior").id
        origin = d.cam_xpos[camera]
        point = d.site_xpos[witness]
        direction = point - origin
        hit = np.array([-1], np.int32)
        distance = mujoco.mj_ray(
            m,
            d,
            origin,
            direction / np.linalg.norm(direction),
            np.array([1, 1, 1, 0, 0, 0], np.uint8),
            1,
            -1,
            hit,
        )
        patch = m.geom(PREFIX + "inert_registration_patch").id
        local = d.cam_xmat[camera].reshape(3, 3).T @ direction
        depth = -float(local[2])
        focal = 350 / math.tan(math.radians(m.cam_fovy[camera]) / 2)
        pixel = (
            [
                560 + focal * float(local[0]) / depth,
                350 - focal * float(local[1]) / depth,
            ]
            if depth > 0
            else None
        )
        visible = bool(
            pixel and 0 < pixel[0] < 1120 and 0 < pixel[1] < 700 and hit[0] == patch
        )
        return dict(
            platform_translation_m=(d.xpos[body] - PLATFORM_ORIGIN).tolist(),
            platform_quaternion=d.xquat[body].tolist(),
            leg_displacements_m=d.qpos[self.addresses].tolist(),
            witness_world_m=point.tolist(),
            witness_visible_by_center_ray=visible,
            camera="interior",
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[camera].tolist(),
            camera_fovy_deg=float(m.cam_fovy[camera]),
            resolution=[1120, 700],
            witness_pixel=pixel,
            camera_first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            ray_distance_m=float(distance),
            scope="Actual inert surface position and a geometric ray; no pixel-contrast or structural-response claim",
            **self.state(),
        )

    def _hold(self):
        self.d.ctrl[self.actuators] = np.clip(self.d.qpos[self.addresses], -0.15, 0.15)

    def move(self, pose, *, ramp_s=4.0, settle_s=1.0, step_observer=None):
        self.preflight()
        if (
            not math.isfinite(ramp_s)
            or not math.isfinite(settle_s)
            or not 0.05 <= ramp_s <= 30
            or not 0 <= settle_s <= 30
        ):
            raise ValueError("Finite ramp 0.05–30 s and settling 0–30 s are required")
        target = np.asarray(pose, dtype=float)
        scratch = mujoco.MjData(self.m)
        diagnostic_pose(self.m, scratch, target)
        # Observe actual pose, but continue the existing compatible command.
        # Actual endpoint compliance must not become an out-of-stroke target.
        start = self.d.xpos[self.platform] - PLATFORM_ORIGIN
        q = self.d.xquat[self.platform]
        angle = 2 * math.acos(float(np.clip(q[0], -1, 1)))
        rot = (
            q[1:] / np.linalg.norm(q[1:]) * angle
            if np.linalg.norm(q[1:]) > 1e-14
            else np.zeros(3)
        )
        measured = np.r_[start, rot]
        if np.max(abs(measured[:3])) > 0.151 or np.linalg.norm(measured[3:]) > 0.041:
            raise ValueError("Measured start is outside the task envelope")
        beginning = commanded_pose(self.d.ctrl[self.actuators], measured)
        if (
            np.max(abs(beginning[:3] - measured[:3])) > 0.003
            or np.linalg.norm(beginning[3:] - measured[3:]) > 0.002
        ):
            raise ValueError(
                "Actual start has not settled near its existing compatible command"
            )
        # A finite screening grid is a plan check, not the executed coverage.
        for fraction in np.linspace(0, 1, 33):
            candidate = (
                target.copy()
                if fraction == 1
                else beginning + (target - beginning) * fraction
            )
            # Numerical compliance may leave measured endpoints <1 mm outside
            # the command box. Do not clamp live state or silently remap it.
            if fraction == 0:
                continue
            diagnostic_pose(self.m, scratch, candidate)
            self.direct_clearance(scratch)
        self.reset_counts()
        initial = self.state()
        reason = None
        self.check()
        self.reset_counts()
        warnings_before = _warning_state(self.d)
        if np.any(warnings_before[:, 0]):
            raise ValueError("Clear native warnings before starting the inspection")
        physics_events = []
        steps = round((ramp_s + settle_s) / self.m.opt.timestep)
        try:
            for index in range(steps):
                fraction = min(1.0, (index + 1) * self.m.opt.timestep / ramp_s)
                fraction = fraction * fraction * (3 - 2 * fraction)
                requested = (
                    target.copy()
                    if fraction == 1
                    else beginning + (target - beginning) * fraction
                )
                rotation, _ = rotation_vector(requested[3:])
                lengths = leg_lengths(requested[:3], rotation)
                if np.max(abs(lengths)) > 0.15000000001:
                    raise RuntimeError("Coordinated actuator request exceeds stroke")
                self.d.ctrl[self.actuators] = lengths
                before_time = float(self.d.time)
                reset_inputs = {
                    name: getattr(self.d, name).copy()
                    for name in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(self.m, self.d)
                self.executed += 1
                mujoco.mj_forward(self.m, self.d)
                event, warnings_before = _physics_event(
                    self.m, self.d, before_time, warnings_before, reset_inputs
                )
                if event is not None:
                    physics_events.append(dict(executed_step=self.executed, **event))
                    raise RuntimeError(
                        "Native physics warning, reset evidence or time discontinuity during platform motion"
                    )
                self.check()
                if step_observer is not None:
                    before = self.state()
                    step_observer(self)
                    if (
                        before != self.state()
                        or _model_signature(self.m) != self.signature
                    ):
                        raise RuntimeError("Observer mutated platform model/state")
        except (RuntimeError, ValueError) as exc:
            reason = str(exc)
            self._hold()
        observed = self.observe() if reason is None else self.state()
        if reason is None:
            position_error = float(
                np.max(abs(np.asarray(observed["platform_translation_m"]) - target[:3]))
            )
            rotation, _ = rotation_vector(target[3:])
            angular_error = math.acos(
                float(
                    np.clip(
                        (
                            np.trace(
                                rotation.T @ self.d.xmat[self.platform].reshape(3, 3)
                            )
                            - 1
                        )
                        / 2,
                        -1,
                        1,
                    )
                )
            )
            if position_error > 0.003 or angular_error > 0.002:
                reason = "Platform did not settle within the declared 3 mm / 2 mrad inspection tolerance"
                self._hold()
            elif not observed["witness_visible_by_center_ray"]:
                reason = "The actual inert witness is not visible through the physical interior view"
                self._hold()
        else:
            position_error = angular_error = None
        complete = self.executed > 0 and self.checked == self.executed
        native_reset = any(e["automatic_reset_detected"] for e in physics_events)
        reset_unknown = any(e["reset_performed"] is None for e in physics_events)
        reset_performed = True if native_reset else (None if reset_unknown else False)
        return dict(
            status="succeeded" if reason is None else "failed",
            reason=reason,
            requested_pose=target.tolist(),
            initial_state=initial,
            final_state=self.state(),
            observation=observed,
            position_error_m=position_error,
            orientation_error_rad=angular_error,
            direct_clearance=dict(
                executed_steps=self.executed,
                observer_entries=self.entries,
                checked_steps=self.checked,
                unchecked_steps=self.executed - self.checked,
                coverage_complete=complete,
                full_pair_count=len(self.all_pairs),
                retained_pair_count=len(self.pairs),
                near_queries=self.queries,
                capped_clearance_lower_bound_m=self.minimum if self.checked else None,
                whole_leg_clearance_lower_bound_m=(
                    self.minimum if complete and reason is None else None
                ),
                cap_m=self.cap,
            ),
            maximum_actuator_force_N=self.max_force,
            maximum_spherical_constraint_error_m=self.max_constraint_error,
            hold_requested=reason is not None,
            physics_monitor_version=2,
            reset_used=reset_performed,
            reset_performed=reset_performed,
            explicit_reset_performed=False,
            automatic_reset_detected=native_reset,
            time_discontinuity_detected=any(
                not e["time_continuous"] for e in physics_events
            ),
            physics_events=physics_events,
        )


def inspect_bristol_table(session, step_observer=None):
    task = TableInspection(session)
    initial = task.observe()
    phases = []
    route = (
        ("settle", (0, 0, 0, 0, 0, 0)),
        ("positive_x", (0.15, 0, 0, 0, 0, 0)),
        ("negative_x", (-0.15, 0, 0, 0, 0, 0)),
        ("center_x", (0, 0, 0, 0, 0, 0)),
        ("positive_y", (0, 0.15, 0, 0, 0, 0)),
        ("negative_y", (0, -0.15, 0, 0, 0, 0)),
        ("center_y", (0, 0, 0, 0, 0, 0)),
        ("raise", (0, 0, 0.15, 0, 0, 0)),
        ("lower", (0, 0, -0.15, 0, 0, 0)),
        ("center_z", (0, 0, 0, 0, 0, 0)),
        ("coupled_inspection", (0.06, -0.04, 0.025, 0.01, -0.008, 0.012)),
        ("return", (0, 0, 0, 0, 0, 0)),
    )
    for name, pose in route:
        report = task.move(pose, step_observer=step_observer)
        phases.append(dict(name=name, **report))
        if report["status"] != "succeeded":
            break
    native_reset = any(p["automatic_reset_detected"] for p in phases)
    reset_unknown = any(p["reset_performed"] is None for p in phases)
    reset_performed = True if native_reset else (None if reset_unknown else False)
    return dict(
        physics_monitor_version=2,
        reset_used=reset_performed,
        reset_performed=reset_performed,
        explicit_reset_performed=False,
        automatic_reset_detected=native_reset,
        time_discontinuity_detected=any(
            p["time_discontinuity_detected"] for p in phases
        ),
        physics_events=[
            dict(phase=p["name"], **event)
            for p in phases
            for event in p["physics_events"]
        ],
        passed=len(phases) == len(route)
        and all(p["status"] == "succeeded" for p in phases),
        scene_sha256=task.binding["scene_sha256"],
        task_sha256=task.task_sha,
        model_signature=task.signature,
        initial=initial,
        phases=phases,
        final=phases[-1]["observation"] if phases else initial,
        scope="Low-speed cold inert displacement of an authored eight-leg rigid table; no hydraulic/earthquake/structural qualification",
    )
