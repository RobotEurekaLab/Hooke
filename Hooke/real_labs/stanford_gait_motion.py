"""Measured unloaded roller indexing and lift inspection, not treadmill gait.

Velocity servos index the original independent rollers using measured feedback.
Stationary belt decks do not convey a payload. Ideal guides and motors are not
hardware calibration or safety-rated stops; the caller owns the session.
"""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding

SCENE = "stanford_biomechatronics_gait"
EQUIPMENT = "gait_test_rig"
CONTROLS = ("drive_left", "drive_right", "fixture_lift")


def full_state(session):
    d = session.data
    return dict(
        time_s=float(d.time),
        **{
            k: getattr(d, k).tolist()
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "eq_active",
            )
        },
    )


class GaitInspection:
    """All enabled cross-weld pairs retained, including filtered parent contacts."""

    cap = 0.003
    minimum = 0.0002

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session, self.m, self.d = session, session.model, session.data
        m, d = self.m, self.d
        if not np.isfinite(d.time):
            raise ValueError("Gait inspection requires a finite simulation clock")
        self.controls = [session.controls[EQUIPMENT, k] for k in CONTROLS]
        self.q = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.v = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        self.a = np.array([c.actuator for c in self.controls])
        moving = {m.body_weldid[m.jnt_bodyid[c.joint]] for c in self.controls}
        explicit = set(m.pair_geom1) | set(m.pair_geom2)
        geoms = [
            i
            for i in range(m.ngeom)
            if m.geom_contype[i] or m.geom_conaffinity[i] or i in explicit
        ]
        self.pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(geoms)
                for b in geoms[i + 1 :]
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
        self.warnings, self.eq = d.warning.number.copy(), d.eq_active.copy()
        self.time, self.dt = float(d.time), float(m.opt.timestep)
        self.signature = _model_signature(m)
        self.report = dict(
            success=False,
            failure=None,
            physics_steps=0,
            observer_entries=0,
            geometry_checks=0,
            completed_checks=0,
            direct_queries=0,
            retained_pair_count=len(self.pairs),
            pairs_pruned=0,
            checked_clearance_lower_bound_m=None,
            whole_run_clearance_lower_bound_m=None,
            phases=[],
            telemetry=[],
            hold_requested=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            physics_events=[],
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_state=full_state(session),
        )

    def owned(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Gait inspection model/state ownership changed")

    def distances(self):
        m, d = self.m, self.d
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = np.flatnonzero(
            np.linalg.norm(
                np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap
        )
        gaps = np.full(len(self.pairs), self.cap)
        for index in near:
            a, b = self.pairs[index]
            gaps[index] = min(
                self.cap,
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap, None)),
            )
        return gaps, len(near)

    def observe(self, reference, active, held_q, held_ctrl):
        self.owned()
        m, d, r = self.m, self.d, self.report
        r["observer_entries"] += 1
        if self.flags != (int(m.opt.disableflags), int(m.opt.enableflags)) or not all(
            np.array_equal(a, b)
            for a, b in zip(self.masks, (m.geom_contype, m.geom_conaffinity))
        ):
            raise RuntimeError("contact_configuration_changed")
        if not all(
            np.isfinite(a).all()
            for a in (d.qpos, d.qvel, d.ctrl, d.qfrc_applied, d.xfrc_applied)
        ):
            raise RuntimeError("nonfinite_physics_state")
        if not np.array_equal(self.warnings, d.warning.number) or not np.array_equal(
            self.eq, d.eq_active
        ):
            raise RuntimeError("physics_warning_or_equality_changed")
        if abs(d.time - self.time - r["physics_steps"] * self.dt) > 1e-7:
            raise RuntimeError("physics_time_discontinuity")
        if not -0.14002 <= d.qpos[self.q[2]] <= 0.14002:
            raise RuntimeError("lift_limit_violation")
        error = abs(d.qpos[self.q] - reference)
        if np.any(error > np.array([0.02, 0.02, 0.012])) or np.any(
            abs(d.qvel[self.v]) > np.array([1.0, 1.0, 0.075])
        ):
            raise RuntimeError("trajectory_tracking_or_velocity_limit")
        inactive = np.array([i for i in range(3) if i not in active], dtype=int)
        if np.any(d.ctrl[self.a[inactive]] != held_ctrl[inactive]) or np.any(
            abs(d.qpos[self.q[inactive]] - held_q[inactive])
            > np.array([0.001, 0.001, 0.0002])[inactive]
        ):
            raise RuntimeError("inactive_axis_changed_or_drifted")
        gaps, queries = self.distances()
        r["geometry_checks"] += 1
        r["direct_queries"] += queries
        bad = np.flatnonzero(gaps < self.minimum)
        if len(bad):
            a, b = self.pairs[bad[0]]
            raise RuntimeError(
                f"direct_clearance_lost:{m.geom(int(a)).name}:{m.geom(int(b)).name}"
            )
        gap = float(min(gaps))
        old = r["checked_clearance_lower_bound_m"]
        r["checked_clearance_lower_bound_m"] = gap if old is None else min(gap, old)
        r["completed_checks"] += 1
        if r["physics_steps"] % max(1, round(0.02 / self.dt)) == 0:
            r["telemetry"].append(
                dict(
                    time_s=float(d.time),
                    q=d.qpos[self.q].tolist(),
                    velocity=d.qvel[self.v].tolist(),
                    command=d.ctrl[self.a].tolist(),
                    clearance_lower_bound_m=gap,
                )
            )

    def phase(self, name, target, active, ramp=4.0, settle=1.0):
        self.owned()
        require_binding(self.session, SCENE)
        m, d, r = self.m, self.d, self.report
        if not np.isfinite(d.time):
            raise ValueError("Gait inspection requires a finite simulation clock")
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
            or abs(target[2]) > 0.14
        ):
            raise ValueError("Invalid inert inspection target")
        start = d.qpos[self.q].copy()
        held_ctrl = d.ctrl[self.a].copy()
        if np.any(
            abs(
                target[[i for i in range(3) if i not in active]]
                - start[[i for i in range(3) if i not in active]]
            )
            > 1e-7
        ):
            raise ValueError("Inactive axes may not receive a new target")
        n, hold = max(1, round(ramp / self.dt)), round(settle / self.dt)
        record = dict(
            name=name,
            target=target.tolist(),
            active_controls=[CONTROLS[i] for i in active],
            success=False,
            executed_steps=0,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
        )
        r["phases"].append(record)
        before = {
            k: r[k] for k in ("observer_entries", "geometry_checks", "completed_checks")
        }
        try:
            for step in range(n + hold):
                self.owned()
                t = min(1, (step + 1) / n)
                blend = t**3 * (10 - 15 * t + 6 * t * t)
                speed = 30 * t * t * (1 - t) ** 2 / (n * self.dt) if step < n else 0
                reference = start + (target - start) * blend
                for i in active:
                    if i < 2:
                        command = (target[i] - start[i]) * speed * 1.01 + 4 * (
                            reference[i] - d.qpos[self.q[i]]
                        )
                        if abs(command) > 0.85:
                            raise RuntimeError("bounded_inspection_speed_exceeded")
                        d.ctrl[self.a[i]] = command
                    else:
                        d.ctrl[self.a[i]] = reference[i]
                if (
                    not np.isfinite(d.qfrc_applied).all()
                    or not np.isfinite(d.xfrc_applied).all()
                ):
                    raise RuntimeError("nonfinite_applied_force")
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
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, reset_inputs
                )
                if event is not None:
                    r["physics_events"].append(
                        dict(executed_step=r["physics_steps"], **event)
                    )
                    r.update(combine_reset_reports(r["physics_events"]))
                    record.update(combine_reset_reports([event]), physics_event=event)
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                self.observe(reference, active, start, held_ctrl)
            if np.any(
                abs(d.qpos[self.q] - target) > np.array([0.0002, 0.0002, 0.0001])
            ) or np.any(abs(d.qvel[self.v]) > np.array([0.0003, 0.0003, 0.0002])):
                raise RuntimeError("inspection_endpoint_not_reached_or_stopped")
            for i in active:
                if i < 2:
                    d.ctrl[self.a[i]] = 0
            record["success"] = True
        finally:
            record.update({k: r[k] - v for k, v in before.items()})
            record["coverage_complete"] = (
                record["executed_steps"] > 0
                and record["executed_steps"] == record["completed_checks"]
            )
            record["state"] = full_state(self.session)
            record["measured_displacement"] = (d.qpos[self.q] - start).tolist()

    def execute(self):
        self.owned()
        require_binding(self.session, SCENE)
        m, d, r = self.m, self.d, self.report
        if (
            r["physics_steps"]
            or not np.isfinite(d.time)
            or not all(
                np.isfinite(a).all()
                for a in (d.qpos, d.qvel, d.ctrl, d.qfrc_applied, d.xfrc_applied)
            )
        ):
            raise ValueError("A fresh finite resting episode is required")
        if (
            np.any(abs(d.qpos[self.q]) > 0.0001)
            or np.any(abs(d.qvel[self.v]) > 0.0001)
            or np.any(abs(d.ctrl[self.a]) > 0.0001)
        ):
            raise ValueError("Inspection requires measured home with stopped drives")
        gaps, _ = self.distances()
        if min(gaps) < self.minimum:
            raise ValueError("Unsafe starting direct geometry")
        home = d.qpos[self.q].copy()
        try:
            target = d.qpos[self.q].copy()
            target[0] = 1.2
            self.phase("index_left_roller", target, (0,))
            target = d.qpos[self.q].copy()
            target[1] = -0.8
            self.phase("index_right_roller", target, (1,))
            target = d.qpos[self.q].copy()
            target[2] = 0.10
            self.phase("lift_clearance_inspection", target, (2,))
            target = d.qpos[self.q].copy()
            target[2] = home[2]
            self.phase("return_lift", target, (2,))
            target = d.qpos[self.q].copy()
            target[:2] = home[:2]
            self.phase("return_roller_registration", target, (0, 1))
            r["success"] = True
        except RuntimeError as exc:
            r["failure"] = str(exc)
            # A stop request changes commands only; no state reset or pose jump.
            self.owned()
            d.ctrl[self.a[:2]] = 0
            d.ctrl[self.a[2]] = float(d.qpos[self.q[2]])
            r["hold_requested"] = True
        finally:
            r["coverage_complete"] = (
                r["physics_steps"] > 0 and r["completed_checks"] == r["physics_steps"]
            )
            if r["coverage_complete"]:
                r["whole_run_clearance_lower_bound_m"] = r[
                    "checked_clearance_lower_bound_m"
                ]
            r["return_error"] = (d.qpos[self.q] - home).tolist()
            r["final_state"] = full_state(self.session)
            r["model_unchanged"] = _model_signature(m) == self.signature
            r["scope"] = (
                "Unloaded independent roller indexing/stopping plus guided inert lift inspection and return; no belt transport"
            )
            r["limitations"] = [
                "Fixed belt decks do not convey contact or payloads",
                "Original ideal joints, gains, masses and gravity compensation; no physical bearing calibration",
                "No walking, human biomechanics, force-plate, belt-tension or safety-rated emergency-stop claims",
                "Sampled completed-step checks are not continuous-time collision certification",
                "Source-derived exterior cues; internal mechanisms and dimensions remain authored estimates",
            ]
        return r


def inspect_stanford_gait(session):
    return GaitInspection(session).execute()
