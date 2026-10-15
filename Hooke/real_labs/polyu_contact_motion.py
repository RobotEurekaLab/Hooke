"""Actual low-load reciprocation on an inert retained coupon.

Only position commands advance the live state. Contact forces and sliding are
measured constraint responses, not calibrated material properties. The exact
ball/coupon pair alone may contact; all other moving cross-weld pairs retain a
positive running clearance, including native parent-filtered pairs.
"""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from real_labs.motion import _physics_event, _warning_state
from real_labs.polyu_contact_geometry import PREFIX
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding

SCENE = "polyu_materials_tribology"
EQUIPMENT = "umt_station"
CONTROLS = ("coupon_x", "probe_height")


class LoadedReciprocation:
    """Authored inspection load, preserving original servo and global solver settings."""

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session, self.model, self.data = session, session.model, session.data
        m, d = self.model, self.data
        self.controls = tuple(session.controls[EQUIPMENT, n] for n in CONTROLS)
        self.qadr = np.array([m.jnt_qposadr[c.joint] for c in self.controls])
        self.dadr = np.array([m.jnt_dofadr[c.joint] for c in self.controls])
        self.actuators = np.array([c.actuator for c in self.controls])
        self.tip = m.geom(PREFIX + "counterface").id
        self.coupon = m.geom(PREFIX + "coupon").id
        self.contact_pair = frozenset((self.tip, self.coupon))
        groups = {int(m.body_weldid[m.jnt_bodyid[c.joint]]) for c in self.controls}
        ids = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(ids)
                for b in ids[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (
                    m.body_weldid[m.geom_bodyid[a]] in groups
                    or m.body_weldid[m.geom_bodyid[b]] in groups
                )
                and frozenset((a, b)) != self.contact_pair
            ],
            dtype=int,
        ).reshape(-1, 2)
        self.held = {
            key: (c, float(d.ctrl[c.actuator]), session._measured(c))
            for key, c in session.controls.items()
            if c.actuator not in self.actuators
        }
        self.jacobian = np.zeros((2, 3, m.nv))
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.preflight()

    def preflight(self):
        if self.session.model is not self.model or self.session.data is not self.data:
            raise ValueError("Contact operation session model/state was replaced")
        require_binding(self.session, SCENE)
        d = self.data
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
            raise ValueError("Contact operation requires finite state and time")
        if np.max(abs(d.qvel), initial=0) > 0.03:
            raise ValueError(
                "Initial velocity exceeds the authored inspection envelope"
            )
        if self.session.equipment[EQUIPMENT]["contact_interface"]["revision"] != 2:
            raise ValueError("Loaded contact requires explicit fixture revision2")

    def state(self):
        return dict(
            time_s=float(self.data.time),
            **{
                n: getattr(self.data, n).tolist()
                for n in (
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
            }
        )

    def _hold(self):
        if self.session.model is not self.model or self.session.data is not self.data:
            return False
        for c, q in zip(self.controls, self.qadr):
            value = float(self.data.qpos[q])
            if not np.isfinite(value):
                return False
            self.data.ctrl[c.actuator] = np.clip(value, c.lower, c.upper)
        return True

    def observe(self):
        m, d = self.model, self.data
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("contact_model_state_binding_changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("contact_configuration_changed")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("nonfinite_physics_state")
        if np.max(abs(d.qvel), initial=0) > 0.08:
            raise RuntimeError("contact_velocity_limit")
        for c, q in zip(self.controls, self.qadr):
            if not c.lower - 1e-6 <= d.qpos[q] <= c.upper + 1e-6:
                raise RuntimeError("contact_joint_limit")
        for key, (c, target, measured) in self.held.items():
            if (
                d.ctrl[c.actuator] != target
                or abs(self.session._measured(c) - measured) > 0.0002
            ):
                raise RuntimeError("inactive_control_changed:" + str(key))
        # Conservative world AABBs retain all near geometry, including parent
        # pairs. A positive cap is a lower bound, not a measured far distance.
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < 0.002
        )
        minimum, queries = 0.002, 0
        for a, b in self.pairs[near]:
            gap = min(
                0.002, float(mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None))
            )
            queries += 1
            minimum = min(minimum, gap)
            if gap < 0.00025:
                raise RuntimeError(
                    "unintended_clearance_lost:"
                    + m.geom(int(a)).name
                    + ":"
                    + m.geom(int(b)).name
                )
        normal = tangential = penetration = speed = 0.0
        world_force = np.zeros(3)
        contacts = 0
        for index in range(d.ncon):
            c = d.contact[index]
            pair = frozenset((c.geom1, c.geom2))
            if pair != self.contact_pair:
                if c.dist < -1e-7 and (c.geom1 in self.pairs or c.geom2 in self.pairs):
                    raise RuntimeError("unintended_native_contact")
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, f)
            frame = c.frame.reshape(3, 3)
            normal += max(0, float(f[0]))
            tangential += float(np.linalg.norm(f[1:3]))
            world_force += (1 if c.geom2 == self.tip else -1) * (frame.T @ f[:3])
            penetration = max(penetration, -float(c.dist))
            for k, geom in enumerate((self.tip, self.coupon)):
                mujoco.mj_jac(
                    m, d, self.jacobian[k], None, c.pos, int(m.geom_bodyid[geom])
                )
            relative = (self.jacobian[0] - self.jacobian[1]) @ d.qvel
            speed += float(
                np.linalg.norm(relative - frame[0] * np.dot(frame[0], relative))
            )
            contacts += 1
        self.latest_contact_sample = dict(
            time_s=float(d.time),
            normal_force_N=normal,
            tangential_force_N=tangential,
            penetration_m=penetration,
            native_contact_count=contacts,
            force_on_counterface_world_N=world_force.tolist(),
        )
        if normal > 2 or penetration > 0.0001:
            raise RuntimeError("contact_load_or_penetration_limit")
        if abs(d.qpos[self.qadr[0]] - d.ctrl[self.actuators[0]]) > 0.0015:
            raise RuntimeError("coupon_tracking_limit")
        if abs(d.qpos[self.qadr[1]] - d.ctrl[self.actuators[1]]) > 0.003:
            raise RuntimeError("load_axis_tracking_limit")
        return dict(
            normal_force_N=normal,
            tangential_force_N=tangential,
            force_on_counterface_world_N=world_force.tolist(),
            penetration_m=penetration,
            relative_tangential_speed_m_s=speed,
            native_contact_count=contacts,
            direct_clearance_m=minimum,
            direct_queries=queries,
            tip_bottom_world_m=d.site_xpos[
                m.site(EQUIPMENT + "__site_counterface_bottom").id
            ].tolist(),
            coupon_top_world_m=d.site_xpos[
                m.site(EQUIPMENT + "__site_tribology_coupon").id
            ].tolist(),
        )

    def move(self, target, *, duration_s=2, settle_s=0.5, loaded=False):
        self.preflight()
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or any(not c.lower <= t <= c.upper for c, t in zip(self.controls, target))
        ):
            raise ValueError("Target must lie within the unchanged public joint limits")
        if (
            not np.isfinite([duration_s, settle_s]).all()
            or duration_s <= 0
            or settle_s < 0
        ):
            raise ValueError("Invalid positive move duration or settling time")
        m, d = self.model, self.data
        start = d.ctrl[self.actuators].copy()
        initial_measured = d.qpos[self.qadr].copy()
        self.observe()
        dt = float(m.opt.timestep)
        ramp, settle = max(1, round(duration_s / dt)), round(settle_s / dt)
        report = dict(
            status="failed",
            physics_steps=0,
            observer_entries=0,
            checked_steps=0,
            failure=None,
            trace=[],
            initial_state=self.state(),
            target_m=target.tolist(),
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            hold_requested=False,
            loaded_required=loaded,
            direct_pair_count=len(self.pairs),
        )
        minimum, travel, max_load, max_depth, queries = 0.002, 0.0, 0.0, 0.0, 0
        loaded_checks = 0
        before_warnings = _warning_state(d)
        last = None
        try:
            for step in range(ramp + settle):
                fraction = min(1.0, (step + 1) / ramp)
                smooth = fraction * fraction * (3 - 2 * fraction)
                d.ctrl[self.actuators] = start + (target - start) * smooth
                before_time = float(d.time)
                inputs = {
                    n: getattr(d, n).copy()
                    for n in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                report["physics_steps"] += 1
                event, before_warnings = _physics_event(
                    m, d, before_time, before_warnings, inputs
                )
                if event is not None:
                    report.update(
                        {
                            k: event[k]
                            for k in ("reset_performed", "automatic_reset_detected")
                        }
                    )
                    report["physics_event"] = event
                    raise RuntimeError("physics_warning_or_clock_discontinuity")
                mujoco.mj_forward(m, d)
                report["observer_entries"] += 1
                last = self.observe()
                if loaded and last["normal_force_N"] < 0.02:
                    raise RuntimeError("loaded_contact_lost")
                if loaded and abs(d.qpos[self.qadr[1]] + 0.04) > 0.0001:
                    raise RuntimeError("loaded_axis_position_changed")
                # Mark completed only after every invariant, including loaded state.
                report["checked_steps"] += 1
                minimum = min(minimum, last["direct_clearance_m"])
                queries += last["direct_queries"]
                travel += (
                    last["relative_tangential_speed_m_s"] * dt
                    if last["normal_force_N"] > 0.02
                    else 0
                )
                loaded_checks += int(last["normal_force_N"] > 0.02)
                max_load = max(max_load, last["normal_force_N"])
                max_depth = max(max_depth, last["penetration_m"])
                if step % 10 == 0 or step + 1 == ramp + settle:
                    report["trace"].append(dict(**self.state(), **last))
            if abs(d.qpos[self.qadr[0]] - target[0]) > 0.0006:
                raise RuntimeError("coupon_endpoint_not_reached")
            if target[1] < -0.04:
                if last["normal_force_N"] < 0.05 or last["normal_force_N"] > 0.3:
                    raise RuntimeError("settled_load_not_established")
            elif (
                abs(d.qpos[self.qadr[1]] - target[1]) > 0.0001
                or last["native_contact_count"]
            ):
                raise RuntimeError("retracted_endpoint_not_reached")
            if loaded and travel < abs(target[0] - initial_measured[0]) - 0.0008:
                raise RuntimeError("measured_sliding_not_reached")
            report["status"] = "succeeded"
        except RuntimeError as exc:
            report["failure"] = str(exc)
            sample = getattr(self, "latest_contact_sample", None)
            report["failure_contact_sample"] = (
                sample if sample and sample["time_s"] == float(d.time) else None
            )
            report["hold_requested"] = self._hold()
        complete = (
            report["physics_steps"] > 0
            and report["physics_steps"] == report["checked_steps"]
        )
        report.update(
            coverage_complete=complete,
            final_state=self.state(),
            final_observation=last,
            sampled_relative_sliding_m=travel,
            sampled_loaded_steps=loaded_checks,
            maximum_normal_force_N=max_load,
            maximum_penetration_m=max_depth,
            direct_queries=queries,
            checked_steps_clearance_lower_bound_m=(
                minimum if report["checked_steps"] else None
            ),
            whole_leg_clearance_lower_bound_m=minimum if complete else None,
        )
        return report

    def run(self):
        self.preflight()
        if (
            np.max(abs(self.data.qpos[self.qadr])) > 0.0001
            or np.max(abs(self.data.ctrl[self.actuators])) > 0.0001
        ):
            raise ValueError("Begin with the centered, retracted fixture holding home")
        phases = []
        for name, target, duration, loaded in (
            ("approach_and_load", (0, -0.0405), 3, False),
            ("slide_negative", (-0.010, -0.0405), 8, True),
            ("slide_positive", (0.010, -0.0405), 16, True),
            ("slide_center", (0, -0.0405), 8, True),
            ("retract", (0, 0), 3, False),
        ):
            report = self.move(target, duration_s=duration, loaded=loaded)
            phases.append(dict(phase=name, **report))
            if report["status"] != "succeeded":
                break
        executed = sum(p["physics_steps"] for p in phases)
        checked = sum(p["checked_steps"] for p in phases)
        return dict(
            status=(
                "succeeded"
                if len(phases) == 5 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            phases=phases,
            executed_steps=executed,
            checked_steps=checked,
            coverage_complete=executed > 0 and executed == checked,
            sampled_relative_sliding_m=sum(
                p["sampled_relative_sliding_m"] for p in phases
            ),
            final_state=self.state(),
            **combine_reset_reports(phases),
            limitations=[
                "Generic position-loaded rigid contact with unchanged default 8 ms MuJoCo contact compliance and generic friction; no calibrated force control, friction coefficient or wear prediction.",
                "Ideal gravity-compensated lumped assemblies and fixed enclosure; no door/loading protocol or physical sensor model.",
            ]
        )


def run_loaded_reciprocation(session):
    return LoadedReciprocation(session).run()
