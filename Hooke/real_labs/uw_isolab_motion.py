"""Measured dry inspection of the version 3 UW laboratory autosampler.

This is a checked authored corridor, not a general motion planner or an OEM
operation sequence. Every actual step checks enabled parent-filtered geometry,
held axes and the physically free inert vial. No live poses are assigned.
"""

from itertools import product
from pathlib import Path
import hashlib
import json
import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding
from real_labs.uw_isolab_geometry import CONTROLS, ID

SCENE = "uw_isolab"
EQUIPMENT = "isotope_analyzer"
NAMES = CONTROLS
FIELDS = (
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IsolabInspection:
    def __init__(self, session, controls=CONTROLS):
        if tuple(controls) != CONTROLS:
            raise ValueError("Version 3 requires its three explicit v3 control keys")
        self.binding = require_binding(session, SCENE)
        self.session = session
        self.m, self.d = self.session.model, self.session.data
        if not np.isfinite(self.d.time):
            raise ValueError("Isolab inspection requires a finite simulation clock")
        self.model_signature = _model_signature(self.m)
        self.scene_sha = self.binding["scene_sha256"]
        self.task_sha = sha(__file__)
        self.other_controls = [
            c for (owner, _), c in session.controls.items() if owner != EQUIPMENT
        ]
        self.controls = [self.session.controls[EQUIPMENT, key] for key in NAMES]
        self.addresses = np.array([self.m.jnt_qposadr[c.joint] for c in self.controls])
        self.actuators = np.array([c.actuator for c in self.controls])
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        self.masks = self.m.geom_contype.copy(), self.m.geom_conaffinity.copy()
        self.flags = int(self.m.opt.disableflags), int(self.m.opt.enableflags)
        self.vial = self.m.geom(ID + "__vial_collision").id
        self.vial_body = self.m.body(ID + "__inert_vial").id
        self.tip = self.m.site(ID + "__site_physical_tip").id
        self.vial_xy = self.d.geom_xpos[self.vial, :2].copy()
        self.support = self.m.geom("isotope_analyzer__oem_tray_shelf").id
        self.support_top = float(
            self.d.geom_xpos[self.support, 2] + self.m.geom_size[self.support, 2]
        )
        self.cap = 0.002
        self.threshold = 0.00025
        self.trace = []
        self.step_observer = None
        self._pairs()
        self._reset_counts()
        self.held_other = []

    def _pairs(self):
        m = self.m
        weld = m.body_weldid[m.geom_bodyid]
        groups = {int(m.body_weldid[m.jnt_bodyid[c.joint]]) for c in self.controls}
        enabled = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))
        pairs = np.array(
            [
                (a, b)
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if weld[a] != weld[b] and (weld[a] in groups or weld[b] in groups)
            ],
            dtype=int,
        ).reshape(-1, 2)
        self.full_pair_count = len(pairs)
        scratch = mujoco.MjData(m)
        lower = np.full((m.ngeom, 3), np.inf)
        upper = -lower.copy()
        for q in product(*zip(self.lower - 0.0001, self.upper + 0.0001)):
            scratch.qpos[self.addresses] = q
            mujoco.mj_forward(m, scratch)
            rotation = scratch.geom_xmat.reshape(-1, 3, 3)
            center = scratch.geom_xpos + np.einsum(
                "nij,nj->ni", rotation, m.geom_aabb[:, :3]
            )
            half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
            lower = np.minimum(lower, center - half)
            upper = np.maximum(upper, center + half)
        a, b = pairs.T
        separation = np.linalg.norm(
            np.maximum(np.maximum(lower[a] - upper[b], lower[b] - upper[a]), 0), axis=1
        )
        unknown = (weld != 0) & ~np.isin(weld, list(groups))
        self.pairs = pairs[(separation <= self.cap + 1e-9) | unknown[a] | unknown[b]]

    def _reset_counts(self):
        self.executed = self.checked = self.queries = 0
        self.physics_events = []
        self.minimum = self.cap
        self.maximum_support_penetration = 0.0

    def state(self):
        return dict(
            **{key: getattr(self.d, key).tolist() for key in FIELDS},
            time_s=float(self.d.time),
            model_signature=self.model_signature,
        )

    def measured(self):
        return self.d.qpos[self.addresses].copy()

    def preflight(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Candidate model/data ownership changed")
        require_binding(self.session, SCENE)
        if (
            sha(__file__) != self.task_sha
            or _model_signature(self.m) != self.model_signature
        ):
            raise ValueError("Candidate source/model binding changed")
        if not np.isfinite(self.d.time) or not all(
            np.isfinite(getattr(self.d, k)).all() for k in FIELDS
        ):
            raise ValueError("Nonfinite candidate state")

    def _hold(self):
        self.d.ctrl[self.actuators] = np.clip(self.measured(), self.lower, self.upper)

    def _observe_step(self, held, held_targets):
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
            or (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
        ):
            raise RuntimeError("Candidate geometry/state binding changed")
        q = self.measured()
        if (
            not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or np.any(q < self.lower - 0.0001)
            or np.any(q > self.upper + 0.0001)
        ):
            raise RuntimeError("Candidate swept domain exceeded")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        center = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(abs(center[a] - center[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap
        )
        current = self.cap
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap, None)),
                self.cap,
            )
            self.queries += 1
            current = min(current, gap)
            if gap < self.threshold:
                raise RuntimeError(
                    "Clearance lost:" + m.geom(a).name + ":" + m.geom(b).name
                )
        if held.size and np.max(abs(q[held] - held_targets[held])) > 0.00005:
            raise RuntimeError("Held axis drift")
        if np.max(abs(q - d.ctrl[self.actuators])) > 0.002:
            raise RuntimeError("Moving axis tracking error")
        axis_z = abs(float(d.geom_xmat[self.vial].reshape(3, 3)[2, 2]))
        bottom = float(
            d.geom_xpos[self.vial, 2]
            - axis_z * m.geom_size[self.vial, 1]
            - math.sqrt(max(0, 1 - axis_z * axis_z)) * m.geom_size[self.vial, 0]
        )
        gap = bottom - self.support_top
        self.maximum_support_penetration = max(self.maximum_support_penetration, -gap)
        if (
            np.linalg.norm(d.geom_xpos[self.vial, :2] - self.vial_xy) > 0.00014
            or axis_z < math.cos(0.01)
            or not -0.0001 <= gap <= 0.0001
        ):
            raise RuntimeError("Free vial support lost")
        for control, value, target in self.held_other:
            measured = self.session._measured(control)
            tolerance = max(1e-8, (control.upper - control.lower) * 2.5e-4)
            if abs(measured - value) > tolerance or d.ctrl[control.actuator] != target:
                raise RuntimeError(
                    "Other instrument control moved during UW inspection"
                )
        if self.step_observer is not None:
            before = self.state()
            self.step_observer(m, d)
            if _model_signature(m) != self.model_signature:
                raise RuntimeError("Read-only observer mutated the simulation model")
            if self.state() != before:
                raise RuntimeError("Read-only observer mutated simulation state")
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def step(self, held, targets):
        before_time = float(self.d.time)
        before_warnings = _warning_state(self.d)
        reset_inputs = {
            key: getattr(self.d, key).copy()
            for key in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.executed += 1
        event, _ = _physics_event(
            self.m, self.d, before_time, before_warnings, reset_inputs
        )
        if event is not None:
            self.physics_events.append(dict(executed_step=self.executed, **event))
            raise RuntimeError("Native physics warning or time discontinuity")
        self._observe_step(held, targets)

    def _require_guarded_target(self, start, target):
        """Reject unqualified low XY routes before commands or physics change.

        This deliberately exposes bounded inspection corridors, not a general
        Cartesian planner. Full control travel and all physical contacts remain.
        """
        if np.any(start < self.lower - 1e-12) or np.any(start > self.upper + 1e-12):
            raise ValueError("Measured start lies outside the version 3 joint limits")
        moving_xy = np.max(abs(target[:2] - start[:2])) > 1e-8
        if moving_xy:
            if abs(start[2]) > 1e-8 or target[2] != 0:
                raise ValueError("Retract to the raised plane before any XY move")
            return
        q = self.measured()
        tip = self.d.site_xpos[self.tip]
        maximum = None
        for key in ("vial", "port"):
            if key == "vial":
                surface = (
                    self.d.geom_xpos[self.vial]
                    + self.d.geom_xmat[self.vial].reshape(3, 3)[:, 2] * 0.016
                )
            else:
                surface = self.d.site_xpos[
                    self.m.site(ID + "__site_vaporizer_dry_datum").id
                ]
            station = q[:2] + (surface - tip)[:2]
            if np.linalg.norm(start[:2] - station) <= 0.00005:
                maximum = float(q[2] + tip[2] - surface[2] - 0.0045)
                break
        if np.max(abs(start[:2])) <= 0.00005:
            maximum = self.upper[2]
        if maximum is None or target[2] > maximum + 0.00005:
            raise ValueError("Target is outside a measured dry approach corridor")

    def move(self, target, *, step_observer=None):
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        self.step_observer = step_observer
        return self.leg("requested_v3_move", target)

    def leg(self, name, target=None, settle_s=None):
        self.preflight()
        before = self.state()
        self.held_other = [
            (c, self.session._measured(c), float(self.d.ctrl[c.actuator]))
            for c in self.other_controls
        ]
        start = self.measured()
        self._reset_counts()
        failure = None
        if target is None:
            target = self.d.ctrl[self.actuators].copy()
            move_steps, dwell_steps = 0, round(settle_s / self.m.opt.timestep)
        else:
            target = np.asarray(target, dtype=float)
            if (
                target.shape != start.shape
                or not np.isfinite(target).all()
                or np.any(target < self.lower)
                or np.any(target > self.upper)
            ):
                raise ValueError("Strict version3 target limits exceeded")
            self._require_guarded_target(start, target)
            # Bound requested speed by the existing position-servo damping/gain.
            # The2mm tracking guard is retained; this is an authored slow inspection.
            gain = self.m.actuator_gainprm[self.actuators, 0]
            damping = -self.m.actuator_biasprm[self.actuators, 2] + np.array(
                [self.m.dof_damping[self.m.jnt_dofadr[c.joint]] for c in self.controls]
            )
            speed = 0.0014 * gain / damping
            duration = max(0.8, 1.875 * float(np.max(abs(target - start) / speed)))
            move_steps = math.ceil(duration / self.m.opt.timestep)
            dwell_steps = round(0.7 / self.m.opt.timestep)
        held = np.flatnonzero(abs(target - start) < 1e-8)
        held_targets = self.d.ctrl[self.actuators].copy()
        try:
            for index in range(move_steps):
                u = (index + 1) / move_steps
                smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
                self.d.ctrl[self.actuators] = start + (target - start) * smooth
                self.step(held, held_targets)
            self.d.ctrl[self.actuators] = target
            for _ in range(dwell_steps):
                self.step(held, held_targets)
            if np.max(abs(self.measured() - target)) > 0.00001:
                raise RuntimeError("Measured endpoint not reached")
        except RuntimeError as exc:
            failure = str(exc)
            self._hold()
        complete = self.executed > 0 and self.checked == self.executed
        return dict(
            phase=name,
            status="failed" if failure else "succeeded",
            failure=failure,
            target_m=target.tolist(),
            before=before,
            after=self.state(),
            physics_steps=self.executed,
            checked_steps=self.checked,
            coverage_complete=complete,
            **combine_reset_reports(
                self.physics_events or [dict(reset_performed=False)]
            ),
            physics_events=self.physics_events.copy(),
            hold_requested=bool(failure),
            direct_queries=self.queries,
            maximum_support_penetration_m=self.maximum_support_penetration,
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
        )

    def observation(self, target):
        self.preflight()
        tip = self.d.site_xpos[self.tip].copy()
        if target == "vial":
            center = self.d.geom_xpos[self.vial].copy()
            surface = (
                center
                + self.d.geom_xmat[self.vial].reshape(3, 3)[:, 2]
                * self.m.geom_size[self.vial, 1]
            )
            expected = self.vial
            visible_cap = self.m.geom(ID + "__vial_cap").id
            # The visible cap and enabled solid surrogate terminate at the same
            # physical top plane. World translation can change MuJoCo's ray
            # tie ordering; both are source-bound parts of the same free body.
            accepted_hits = {expected, visible_cap}
        else:
            expected = self.m.geom("isotope_analyzer__oem_vaporizer_port").id
            surface = self.d.geom_xpos[expected].copy()
            surface[2] += self.m.geom_size[expected, 1]
            accepted_hits = {expected}
        hit = np.array([-1], dtype=np.int32)
        distance = float(
            mujoco.mj_ray(
                self.m,
                self.d,
                tip - [0, 0, 1e-6],
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                hit,
            )
        )
        return dict(
            target=target,
            tip_world_m=tip.tolist(),
            surface_world_m=surface.tolist(),
            lateral_error_m=float(np.linalg.norm((tip - surface)[:2])),
            standoff_m=float(tip[2] - surface[2]),
            first_hit=None if hit[0] < 0 else self.m.geom(hit[0]).name,
            registered=bool(
                np.linalg.norm((tip - surface)[:2]) < 0.00005
                and abs(tip[2] - surface[2] - 0.0045) < 0.00005
                and hit[0] in accepted_hits
                and abs(distance + 1e-6 - (tip[2] - surface[2])) < 1e-6
            ),
            ray_distance_m=distance,
            collision_surface_geom=self.m.geom(expected).name,
            equivalent_top_surface_geoms=[
                self.m.geom(i).name for i in sorted(accepted_hits)
            ],
            ray_surface_error_m=float(abs(distance + 1e-6 - (tip[2] - surface[2]))),
            actual_state=self.state(),
            rendered_visibility_qualified=False,
        )

    def run(self, *, progress=None, step_observer=None):
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        self.step_observer = step_observer
        self.preflight()
        start = self.measured()
        if (
            np.max(abs(start)) > 0.00005
            or np.any(start < self.lower - 1e-12)
            or np.any(start > self.upper + 1e-12)
        ):
            raise ValueError("Version3 route requires measured near-home start")
        saved = np.clip(start, self.lower, self.upper)
        initial = self.state()
        phases = [self.leg("settle_free_vial", settle_s=0.3)]
        observations = []
        if phases[-1]["status"] == "succeeded":
            surface = (
                self.d.geom_xpos[self.vial]
                + self.d.geom_xmat[self.vial].reshape(3, 3)[:, 2] * 0.016
            )
            q = self.measured()
            tip = self.d.site_xpos[self.tip].copy()
            xy = q[:2] + (surface - tip)[:2]
            approach = float(q[2] + tip[2] - surface[2] - 0.0045)
            port = self.d.site_xpos[
                self.m.site(ID + "__site_vaporizer_dry_datum").id
            ].copy()
            port_xy = q[:2] + (port - tip)[:2]
            port_z = float(q[2] + tip[2] - port[2] - 0.0045)
            route = [
                ("vial_x", (xy[0], 0, 0)),
                ("vial_y", (*xy, 0)),
                ("inspect_vial", (*xy, approach)),
                ("retract_vial", (*xy, 0)),
                ("port_x", (port_xy[0], xy[1], 0)),
                ("port_y", (*port_xy, 0)),
                ("inspect_port", (*port_xy, port_z)),
                ("retract_port", (*port_xy, 0)),
                ("return_y", (port_xy[0], 0, 0)),
                ("return_x", (0, 0, 0)),
                ("full_service_stroke", (0, 0, self.upper[2])),
                ("return_service", saved),
            ]
            for name, target in route:
                phase = self.leg(name, target)
                phases.append(phase)
                if progress is not None:
                    progress(phase)
                if phase["status"] != "succeeded":
                    break
                if name in ("inspect_vial", "inspect_port"):
                    observation = self.observation(
                        "vial" if name == "inspect_vial" else "port"
                    )
                    observations.append(observation)
                    if not observation["registered"]:
                        self._hold()
                        phase.update(
                            status="failed",
                            failure="Surface registration missing",
                            hold_requested=True,
                        )
                        break
        success = len(phases) == 13 and all(p["status"] == "succeeded" for p in phases)
        return dict(
            status="succeeded" if success else "failed",
            scene_sha256=self.scene_sha,
            task_sha256=self.task_sha,
            model_signature=self.model_signature,
            initial_state=initial,
            final_state=self.state(),
            saved_return_target_m=saved.tolist(),
            phases=phases,
            observations=observations,
            **combine_reset_reports(phases),
            physics_events=[
                dict(phase=p["phase"], **event)
                for p in phases
                for event in p["physics_events"]
            ],
            return_error_m=float(np.max(abs(self.measured() - start))),
            executed_steps=sum(p["physics_steps"] for p in phases),
            checked_steps=sum(p["checked_steps"] for p in phases),
            complete_pair_count=self.full_pair_count,
            retained_pair_count=len(self.pairs),
            limits=[
                "Source-informed version 3 mechanics in an estimated UW room; not an OEM operating protocol",
                "True vial body dimensions but uncalibrated cap, mass, guides, forces and layout",
                "No fluid, isotope, vacuum or analytical process; no image contrast qualification",
                "Discrete checked trajectory, not continuous collision certification",
                "No concurrent external model/state writers supported",
                "Direct pairs cover all enabled moving-assembly/world interfaces, including parent-filtered pairs; free vial/static support is observed separately",
                "Only raised XY and named dry approach corridors are accepted; the full joint box is obstructed",
            ],
        )
