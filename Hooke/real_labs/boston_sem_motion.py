"""Measured revision-three SEM mechanics in the source-bound Boston laboratory.

This authored route proves a bounded mechanical capability. It does not model
vacuum, electron optics, specimen charging, or calibrated SEM focus. All seven
coordinates are actually integrated. The task never explicitly resets state;
native numerical resets are detected and reported as failed, unchecked steps.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.boston_sem_compact_geometry import ID, SCENE
from real_labs.boston_sem_bounds import candidate_pairs
from real_labs.rigid_access import require_binding
from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession

EQUIPMENT = ID
CONTROLS = (
    "door_slide_outer",
    "door_slide_inner",
    "stage_x",
    "stage_y",
    "stage_z",
    "stage_tilt",
    "stage_rotation",
)
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
NAMES = (
    "door_slide_outer",
    "door_slide_inner",
    "stage_x",
    "stage_y",
    "stage_z",
    "stage_tilt",
    "stage_rotation",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StageInspection:
    """Exact candidate binding and complete post-integration observation.

    The public targets have strict original candidate limits. Whole-range
    pruning covers an explicitly enlarged 100 micrometre / 100 microradian numerical envelope;
    leaving that envelope fails before using its conservative distance bound.
    Source changes are checked before each leg. No external state-mutating
    callbacks or concurrent model edits are supported by this local recorder.
    """

    def __init__(self, session):
        self.binding = require_binding(session, SCENE)
        self.session = session
        self.m, self.d = session.model, session.data
        revision = session.equipment[ID].get("mechanism_revision", {})
        if (
            revision.get("version") != 3
            or revision.get("automatic_legacy_mapping") is not False
        ):
            raise ValueError("Explicit SEM revision 3 metadata is required")
        self.signature = _model_signature(self.m)
        directory = Path(__file__).parent
        self.paths = [
            Path(__file__),
            directory / "boston_sem_compact_geometry.py",
            directory / "boston_sem_bounds.py",
            directory / "university_americas_wave5.py",
            directory / "scenes/university_americas_wave5.json",
        ]
        if session.equipment[ID].get("exterior_revision") == 1:
            self.paths.append(directory / "boston_sem_exterior.py")
        if session.equipment[ID].get("chamber_observation", {}).get("revision") == 1:
            self.paths.append(directory / "boston_sem_observation.py")
        self.sources = {str(p): sha(p) for p in self.paths}
        self.registry = deepcopy(self.session.controls)
        self.metadata = deepcopy(self.session.manifest)
        self.controls = [self.session.controls[ID, key] for key in NAMES]
        self.addresses = np.array([self.m.jnt_qposadr[c.joint] for c in self.controls])
        self.actuators = np.array([c.actuator for c in self.controls])
        self.dofs = np.array([self.m.jnt_dofadr[c.joint] for c in self.controls])
        self.lower = np.array([c.lower for c in self.controls])
        self.upper = np.array([c.upper for c in self.controls])
        # The compiled range is serialized independently; intersect both contracts.
        joints = np.array([c.joint for c in self.controls])
        self.lower = np.maximum(self.lower, self.m.jnt_range[joints, 0])
        self.upper = np.minimum(self.upper, self.m.jnt_range[joints, 1])
        if set(joints) != set(range(self.m.njnt)):
            raise ValueError(
                "This SEM task requires a stationary room with only its seven authored joints"
            )
        self.domain_padding = 0.0001
        self.cap, self.threshold = 0.002, 0.00025
        self.pairs, excluded, _ = candidate_pairs(self.m, self.cap, self.domain_padding)
        self.excluded_count = len(excluded)
        self.masks = self.m.geom_contype.copy(), self.m.geom_conaffinity.copy()
        self.flags = int(self.m.opt.disableflags), int(self.m.opt.enableflags)
        self.trace = []
        self.tracking = np.array([0.002] * 5 + [0.020] * 2)
        self.held_tolerance = np.array([0.00005] * 5 + [0.0005] * 2)
        self.endpoint = np.array([0.00001] * 5 + [0.0001] * 2)
        # Validity limits for inspection, not reduced joint travel or gains.
        self.velocity_limit = np.array([1.0] * 5 + [10.0] * 2)
        self.last_finite_command = self.d.ctrl[self.actuators].copy()
        self.reset_counts()
        self.preflight()

    def reset_counts(self):
        self.executed = self.checked = self.queries = 0
        self.minimum = self.cap
        self.peak_tracking = np.zeros(7)
        self.peak_held = np.zeros(7)
        self.physics_events = []
        self.automatic_reset_detected = False
        self.time_discontinuity_detected = False

    def measured(self):
        return self.d.qpos[self.addresses].copy()

    def state(self):
        return dict(
            **{key: getattr(self.d, key).tolist() for key in FIELDS},
            time_s=float(self.d.time),
            model_signature=self.signature,
            physics_warning_counts=self.warnings().tolist(),
        )

    def preflight(self):
        require_binding(self.session, SCENE)
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            raise ValueError("Candidate model/state identity changed")
        if (
            any(sha(p) != digest for p, digest in self.sources.items())
            or _model_signature(self.m) != self.signature
            or self.session.controls != self.registry
            or self.session.manifest != self.metadata
        ):
            raise ValueError("Candidate source, model or public interface changed")
        problem = self.state_problem()
        if problem:
            raise ValueError(problem)

    def state_problem(self):
        # A prior native reset may leave the same warning count at time zero,
        # concealing a repeated reset. Require explicit session recovery before
        # reusing such state. No warning counter or live state is cleared here.
        bad_state_warnings = (
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        )
        if any(self.d.warning[k].number for k in bad_state_warnings):
            return "Native bad-state warning history requires explicit session recovery"
        if not math.isfinite(float(self.d.time)):
            return "Nonfinite candidate simulation time"
        if not all(np.isfinite(getattr(self.d, key)).all() for key in FIELDS):
            return "Nonfinite candidate state"
        if np.any(abs(self.d.qvel[self.dofs]) > self.velocity_limit):
            return "Candidate inspection velocity limit exceeded"
        return None

    def warnings(self):
        return np.array([warning.number for warning in self.d.warning])

    def geometry(self, d):
        m = self.m
        q = d.qpos[self.addresses]
        if (
            not np.isfinite(q).all()
            or np.any(q < self.lower - self.domain_padding)
            or np.any(q > self.upper + self.domain_padding)
        ):
            raise RuntimeError("Swept geometry domain exceeded")
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        centers = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        near = (
            np.linalg.norm(
                np.maximum(abs(centers[a] - centers[b]) - half[a] - half[b], 0), axis=1
            )
            < self.cap
        )
        minimum = self.cap
        queries = 0
        for a, b in self.pairs[near]:
            gap = min(
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap, None)),
                self.cap,
            )
            queries += 1
            minimum = min(minimum, gap)
            if gap < self.threshold:
                raise RuntimeError(
                    "Clearance lost:" + m.geom(a).name + ":" + m.geom(b).name
                )
        return minimum, queries

    def hold(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            return False
        measured = self.measured()
        self.d.ctrl[self.actuators] = np.where(
            np.isfinite(measured),
            np.clip(measured, self.lower, self.upper),
            self.last_finite_command,
        )
        return True

    def step(self, held, held_targets):
        m, d = self.m, self.d
        problem = self.state_problem()
        if problem:
            raise RuntimeError(problem)
        before_time = float(d.time)
        before_warnings = self.warnings()
        self.last_finite_command = d.ctrl[self.actuators].copy()
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.executed += 1
        self.trace.append(np.r_[d.time, d.qpos, d.qvel, d.ctrl])
        after_warnings = self.warnings()
        warning_changes = np.flatnonzero(after_warnings != before_warnings)
        tolerance = min(
            m.opt.timestep * 1e-6,
            max(1e-12, 8 * abs(float(np.spacing(before_time)))),
        )
        time_continuous = (
            math.isfinite(float(d.time))
            and float(d.time) > before_time
            and math.isclose(
                float(d.time) - before_time,
                m.opt.timestep,
                rel_tol=0,
                abs_tol=tolerance,
            )
        )
        if warning_changes.size or not time_continuous:
            reset_warnings = {
                int(mujoco.mjtWarning.mjWARN_BADQPOS),
                int(mujoco.mjtWarning.mjWARN_BADQVEL),
                int(mujoco.mjtWarning.mjWARN_BADQACC),
            }
            reset_enabled = not (
                m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
            )
            self.automatic_reset_detected |= bool(
                reset_enabled and reset_warnings.intersection(warning_changes)
            )
            self.time_discontinuity_detected |= not time_continuous
            self.physics_events.append(
                dict(
                    executed_step=self.executed,
                    before_time_s=before_time,
                    after_time_s=float(d.time),
                    expected_timestep_s=float(m.opt.timestep),
                    time_continuous=bool(time_continuous),
                    warning_indices=warning_changes.tolist(),
                    warnings_before=before_warnings.tolist(),
                    warnings_after=after_warnings.tolist(),
                )
            )
            raise RuntimeError(
                "Native physics warning or simulation-time discontinuity"
            )
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
            or (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("Runtime source/state configuration changed")
        problem = self.state_problem()
        if problem:
            raise RuntimeError(problem)
        minimum, queries = self.geometry(d)
        self.queries += queries
        tracking = abs(self.measured() - d.ctrl[self.actuators])
        self.peak_tracking = np.maximum(self.peak_tracking, tracking)
        if np.any(tracking > self.tracking):
            raise RuntimeError("Measured trajectory tracking limit")
        drift = abs(self.measured() - held_targets)
        self.peak_held[held] = np.maximum(self.peak_held[held], drift[held])
        if np.any(drift[held] > self.held_tolerance[held]):
            raise RuntimeError("Held axis drift")
        self.minimum = min(self.minimum, minimum)
        self.checked += 1

    def plan_line(self, start, target):
        """Read-only planning; sampled geometric clearance, not continuous proof."""
        scratch = mujoco.MjData(self.m)
        scratch.qpos[:] = self.d.qpos
        resolution = np.array([0.0005] * 5 + [math.radians(0.5)] * 2)
        count = max(1, math.ceil(float(np.max(abs(target - start) / resolution))))
        for fraction in np.linspace(0, 1, count + 1):
            scratch.qpos[self.addresses] = start + (target - start) * fraction
            mujoco.mj_forward(self.m, scratch)
            self.geometry(scratch)
        return count + 1

    def leg(self, name, target):
        self.preflight()
        target = np.asarray(target, dtype=float)
        start = self.measured()
        if (
            target.shape != (7,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("Strict version3 target limits exceeded")
        before = self.state()
        self.reset_counts()
        planned = 0
        failure = None
        try:
            planned = self.plan_line(start, target)
            gain = self.m.actuator_gainprm[self.actuators, 0]
            damping = (
                -self.m.actuator_biasprm[self.actuators, 2]
                + self.m.dof_damping[self.dofs]
            )
            speed = 0.50 * self.tracking * gain / damping
            duration = max(1.5, 1.875 * float(np.max(abs(target - start) / speed)))
            count = math.ceil(duration / self.m.opt.timestep)
            held = np.flatnonzero(abs(target - start) < 1e-8)
            held_targets = self.d.ctrl[self.actuators].copy()
            for i in range(count):
                u = (i + 1) / count
                smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
                self.d.ctrl[self.actuators] = start + (target - start) * smooth
                self.step(held, held_targets)
            self.d.ctrl[self.actuators] = target
            for _ in range(round(0.8 / self.m.opt.timestep)):
                self.step(held, held_targets)
            if np.any(abs(self.measured() - target) > self.endpoint):
                raise RuntimeError("Measured endpoint not reached")
        except RuntimeError as exc:
            failure = str(exc)
            self.hold()
        complete = self.executed > 0 and self.executed == self.checked
        return dict(
            phase=name,
            status="failed" if failure else "succeeded",
            failure=failure,
            target=target.tolist(),
            control_order=NAMES,
            before=before,
            after=self.state(),
            physics_steps=self.executed,
            checked_steps=self.checked,
            whole_leg_coverage=complete,
            reset_performed=(
                True
                if self.automatic_reset_detected
                else None if self.time_discontinuity_detected else False
            ),
            explicit_reset_performed=False,
            automatic_reset_detected=self.automatic_reset_detected,
            time_discontinuity_detected=self.time_discontinuity_detected,
            physics_events=self.physics_events,
            inspection_velocity_limits=self.velocity_limit.tolist(),
            hold_requested=bool(failure),
            planned_samples=planned,
            direct_near_queries=self.queries,
            retained_pairs=len(self.pairs),
            excluded_pairs=self.excluded_count,
            distance_cap_m=self.cap,
            swept_domain_padding=self.domain_padding,
            peak_tracking=self.peak_tracking.tolist(),
            peak_held_drift=self.peak_held.tolist(),
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
        )

    def observation(self, expected_xy=(0.0, 0.0)):
        self.preflight()
        m, d = self.m, self.d
        pole = d.site_xpos[m.site(ID + "__site_pole_face").id].copy()
        stub = m.geom(ID + "__retained_stub").id
        rotation = d.geom_xmat[stub].reshape(3, 3)
        center = d.geom_xpos[stub]
        normal = rotation[:, 2]
        surface = center + normal * m.geom_size[stub, 1]
        base = m.body(ID).id
        instrument_axes = d.xmat[base].reshape(3, 3)
        beam_axis = instrument_axes[:, 2]
        origin = pole - beam_axis * 1e-6
        direction = -beam_axis
        plane_distance = float(
            np.dot(surface - origin, normal) / np.dot(direction, normal)
        )
        intersection = origin + plane_distance * direction
        local_intersection = rotation.T @ (intersection - center)
        hit = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(m, d, origin, direction, None, 1, -1, hit))
        name = m.geom(hit[0]).name if hit[0] >= 0 else None
        gap = float(np.dot(pole - intersection, beam_axis))
        local_offset = instrument_axes.T @ (surface - pole)
        registration_error = float(np.linalg.norm(local_offset[:2] - expected_xy))
        return dict(
            pole_face_world_m=pole.tolist(),
            stub_surface_center_m=surface.tolist(),
            observed_surface_world_m=intersection.tolist(),
            physical_stub_local_surface_m=local_intersection.tolist(),
            requested_stage_offset_m=list(expected_xy),
            physical_axial_gap_m=gap,
            registration_error_m=registration_error,
            first_hit=name,
            ray_distance_m=distance,
            registered=bool(
                abs(gap - 0.008) < 0.00001
                and registration_error < 0.00001
                and np.linalg.norm(local_intersection[:2]) < 0.015
                and name == ID + "__retained_stub"
                and abs(distance - plane_distance) < 1e-7
            ),
            actual_state=self.state(),
            rendered_visibility_qualified=False,
            limitation="Geometric registration of a real stub surface; no electron or calibrated SEM image.",
        )

    def route(self, progress=None):
        self.preflight()
        start = self.measured()
        if (
            np.any(abs(start) > self.endpoint)
            or np.any(start < self.lower - 1e-12)
            or np.any(start > self.upper + 1e-12)
        ):
            raise ValueError("Authored route requires measured near-home state")
        saved = np.clip(start, self.lower, self.upper)
        initial = self.state()
        phases, observations = [], []
        q = np.zeros(7)
        route = []

        def add(name, index, value):
            q[index] = value
            route.append((name, q.copy()))

        add("open_outer", 0, 0.210)
        add("open_inner", 1, 0.210)
        add("close_inner_for_stage_work", 1, 0)
        add("close_outer_for_stage_work", 0, 0)
        for name, index in [("x", 2), ("y", 3), ("z", 4), ("tilt", 5), ("rotation", 6)]:
            if self.lower[index] < 0:
                add("index_" + name + "_lower", index, self.lower[index])
            add("index_" + name + "_upper", index, self.upper[index])
            add("center_" + name, index, 0)
        add("inspect_eight_mm", 4, 0.045)
        for x, y in (
            (-0.008, -0.008),
            (-0.008, 0),
            (-0.008, 0.008),
            (0, 0.008),
            (0.008, 0.008),
            (0.008, 0),
            (0.008, -0.008),
            (0, -0.008),
        ):
            q[2:4] = [x, y]
            route.append((f"inspect_surface_{x}_{y}", q.copy()))
        q[2:4] = 0
        route.append(("center_inspection_xy", q.copy()))
        add("lower_specimen", 4, 0)
        route.append(("return_actual_initial", saved))
        return self._execute_route(route, start, progress, expected_observations=9)

    def specimen_location(self):
        """Measure the complete retained stub against the real front opening."""
        self.preflight()
        m, d = self.m, self.d
        base = m.body(ID).id
        axes = d.xmat[base].reshape(3, 3)

        def local_bounds(name):
            geom = m.geom(ID + "__" + name).id
            rotation = axes.T @ d.geom_xmat[geom].reshape(3, 3)
            center = axes.T @ (d.geom_xpos[geom] - d.xpos[base])
            center += rotation @ m.geom_aabb[geom, :3]
            half = abs(rotation) @ m.geom_aabb[geom, 3:]
            return center - half, center + half

        low, high = local_bounds("retained_stub")
        floor_low, floor_high = local_bounds("chamber_floor")
        roof_low, _ = local_bounds("chamber_roof")
        return dict(
            specimen_bounds_instrument_m=[low.tolist(), high.tolist()],
            front_opening_y_m=float(floor_low[1]),
            whole_stub_outside_front=bool(high[1] < floor_low[1]),
            within_nominal_height=bool(
                low[2] > floor_high[2] and high[2] < roof_low[2]
            ),
            scope="Actual retained-stub bounds and physical opening; no removable loading or vacuum seal claim",
        )

    def _execute_route(self, route, start, progress, *, expected_observations):
        initial = self.state()
        phases, observations = [], []
        for name, target in route:
            result = self.leg(name, target)
            phases.append(result)
            if result["status"] == "succeeded":
                result["specimen_location"] = self.specimen_location()
                if (
                    name == "open_inner"
                    and not result["specimen_location"]["whole_stub_outside_front"]
                ):
                    self.hold()
                    result.update(
                        status="failed",
                        failure="The complete retained stub did not reach exterior access",
                        hold_requested=True,
                        after=self.state(),
                    )
            if name.startswith("inspect_") and result["status"] == "succeeded":
                observed = self.observation(target[2:4])
                observations.append(observed)
                if not observed["registered"]:
                    self.hold()
                    result.update(
                        status="failed",
                        failure="Physical surface registration missing",
                        hold_requested=True,
                        after=self.state(),
                    )
            report = dict(
                scene_sha256=self.binding["scene_sha256"],
                source_hashes=self.sources,
                model_signature=self.signature,
                initial=initial,
                final=self.state(),
                status="running",
                phases=phases,
                observations=observations,
                physics_steps=sum(p["physics_steps"] for p in phases),
                checked_steps=sum(p["checked_steps"] for p in phases),
                trace_columns=["time"]
                + [f"qpos{i}" for i in range(self.m.nq)]
                + [f"qvel{i}" for i in range(self.m.nv)]
                + [f"ctrl{i}" for i in range(self.m.nu)],
            )
            if progress:
                before_callback = self.state()
                progress(deepcopy(report))
                try:
                    self.preflight()
                    if self.state() != before_callback:
                        raise ValueError("Progress observer changed the live state")
                except (ValueError, RuntimeError) as error:
                    held = self.hold()
                    result.update(
                        status="failed",
                        failure=str(error),
                        hold_requested=held,
                        after=self.state(),
                    )
                    report["final"] = self.state()
            if result["status"] != "succeeded":
                break
        report["status"] = (
            "succeeded"
            if (
                len(phases) == len(route)
                and all(p["status"] == "succeeded" for p in phases)
                and len(observations) == expected_observations
                and all(o["registered"] for o in observations)
            )
            else "failed"
        )
        report.update(
            executed_steps=report["physics_steps"],
            coverage_complete=bool(
                report["physics_steps"]
                and report["physics_steps"] == report["checked_steps"]
            ),
            whole_run_clearance_lower_bound_m=(
                min(p["whole_leg_clearance_lower_bound_m"] for p in phases)
                if report["status"] == "succeeded"
                and all(p["whole_leg_coverage"] for p in phases)
                else None
            ),
            **combine_reset_reports(phases),
        )
        report["return_errors"] = abs(self.measured() - start).tolist()
        report["limitations"] = [
            "Dry mechanical inspection only; no vacuum, electron, detector or calibrated image physics.",
            "Clamped30mm inert stub; no autonomous specimen gripping or insertion.",
            "The whole joint box is not collision-free. Only checked paths are supported.",
            "Joint axes/gains and hidden telescopes are authored estimates, not recovered OEM CAD.",
            "Rotation spans360degrees with finite limits; not continuous multi-turn rotation.",
            "Distance reports are capped lower bounds; interpolation is discrete, not continuous collision proof.",
        ]
        return report

    def run(self, progress=None):
        """Open fully, observe a real 8 mm surface gap, and physically return.

        Both telescopes traverse 210 mm for 420 mm combined access. The
        stage remains rigidly retained; no autonomous sample loading.
        """
        self.preflight()
        start = self.measured()
        if (
            np.any(abs(start) > self.endpoint)
            or np.any(start < self.lower - 1e-12)
            or np.any(start > self.upper + 1e-12)
        ):
            raise ValueError("Begin SEM inspection at a measured near-home pose")
        saved = np.clip(start, self.lower, self.upper)
        q = np.zeros(7)
        route = []
        for name, index, value in (
            ("open_outer", 0, 0.210),
            ("open_inner", 1, 0.210),
            ("close_inner", 1, 0),
            ("close_outer", 0, 0),
            ("inspect_eight_mm", 4, 0.045),
            ("inspect_offset", 2, 0.003),
            ("inspect_center", 2, 0),
            ("lower_specimen", 4, 0),
        ):
            q[index] = value
            route.append((name, q.copy()))
        route.append(("return_actual_initial", saved))
        report = self._execute_route(route, start, progress, expected_observations=3)
        report["operation_scope"] = (
            "Measured 420 mm combined full door access, three actual surface registrations at 8 mm, and physical return; the specimen remains clamped"
        )
        return report


class SpecimenAccess:
    """Explicit retirement notice for the incompatible revision-two task."""

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "SpecimenAccess revision 2 is retired. Use StageInspection with the explicit seven-axis revision 3 interface; old action keys are not mapped."
        )
