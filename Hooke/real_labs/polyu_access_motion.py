"""Version-three PolyU access and contact operations on an InstrumentSession.

Commands step the caller's actual model/data. The free coupon is retained by
physical support/jaws/lips; no weld, live teleport or calibrated material model
is used. Arbitrary raw controls remain capable of colliding.
"""

import itertools
import math

import mujoco
import numpy as np

from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.polyu_access_geometry import OLD, NEW, EQUIPMENT
from real_labs.polyu_fixture_contacts import finite_face_gap, retain_pairs
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import require_binding

SCENE = "polyu_materials_tribology"
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


class _FixtureMotion:
    def __init__(self, session, *, extra_contact_geoms=()):
        self.binding = require_binding(session, SCENE)
        self.session = session
        self.m, self.d = session.model, session.data
        m, d = self.m, self.d
        self.metadata = session.manifest
        item = session.equipment[EQUIPMENT]
        if (
            item.get("interface_revision") != 3
            or item["contact_interface"]["revision"] != 3
        ):
            raise ValueError("PolyU free-coupon operations require fixture revision 3")
        self.bound_model, self.bound_data = m, d
        self.signature = _model_signature(m)
        item = next(v for v in self.metadata["equipment"] if v["id"] == "umt_station")
        self.keys = tuple(item["joints"])
        self.joints = np.array([m.joint(item["joints"][k]).id for k in self.keys])
        self.qa = m.jnt_qposadr[self.joints].copy()
        self.da = m.jnt_dofadr[self.joints].copy()
        self.aa = np.array([m.actuator(item["actuators"][k]).id for k in self.keys])
        self.limits = m.jnt_range[self.joints].copy()
        self.ids = {k: i for i, k in enumerate(self.keys)}
        self.coupon = m.geom(OLD + "coupon").id
        self.puck = m.geom(OLD + "specimen_puck").id
        self.slider = m.body("umt_station__body_coupon_x").id
        self.free = m.body(NEW + "free_coupon").id
        self.pads = [m.geom(NEW + "jaw_pad_" + s).id for s in ("front", "rear")]
        self.stops = [
            m.geom("umt_retention__stop_" + s).id for s in ("negative", "positive")
        ]
        self.contact_counterparts = (
            [self.puck] + self.pads + [m.geom(name).id for name in extra_contact_geoms]
        )
        self.allowed = {frozenset((self.coupon, g)) for g in self.contact_counterparts}
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        weld = m.body_weldid[m.geom_bodyid]
        self.pairs = np.array(
            [
                (int(a), int(b))
                for a, b in itertools.combinations(enabled, 2)
                if weld[a] != weld[b] and frozenset((a, b)) not in self.allowed
            ]
        )
        self.preflight()
        # Constructor caches use private derived frames only. Do not run
        # dynamics or normalize the caller's integration state at a handoff.
        self._initial_frames = mujoco.MjData(m)
        self._initial_frames.qpos[:] = d.qpos
        mujoco.mj_kinematics(m, self._initial_frames)
        self.relative = (
            self._initial_frames.site_xpos[
                m.site(EQUIPMENT + "__site_loading_center").id
            ]
            - self._initial_frames.xpos[self.slider]
        )
        self.trace = []
        self.observations = []
        self.preflight()

    def preflight(self):
        require_binding(self.session, SCENE)
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or m is not self.bound_model
            or d is not self.bound_data
            or d.model is not m
            or _model_signature(m) != self.signature
        ):
            raise ValueError("Candidate model/state binding changed")
        if not all(np.isfinite(getattr(d, f)).all() for f in FIELDS) or not np.isfinite(
            d.time
        ):
            raise ValueError("Candidate requires finite complete state and clock")
        if np.max(abs(d.qvel), initial=0) > 0.1:
            raise ValueError("Candidate requires a settled initial state")
        if any(
            d.warning[w].number
            for w in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
        ):
            raise ValueError("Candidate requires clean native bad-state history")

    def state(self):
        return dict(
            time_s=float(self.d.time),
            **{f: getattr(self.d, f).tolist() for f in FIELDS}
        )

    def observe(self):
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or m is not self.bound_model
            or d is not self.bound_data
            or d.model is not m
            or not np.isfinite(d.time)
            or not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
        ):
            raise RuntimeError("state_binding_or_finiteness")
        q = d.qpos[self.qa]
        if np.any(q < self.limits[:, 0] - 1e-6) or np.any(q > self.limits[:, 1] + 1e-6):
            raise RuntimeError("joint_limit")
        for k, i in self.ids.items():
            speed_limit = 2.5 if k.startswith("guard") else 0.1
            error_limit = 0.03 if k.startswith("guard") else 0.003
            if abs(d.qvel[self.da[i]]) > speed_limit:
                raise RuntimeError("velocity:" + k)
            if abs(q[i] - d.ctrl[self.aa[i]]) > error_limit:
                raise RuntimeError("tracking:" + k)
        rotation = d.geom_xmat.reshape(-1, 3, 3)
        centers = d.geom_xpos + np.einsum("nij,nj->ni", rotation, m.geom_aabb[:, :3])
        half = np.einsum("nij,nj->ni", abs(rotation), m.geom_aabb[:, 3:])
        a, b = self.pairs.T
        lower = np.linalg.norm(
            np.maximum(abs(centers[a] - centers[b]) - half[a] - half[b], 0), axis=1
        )
        minimum = 0.002
        queries = 0
        for a, b in self.pairs[lower < 0.002]:
            gap = min(
                0.002, float(mujoco.mj_geomDistance(m, d, int(a), int(b), 0.002, None))
            )
            minimum = min(minimum, gap)
            queries += 1
            if gap < 0.00025:
                raise RuntimeError(
                    "clearance:"
                    + m.geom(int(a)).name
                    + ":"
                    + m.geom(int(b)).name
                    + ":"
                    + str(gap)
                )
        depth = 0.0
        forces = {m.geom(g).name: 0.0 for g in self.contact_counterparts}
        for i in range(d.ncon):
            c = d.contact[i]
            pair = frozenset((c.geom1, c.geom2))
            if pair in self.allowed:
                depth = max(depth, -float(c.dist))
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                g = c.geom1 if c.geom2 == self.coupon else c.geom2
                forces[m.geom(g).name] += float(f[0])
            elif c.dist < -1e-7:
                if (
                    m.body_weldid[m.geom_bodyid[c.geom1]]
                    != m.body_weldid[m.geom_bodyid[c.geom2]]
                ):
                    raise RuntimeError("unexpected_native_contact")
        # Exact box/box tangency has a native distance degeneracy (the reset
        # query returns -12 mm despite coincident top/bottom faces and zero
        # overlap). For this declared horizontal support, compute its actual
        # plane separation, after proving the entire coupon projection lies
        # inside the finite support footprint; retain native contact forces.
        coupon_half = (
            abs(d.geom_xmat[self.coupon].reshape(3, 3)) @ m.geom_size[self.coupon]
        )
        puck_half = m.geom_size[self.puck]
        offset = d.geom_xpos[self.coupon] - d.geom_xpos[self.puck]
        if np.any(abs(offset[:2]) + coupon_half[:2] > puck_half[:2]):
            raise RuntimeError("coupon_leaves_finite_support_footprint")
        if not np.allclose(
            d.geom_xmat[self.puck].reshape(3, 3), np.eye(3), atol=1e-12, rtol=0
        ):
            raise RuntimeError("support_plane_not_horizontal")
        direct_support = float(offset[2] - coupon_half[2] - puck_half[2])
        depth = max(depth, -direct_support)
        face_gaps = {}
        for sign, pad in zip((-1, 1), self.pads):
            gap = finite_face_gap(m, d, self.coupon, pad, sign)
            face_gaps[m.geom(pad).name] = gap
            if gap is not None:
                depth = max(depth, -gap)
        stop_gaps = {}
        for sign, stop in zip((-1, 1), self.stops):
            gap = finite_face_gap(m, d, self.coupon, stop, sign, axis=0)
            stop_gaps[m.geom(stop).name] = gap
            if gap is not None:
                depth = max(depth, -gap)
        if depth > 0.0001:
            raise RuntimeError("support_or_clamp_penetration")
        relative = d.geom_xpos[self.coupon] - d.xpos[self.slider]
        if (
            np.linalg.norm((relative - self.relative)[:2]) > 0.0005
            or abs(relative[2] - self.relative[2]) > 0.0001
        ):
            raise RuntimeError("coupon_retention")
        tilt = math.acos(np.clip(d.geom_xmat[self.coupon].reshape(3, 3)[2, 2], -1, 1))
        if tilt > math.radians(0.5):
            raise RuntimeError("coupon_tilt")
        return dict(
            minimum_capped_clearance_m=minimum,
            direct_queries=queries,
            coupon_relative_m=relative.tolist(),
            coupon_tilt_rad=tilt,
            maximum_intended_contact_depth_m=depth,
            normal_forces_N=forces,
            direct_support_gap_m=direct_support,
            finite_jaw_face_gaps_m=face_gaps,
            finite_endstop_face_gaps_m=stop_gaps,
        )

    def hold(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.m is not self.bound_model
            or self.d is not self.bound_data
            or self.d.model is not self.m
        ):
            return False
        self.d.ctrl[self.aa] = np.clip(
            self.d.qpos[self.qa], self.limits[:, 0], self.limits[:, 1]
        )
        return True

    def check_endpoint(self, target, observation, *, ignore=()):
        for key, index in self.ids.items():
            if key in ignore:
                continue
            tolerance = 0.002 if key.startswith("guard") else 0.00015
            if abs(self.d.qpos[self.qa[index]] - target[index]) > tolerance:
                raise RuntimeError("endpoint:" + key)

    def move(self, changes, duration, settle=0.5):
        self.preflight()
        if any(k not in self.ids for k in changes):
            raise ValueError("Unknown version3 key")
        target = self.d.ctrl[self.aa].copy()
        for k, v in changes.items():
            i = self.ids[k]
            if not np.isfinite(v) or not self.limits[i, 0] <= v <= self.limits[i, 1]:
                raise ValueError("Target outside full public range")
            target[i] = v
        if not np.isfinite([duration, settle]).all() or duration <= 0 or settle < 0:
            raise ValueError("Invalid duration")
        m, d = self.m, self.d
        dt = m.opt.timestep
        start = d.ctrl[self.aa].copy()
        ramp = max(1, round(duration / dt))
        dwell = round(settle / dt)
        report = dict(
            status="failed",
            changes=changes,
            physics_steps=0,
            checked_steps=0,
            observer_entries=0,
            initial_state=self.state(),
            failure=None,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            hold_requested=False,
            trace=[],
        )
        minimum = 0.002
        depth = 0.0
        queries = 0
        obs = None
        before_warning = _warning_state(d)
        try:
            self.observe()
            for step in range(ramp + dwell):
                t = min(1, (step + 1) / ramp)
                blend = t * t * (3 - 2 * t)
                d.ctrl[self.aa] = start + blend * (target - start)
                before = float(d.time)
                inputs = {
                    f: getattr(d, f).copy()
                    for f in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                report["physics_steps"] += 1
                event, before_warning = _physics_event(
                    m, d, before, before_warning, inputs
                )
                if event is not None:
                    report.update(
                        {
                            k: event[k]
                            for k in ("reset_performed", "automatic_reset_detected")
                        }
                    )
                    report["physics_event"] = event
                    raise RuntimeError("native_warning_or_clock")
                mujoco.mj_forward(m, d)
                report["observer_entries"] += 1
                obs = self.observe()
                report["checked_steps"] += 1
                minimum = min(minimum, obs["minimum_capped_clearance_m"])
                depth = max(depth, obs["maximum_intended_contact_depth_m"])
                queries += obs["direct_queries"]
                self.trace.append(np.r_[d.time, d.qpos, d.qvel, d.ctrl])
                self.observations.append(obs)
                if step % 100 == 0 or step + 1 == ramp + dwell:
                    report["trace"].append(dict(**self.state(), observation=obs))
            self.check_endpoint(target, obs)
            report["status"] = "succeeded"
        except RuntimeError as e:
            report["failure"] = str(e)
            report["hold_requested"] = self.hold()
        complete = (
            report["physics_steps"] > 0
            and report["physics_steps"] == report["checked_steps"]
        )
        report.update(
            final_state=self.state(),
            final_observation=obs,
            coverage_complete=complete,
            direct_queries=queries,
            maximum_intended_contact_depth_m=depth,
            whole_leg_capped_clearance_lower_bound_m=minimum if complete else None,
            checked_steps_capped_clearance_lower_bound_m=(
                minimum if report["checked_steps"] else None
            ),
        )
        return report


class _ContactFixture(_FixtureMotion):
    def __init__(self, session):
        self.mode = "access"
        super().__init__(
            session,
            extra_contact_geoms=(
                OLD + "counterface",
                "umt_retention__stop_negative",
                "umt_retention__stop_positive",
            ),
        )
        self.tip = self.m.geom(OLD + "counterface").id
        self.ball_pair = frozenset((self.tip, self.coupon))
        self.full_pairs = self.pairs.copy()
        self.pairs, self.pruning = retain_pairs(
            self.m, self._initial_frames, self.full_pairs
        )
        del self._initial_frames
        self.jac = np.zeros((2, 3, self.m.nv))

    def observe(self):
        observation = super().observe()
        m, d = self.m, self.d
        force = friction = speed = depth = 0.0
        world = np.zeros(3)
        count = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if frozenset((c.geom1, c.geom2)) != self.ball_pair:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            frame = c.frame.reshape(3, 3)
            force += float(f[0])
            friction += float(np.linalg.norm(f[1:3]))
            depth = max(depth, -float(c.dist))
            count += 1
            world += (1 if c.geom2 == self.tip else -1) * (frame.T @ f[:3])
            for k, g in enumerate((self.tip, self.coupon)):
                mujoco.mj_jac(m, d, self.jac[k], None, c.pos, int(m.geom_bodyid[g]))
            relative = (self.jac[0] - self.jac[1]) @ d.qvel
            speed += float(
                np.linalg.norm(relative - frame[0] * np.dot(relative, frame[0]))
            )
        gap = min(
            0.002,
            float(mujoco.mj_geomDistance(m, d, self.tip, self.coupon, 0.002, None)),
        )
        if self.mode == "access":
            if gap < 0.00025 or force > 1e-7:
                raise RuntimeError("access_counterface_not_clear")
        else:
            for k in ("guard_left", "guard_right"):
                i = self.ids[k]
                if abs(d.qpos[self.qa[i]]) > 0.001 or abs(d.ctrl[self.aa[i]]) > 1e-12:
                    raise RuntimeError("loaded_guard_not_closed:" + k)
            for k, pad in zip(("clamp_front", "clamp_rear"), self.pads):
                i = self.ids[k]
                if (
                    abs(d.ctrl[self.aa[i]] + 0.0001) > 1e-12
                    or d.qpos[self.qa[i]] > 0.00005
                ):
                    raise RuntimeError("loaded_clamp_not_engaged:" + k)
                if observation["normal_forces_N"][m.geom(pad).name] < 0.02:
                    raise RuntimeError("loaded_jaw_contact_lost:" + k)
            if force > 2 or max(depth, -gap) > 0.0001:
                raise RuntimeError("ball_load_or_depth_limit")
            if self.mode == "slide" and force < 0.02:
                raise RuntimeError("loaded_ball_contact_lost")
        observation.update(
            normal_force_N=force,
            tangential_force_N=friction,
            force_on_counterface_world_N=world.tolist(),
            native_ball_contacts=count,
            ball_penetration_m=max(0.0, depth, -gap),
            ball_gap_capped_m=gap,
            relative_tangential_speed_m_s=speed / count if count else 0.0,
            operation_mode=self.mode,
        )
        return observation

    def check_endpoint(self, target, observation, *, ignore=()):
        if self.mode in ("load", "slide"):
            super().check_endpoint(target, observation, ignore=("probe_height",))
            if not 0.05 <= observation["normal_force_N"] <= 0.3:
                raise RuntimeError("settled_ball_load_not_established")
            if (
                observation["ball_gap_capped_m"] > 0
                or observation["ball_penetration_m"] > 0.0001
            ):
                raise RuntimeError("physical_counterface_contact_not_established")
        else:
            super().check_endpoint(target, observation)
            if self.mode == "retract" and (
                observation["native_ball_contacts"]
                or observation["ball_gap_capped_m"] < 0.001
            ):
                raise RuntimeError("counterface_not_physically_retracted")

    def move_contact(self, changes, duration, *, mode, settle=0.5):
        if mode not in ("access", "load", "slide", "retract"):
            raise ValueError("Unknown inspection mode")
        if mode != "access" and any(
            k not in ("coupon_x", "probe_height") for k in changes
        ):
            raise ValueError("Loaded phases may command only the original X/Z axes")
        self.mode = mode
        return self.move(changes, duration, settle)

    def _summary(self, phases, count):
        executed = sum(p["physics_steps"] for p in phases)
        checked = sum(p["checked_steps"] for p in phases)
        return dict(
            status=(
                "succeeded"
                if len(phases) == count
                and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            phases=phases,
            physics_steps=executed,
            checked_steps=checked,
            coverage_complete=executed > 0 and executed == checked,
            pruning=self.pruning,
            **combine_reset_reports(phases),
            limitations=[
                "Authored source-family fixture, not exact OEM hardware",
                "Uncalibrated rigid contact/friction, no wear or material-science model",
                "No autonomous sample pickup/transfer",
                "Discrete checked poses do not prove continuous clearance",
            ]
        )


class FreeCouponReciprocation(_ContactFixture):
    """Observed loaded reciprocation of a physically retained free coupon."""

    def run(self):
        """Clamp, approach, slide -10/+10/0 mm under measured contact, retract."""
        phases = []
        for name, changes, duration, mode in (
            (
                "clamp_free_coupon",
                {"clamp_front": -0.0001, "clamp_rear": -0.0001},
                0.5,
                "access",
            ),
            ("approach_and_load", {"coupon_x": 0, "probe_height": -0.0405}, 3, "load"),
            (
                "slide_negative",
                {"coupon_x": -0.010, "probe_height": -0.0405},
                8,
                "slide",
            ),
            (
                "slide_positive",
                {"coupon_x": 0.010, "probe_height": -0.0405},
                16,
                "slide",
            ),
            ("slide_center", {"coupon_x": 0, "probe_height": -0.0405}, 8, "slide"),
            ("retract", {"coupon_x": 0, "probe_height": 0}, 3, "retract"),
        ):
            before = len(self.observations)
            phase = self.move_contact(changes, duration, mode=mode)
            actual = self.observations[before:]
            phase["actual_loaded_steps"] = sum(
                o["normal_force_N"] > 0.02 for o in actual
            )
            phase["measured_loaded_relative_path_m"] = sum(
                o["relative_tangential_speed_m_s"] * self.m.opt.timestep
                for o in actual
                if o["normal_force_N"] > 0.02
            )
            phases.append(dict(phase=name, **phase))
            if phase["status"] != "succeeded":
                break
        return self._summary(phases, 6)


class CouponAccess(_ContactFixture):
    """Raise tool, expose and release the supported coupon, then return."""

    def run(self):
        phases = []
        for name, changes, duration in (
            ("settle_and_clamp", {"clamp_front": -0.0001, "clamp_rear": -0.0001}, 0.5),
            ("retention_positive", {"coupon_x": 0.003}, 0.8),
            ("retention_return", {"coupon_x": 0}, 0.8),
            ("raise_counterface", {"probe_height": 0.06}, 3),
            (
                "open_guards",
                {"guard_left": math.radians(100), "guard_right": math.radians(100)},
                4,
            ),
            ("release_coupon", {"clamp_front": 0.015, "clamp_rear": 0.015}, 1),
            ("released_support", {}, 0.5),
            ("reclamp", {"clamp_front": -0.0001, "clamp_rear": -0.0001}, 1),
            ("close_guards", {"guard_left": 0, "guard_right": 0}, 4),
            ("return_counterface", {"probe_height": 0}, 3),
        ):
            phase = self.move_contact(changes, duration, mode="access")
            phases.append(dict(phase=name, **phase))
            if phase["status"] != "succeeded":
                break
        return self._summary(phases, 10)
