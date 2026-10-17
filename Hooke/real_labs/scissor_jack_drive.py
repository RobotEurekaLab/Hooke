"""Measured bath-height control through the authored jack's real torque motor.

The session owns forwarding, stepping and native warning/reset detection. This
controller only writes its own motor control; it never changes a physical pose.
The ideal screw lead has no thread friction or self-locking model.
"""

import math
from numbers import Real

import mujoco
import numpy as np


TORQUE_LIMIT_NM = 0.2
HEIGHT_TOLERANCE_M = 0.0005
CLOSURE_TOLERANCE_M = 0.00005
_ROUND_OFF_M = 1e-8
_STIFFNESS = 40000.0
_JOINTS = (
    "nut_translation", "top_slot_translation", "screw_rotation", "a1_angle",
    "b1_angle", "a2_angle", "b2_angle", "deck_pitch",
)
_EQUALITIES = ("lower_scissor", "upper_scissor", "deck_slot", "screw_transmission")
_ANGLE_FACTORS = (("a1_angle", 1), ("b1_angle", 1), ("a2_angle", 2),
                  ("b2_angle", 2), ("deck_pitch", 1))


def _finite(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


class ScissorJackDrive:
    """One height trajectory and hold controller; all heights use the bench datum."""

    units = "m"
    mode = "mechanism_position"

    def __init__(self, model, metadata):
        # The geometry module owns the kinematic constants. Importing this
        # adapter alone does not require constructing or importing a scene.
        from real_labs import epfl_jack_geometry as jack

        self.model, self.jack = model, jack
        if not isinstance(metadata, dict) or metadata.get("kind") != "scissor_jack_v1":
            raise ValueError("Expected scissor_jack_v1 mechanism metadata")
        self.joints = self._objects(metadata.get("joints"), _JOINTS, mujoco.mjtObj.mjOBJ_JOINT)
        self.equalities = self._objects(metadata.get("equalities"), _EQUALITIES, mujoco.mjtObj.mjOBJ_EQUALITY)
        self.actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, metadata.get("actuator"))
        self.deck_site = self._id(mujoco.mjtObj.mjOBJ_SITE, metadata.get("deck_site"))
        self.datum_site = self._id(mujoco.mjtObj.mjOBJ_SITE, metadata.get("datum_site"))
        self.observation_site = self._id(mujoco.mjtObj.mjOBJ_SITE, metadata.get("observation_site"))
        limits = metadata.get("range")
        if not isinstance(limits, (tuple, list)) or len(limits) != 2:
            raise ValueError("Bath height range must have two finite bounds")
        self.lower, self.upper = (_finite(v, "Height bound") for v in limits)
        self.offset = _finite(metadata.get("deck_to_observation_offset_m"), "Observation offset")
        if not math.isclose(self.offset, .140, rel_tol=0, abs_tol=1e-12) or not np.allclose(
            [self.lower, self.upper], [jack.MIN_HEIGHT + self.offset, jack.MAX_HEIGHT + self.offset],
            rtol=0, atol=1e-12,
        ):
            raise ValueError("Bath height range/offset is incompatible with the authored jack")
        self.qpos = {n: int(model.jnt_qposadr[j]) for n, j in self.joints.items()}
        self.dof = {n: int(model.jnt_dofadr[j]) for n, j in self.joints.items()}
        self.bodies = {n: int(model.jnt_bodyid[j]) for n, j in self.joints.items()}
        self._validate_geometry()
        self._validate_motor()
        self._validate_equalities()
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        self._datum_position = probe.site_xpos[self.datum_site].copy()
        self._datum_rotation = probe.site_xmat[self.datum_site].copy()
        self._check_datum(probe)
        if not np.allclose(probe.xmat[self.base_body].reshape(3, 3)[:, 2], [0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("Jack base must be upright in world coordinates")
        if not np.allclose(
            probe.site_xpos[self.observation_site] - probe.site_xpos[self.deck_site],
            [0, 0, self.offset], rtol=0, atol=1e-9,
        ) or not math.isclose(self.measured(probe), self.lower, rel_tol=0, abs_tol=_ROUND_OFF_M):
            raise ValueError("Bath/deck/datum sites do not describe the authored reset geometry")
        self._configuration = self._configuration_arrays()
        self.inertia = np.empty((model.nv, model.nv))
        self.tangent = np.zeros(model.nv)
        self.target = None
        self.requested_duration_s = self.motion_duration_s = 0.0
        self.requested_torque_Nm = self.applied_torque_Nm = 0.0
        self.saturated = False
        self.effective_mass_kg = None
        self._last_time = None
        self._last_control = None

    def _id(self, kind, name):
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            raise ValueError("Mechanism bindings require nonempty object names")
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Unknown mechanism object: {name}")
        return int(result)

    def _objects(self, mapping, expected, kind):
        if not isinstance(mapping, dict) or set(mapping) != set(expected):
            raise ValueError("Mechanism bindings must declare every semantic exactly once")
        result = {n: self._id(kind, mapping[n]) for n in expected}
        if len(set(result.values())) != len(expected):
            raise ValueError("Mechanism object bindings must be distinct")
        return result

    def _validate_motor(self):
        m, a = self.model, self.actuator
        if (
            m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
            or m.actuator_trnid[a, 0] != self.joints["screw_rotation"]
            or not np.array_equal(m.actuator_gear[a], [1, 0, 0, 0, 0, 0])
            or m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
            or m.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
            or m.actuator_gainprm[a, 0] != 1
            or m.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE
            or not m.actuator_ctrllimited[a]
            or not np.allclose(m.actuator_ctrlrange[a], [-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM], rtol=0, atol=1e-12)
            or (m.actuator_forcelimited[a] and (m.actuator_forcerange[a, 0] > -TORQUE_LIMIT_NM or m.actuator_forcerange[a, 1] < TORQUE_LIMIT_NM))
        ):
            raise ValueError("Jack requires its unit-gear torque motor with +/-0.2 Nm limits")

    def _validate_geometry(self):
        m, j = self.model, self.jack
        for name, joint in self.joints.items():
            slide = name in {"nut_translation", "top_slot_translation"}
            kind = mujoco.mjtJoint.mjJNT_SLIDE if slide else mujoco.mjtJoint.mjJNT_HINGE
            axis = [1, 0, 0] if slide or name == "screw_rotation" else [0, -1 if name in {"a1_angle", "a2_angle", "deck_pitch"} else 1, 0]
            if (m.jnt_type[joint] != kind or not np.allclose(m.jnt_axis[joint], axis, rtol=0, atol=1e-12)
                    or not np.allclose(m.jnt_pos[joint], 0, rtol=0, atol=1e-12)
                    or m.qpos0[self.qpos[name]] != 0
                    or m.body_jntnum[self.bodies[name]] != 1
                    or m.body_gravcomp[self.bodies[name]] != 0):
                raise ValueError("Jack joint geometry differs from the authored scalar mechanism")
        nut = self.joints["nut_translation"]
        if not m.jnt_limited[nut] or not np.allclose(m.jnt_range[nut], [j.MIN_WIDTH-j.WIDTH, 0], rtol=0, atol=1e-12):
            raise ValueError("Jack nut travel differs from the authored range")
        if any(m.jnt_limited[joint] for name, joint in self.joints.items() if name != "nut_translation"):
            raise ValueError("Only the jack nut may have an authored joint limit")
        for name, parent, position, angle in (
            ("b1_angle", "nut_translation", (0, 0, 0), j.ANGLE),
            ("a2_angle", "b1_angle", (-j.LENGTH, 0, 0), -2*j.ANGLE),
            ("b2_angle", "a1_angle", (j.LENGTH, 0, 0), 2*j.ANGLE),
            ("deck_pitch", "b2_angle", (-j.LENGTH, 0, 0), -j.ANGLE),
            ("top_slot_translation", "deck_pitch", (j.WIDTH, 0, 0), 0),
        ):
            body = self.bodies[name]
            quat = np.array([math.cos(angle/2), 0, math.sin(angle/2), 0])
            if (m.body_parentid[body] != self.bodies[parent]
                    or not np.allclose(m.body_pos[body], position, rtol=0, atol=1e-12)
                    or not math.isclose(abs(float(m.body_quat[body] @ quat)), 1, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Jack link topology or dimensions are incompatible")
        base = int(m.body_parentid[self.bodies["a1_angle"]])
        self.base_body = base
        mount = int(m.body_parentid[base])
        for body, parent, position, angle in (
            (self.bodies["a1_angle"], base, (0, 0, 0), -j.ANGLE),
            (self.bodies["nut_translation"], base, (j.WIDTH, 0, 0), 0),
            (self.bodies["screw_rotation"], mount, (0, 0, j.SCREW_Z), 0),
            (base, mount, (-.075, 0, j.PIVOT_Z), 0),
        ):
            quat = np.array([math.cos(angle/2), 0, math.sin(angle/2), 0])
            if (m.body_parentid[body] != parent
                    or not np.allclose(m.body_pos[body], position, rtol=0, atol=1e-12)
                    or not math.isclose(abs(float(m.body_quat[body] @ quat)), 1, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Jack base, nut or screw geometry is incompatible")
        if (m.body_parentid[self.bodies["nut_translation"]] != base
                or not np.allclose(m.body_pos[self.bodies["nut_translation"]], [j.WIDTH, 0, 0], rtol=0, atol=1e-12)
                or m.site_bodyid[self.deck_site] != self.bodies["deck_pitch"]
                or not np.allclose(m.site_pos[self.deck_site], [.075, 0, j.DECK_OFFSET], rtol=0, atol=1e-12)
                or m.body_weldid[m.site_bodyid[self.observation_site]] != m.body_weldid[self.bodies["deck_pitch"]]):
            raise ValueError("Jack deck and observation bindings cross mechanism ownership")
        for body in (base, int(m.site_bodyid[self.datum_site])):
            while body:
                if m.body_jntnum[body]:
                    raise ValueError("Jack mount and bench datum must be stationary")
                body = int(m.body_parentid[body])

    def _validate_equalities(self):
        m, j = self.model, self.jack
        for semantic, first, second, anchor, second_anchor in (
            ("lower_scissor", "a1_angle", "b1_angle", (j.LENGTH/2, 0, 0), (-j.LENGTH/2, 0, 0)),
            ("upper_scissor", "a2_angle", "b2_angle", (j.LENGTH/2, 0, 0), (-j.LENGTH/2, 0, 0)),
            ("deck_slot", "a2_angle", "top_slot_translation", (j.LENGTH, 0, 0), (0, 0, 0)),
        ):
            eq = self.equalities[semantic]
            if (m.eq_type[eq] != mujoco.mjtEq.mjEQ_CONNECT
                    or m.eq_obj1id[eq] != self.bodies[first]
                    or m.eq_obj2id[eq] != self.bodies[second]
                    or not np.allclose(m.eq_data[eq, :3], anchor, rtol=0, atol=1e-12)
                    or not np.allclose(m.eq_data[eq, 3:6], second_anchor, rtol=0, atol=1e-12)):
                raise ValueError("Jack loop equality has incompatible endpoints or anchors")
        eq = self.equalities["screw_transmission"]
        if (m.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT
                or m.eq_obj1id[eq] != self.joints["screw_rotation"]
                or m.eq_obj2id[eq] != self.joints["nut_translation"]
                or not np.allclose(m.eq_data[eq, :5], [0, 1/j.METRES_PER_RADIAN, 0, 0, 0], rtol=1e-12, atol=1e-12)):
            raise ValueError("Jack screw transmission must use the authored radian equation")
        if not all(m.eq_active0[e] for e in self.equalities.values()):
            raise ValueError("All jack equalities must be active at reset")

    def _configuration_arrays(self):
        m = self.model
        ids = list(self.joints.values()); bodies = list(self.bodies.values())+[self.base_body]
        eq = list(self.equalities.values()); sites = [self.deck_site, self.datum_site, self.observation_site]
        selections = [(name, ids) for name in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_bodyid")]
        selections += [(name, bodies) for name in ("body_parentid", "body_pos", "body_quat", "body_gravcomp")]
        selections += [(name, eq) for name in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_active0", "eq_solref", "eq_solimp")]
        selections += [(name, sites) for name in ("site_bodyid", "site_pos", "site_quat")]
        selections += [(name, [self.actuator]) for name in ("actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_dyntype", "actuator_gaintype", "actuator_gainprm", "actuator_biastype", "actuator_biasprm", "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange")]
        return tuple((name, indices, getattr(m, name)[indices].copy()) for name, indices in selections)

    def _check_datum(self, data):
        z = data.site_xmat[self.datum_site].reshape(3, 3)[:, 2]
        if not np.allclose(z, [0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("Jack bench datum must be upright in world coordinates")
        if (not np.allclose(data.site_xpos[self.datum_site], self._datum_position, rtol=0, atol=1e-10)
                or not np.allclose(data.site_xmat[self.datum_site], self._datum_rotation, rtol=0, atol=1e-10)):
            raise RuntimeError("scissor_jack_datum_moved")

    def _state(self, data, *, configuration=True):
        if data.model is not self.model:
            raise ValueError("Jack controller requires data owned by its model")
        if not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl, data.qfrc_bias, data.qM, np.asarray(data.time))):
            raise RuntimeError("scissor_jack_nonfinite_state")
        self._check_datum(data)
        if configuration and any(not np.array_equal(getattr(self.model, n)[ids], saved) for n, ids, saved in self._configuration):
            raise RuntimeError("scissor_jack_configuration_changed")
        if not all(data.eq_active[e] for e in self.equalities.values()):
            raise RuntimeError("scissor_jack_equality_disabled")

    def measured(self, data):
        if data.model is not self.model:
            raise ValueError("Jack controller requires data owned by its model")
        result = float(data.site_xpos[self.observation_site, 2] - data.site_xpos[self.datum_site, 2])
        if not math.isfinite(result):
            raise RuntimeError("scissor_jack_nonfinite_measurement")
        return result

    def reset(self, data):
        """Capture a holding target after the session forwards its reset state."""
        self._state(data)
        measured = self.measured(data)
        if not self.lower-_ROUND_OFF_M <= measured <= self.upper+_ROUND_OFF_M:
            raise ValueError("Reset bath height is outside the authored range")
        self.target = min(self.upper, max(self.lower, measured))
        self._start_height, self._start_time = self.target, float(data.time)
        self.requested_duration_s = self.motion_duration_s = 0.0
        self.requested_torque_Nm = self.applied_torque_Nm = 0.0
        self.saturated = False
        self.effective_mass_kg = None
        self._last_time = float(data.time)
        self._last_control = float(data.ctrl[self.actuator])

    def start(self, data, target, duration):
        """Start a cubic trajectory and return its possibly extended duration."""
        self._state(data)
        target, duration = _finite(target, "Target"), _finite(duration, "Duration")
        if not self.lower <= target <= self.upper:
            raise ValueError("Bath height target is outside the authored range")
        if not 0 < duration <= 30:
            raise ValueError("Duration must be in (0, 30] seconds")
        if self.target is None:
            raise RuntimeError("scissor_jack_requires_reset")
        start = self.measured(data)
        if not self.lower-HEIGHT_TOLERANCE_M <= start <= self.upper+HEIGHT_TOLERANCE_M:
            raise RuntimeError("scissor_jack_height_out_of_range")
        self._start_height = min(self.upper, max(self.lower, start))
        self._start_time = float(data.time)
        self.target = target
        self.requested_duration_s = duration
        self.motion_duration_s = max(duration, 15*abs(target-self._start_height)/(self.upper-self.lower))
        return self.motion_duration_s

    def _reference(self, time):
        if not self.motion_duration_s:
            return self.target, 0.0
        u = min(1.0, max(0.0, (time-self._start_time)/self.motion_duration_s))
        delta = self.target-self._start_height
        return (self._start_height+delta*(3*u*u-2*u*u*u),
                delta*(6*u-6*u*u)/self.motion_duration_s)

    def update(self, data):
        """Write one torque from forwarded state; fail explicitly on saturation."""
        self._state(data)
        if self.target is None:
            raise RuntimeError("scissor_jack_requires_reset")
        if data.time < self._last_time-1e-12:
            raise RuntimeError("scissor_jack_time_reversed")
        if not math.isclose(float(data.ctrl[self.actuator]), self._last_control, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("scissor_jack_control_changed")
        height, speed = self._reference(float(data.time))
        if abs(self.measured(data)-height) > HEIGHT_TOLERANCE_M:
            raise RuntimeError("scissor_jack_height_tracking_error")
        equality_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        for name, eq in self.equalities.items():
            rows = equality_rows & (data.efc_id == eq)
            factor = self.jack.METRES_PER_RADIAN if name == "screw_transmission" else 1
            if not np.any(rows) or np.max(np.abs(data.efc_pos[rows]))*factor > CLOSURE_TOLERANCE_M:
                raise RuntimeError("scissor_jack_loop_closure_error")
        j = self.jack
        deck_height = min(j.MAX_HEIGHT, max(j.MIN_HEIGHT, height-self.offset))
        target = j.height_to_nut(deck_height)
        theta = j.ANGLE+data.qpos[self.qpos["a1_angle"]]
        if not math.isfinite(theta) or math.sin(theta) <= 1e-5:
            raise RuntimeError("scissor_jack_singular_configuration")
        tangent = self.tangent
        tangent.fill(0)
        tangent[self.dof["nut_translation"]] = tangent[self.dof["top_slot_translation"]] = 1
        tangent[self.dof["screw_rotation"]] = 1/j.METRES_PER_RADIAN
        for name, factor in _ANGLE_FACTORS:
            tangent[self.dof[name]] = -factor/(j.LENGTH*np.sin(theta))
        target_speed = -(deck_height-j.PIVOT_Z-j.DECK_OFFSET)*speed/(4*(j.WIDTH+target))
        mujoco.mj_fullM(self.model, self.inertia, data.qM)
        effective_mass = float(tangent @ self.inertia @ tangent)
        if not math.isfinite(effective_mass) or effective_mass <= 0:
            raise RuntimeError("scissor_jack_invalid_effective_mass")
        damping = 2*np.sqrt(_STIFFNESS*effective_mass)
        measured = j.METRES_PER_RADIAN*data.qpos[self.qpos["screw_rotation"]]
        measured_speed = j.METRES_PER_RADIAN*data.qvel[self.dof["screw_rotation"]]
        force = float(tangent @ data.qfrc_bias)+_STIFFNESS*(target-measured)+damping*(target_speed-measured_speed)
        requested = float(force*j.METRES_PER_RADIAN)
        if not math.isfinite(requested):
            raise RuntimeError("scissor_jack_nonfinite_torque")
        applied = float(np.clip(requested, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM))
        self.requested_torque_Nm, self.applied_torque_Nm = requested, applied
        self.effective_mass_kg = effective_mass
        self.saturated = abs(requested) > TORQUE_LIMIT_NM
        data.ctrl[self.actuator] = applied
        self._last_control, self._last_time = applied, float(data.time)
        if self.saturated:
            raise RuntimeError("scissor_jack_torque_saturation")

    def inspect(self, data):
        return dict(measured=self.measured(data), target=self.target,
                    range=[self.lower, self.upper], units=self.units, mode=self.mode,
                    requested_torque_Nm=self.requested_torque_Nm,
                    applied_torque_Nm=self.applied_torque_Nm, saturated=self.saturated,
                    torque_limit_Nm=TORQUE_LIMIT_NM, effective_mass_kg=self.effective_mass_kg,
                    requested_duration_s=self.requested_duration_s,
                    motion_duration_s=self.motion_duration_s,
                    duration_extended=self.motion_duration_s > self.requested_duration_s,
                    observation="Bath rim height above the stationary bench datum",
                    torque_observation="Requested and bounded motor control input; not calibrated shaft torque")
