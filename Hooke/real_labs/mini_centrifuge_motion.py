"""Measured inert-gauge loading, released rotation and return to a cradle.

Handling is a bounded external wrench on a real free body. No biological
procedure, robot grasp, manufacturer operating speed or separation model is
implemented. Every success follows observed contacts, endpoints and held axes.
"""

import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.mini_centrifuge_contacts import ContactCheck
from real_labs.mini_centrifuge_fixture import GAUGE_MASS, build_inspection_scene
from real_labs.motion import _model_signature, _physics_event, _warning_state
from real_labs.reset_reporting import combine_reset_reports


@lru_cache(maxsize=24)
def _binding(scene, port):
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_inspection_scene(scene, port=port)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml, {item["file"]: Path(item["source"]).read_bytes() for item in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures={_model_signature(raw), _model_signature(portable)},
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        metadata={
            key: manifest.get(key)
            for key in (
                "camera",
                "camera_names",
                "mounted_cameras",
                "specimens",
                "inspection_variant",
            )
        },
    )


def _require_binding(session):
    if session.data.model is not session.model:
        raise ValueError("Inspection data must belong to the exact supplied model")
    fixture = session.manifest.get("inspection_variant", {})
    if fixture.get("kind") != "mini_inert_gauge_loading":
        raise ValueError("Use the supported free-gauge inspection scene variant")
    expected = _binding(session.manifest.get("id"), fixture.get("port"))
    if (
        _model_signature(session.model) not in expected["signatures"]
        or session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or any(
            session.manifest.get(key) != value
            for key, value in expected["metadata"].items()
        )
        or session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError(
            "Authored mini geometry, physics, controls or fixture binding changed"
        )
    return expected


class MiniInspection:
    """Execute the bound fixture without live pose writes or simulation resets."""

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.session, self.m, self.d = session, session.model, session.data
        m, d = self.m, self.d
        self.fixture = session.manifest["inspection_variant"]
        self.interface = session.equipment["centrifuge"]["mechanical_interface"]
        self.gauge = m.body(self.fixture["gauge_body"]).id
        self.port = m.site(self.fixture["port_site"]).id
        self.park = m.site(self.fixture["park_site"]).id
        self.lid = session.controls["centrifuge", "lid"]
        self.rotor = session.controls["centrifuge", "rotor"]
        self.selected = np.array([self.lid.actuator, self.rotor.actuator])
        joint = int(m.body_jntadr[self.gauge])
        if (
            m.body_jntnum[self.gauge] != 1
            or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("The inert gauge must have six free degrees of freedom")
        if any(
            kind in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD)
            and self.gauge in (m.eq_obj1id[index], m.eq_obj2id[index])
            for index, kind in enumerate(m.eq_type)
        ):
            raise ValueError("The inert gauge cannot be welded or tethered")
        self._preflight()
        probe = copy.copy(d)
        mujoco.mj_forward(m, probe)
        if (
            np.linalg.norm(probe.xpos[self.gauge] - probe.site_xpos[self.park]) > 0.0003
            or abs(session._measured(self.lid)) > 0.001
            or abs(session._measured(self.rotor)) > 0.01
            or np.any(d.xfrc_applied[self.gauge])
        ):
            raise ValueError(
                "Start with the free gauge parked, lid open and rotor stopped"
            )
        self.contacts = ContactCheck(m, probe, self.interface, self.fixture)
        self.contacts.check(probe)
        self.held = {
            key: (control, session._measured(control))
            for key, control in session.controls.items()
            if control.actuator not in self.selected
        }
        self.initial_ctrl = d.ctrl.copy()
        self.initial_time = float(d.time)
        self.initial_eq = d.eq_active.copy()
        self.contact_flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.contract_arrays = {
            name: getattr(m, name).copy()
            for name in (
                "geom_contype",
                "geom_conaffinity",
                "geom_pos",
                "geom_quat",
                "geom_size",
                "geom_aabb",
                "body_pos",
                "body_quat",
                "jnt_pos",
                "jnt_axis",
                "jnt_range",
                "mesh_vert",
                "mesh_face",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
                "dof_damping",
                "dof_armature",
            )
        }
        self.executed = self.entries = self.checked = 0
        self.phases, self.telemetry = [], []
        self.physics_events = []
        self.peak_force = self.peak_torque = 0.0
        self.contacts.queries = self.contacts.fallbacks = 0

    def _preflight(self):
        if self.session.model is not self.m or self.session.data is not self.d:
            raise ValueError("Inspection session model or state was replaced")
        _require_binding(self.session)
        if not np.array_equal(self.d.eq_active, self.m.eq_active0):
            raise ValueError("Authored equality activation changed")
        if not all(
            np.isfinite(values).all()
            for values in (
                self.d.qpos,
                self.d.qvel,
                self.d.ctrl,
                self.d.qfrc_applied,
                self.d.xfrc_applied,
                np.array([self.d.time]),
            )
        ):
            raise ValueError("Inspection requires finite simulation state")

    def _state(self):
        return {
            **{
                name: getattr(self.d, name).tolist()
                for name in (
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
            "time_s": float(self.d.time),
        }

    def _port_pose(self):
        rotation = self.d.site_xmat[self.port].reshape(3, 3)
        return (
            self.d.site_xpos[self.port].copy(),
            rotation[:, 2].copy(),
            rotation.copy(),
        )

    def _hold(self):
        if (
            self.session.model is not self.m
            or self.session.data is not self.d
            or self.d.model is not self.m
        ):
            return False
        self.d.ctrl[self.lid.actuator] = np.clip(
            self.session._measured(self.lid), self.lid.lower, self.lid.upper
        )
        self.d.ctrl[self.rotor.actuator] = 0
        self.d.xfrc_applied[self.gauge] = 0
        return True

    def _observe(self, command, retained, parked):
        self.entries += 1
        m, d = self.m, self.d
        if (
            self.session.model is not m
            or self.session.data is not d
            or d.model is not m
        ):
            raise RuntimeError("inspection_model_state_binding_changed")
        if (
            abs(d.time - self.initial_time - self.executed * m.opt.timestep) > 1e-8
            or any(w.number for w in d.warning)
            or not np.isfinite(np.r_[d.qpos, d.qvel]).all()
        ):
            raise RuntimeError("invalid_physics_or_time")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.contact_flags
            or not np.array_equal(d.eq_active, self.initial_eq)
            or any(
                not np.array_equal(getattr(m, key), value)
                for key, value in self.contract_arrays.items()
            )
        ):
            raise RuntimeError("inspection_contact_or_mechanical_contract_changed")
        expected = self.initial_ctrl.copy()
        expected[self.selected] = command
        if not np.array_equal(d.ctrl, expected):
            raise RuntimeError("inspection_held_controls_changed")
        for key, (control, start) in self.held.items():
            tolerance = 0.0005 if control.units == "m" else 0.005
            if abs(self.session._measured(control) - start) > tolerance:
                raise RuntimeError("held_axis_tracking:" + ".".join(key))
        lid = self.session._measured(self.lid)
        if (
            not self.lid.lower - self.contacts.limit_padding
            <= lid
            <= self.lid.upper + self.contacts.limit_padding
        ):
            raise RuntimeError("lid_left_verified_motion_envelope")
        result = self.contacts.check(d)
        center, axis, _ = self._port_pose()
        delta = d.xpos[self.gauge] - center
        axial = float(delta @ axis)
        lateral = float(np.linalg.norm(delta - axis * axial))
        if retained and not (-0.008 < axial < 0.001 and lateral < 0.0015):
            raise RuntimeError("released_gauge_not_retained")
        if (
            parked
            and np.linalg.norm(d.xpos[self.gauge] - d.site_xpos[self.park]) > 0.0005
        ):
            raise RuntimeError("released_gauge_left_cradle")
        result.update(port_axial_offset_m=axial, port_lateral_offset_m=lateral)
        self.checked += 1
        return result

    def _phase(
        self,
        name,
        duration,
        *,
        target=None,
        rotation=None,
        lid=0.0,
        speed=0.0,
        retained=False,
        parked=False,
        tolerance=0.0003
    ):
        self._preflight()
        m, d = self.m, self.d
        command = np.array([lid, speed])
        d.ctrl[self.selected] = command
        start_position = d.xpos[self.gauge].copy()
        start_quat = d.xquat[self.gauge].copy()
        target_quat = start_quat.copy()
        if rotation is not None:
            mujoco.mju_mat2Quat(target_quat, np.asarray(rotation).ravel())
        if start_quat @ target_quat < 0:
            target_quat *= -1
        begin, checked_begin = self.executed, self.checked
        record = dict(
            phase=name,
            success=False,
            reset_performed=False,
            explicit_reset_performed=False,
            automatic_reset_detected=False,
            target_center_m=None if target is None else np.asarray(target).tolist(),
        )
        self.phases.append(record)
        velocity = np.empty(6)
        try:
            for index in range(round(duration / m.opt.timestep)):
                d.xfrc_applied[self.gauge] = 0
                if target is not None:
                    u = min(1.0, (index + 1) * m.opt.timestep / duration)
                    blend = u**3 * (10 + u * (-15 + 6 * u))
                    position = start_position * (1 - blend) + np.asarray(target) * blend
                    quat = start_quat * (1 - blend) + target_quat * blend
                    quat /= np.linalg.norm(quat)
                    desired = np.empty(9)
                    mujoco.mju_quat2Mat(desired, quat)
                    desired = desired.reshape(3, 3)
                    mujoco.mj_objectVelocity(
                        m, d, mujoco.mjtObj.mjOBJ_BODY, self.gauge, velocity, 0
                    )
                    force = (
                        50 * (position - d.xpos[self.gauge])
                        - 0.20 * velocity[3:]
                        - GAUGE_MASS * m.opt.gravity
                    )
                    current = d.xmat[self.gauge].reshape(3, 3)
                    error = 0.5 * sum(
                        np.cross(current[:, i], desired[:, i]) for i in range(3)
                    )
                    torque = 0.005 * error - 0.00005 * velocity[:3]
                    force *= min(1, 0.08 / max(np.linalg.norm(force), 1e-30))
                    torque *= min(1, 0.0001 / max(np.linalg.norm(torque), 1e-30))
                    d.xfrc_applied[self.gauge] = np.r_[force, torque]
                    self.peak_force = max(self.peak_force, float(np.linalg.norm(force)))
                    self.peak_torque = max(
                        self.peak_torque, float(np.linalg.norm(torque))
                    )
                before_time = float(d.time)
                before_warnings = _warning_state(d)
                reset_inputs = {
                    key: getattr(d, key).copy()
                    for key in ("ctrl", "qfrc_applied", "xfrc_applied")
                }
                mujoco.mj_step(m, d)
                self.executed += 1
                mujoco.mj_forward(m, d)
                event, _ = _physics_event(
                    m, d, before_time, before_warnings, reset_inputs
                )
                if event is not None:
                    self.physics_events.append(
                        dict(executed_step=self.executed, phase=name, **event)
                    )
                    record.update(combine_reset_reports([event]), physics_event=event)
                    raise RuntimeError("native_physics_warning_or_time_discontinuity")
                observation = self._observe(command, retained, parked)
                if self.executed % 10 == 0:
                    self.telemetry.append(
                        dict(
                            phase=name,
                            time_s=float(d.time),
                            qpos=d.qpos.tolist(),
                            qvel=d.qvel.tolist(),
                            applied_wrench=d.xfrc_applied[self.gauge].tolist(),
                            **observation,
                        )
                    )
            if (
                abs(self.session._measured(self.lid) - lid) > 0.001
                or abs(self.session._measured(self.rotor) - speed) > 0.02
            ):
                raise RuntimeError("instrument_endpoint_not_reached")
            if target is not None:
                error = float(np.linalg.norm(d.xpos[self.gauge] - target))
                record["gauge_target_error_m"] = error
                if error > tolerance:
                    raise RuntimeError("gauge_endpoint_not_reached")
                if rotation is not None:
                    alignment = float(
                        d.xmat[self.gauge].reshape(3, 3)[:, 2]
                        @ np.asarray(rotation)[:, 2]
                    )
                    record["gauge_axis_error_rad"] = float(
                        np.arccos(np.clip(alignment, -1, 1))
                    )
                    if record["gauge_axis_error_rad"] > 0.05:
                        raise RuntimeError("gauge_axis_not_aligned")
            self._preflight()
            record["success"] = True
        finally:
            record.update(
                executed_steps=self.executed - begin,
                completed_checks=self.checked - checked_begin,
                state=self._state(),
            )

    def run(self):
        self._preflight()
        mujoco.mj_forward(self.m, self.d)
        initial = self._state()
        park = self.d.site_xpos[self.park].copy()
        park_rotation = self.d.xmat[self.gauge].reshape(3, 3).copy()
        center, axis, rotation = self._port_pose()
        above_port = center + axis * 0.024
        failure = None
        hold = False
        try:
            self._phase("settle_on_cradle", 0.4, parked=True)
            self._phase(
                "lift_from_cradle",
                2.0,
                target=park + [0, 0, 0.17],
                rotation=park_rotation,
            )
            self._phase(
                "align_above_port",
                1.6,
                target=above_port + [0, 0, 0.04],
                rotation=rotation,
            )
            self._phase("approach_port", 1.0, target=above_port, rotation=rotation)
            self._phase(
                "insert_gauge",
                1.5,
                target=center - axis * 0.004,
                rotation=rotation,
                tolerance=0.001,
            )
            self._phase(
                "seat_gauge",
                0.3,
                target=center - axis * 0.004,
                rotation=rotation,
                tolerance=0.001,
            )
            self._phase("release_in_port", 1.5, retained=True)
            self._phase("close_lid", 3, lid=1.712, retained=True)
            self._phase("forward_rotation", 3, lid=1.712, speed=math.tau, retained=True)
            self._phase(
                "reverse_rotation", 4, lid=1.712, speed=-math.tau, retained=True
            )
            self._phase("stop_rotor", 3, lid=1.712, retained=True)
            self._phase("open_lid", 3, retained=True)
            center, axis, rotation = self._port_pose()
            self._phase(
                "extract_gauge", 1.5, target=center + axis * 0.024, rotation=rotation
            )
            self._phase(
                "clear_rotor",
                1.0,
                target=center + axis * 0.024 + [0, 0, 0.04],
                rotation=rotation,
            )
            self._phase(
                "return_above_cradle",
                1.6,
                target=park + [0, 0, 0.17],
                rotation=park_rotation,
            )
            self._phase("lower_to_cradle", 2.0, target=park, rotation=park_rotation)
            self._phase("seat_on_cradle", 0.3, target=park, rotation=park_rotation)
            self._phase("release_on_cradle", 1.0, parked=True)
            if np.linalg.norm(self.d.xpos[self.gauge] - park) > 0.0003:
                raise RuntimeError("released_gauge_not_on_cradle")
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            hold = self._hold()
            if self.phases:
                self.phases[-1].update(success=False, failure=failure)
        return dict(
            status="succeeded" if failure is None else "failed",
            failure=failure,
            scene=self.session.manifest["id"],
            scene_sha256=self.binding["scene_sha256"],
            initial=initial,
            final=self._state(),
            phases=self.phases,
            telemetry=self.telemetry,
            executed_steps=self.executed,
            observer_entries=self.entries,
            completed_checks=self.checked,
            coverage_complete=self.executed > 0 and self.executed == self.checked,
            direct_pair_count=len(self.contacts.pairs),
            unculled_pair_count=len(self.contacts.unculled_pairs),
            conservatively_excluded_pairs=len(self.contacts.excluded_pairs),
            direct_queries=self.contacts.queries,
            support_bound_fallbacks=self.contacts.fallbacks,
            contact_check_contract=dict(
                broad_phase_padding_m=self.contacts.screen_padding,
                exact_query_cap_m=self.contacts.cap,
                instrument_overlap_limit_m=self.contacts.instrument_overlap,
                allowed_gauge_overlap_limit_m=self.contacts.gauge_overlap,
                scope="All enabled cross-weld pairs involving lid, rotor or gauge; separating bounds are conservative",
            ),
            maximum_checked_overlap_bound_m=self.contacts.maximum_penetration,
            peak_handling_force_N=self.peak_force,
            peak_handling_torque_Nm=self.peak_torque,
            hold_requested=hold,
            **combine_reset_reports(self.phases),
            physics_events=self.physics_events.copy(),
            model_signature_unchanged=_model_signature(self.m)
            in self.binding["signatures"],
            limitations=[
                "The inert gauge is a free rigid body, not a sample tube or biological specimen",
                "Handling uses a bounded external wrench; no robot gripper is modeled",
                "Released rotation has zero handling wrench; original finite inspection drives remain",
                "A failure requests lid hold and rotor stop and clears the handling wrench; it does not claim settled recovery",
                "No calibrated separation, imbalance sensing, lid interlock or manufacturer-speed claim",
            ],
        )


def inspect_mini_centrifuge(session):
    return MiniInspection(session).run()
