"""Measured dry hood/cassette access; no chemistry or object transfer.

Plans assume a stationary world and are finite sampled. Execution uses the live
actuators without state teleportation. Failure requests a measured position hold;
only the caller's explicit session.reset() starts a new episode.
"""

from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import JointMotion, PlanningError, _model_signature


@lru_cache(maxsize=3)
def _reference(variant):
    from real_labs.caltech_details import add_hood
    from real_labs.equipment import _Builder

    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    base = ET.SubElement(world, "body", name="access_reference")
    add_hood(_Builder(root, "access_reference"), base, {"apparatus_variant": variant})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _require_state_binding(session):
    if session.data.model is not session.model:
        raise ValueError("Hood access requires state belonging to its physical model")


def _require_interface(session, equipment):
    """Bind essential geometry, hierarchy, axes and datums to the authored fixture."""
    _require_state_binding(session)
    item = session.equipment[equipment]
    if session.manifest.get("id") != "caltech_stoltz_schlinger":
        raise ValueError("Hood access requires the authored Caltech scene")
    contract = item.get("caltech_access", {})
    if contract.get("version") != 1:
        raise ValueError("No supported hood access contract")
    expected = _reference(contract.get("apparatus_variant", 0))
    actual = session.model

    def name(original):
        return equipment + original.removeprefix("access_reference")

    def close(first, second):
        return np.allclose(first, second, rtol=0, atol=1e-10)

    def rotation(first, second):
        return abs(abs(float(np.dot(first, second))) - 1) < 1e-10

    bodies = set()
    for index in range(1, expected.nbody):
        body = actual.body(name(expected.body(index).name)).id
        bodies.add(body)
        if index == 1:
            if actual.body_weldid[body] != 0:
                raise ValueError("Hood base must remain fixed to the world")
            continue
        if (
            actual.body(actual.body_parentid[body]).name
            != name(expected.body(expected.body_parentid[index]).name)
            or not close(actual.body_pos[body], expected.body_pos[index])
            or not rotation(actual.body_quat[body], expected.body_quat[index])
        ):
            raise ValueError("Authored hood body binding changed")
    for index in range(expected.ngeom):
        geom = actual.geom(name(expected.geom(index).name)).id
        if (
            actual.body(actual.geom_bodyid[geom]).name
            != name(expected.body(expected.geom_bodyid[index]).name)
            or actual.geom_type[geom] != expected.geom_type[index]
            or actual.geom_contype[geom] != expected.geom_contype[index]
            or actual.geom_conaffinity[geom] != expected.geom_conaffinity[index]
            or not close(actual.geom_pos[geom], expected.geom_pos[index])
            or not close(actual.geom_size[geom], expected.geom_size[index])
            or not rotation(actual.geom_quat[geom], expected.geom_quat[index])
        ):
            raise ValueError("Authored hood collider shape, mask or binding changed")
    for index in range(expected.njnt):
        joint_name = name(expected.joint(index).name)
        joint = actual.joint(joint_name).id
        semantic = joint_name.split("__joint_", 1)[1]
        control = session.controls[equipment, semantic]
        if (
            control.joint != joint
            or control.lower != expected.jnt_range[index, 0]
            or control.upper != expected.jnt_range[index, 1]
            or actual.actuator_trnid[control.actuator, 0] != joint
            or actual.jnt_type[joint] != expected.jnt_type[index]
            or actual.jnt_limited[joint] != expected.jnt_limited[index]
            or actual.body(actual.jnt_bodyid[joint]).name
            != name(expected.body(expected.jnt_bodyid[index]).name)
            or not close(actual.jnt_pos[joint], expected.jnt_pos[index])
            or not close(actual.jnt_axis[joint], expected.jnt_axis[index])
            or not close(actual.jnt_range[joint], expected.jnt_range[index])
        ):
            raise ValueError("Authored hood control binding or travel changed")
    for index in range(expected.nsite):
        site_name = name(expected.site(index).name)
        site = actual.site(site_name).id
        semantic = site_name.split("__site_", 1)[1]
        if (
            item["sites"].get(semantic) != site_name
            or actual.body(actual.site_bodyid[site]).name
            != name(expected.body(expected.site_bodyid[index]).name)
            or not close(actual.site_pos[site], expected.site_pos[index])
        ):
            raise ValueError("Authored hood datum binding changed")
    if item.get("sample_site") != name("access_reference__site_sample_carrier"):
        raise ValueError("Cassette target must use the authored sample datum")
    for signature in actual.exclude_signature:
        if int(signature) >> 16 in bodies or int(signature) & 65535 in bodies:
            raise ValueError("Collision exclusions affect the hood")


def full_state(session):
    """A measured checkpoint sufficient to replay this rigid mechanism's state."""
    data = session.data
    return {
        "time_s": float(data.time),
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
        "act": data.act.tolist(),
        "qacc_warmstart": data.qacc_warmstart.tolist(),
        "qfrc_applied": data.qfrc_applied.tolist(),
        "xfrc_applied": data.xfrc_applied.tolist(),
        "mocap_pos": data.mocap_pos.tolist(),
        "mocap_quat": data.mocap_quat.tolist(),
        "eq_active": data.eq_active.tolist(),
        "model_signature": _model_signature(session.model),
    }


class HoodAccess:
    """Preserve original travel and check filtered pairs every executed step."""

    def __init__(self, session, equipment="hood_primary"):
        _require_interface(session, equipment)
        self.session, self.equipment = session, equipment
        self.motion = JointMotion(
            session, equipment, ("supply_drawer", "sash"), clearance_m=0.0005
        )
        self.motion.resolution[:] = 0.001
        model = session.model
        enabled = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
        moving = np.array(sorted(self.motion.moving_geoms))
        # Include all other enabled bodies, not merely contacts surviving MuJoCo
        # same/parent-body filtering. Same-body assembly overlaps are intentional.
        first, second = [], []
        for geom in moving:
            for other in enabled:
                if model.geom_bodyid[geom] == model.geom_bodyid[other]:
                    continue
                if other in self.motion.moving_geoms and other < geom:
                    continue
                first.append(geom)
                second.append(other)
        self.first = np.array(first, dtype=int)
        self.second = np.array(second, dtype=int)
        self.bounds = model.geom_rbound[self.first] + model.geom_rbound[self.second]
        self.site = model.site(session.equipment[equipment]["sample_site"]).id
        self.reset_observations()

    def reset_observations(self):
        """Reset counters only; never change the physical session."""
        self.observed_steps = self.checked_steps = 0
        self.minimum_gap = 0.01
        self.minimum_pair = None
        self.direct_pair_evaluations = 0

    def _observe(self, model, data):
        self.observed_steps += 1
        # A bounding sphere distance above 10 mm proves at least that clearance.
        # Far/sentinel and uncapped analytic results are uniformly capped at 10 mm.
        centers = np.linalg.norm(
            data.geom_xpos[self.first] - data.geom_xpos[self.second], axis=1
        )
        for index in np.flatnonzero(centers <= self.bounds + 0.01):
            first, second = int(self.first[index]), int(self.second[index])
            distance = min(
                float(mujoco.mj_geomDistance(model, data, first, second, 0.01, None)),
                0.01,
            )
            self.direct_pair_evaluations += 1
            if distance < self.minimum_gap:
                self.minimum_gap = distance
                self.minimum_pair = [model.geom(first).name, model.geom(second).name]
            if distance < 0.0005 - 1e-9:
                raise RuntimeError("measured_hood_clearance_lost")
        self.checked_steps += 1

    def move(self, target, *, step_observer=None):
        """Move from actual joints; an optional observer must remain read-only."""
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Step observer must be callable")
        # A foreign state is an invalid precondition, not a simulated failure.
        # Reject before counters, controls, time or hold requests can change.
        _require_state_binding(self.session)
        self.reset_observations()
        before = full_state(self.session)
        try:
            _require_interface(self.session, self.equipment)
            plan = self.motion.plan_joint_target(target)
        except (PlanningError, ValueError) as error:
            return {
                "status": "blocked",
                "failure": str(error),
                "hold_requested": self.motion._request_hold(),
                "reset_performed": False,
                "physics_steps": 0,
                "observed_steps": 0,
                "fully_checked_steps": 0,
                "whole_leg_clearance_observed": False,
                "minimum_clearance_lower_bound_m": None,
                "before": before,
                "after": full_state(self.session),
            }

        def observe(model, data):
            self._observe(model, data)
            if step_observer is not None:
                step_observer(model, data)

        result = self.motion.execute(plan, step_observer=observe)
        result.update(
            observed_steps=self.observed_steps,
            fully_checked_steps=self.checked_steps,
            whole_leg_clearance_observed=bool(
                result["status"] == "succeeded"
                and result["physics_steps"] > 0
                and self.checked_steps == result["physics_steps"]
            ),
            minimum_clearance_lower_bound_m=(
                self.minimum_gap if self.observed_steps else None
            ),
            closest_observed_pair=self.minimum_pair,
            direct_pair_evaluations=self.direct_pair_evaluations,
            distance_reporting="Minimum observed lower bound, capped at 10 mm; partial coverage is not a whole-leg bound",
            before=before,
            after=full_state(self.session),
        )
        result["cassette_position_m"] = self.session.data.site_xpos[self.site].tolist()
        result["aperture_height_m"] = 0.023 + float(self.motion.measured()[1])
        return result

    def open_inspect_return(self, *, step_observer=None):
        """Expose the retained cassette and aperture, then return to measured start."""
        start = self.motion.measured().copy()
        initial = full_state(self.session)
        legs = []
        targets = (
            ("sash_open", (start[0], 0.4)),
            ("cassette_exposed", (0.23, 0.4)),
            ("drawer_return", (start[0], 0.4)),
            ("sash_return", tuple(start)),
        )
        for phase, target in targets:
            leg = self.move(target, step_observer=step_observer)
            leg["phase"] = phase
            legs.append(leg)
            if leg["status"] != "succeeded":
                break
        success = len(legs) == 4 and all(leg["status"] == "succeeded" for leg in legs)
        return {
            "status": "succeeded" if success else "failed",
            "scope": "Retained inert cassette access and sash aperture inspection only; no retrieval, chemistry, ventilation or containment validation",
            "equipment": self.equipment,
            "model_signature": self.motion.signature,
            "scene_sha256": self.session.manifest.get("scene_sha256"),
            "initial_state": initial,
            "legs": legs,
            "return_error_m": np.abs(self.motion.measured() - start).tolist(),
            "physics_steps": sum(leg["physics_steps"] for leg in legs),
            "fully_checked_steps": sum(leg["fully_checked_steps"] for leg in legs),
            **combine_reset_reports(legs),
            "hold_requested": any(leg.get("hold_requested", False) for leg in legs),
            "final_state": full_state(self.session),
            "limitations": [
                "Ideal constrained bearings have 1 mm authored running clearance; friction, compliance and counterweights are not calibrated.",
                "Finite sampled planning plus per-step measured distances is not a continuous-time clearance proof.",
                "Full states are recorded at each leg boundary; intermediate control/position telemetry is sampled, while clearance checks run every actual physics step.",
                "Failure hold requests the current position through the existing servo and does not guarantee an instantaneous stop.",
            ],
        }
