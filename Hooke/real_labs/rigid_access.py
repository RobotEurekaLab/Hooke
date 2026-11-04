"""Source-bound motion and direct clearances for authored rigid instruments."""

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import JointMotion, _model_signature


@lru_cache(maxsize=16)
def authored_binding(scene):
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(scene)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml, {r["file"]: Path(r["source"]).read_bytes() for r in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures={_model_signature(raw), _model_signature(portable)},
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        controls=session.controls,
        equipment=session.equipment,
        metadata={
            key: manifest.get(key)
            for key in ("camera", "camera_names", "mounted_cameras", "specimens")
        },
    )


def require_binding(session, scene):
    expected = authored_binding(scene)
    if session.manifest.get("id") != scene or session.data.model is not session.model:
        raise ValueError("Inspection requires the authored scene and owned state")
    if _model_signature(session.model) not in expected["signatures"]:
        raise ValueError("Authored geometry, contacts, physics or drives changed")
    if (
        session.controls != expected["controls"]
        or session.equipment != expected["equipment"]
        or any(session.manifest.get(k) != v for k, v in expected["metadata"].items())
        or session.manifest.get("scene_sha256", expected["scene_sha256"])
        != expected["scene_sha256"]
    ):
        raise ValueError(
            "Authored control, equipment, camera or source binding changed"
        )
    return expected


def near_geometry_pairs(model, data, pairs, cap_m):
    """Conservatively cull AABB pairs without large three-column temporaries.

    A Euclidean gap below the cap must also be below it on each axis. Applying
    those necessary tests first preserves the original final distance test and
    pair order, while avoiding most gathered arrays in dense articulated scenes.
    """
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    center = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    half = np.einsum("nij,nj->ni", np.abs(rotation), model.geom_aabb[:, 3:])
    candidates = pairs
    for axis in range(3):
        first, second = candidates.T
        gap = np.abs(center[first, axis] - center[second, axis]) - half[first, axis] - half[second, axis]
        candidates = candidates[gap < cap_m]
    first, second = candidates.T
    distance = np.linalg.norm(
        np.maximum(np.abs(center[first] - center[second]) - half[first] - half[second], 0), axis=1
    )
    return candidates[distance < cap_m]


class RigidAccess:
    """Check every completed step, including pairs excluded by parent filtering.

    An AABB lower bound only culls distant pairs. Near pairs use MuJoCo's direct
    geometry distance. Ideal joints carry the mechanism; a clearance check does
    not establish real bearing preload, stiffness or instrument calibration.
    """

    def __init__(self, session, scene, equipment, controls, *, minimum_m=0.00025):
        self.binding = require_binding(session, scene)
        self.session, self.scene = session, scene
        self.model, self.data = session.model, session.data
        self.minimum_m, self.cap_m = minimum_m, max(0.002, minimum_m)
        self.motion = JointMotion(session, equipment, controls, clearance_m=minimum_m)
        m = self.model
        groups = {int(m.body_weldid[b]) for b in self.motion.moving_bodies}
        explicit = set(m.pair_geom1) | set(m.pair_geom2)
        ids = [
            i
            for i in range(m.ngeom)
            if m.geom_contype[i] or m.geom_conaffinity[i] or i in explicit
        ]
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
            ],
            dtype=int,
        ).reshape(-1, 2)
        self.masks = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self.flags = (int(m.opt.disableflags), int(m.opt.enableflags))
        self.reset_counts()

    def reset_counts(self):
        self.entries = self.checked = self.queries = 0
        self.minimum = self.cap_m

    def check(self, m, d):
        self.entries += 1
        if (
            self.session.model is not self.model
            or self.session.data is not self.data
            or m is not self.model
            or d is not self.data
            or d.model is not m
        ):
            raise RuntimeError("inspection_model_state_binding_changed")
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("inspection_contact_configuration_changed")
        current = self.cap_m
        for a, b in near_geometry_pairs(m, d, self.pairs, self.cap_m):
            gap = min(
                float(mujoco.mj_geomDistance(m, d, int(a), int(b), self.cap_m, None)),
                self.cap_m,
            )
            self.queries += 1
            current = min(current, gap)
            if not np.isfinite(gap) or gap < self.minimum_m:
                raise RuntimeError(
                    "inspection_clearance_lost:"
                    + m.geom(int(a)).name
                    + ":"
                    + m.geom(int(b)).name
                )
        self.minimum = min(self.minimum, current)
        self.checked += 1

    def preflight(self):
        require_binding(self.session, self.scene)
        if self.session.model is not self.model or self.session.data is not self.data:
            raise ValueError("Inspection session state was replaced")
        if any(
            not np.isfinite(v).all()
            for v in (
                self.data.qpos,
                self.data.qvel,
                self.data.ctrl,
                self.data.qfrc_applied,
                self.data.xfrc_applied,
                np.array([self.data.time]),
            )
        ):
            raise ValueError("Inspection requires finite simulation state")

    def move(self, target, *, speed_scale=1.0):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        plan = replace(plan, waypoints=plan.waypoints + (plan.waypoints[-1],) * 4)
        self.reset_counts()
        report = self.motion.execute(plan, step_observer=self.check, speed_scale=speed_scale)
        executed = report["physics_steps"]
        complete = executed > 0 and executed == self.checked
        report["direct_clearance"] = dict(
            executed_steps=executed,
            observer_entries=self.entries,
            checked_steps=self.checked,
            unchecked_steps=executed - self.checked,
            coverage_complete=complete,
            pair_count=len(self.pairs),
            direct_near_queries=self.queries,
            clearance_cap_m=self.cap_m,
            checked_steps_clearance_lower_bound_m=(
                self.minimum if self.checked else None
            ),
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
        )
        return report

    def state(self):
        return dict(
            **{
                name: getattr(self.data, name).tolist()
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
            time_s=float(self.data.time),
            model_signature=_model_signature(self.model),
        )
