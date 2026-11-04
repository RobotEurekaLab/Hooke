"""Source-bound complete-arm clearance planning and measured empty-tool motion.

Authored open bearing pairs use their declared running clearance. All other
arm/self/world pairs retain 10 mm. Direct queries include parent-filtered pairs;
finite interpolation and per-step observations are not continuous guarantees.
"""

from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.motion import (
    JointMotion,
    PlanningError,
    _ARM_CONTROLS,
    _model_signature,
    inspect_point_and_return,
)


CONSUMERS = {
    "waterloo_robohub": ("robot_arm", ""),
    "nasa_planetary": ("rover", "arm_"),
}


@lru_cache(maxsize=2)
def _binding(scene):
    from real_labs.builder import build_scene
    from real_labs.export import _portable_resources
    from real_labs.runtime import InstrumentSession

    root, manifest = build_scene(scene)
    raw = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    resources = _portable_resources(root)
    xml = ET.tostring(root, encoding="unicode")
    portable = mujoco.MjModel.from_xml_string(
        xml, {i["file"]: Path(i["source"]).read_bytes() for i in resources}
    )
    session = InstrumentSession(raw, manifest)
    return dict(
        signatures=(_model_signature(raw), _model_signature(portable)),
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        equipment=session.equipment,
        controls=session.controls,
        benches=manifest.get("benches"),
        camera=manifest.get("camera"),
    )


def _require_binding(session):
    scene = session.manifest.get("id")
    if scene not in CONSUMERS:
        raise ValueError("Complete-arm inspection requires an authored consumer scene")
    if session.data.model is not session.model:
        raise ValueError("Arm state must belong to its exact model")
    ref = _binding(scene)
    if session.equipment != ref["equipment"] or session.controls != ref["controls"]:
        raise ValueError("Arm controls, geometry classification or scale changed")
    if any(session.manifest.get(k) != ref[k] for k in ("benches", "camera")):
        raise ValueError("Arm scene layout or review cameras changed")
    if _model_signature(session.model) not in ref["signatures"]:
        raise ValueError("Arm geometry, collisions, physics or drives changed")
    if session.manifest.get("scene_sha256", ref["scene_sha256"]) != ref["scene_sha256"]:
        raise ValueError("Arm scene digest changed")
    return ref


def _state(session):
    d = session.data
    return dict(
        time_s=float(d.time),
        **{
            name: getattr(d, name).tolist()
            for name in (
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


class ArmMotion(JointMotion):
    """Full exterior checks with exact named, source-bound bearing interfaces.

    The caller exclusively owns model/state/controls. Model changes and state
    replacement between calls are rejected; arbitrary concurrent writers are
    unsupported. Every measured step includes all enabled distinct-weld pairs
    involving a moving arm surface, even when native parent filtering omits it.
    """

    def __init__(self, session):
        self.binding = _require_binding(session)
        self.live_model, self.live_data = session.model, session.data
        scene = session.manifest["id"]
        equipment, prefix = CONSUMERS[scene]
        names = tuple(prefix + name for name in _ARM_CONTROLS)
        super().__init__(session, equipment, names, clearance_m=0.01)
        m = session.model
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        self.pairs = np.array(
            [
                (int(a), int(b))
                for i, a in enumerate(enabled)
                for b in enabled[i + 1 :]
                if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
                and (a in self.moving_geoms or b in self.moving_geoms)
            ],
            dtype=int,
        )
        interface = session.equipment[equipment]["arm_interfaces"][prefix or "main"]
        named = {}
        for group in interface["engineered_pair_classes"]:
            for first in group["fixed_or_proximal"]:
                for second in group["moving_or_distal"]:
                    key = tuple(sorted((m.geom(first).id, m.geom(second).id)))
                    named[key] = (group["minimum_clearance_m"], group["name"])
        self.thresholds = np.array(
            [named.get(tuple(pair), (self.clearance, None))[0] for pair in self.pairs]
        )
        self.engineered = self.thresholds < self.clearance
        self.masks = m.geom_contype.copy(), m.geom_conaffinity.copy()
        self.flags = int(m.opt.disableflags), int(m.opt.enableflags)
        self._clear_observations()

    def _identity(self):
        if (
            self.session.model is not self.live_model
            or self.session.data is not self.live_data
            or self.live_data.model is not self.live_model
        ):
            raise ValueError("Arm model or state object replaced")

    def _refresh_checker(self):
        self._identity()
        _require_binding(self.session)
        super()._refresh_checker()

    def _distances(self, model, data):
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum(
            "nij,nj->ni", rotation, model.geom_aabb[:, :3]
        )
        half = np.einsum("nij,nj->ni", np.abs(rotation), model.geom_aabb[:, 3:])
        a, b = self.pairs.T
        separation = np.maximum(np.abs(center[a] - center[b]) - half[a] - half[b], 0)
        lower = np.linalg.norm(separation, axis=1)
        # Uniform cap: distant analytic results and MuJoCo sentinels never
        # masquerade as uncapped measured minimum distances.
        gaps = np.full(len(self.pairs), self.clearance)
        near = np.flatnonzero(lower < self.clearance)
        for i in near:
            first, second = self.pairs[i]
            gaps[i] = min(
                float(
                    mujoco.mj_geomDistance(
                        model, data, int(first), int(second), self.clearance, None
                    )
                ),
                self.clearance,
            )
        return gaps, len(near)

    def configuration_contacts(self, values):
        q = self._configuration(values)
        self._pose(q)
        gaps, _ = self._distances(self.model, self.data)
        return [
            dict(
                geoms=[self.model.geom(int(g)).name for g in self.pairs[i]],
                distance_m=float(gaps[i]),
                required_clearance_m=float(self.thresholds[i]),
                engineered_bearing_interface=bool(self.engineered[i]),
            )
            for i in np.flatnonzero(gaps < self.thresholds)
        ]

    def _clear_observations(self):
        self.entries = self.checked = self.queries = 0
        self.minimum_external = self.minimum_interface = self.clearance
        self.violations = []

    def _observe(self, m, d):
        self._identity()
        self.entries += 1
        if (
            (int(m.opt.disableflags), int(m.opt.enableflags)) != self.flags
            or not np.array_equal(m.geom_contype, self.masks[0])
            or not np.array_equal(m.geom_conaffinity, self.masks[1])
        ):
            raise RuntimeError("arm_collision_configuration_changed")
        gaps, queries = self._distances(m, d)
        self.queries += queries
        bad = np.flatnonzero(gaps < self.thresholds)
        if len(bad):
            self.violations = [
                dict(
                    geoms=[m.geom(int(g)).name for g in self.pairs[i]],
                    distance_m=float(gaps[i]),
                    required_clearance_m=float(self.thresholds[i]),
                )
                for i in bad[:8]
            ]
            raise RuntimeError("arm_measured_direct_clearance_lost")
        self.minimum_external = min(
            self.minimum_external,
            float(np.min(gaps[~self.engineered], initial=self.clearance)),
        )
        self.minimum_interface = min(
            self.minimum_interface,
            float(np.min(gaps[self.engineered], initial=self.clearance)),
        )
        self.checked += 1

    def execute(self, plan, *, position_tolerance_m=0.001, step_observer=None):
        self._identity()
        _require_binding(self.session)
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        self._clear_observations()
        before = _state(self.session)

        def observe(m, d):
            self._observe(m, d)
            if step_observer is not None:
                step_observer(m, d)

        # An actual endpoint hold lets residual link motion settle below the
        # stricter Cartesian contract; every added physics step is observed.
        plan = replace(plan, waypoints=plan.waypoints + (plan.waypoints[-1],))
        result = super().execute(
            plan, position_tolerance_m=position_tolerance_m, step_observer=observe
        )
        complete = (
            result["physics_steps"] > 0 and result["physics_steps"] == self.checked
        )
        result.update(
            before=before,
            after=_state(self.session),
            endpoint_hold_segments=1,
            cartesian_acceptance_tolerance_m=position_tolerance_m,
            scene_sha256=self.binding["scene_sha256"],
            direct_checks=dict(
                candidate_pair_count=len(self.pairs),
                engineered_pair_count=int(self.engineered.sum()),
                executed_steps=result["physics_steps"],
                observer_entries=self.entries,
                fully_checked_steps=self.checked,
                coverage_complete=complete,
                reporting_cap_m=self.clearance,
                checked_external_clearance_lower_bound_m=(
                    self.minimum_external if self.checked else None
                ),
                checked_interface_clearance_lower_bound_m=(
                    self.minimum_interface if self.checked else None
                ),
                whole_leg_external_clearance_lower_bound_m=(
                    self.minimum_external if complete else None
                ),
                whole_leg_interface_clearance_lower_bound_m=(
                    self.minimum_interface if complete else None
                ),
                direct_distance_queries=self.queries,
                violations=self.violations,
                scope="All enabled moving-arm distinct-weld pairs including native parent-filtered pairs; exact authored bearing classes only",
            ),
        )
        return result


def inspect_waterloo_handoff(session):
    """Inspect 150 mm above the actual bench, then return without state assignment."""
    if session.manifest.get("id") != "waterloo_robohub":
        raise ValueError("Waterloo handoff requires its authored scene")
    motion = ArmMotion(session)
    bench = next(
        row for row in session.manifest["benches"] if row["id"] == "handoff_station"
    )
    yaw = float(bench.get("yaw", 0))
    offset = np.array([0.60, -0.25, 0.15])
    target = np.array(bench["pos"], dtype=float)
    target[:2] += (
        np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        @ offset[:2]
    )
    target[2] += bench["size"][2] + offset[2]
    shoulder = session.controls["robot_arm", "shoulder"]
    high = shoulder.lower + 0.85 * (shoulder.upper - shoulder.lower)
    report = inspect_point_and_return(
        motion,
        session.equipment["robot_arm"]["sites"]["tcp"],
        target,
        fixed_goal_joints={"shoulder": high},
    )
    report.update(
        task="Complete-arm empty-tool handoff-table hover and return",
        bench="handoff_station",
        bench_position_m=bench["pos"],
        bench_size_m=bench["size"],
        goal_offset_from_bench_top_m=offset.tolist(),
        endpoint_shoulder_target_rad=high,
    )
    return report


def inspect_nasa_cache(session, *, cache=3):
    """Retain the historical fixed-base inspection without moving a free rover.

    A grounded rover needs coordinated base, arm and sample handling. The old
    empty-tool planner cannot keep that base stationary or clear the loaded
    pickup cradle; reject before planning instead of executing an unsafe path.
    """
    if session.manifest.get("id") != "nasa_planetary" or cache not in (1, 2, 3):
        raise ValueError("Select an authored NASA cache numbered 1, 2 or 3")
    _require_binding(session)
    base = session.model.body("rover").id
    joints = np.flatnonzero(session.model.jnt_bodyid == base)
    if np.any(session.model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(
            "The fixed-base cache inspection does not support the grounded "
            "rover; use the rover_traverse_and_stage operation"
        )
    motion = ArmMotion(session)
    sites = session.equipment["rover"]["sites"]
    site = sites[f"sample_cache_{cache}"]
    origin = session.data.site_xpos[session.model.site(site).id].copy()
    offset = np.array([0.0, 0.0, 0.15])
    report = inspect_point_and_return(motion, sites["arm_tcp"], origin + offset)
    report.update(
        task="Complete-arm empty-tool on-rover cache hover and return",
        cache_site=site,
        cache_position_m=origin.tolist(),
        goal_offset_from_cache_m=offset.tolist(),
        wheel_motion_requested=False,
    )
    return report
