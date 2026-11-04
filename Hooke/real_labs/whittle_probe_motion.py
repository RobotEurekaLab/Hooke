"""Measured noncontact datum inspection on the authored Whittle dry traverse.

The reference and probe travel together along Y. This checks that fixture's
positioning and physical tip gap; it does not sample the turbine flow field.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from real_labs.motion import PlanningError
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import RigidAccess
from real_labs.whittle_probe_geometry import PREFIX

SCENE = "cambridge_whittle_low_speed_turbine"
EQUIPMENT = "primary_apparatus"
CONTROLS = ("probe_y", "probe_z")


class ReferenceInspection(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.tip = self.model.site(EQUIPMENT + "__site_probe_tip").id
        self.surface = self.model.site(EQUIPMENT + "__site_reference_surface").id
        self.reference = self.model.geom(PREFIX + "reference").id

    def move(self, target):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        # Short settled submoves respect the original low-stiffness drives;
        # their gains, forces, inertia and tracking limits remain unchanged.
        for first, second in zip(plan.waypoints, plan.waypoints[1:]):
            delta = np.asarray(second) - first
            count = max(1, int(np.ceil(np.max(abs(delta) / [0.015, 0.004]))))
            path.extend(
                tuple(np.asarray(first) + delta * i / count)
                for i in range(1, count + 1)
            )
        plan = replace(plan, waypoints=tuple(path) + (path[-1],) * 4)
        self.reset_counts()
        report = self.motion.execute(plan, step_observer=self.check)
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

    def observe(self):
        self.preflight()
        m, d = self.model, self.data
        tip, surface = d.site_xpos[self.tip], d.site_xpos[self.surface]
        gap = float(tip[2] - surface[2])
        start = tip - [0, 0, 1e-6]
        hit = np.array([-1], dtype=np.int32)
        distance = float(
            mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, hit)
        )
        lateral = float(np.linalg.norm(tip[:2] - surface[:2]))
        return dict(
            tip_world_m=tip.tolist(),
            reference_surface_world_m=surface.tolist(),
            gap_m=gap,
            lateral_error_m=lateral,
            first_hit=(m.geom(int(hit[0])).name if hit[0] >= 0 else None),
            ray_distance_m=distance,
            ray_surface_error_m=abs(distance + 1e-6 - gap),
            registered=bool(
                hit[0] == self.reference
                and lateral < 1e-6
                and abs(distance + 1e-6 - gap) < 1e-6
            ),
            state=self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured().copy()
        if np.max(abs(start)) > 1e-4 or np.max(abs(self.data.qvel)) > 0.001:
            raise PlanningError("Begin the dry reference inspection at resting home")
        initial = self.observe()
        phases = []
        for name, target in (
            ("raise_probe", (0, 0.06)),
            ("traverse_positive", (0.23, 0.06)),
            ("approach_reference", (0.23, -0.05)),
            ("traverse_negative_at_datum", (-0.23, -0.05)),
            ("retract_probe", (-0.23, 0.06)),
            ("return_home", start),
        ):
            result = self.move(target)
            observation = self.observe()
            if result["status"] == "succeeded":
                expected_gap = 0.056 + target[1]
                if (
                    not observation["registered"]
                    or abs(observation["gap_m"] - expected_gap) > 0.00005
                    or np.max(abs(self.motion.measured() - target)) > 0.00005
                ):
                    result.update(
                        status="failed",
                        failure="dry_reference_not_registered",
                        hold_requested=self.motion._request_hold(),
                    )
            phases.append(dict(phase=name, **result, observation=observation))
            if result["status"] != "succeeded":
                break
        executed = sum(p["physics_steps"] for p in phases)
        checked = sum(p["direct_clearance"]["checked_steps"] for p in phases)
        complete = executed > 0 and executed == checked
        resets = combine_reset_reports(phases)
        return dict(
            status=(
                "succeeded"
                if len(phases) == 6 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_observation=initial,
            phases=phases,
            final_observation=self.observe(),
            executed_steps=executed,
            checked_steps=checked,
            coverage_complete=complete,
            whole_route_clearance_lower_bound_m=(
                min(
                    p["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
                    for p in phases
                )
                if complete
                else None
            ),
            state_reset=resets["reset_performed"],
            **resets,
            limitations=[
                "Retained reference and probe share Y motion; no independent sample loading or flow traverse is modeled.",
                "Measured geometric tip gap and first surface hit, not a calibrated physical sensor.",
                "Every executed sample must pass direct enabled-pair clearance checks; no continuous-time clearance claim.",
                "Authored ideal guides carry the mechanism across explicit running gaps; no bearing stress calibration.",
                "No turbine flow, pressure, efficiency, rotor dynamics or native Isaac validation.",
            ],
        )


def inspect_whittle_reference(session):
    return ReferenceInspection(session).run()
