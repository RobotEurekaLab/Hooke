"""Measured passage through the ANU profiler's authored open reference ring.

The ring and profiler rotate together. This inspects physical alignment and
height travel; it neither measures fluid motion nor calibrates a thermistor.
"""

from dataclasses import replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from real_labs.annulus_reference_geometry import BORE_RADIUS, TIP_RADIUS
from real_labs.motion import PlanningError
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import RigidAccess

SCENE = "anu_climate_fluid_physics"
EQUIPMENT = "large_rotating_annulus"
CONTROLS = ("tank_rotation", "probe_height")


class ProbeRegistration(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.tip = self.model.site(EQUIPMENT + "__site_probe_tip").id
        self.datum = self.model.site(EQUIPMENT + "__site_calibration_target").id
        self.probe_body = self.model.body(EQUIPMENT + "__body_probe_height").id

    def move(self, target):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        # Short settled moves respect the retained drives and tracking limits.
        for first, second in zip(plan.waypoints, plan.waypoints[1:]):
            delta = np.asarray(second) - first
            count = max(1, int(np.ceil(np.max(abs(delta) / [0.015, 0.008]))))
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
            coverage_complete=complete,
            pair_count=len(self.pairs),
            direct_near_queries=self.queries,
            clearance_cap_m=self.cap_m,
            whole_leg_clearance_lower_bound_m=self.minimum if complete else None,
        )
        return report

    def observe(self):
        self.preflight()
        m, d = self.model, self.data
        tip, datum = d.site_xpos[self.tip], d.site_xpos[self.datum]
        lateral = float(np.linalg.norm(tip[:2] - datum[:2]))
        rays = []
        # This datum is an opening, not a surface a downward ray can hit.
        # Exclude the shaft itself and inspect four radial bore walls.
        origin = datum - [0, 0, 0.003]
        for direction in ((1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0), (0, -1.0, 0)):
            hit = np.array([-1], dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    m, d, origin, np.asarray(direction), None, 1, self.probe_body, hit
                )
            )
            name = m.geom(int(hit[0])).name if hit[0] >= 0 else None
            rays.append(
                dict(direction=list(direction), distance_m=distance, first_hit=name)
            )
        registered = lateral < 1e-6 and all(
            ray["first_hit"] is not None
            and ray["first_hit"].startswith(EQUIPMENT + "__annulus_reference_datum_")
            and abs(ray["distance_m"] - BORE_RADIUS) < 0.000025
            for ray in rays
        )
        return dict(
            tip_world_m=tip.tolist(),
            datum_world_m=datum.tolist(),
            tip_height_above_datum_m=float(tip[2] - datum[2]),
            lateral_error_m=lateral,
            nominal_radial_clearance_m=BORE_RADIUS - TIP_RADIUS - lateral,
            bore_rays=rays,
            registered=bool(registered),
            state=self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured().copy()
        if np.max(abs(start)) > 1e-4 or np.max(abs(self.data.qvel)) > 0.001:
            raise PlanningError("Begin probe registration at resting home")
        initial = self.observe()
        phases = []
        route = (
            ("raise_above_ring", (0, 0.17)),
            ("rotate_positive", (0.30, 0.17)),
            ("approach_datum", (0.30, 0.1264)),
            ("pass_through_bore", (0.30, 0.1034)),
            ("retract_above_ring", (0.30, 0.17)),
            ("rotate_negative", (-0.30, 0.17)),
            ("approach_second_orientation", (-0.30, 0.1264)),
            ("pass_second_orientation", (-0.30, 0.1034)),
            ("return_home", start),
        )
        for name, target in route:
            result = self.move(target)
            observation = self.observe()
            expected_height = initial["tip_height_above_datum_m"] + target[1] - start[1]
            if result["status"] == "succeeded" and (
                not observation["registered"]
                or abs(observation["tip_height_above_datum_m"] - expected_height)
                > 0.00005
                or np.max(abs(self.motion.measured() - target)) > 0.00005
            ):
                result.update(
                    status="failed",
                    failure="probe_registration_not_observed",
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
                if len(phases) == len(route)
                and all(p["status"] == "succeeded" for p in phases)
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
                "The fixed ring is an authored inspection aid, not photographed ANU instrumentation.",
                "Probe and ring share tank rotation; no independent lateral probing or scientific measurement.",
                "Recorded geometric observations are simulator state, not a calibrated optical sensor.",
                "Clearance checks cover each executed physics step, not continuous time or real bearing stiffness.",
                "Water remains static; no hydrodynamics, heat transfer, thermistor response or native Isaac validation.",
            ],
        )


def inspect_annulus_probe(session):
    return ProbeRegistration(session).run()
