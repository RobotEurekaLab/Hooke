"""Measured inert coupon indexing relative to the tunnel's geometric axes.

The coupon is rigidly retained. This is a dry positioner operation with complete
sampled mechanical checks, not airflow, animal or aerodynamic experimentation.
"""

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.brown_pitch_geometry import PREFIX
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "brown_breuer_afam_tunnel"
EQUIPMENT = "afam_test_section"
CONTROLS = ("model_yaw", "model_pitch")


class CouponIndexing(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.coupon = self.model.geom(PREFIX + "inert_coupon").id
        self.floor = next(
            g
            for g in range(self.model.ngeom)
            if self.model.geom(g).name.startswith(EQUIPMENT + "__test_floor_")
        )

    def move(self, target):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        # Bounded settled moves respect original low-stiffness drives without
        # widening tracking tolerances or changing their full useful ranges.
        for a, b in zip(plan.waypoints, plan.waypoints[1:]):
            count = max(1, math.ceil(float(np.max(np.abs(np.asarray(b) - a))) / 0.008))
            path.extend(
                (
                    tuple(b)
                    if i == count
                    else tuple(np.asarray(a) + (np.asarray(b) - a) * i / count)
                )
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
        report["settled_angular_segment_max_rad"] = 0.008
        return report

    def observe(self):
        self.preflight()
        d = self.data
        reference = d.geom_xmat[self.floor].reshape(3, 3)
        rotation = d.geom_xmat[self.coupon].reshape(3, 3)
        relative = reference.T @ rotation
        chord = relative[:, 0]
        angles = [
            float(math.atan2(chord[1], chord[0])),
            float(math.atan2(-chord[2], math.hypot(chord[0], chord[1]))),
        ]
        return dict(
            coupon_center_world_m=d.geom_xpos[self.coupon].tolist(),
            coupon_chord_world=rotation[:, 0].tolist(),
            coupon_normal_world=rotation[:, 2].tolist(),
            tunnel_frame_rotation=reference.tolist(),
            measured_orientation_rad=angles,
            measured_joints_rad=self.motion.measured().tolist(),
            orientation_vs_joint_error_rad=(
                np.asarray(angles) - self.motion.measured()
            ).tolist(),
            reference="Actual longitudinal floor X axis and its geometric normal; no measured wind direction",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured().copy()
        if np.max(abs(start)) > 1e-4 or np.max(abs(self.data.qvel)) > 0.01:
            raise PlanningError("Begin coupon indexing at stationary neutral yaw/pitch")
        initial = self.observe()
        phases, observations = [], []
        route = (
            ("positive_yaw_pitch", (0.45, 0.32)),
            ("negative_pitch", (0.45, -0.32)),
            ("negative_yaw", (-0.45, -0.32)),
            ("positive_pitch", (-0.45, 0.32)),
            ("return_neutral", start),
        )
        for name, target in route:
            report = self.move(target)
            observed = self.observe()
            observations.append(dict(phase=name, **observed))
            if (
                report["status"] == "succeeded"
                and np.max(
                    abs(np.asarray(observed["measured_orientation_rad"]) - target)
                )
                > 0.0001
            ):
                report.update(
                    status="failed",
                    failure="Actual coupon orientation not reached",
                    hold_requested=self.motion._request_hold(),
                )
            phases.append(dict(phase=name, **report))
            if report["status"] != "succeeded":
                break
        executed = sum(p["physics_steps"] for p in phases)
        checked = sum(p["direct_clearance"]["checked_steps"] for p in phases)
        complete = executed > 0 and executed == checked
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
            observations=observations,
            phases=phases,
            executed_steps=executed,
            checked_steps=checked,
            coverage_complete=complete,
            whole_run_clearance_lower_bound_m=(
                min(
                    p["direct_clearance"]["whole_leg_clearance_lower_bound_m"]
                    for p in phases
                )
                if complete
                else None
            ),
            final_state=self.state(),
            measured_return_error_rad=(self.motion.measured() - start).tolist(),
            **combine_reset_reports(phases),
            scope="Actual full-range inert coupon yaw/pitch indexing, geometric orientation observation and return",
            limitations=[
                "The institutional source documents pitch/yaw functionality and test-section dimensions; hidden mechanism and ideal drives are authored.",
                "Retained inert coupon only; no free loading, panel opening, airflow, animal experiment or aerodynamic force model.",
                "0.6 mm radial running gaps and ideal joints do not establish bearing preload, stiffness or physical positioning accuracy.",
                "Every completed step includes enabled cross-weld geometry, including parent-filtered pairs; sampled checks do not prove continuous-time safety.",
                "Failure requests hold without resetting actual state; exclusive session ownership required.",
                "Geometric coupon orientation is not a calibrated angle-of-attack sensor or a rendered image validation.",
            ],
        )


def index_brown_coupon(session):
    return CouponIndexing(session).run()


def main():
    import argparse, json, xml.etree.ElementTree as ET
    import mujoco
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, manifest = build_scene(SCENE)
    s = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = index_brown_coupon(s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "executed_steps",
                    "checked_steps",
                    "coverage_complete",
                    "whole_run_clearance_lower_bound_m",
                    "measured_return_error_rad",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
