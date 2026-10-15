"""Measured guarded dry positioning; no specimen loading or material model."""

import math
from dataclasses import replace

import mujoco
import numpy as np

from real_labs.duke_frame_geometry import PREFIX
from real_labs.motion import PlanningError
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import RigidAccess

SCENE = "duke_joint_mechanical_testing"
CONTROLS = ("axial_chuck_position", "lower_chuck_rotation", "guard_opening")


class GuardedInspection(RigidAccess):
    """Open the authored guard, inspect/index the inert coupon, then return."""

    def __init__(self, session):
        super().__init__(session, SCENE, "biaxial_frame", CONTROLS)

    def move(self, target):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        # Subdivide commanded travel, retaining original gains, force limits
        # and every-step tracking thresholds for these deliberately slow drives.
        for first, last in zip(plan.waypoints, plan.waypoints[1:]):
            delta = np.asarray(last) - first
            count = max(1, math.ceil(float(np.max(abs(delta) / [0.008, 0.03, 0.03]))))
            path.extend(
                (
                    tuple(last)
                    if i == count
                    else tuple(np.asarray(first) + delta * i / count)
                )
                for i in range(1, count + 1)
            )
        plan = replace(plan, waypoints=tuple(path) + (path[-1],) * 4)
        self.reset_counts()
        report = self.motion.execute(plan, step_observer=self.check)
        steps = report["physics_steps"]
        complete = steps > 0 and steps == self.checked
        report["direct_clearance"] = dict(
            executed_steps=steps,
            observer_entries=self.entries,
            checked_steps=self.checked,
            unchecked_steps=steps - self.checked,
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
        upper = [m.geom(PREFIX + f"upper_jaw_{x}").id for x in (-0.048, 0.048)]
        coupon = m.geom(PREFIX + "coupon_end_0.182").id
        bottom = min(d.geom_xpos[g, 2] - m.geom_size[g, 2] for g in upper)
        top = float(d.geom_xpos[coupon, 2] + m.geom_size[coupon, 2])
        site = m.site("biaxial_frame__site_specimen_surface").id
        camera = m.camera("specimen_access").id
        point = d.site_xpos[site]
        origin = d.cam_xpos[camera]
        direction = point - origin
        direction = direction / np.linalg.norm(direction)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(m, d, origin, direction, None, 1, -1, hit)
        return dict(
            measured_controls=self.motion.measured().tolist(),
            coupon_reference_world_m=point.tolist(),
            actual_grip_to_coupon_vertical_gap_m=float(bottom - top),
            guard_open_angle_rad=float(self.motion.measured()[2]),
            camera="specimen_access",
            first_ray_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            ray_distance_m=float(distance),
            visibility_scope="First physical ray only; glass transmission and image readability require separate pixel review",
            actual_material_loading=False,
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.max(abs(start)) > 1e-5 or np.max(abs(self.data.qvel)) > 0.01:
            raise PlanningError("Begin this guarded dry inspection at resting home")
        phases, observations = [], []
        initial = self.observe()
        for name, target in (
            ("open_guard", (0, 0, math.radians(100))),
            ("lower_unloaded_grip", (0.20, 0, math.radians(100))),
            ("index_positive", (0.20, 0.60, math.radians(100))),
            ("index_negative", (0.20, -0.60, math.radians(100))),
            ("return_index", (0.20, 0, math.radians(100))),
            ("raise_grip", (0, 0, math.radians(100))),
            ("close_guard", tuple(start)),
        ):
            before = self.state()
            report = self.move(target)
            report.update(phase=name, initial_state=before, final_state=self.state())
            phases.append(report)
            if report["status"] != "succeeded":
                break
            observations.append(dict(phase=name, **self.observe()))
        succeeded = len(phases) == 7 and all(p["status"] == "succeeded" for p in phases)
        return dict(
            status="succeeded" if succeeded else "failed",
            scene_sha256=self.binding["scene_sha256"],
            initial_observation=initial,
            phases=phases,
            observations=observations,
            physics_steps=sum(p["physics_steps"] for p in phases),
            checked_steps=sum(p["direct_clearance"]["checked_steps"] for p in phases),
            **combine_reset_reports(phases),
            limits=[
                "Original dry controls plus authored guard; not exact OEM motion",
                "The closest unloaded grip/coupon gap is55mm, not a contact test",
                "Ideal joints carry loads; bearing gaps do not establish stiffness or preload",
                "No calibrated force, torque, fatigue, fracture or automatic sample transfer",
                "Discrete checked poses do not prove continuous collision clearance",
            ],
        )
