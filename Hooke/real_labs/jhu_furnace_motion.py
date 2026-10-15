"""Measured cold feed/sample inspection with an authored furnace surrogate.

The retained specimen stays solid. Height and azimuth observations are geometry
measurements, not growth rate, thermal response or furnace calibration.
"""

from dataclasses import replace
from pathlib import Path
import hashlib
import math

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "jhu_paradim_bulk_crystal_growth"
EQUIPMENT = "mirror_furnace"
CONTROLS = ("feed_rod_height", "feed_rod_rotation")
CAMERA = "feed_detail"


class FurnaceInspection(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.interface = session.equipment[EQUIPMENT]["feed_interface"]

    def move(self, target):
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        # Slow the existing ideal drives with real settled submoves, preserving
        # their original inertia, gains, force caps, range and tracking limits.
        for a, b in zip(plan.waypoints, plan.waypoints[1:]):
            delta = np.asarray(b) - a
            count = max(1, math.ceil(float(np.max(abs(delta) / [0.006, 0.05]))))
            path.extend(
                tuple(np.asarray(a) + delta * i / count) for i in range(1, count + 1)
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
        m, d, i = self.model, self.data, self.interface
        witness = m.geom(i["witness_geom"]).id
        site = m.site(i["witness_site"]).id
        reference = m.site(i["height_reference_site"]).id
        coupon = m.geom(i["coupon_geom"]).id
        lower = m.geom(i["lower_spindle_geom"]).id
        point = d.site_xpos[site].copy()
        bottom = float(d.geom_xpos[coupon, 2] - m.geom_size[coupon, 1])
        lower_top = float(d.geom_xpos[lower, 2] + m.geom_size[lower, 1])
        camera = m.camera(CAMERA).id
        origin = d.cam_xpos[camera].copy()
        local = d.cam_xmat[camera].reshape(3, 3).T @ (point - origin)
        depth = -float(local[2])
        focal = 350 / math.tan(math.radians(m.cam_fovy[camera]) / 2)
        pixel = (
            [560 + focal * local[0] / depth, 350 - focal * local[1] / depth]
            if depth > 0
            else None
        )
        direction = point - origin
        direction /= np.linalg.norm(direction)
        hit = np.array([-1], np.int32)
        ray = float(
            mujoco.mj_ray(
                m,
                d,
                origin,
                direction,
                np.array([1, 1, 1, 0, 0, 0], np.uint8),
                1,
                -1,
                hit,
            )
        )
        return dict(
            measured_controls=self.motion.measured().tolist(),
            witness_surface_world_m=point.tolist(),
            coupon_bottom_world_m=bottom,
            lower_spindle_top_world_m=lower_top,
            cold_axial_gap_m=bottom - lower_top,
            height_reference_world_m=d.site_xpos[reference].tolist(),
            height_error_m=float(point[2] - d.site_xpos[reference, 2]),
            camera=CAMERA,
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[camera].tolist(),
            fovy_deg=float(m.cam_fovy[camera]),
            resolution=[1120, 700],
            target_pixel=pixel,
            estimated_witness_diameter_px=(
                i["witness_diameter_m"] * focal / depth if depth > 0 else None
            ),
            visible=bool(
                pixel
                and 0 < pixel[0] < 1120
                and 0 < pixel[1] < 700
                and hit[0] == witness
            ),
            camera_first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            camera_ray_distance_m=ray,
            scope="Physical cold specimen surface, retained feed motion and external-camera pinhole visibility; no thermal/optical process model",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.max(abs(start)) > 1e-4 or np.max(abs(self.data.qvel)) > 0.01:
            raise PlanningError(
                "Begin the cold inspection at a finite resting home pose"
            )
        initial = self.observe()
        phases = []
        observations = []
        height = self.interface["inspection_height_m"]
        angle = self.interface["inspection_azimuth_rad"]
        for name, target in (
            ("raise_to_height_datum", (height, 0)),
            ("index_retained_coupon", (height, angle)),
            ("return_azimuth", (height, 0)),
            ("lower_to_home", start),
        ):
            result = self.move(target)
            if result["status"] == "succeeded":
                obs = self.observe()
                observations.append(dict(phase=name, **obs))
                on_datum = (
                    name == "lower_to_home"
                    or abs(obs["height_error_m"]) < self.interface["height_tolerance_m"]
                )
                if (
                    not obs["visible"]
                    or not on_datum
                    or obs["cold_axial_gap_m"]
                    < self.interface["minimum_lower_gap_m"] - 1e-7
                ):
                    result.update(
                        status="failed",
                        failure="physical_sample_observation_failed",
                        hold_requested=self.motion._request_hold(),
                    )
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
        executed = sum(p["physics_steps"] for p in phases)
        checked = sum(p["direct_clearance"]["checked_steps"] for p in phases)
        complete = executed > 0 and executed == checked
        return dict(
            status=(
                "succeeded"
                if len(phases) == 4 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_observation=initial,
            phases=phases,
            observations=observations,
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
            **combine_reset_reports(phases),
            final_state=self.state(),
            measured_return_error=(self.motion.measured() - start).tolist(),
            scope="Cold retained-sample height registration, azimuth inspection and measured return",
            limitations=[
                "Rigid retained inert coupon; no loading, transfer, molten zone, growth, heating, optical focusing or pressure model",
                "Original ideal joint masses/gains and authored open bearings; no calibrated hardware accuracy or real furnace operating procedure",
                "The physical 40 mm feed-height target and camera are authored inspection interfaces, not OEM specifications",
                "Every completed physics step has direct enabled cross-weld checks; finite samples are not continuous-time collision certification",
                "Source dimensions, room placement and hidden services remain estimates; instrument photo is not registered to the panorama",
                "A failure hold is a command request, not an instantaneous or safety-rated stop; no implicit reset",
            ],
        )


def inspect_jhu_furnace(session):
    return FurnaceInspection(session).run()
