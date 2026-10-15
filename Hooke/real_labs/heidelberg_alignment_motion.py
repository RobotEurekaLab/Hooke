"""Measured dry surface-normal registration of an original two-axis gimbal.

The fixed aperture is an authored geometric reference. No light propagation,
quantum response, vacuum calibration or optical measurement is simulated.
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

SCENE = "heidelberg_jochim_quantum_architecture"
EQUIPMENT = "hqa_alignment"
CONTROLS = ("alignment_tip", "alignment_tilt")
CAMERA = "alignment_detail"


class AlignmentInspection(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.interface = session.equipment[EQUIPMENT]["alignment_interface"]

    def move(self, target):
        # Slow the unchanged surrogate drives with actual settled submoves.
        # The ordinary long ramp exceeded the original 8 mrad tip tracking
        # limit. Gains, ranges, collision margins and that limit stay unchanged.
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        for a, b in zip(plan.waypoints, plan.waypoints[1:]):
            count = max(1, math.ceil(float(np.max(np.abs(np.asarray(b) - a))) / 0.018))
            path.extend(
                tuple(np.asarray(a) + (np.asarray(b) - a) * i / count)
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
        report["authored_settled_submove_span_rad"] = 0.018
        return report

    def observe(self):
        """Read actual rigid surface and reference plane, independent of commands."""
        self.preflight()
        m, d = self.model, self.data
        interface = self.interface
        target = m.geom(interface["target_geom"]).id
        site = m.site(interface["target_site"]).id
        reference = m.site(interface["reference_site"]).id
        camera = m.camera(CAMERA).id
        point, center = d.site_xpos[site].copy(), d.site_xpos[reference].copy()
        normal = d.geom_xmat[target].reshape(3, 3)[:, 2]
        reference_normal = d.site_xmat[reference].reshape(3, 3)[:, 2]
        direction = center - point
        direction /= np.linalg.norm(direction)
        angle = float(np.arccos(np.clip(np.dot(normal, direction), -1, 1)))
        denominator = float(np.dot(normal, reference_normal))
        distance = (
            float(np.dot(center - point, reference_normal) / denominator)
            if abs(denominator) > 1e-10
            else None
        )
        projected = point + distance * normal if distance is not None else None
        miss = (
            float(np.linalg.norm(projected - center)) if projected is not None else None
        )
        ray_hit = np.array([-1], dtype=np.int32)
        clear_distance = float(
            mujoco.mj_ray(
                m,
                d,
                point + normal * 1e-7,
                normal,
                np.array([1, 1, 1, 0, 0, 0], np.uint8),
                1,
                -1,
                ray_hit,
            )
        )
        unobstructed = bool(
            distance is not None
            and distance > 0
            and (clear_distance < 0 or clear_distance > distance + 0.005)
        )
        origin = d.cam_xpos[camera].copy()
        local = d.cam_xmat[camera].reshape(3, 3).T @ (point - origin)
        depth = -float(local[2])
        focal = 350 / math.tan(math.radians(m.cam_fovy[camera]) / 2)
        pixels = (
            [
                560 + focal * float(local[0]) / depth,
                350 - focal * float(local[1]) / depth,
            ]
            if depth > 0
            else None
        )
        visible_ray = point - origin
        hit = np.array([-1], dtype=np.int32)
        camera_distance = float(
            mujoco.mj_ray(
                m,
                d,
                origin,
                visible_ray / np.linalg.norm(visible_ray),
                np.array([1, 1, 1, 0, 0, 0], np.uint8),
                1,
                -1,
                hit,
            )
        )
        visible = bool(
            pixels and 0 < pixels[0] < 1120 and 0 < pixels[1] < 700 and hit[0] == target
        )
        return dict(
            measured_angles_rad=self.motion.measured().tolist(),
            target_surface_world_m=point.tolist(),
            target_normal_world=normal.tolist(),
            reference_center_world_m=center.tolist(),
            reference_normal_world=reference_normal.tolist(),
            surface_normal_error_rad=angle,
            axis_intersection_distance_m=distance,
            axis_intersection_world_m=None if projected is None else projected.tolist(),
            reference_center_miss_m=miss,
            aperture_radial_margin_m=(
                None
                if miss is None
                else interface["reference_aperture_radius_m"] - miss
            ),
            axis_segment_unobstructed=unobstructed,
            axis_segment_clear_through_reference=bool(
                unobstructed
                and miss is not None
                and miss < interface["reference_aperture_radius_m"]
            ),
            axis_ray_first_hit=None if ray_hit[0] < 0 else m.geom(int(ray_hit[0])).name,
            axis_ray_first_distance_m=clear_distance,
            camera=CAMERA,
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[camera].tolist(),
            fovy_deg=float(m.cam_fovy[camera]),
            resolution=[1120, 700],
            target_pixel=pixels,
            estimated_target_diameter_px=(
                interface["target_diameter_m"] * focal / depth if depth > 0 else None
            ),
            visible=visible,
            camera_first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            camera_ray_distance_m=camera_distance,
            scope="Measured geometric surface-normal alignment and pinhole visibility, not optical image contrast or a light/quantum model",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.max(np.abs(start)) > 1e-4 or np.max(np.abs(self.data.qvel)) > 0.01:
            raise PlanningError(
                "Begin the HQA inspection at a finite resting home pose"
            )
        initial = self.observe()
        phases, observations = [], []
        for name, target in (
            ("offset_inspection", (-0.075, 0.090)),
            ("register_surface_normal", self.interface["registered_angles_rad"]),
            ("return_home", start),
        ):
            result = self.move(target)
            if result["status"] == "succeeded":
                observation = self.observe()
                observations.append(dict(phase=name, **observation))
                registered = name != "register_surface_normal" or (
                    observation["surface_normal_error_rad"]
                    < self.interface["normal_registration_tolerance_rad"]
                    and observation["reference_center_miss_m"]
                    < self.interface["ray_center_tolerance_m"]
                    and observation["axis_segment_clear_through_reference"]
                )
                if not observation["visible"] or not registered:
                    result.update(
                        status="failed",
                        failure="physical_alignment_or_observation_failed",
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
                if len(phases) == 3 and all(p["status"] == "succeeded" for p in phases)
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
            measured_return_error_rad=(self.motion.measured() - start).tolist(),
            scope="Actual dry inert-disc orientation toward an authored fixed aperture, observation and return",
            limitations=[
                "Original estimated gimbal, reference aperture and camera; not photographed HQA internals or calibrated hardware",
                "Original mass, gains, ideal gravity compensation and joint ranges retained; bearing stress/preload is not modeled",
                "Complete sampled-state clearance coverage does not prove continuous-time collision safety",
                "Exclusive session ownership required; failure requests hold and preserves actual state/time",
                "Pinhole visibility and geometric normal registration do not prove image contrast, optical propagation, vacuum or quantum response",
            ],
        )


def inspect_heidelberg_alignment(session):
    return AlignmentInspection(session).run()


def main():
    import argparse, json, xml.etree.ElementTree as ET
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene(SCENE)
    session = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = inspect_heidelberg_alignment(session)
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
