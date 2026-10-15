"""Measured roof access, retained inert coupon registration, and actual closure.

This is an authored mechanical inspection of a reference-informed exterior.
There is no laser, mineral response, optical focus, or free-specimen handling.
"""

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.chicago_enclosure_geometry import PREFIX
from real_labs.motion import PlanningError
from real_labs.reset_reporting import combine_reset_reports
from real_labs.rigid_access import RigidAccess

SCENE = "chicago_mineralphysics_laser_bay"
EQUIPMENT = "covered_optical_table"
CONTROLS = ("access_lid_angle", "mineral_coupon_x")
CAMERA = "enclosure_inspection"


class EnclosureInspection(RigidAccess):
    """Exclusive-session dry inspection with enabled, parent-inclusive checks."""

    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)

    def move(self, target):
        # The original long opening ramp exceeded the unchanged 15 mrad
        # tracking guard. Bounded settled segments retain the original drives,
        # full travel, physics and safety tolerances.
        self.preflight()
        self.reset_counts()
        self.check(self.model, self.data)
        plan = self.motion.plan_joint_target(target)
        path = [plan.waypoints[0]]
        for a, b in zip(plan.waypoints, plan.waypoints[1:]):
            count = max(1, math.ceil(abs(b[0] - a[0]) / 0.025))
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
        report["settled_lid_segment_max_rad"] = 0.025
        return report

    def observe(self):
        self.preflight()
        m, d = self.model, self.data
        coupon = m.geom(PREFIX + "inert_coupon").id
        datum = m.geom(PREFIX + "datum_tab").id
        lid = m.geom(PREFIX + "lid_sheet").id
        front = m.site(EQUIPMENT + "__site_cover_front").id
        cam = m.camera(CAMERA).id
        point = d.geom_xpos[coupon].copy()
        index_axis = d.geom_xmat[datum].reshape(3, 3)[:, 0]
        index_error = float(np.dot(point - d.geom_xpos[datum], index_axis))
        lid_normal = d.geom_xmat[lid].reshape(3, 3)[:, 2]
        angle = float(math.atan2(lid_normal[1], lid_normal[2]))
        origin = d.cam_xpos[cam].copy()
        local = d.cam_xmat[cam].reshape(3, 3).T @ (point - origin)
        depth = -float(local[2])
        focal = 350 / math.tan(math.radians(m.cam_fovy[cam]) / 2)
        pixel = (
            [
                560 + focal * float(local[0]) / depth,
                350 - focal * float(local[1]) / depth,
            ]
            if depth > 0
            else None
        )
        hit = np.array([-1], np.int32)
        direction = point - origin
        ray_distance = float(
            mujoco.mj_ray(
                m,
                d,
                origin,
                direction / np.linalg.norm(direction),
                np.array([1, 1, 1, 0, 0, 0], np.uint8),
                1,
                -1,
                hit,
            )
        )
        visible = bool(
            pixel and 0 < pixel[0] < 1120 and 0 < pixel[1] < 700 and hit[0] == coupon
        )
        return dict(
            measured_controls=self.motion.measured().tolist(),
            physical_open_angle_rad=angle,
            cover_front_world_m=d.site_xpos[front].tolist(),
            coupon_center_world_m=point.tolist(),
            datum_center_world_m=d.geom_xpos[datum].tolist(),
            datum_normal_world=index_axis.tolist(),
            coupon_index_plane_error_m=index_error,
            coupon_visible=visible,
            camera=CAMERA,
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[cam].tolist(),
            fovy_deg=float(m.cam_fovy[cam]),
            resolution=[1120, 700],
            coupon_pixel=pixel,
            camera_first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            camera_ray_distance_m=ray_distance,
            scope="X-coordinate registration to the physical index-tab plane, with external pinhole visibility. Not optical alignment or pixel-quality validation.",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.max(np.abs(start)) > 1e-4 or np.max(np.abs(self.data.qvel)) > 0.01:
            raise PlanningError(
                "Begin with the stationary revision-2 roof closed and coupon centered"
            )
        initial = self.observe()
        phases, observations = [], []
        route = (
            ("open_roof", (-0.6, 0)),
            ("inspect_left_travel", (-0.6, -0.07)),
            ("inspect_right_travel", (-0.6, 0.07)),
            ("register_inert_coupon", (-0.6, 0)),
            ("close_roof", (0, 0)),
        )
        for name, target in route:
            report = self.move(target)
            if report["status"] == "succeeded":
                observed = self.observe()
                observations.append(dict(phase=name, **observed))
                endpoint = bool(
                    np.all(
                        np.abs(np.asarray(observed["measured_controls"]) - target)
                        <= [0.003, 0.0002]
                    )
                )
                opening = (
                    abs(
                        observed["physical_open_angle_rad"]
                        - (0 if name == "close_roof" else 0.6)
                    )
                    < 0.003
                )
                registration = (
                    name != "register_inert_coupon"
                    or abs(observed["coupon_index_plane_error_m"]) < 0.0002
                )
                visibility = observed["coupon_visible"] == (name != "close_roof")
                if not (endpoint and opening and registration and visibility):
                    report.update(
                        status="failed",
                        failure="measured_enclosure_or_registration_failed",
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
            measured_return_error=self.motion.measured().tolist(),
            **combine_reset_reports(phases),
            scope="Actual roof opening, full retained-coupon X traverse, physical index-plane registration and closure",
            limitations=[
                "Reference establishes the exterior only; hidden hinges, guides, datum and motorization are authored estimates.",
                "The coupon remains clamped. No free loading, beam alignment, optical focus, laser or mineral experiment is performed.",
                "Revision 2 explicitly changes the physical lid zero while retaining the original numeric control range. Legacy revision 1 remains available.",
                "Bearings retain 0.6 mm geometric clearance and ideal joints; preload, seal integrity and structural compliance are not modeled.",
                "Every executed physics step receives core safety checks; completed direct checks additionally include enabled cross-weld pairs omitted by parent filtering. Sampled checks do not prove continuous-time safety.",
                "Exclusive session ownership required; failure requests servo hold without resetting actual state or time.",
                "External geometric camera visibility is not rendered pixel validation or an instrument camera.",
            ],
        )


def inspect_chicago_enclosure(session):
    return EnclosureInspection(session).run()


def main():
    import argparse
    import json
    import xml.etree.ElementTree as ET
    from real_labs.builder import build_scene
    from real_labs.runtime import InstrumentSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, manifest = build_scene(SCENE)
    session = InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )
    report = inspect_chicago_enclosure(session)
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
                    "measured_return_error",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
