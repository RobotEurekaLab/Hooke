"""Measured mounted-camera inspection of retained inert fiducials.

The images are geometric observations of authored hardware. The decorative
specimen is never manipulated as an organism; no spectral or biological model.
"""

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "purdue_phenotyping"
EQUIPMENT = "phenotyping_booth"
CONTROLS = ("conveyor", "plant_turntable", "camera_height")
CAMERA = EQUIPMENT + "__plant_camera"


class PurdueInspection(RigidAccess):
    """Preserved drives, supported carriers and actual camera observations."""

    def __init__(self, session, controls=CONTROLS):
        if tuple(controls) not in (CONTROLS, CONTROLS[:2], CONTROLS[2:]):
            raise ValueError("Select all axes, the carrier pair, or camera height")
        super().__init__(session, SCENE, EQUIPMENT, tuple(controls))
        self.selected_controls = tuple(controls)
        self.motion.resolution[:] = [
            0.004 if c != "plant_turntable" else 0.025 for c in controls
        ]

    def move(self, target, *, step_observer=None):
        self.preflight()
        if step_observer is not None and not callable(step_observer):
            raise ValueError("Observer must be callable and read-only")
        before = self.state()
        self.reset_counts()
        try:
            self.check(self.model, self.data)
            plan = self.motion.plan_joint_target(target)
            increments = 0.8 * 0.025 * (self.motion.upper - self.motion.lower)
            path = [plan.waypoints[0]]
            for a, b in zip(plan.waypoints, plan.waypoints[1:]):
                a, b = np.asarray(a), np.asarray(b)
                count = max(1, math.ceil(float(np.max(np.abs(b - a) / increments))))
                path.extend(tuple(a + (b - a) * k / count) for k in range(1, count + 1))
            plan = replace(plan, waypoints=tuple(path) + (path[-1],) * 4)
        except (PlanningError, RuntimeError) as error:
            self.reset_counts()
            result = dict(
                status="blocked",
                failure=str(error),
                physics_steps=0,
                hold_requested=self.motion._request_hold(),
                reset_performed=False,
            )
        else:
            self.reset_counts()

            def observe(model, data):
                self.check(model, data)
                if step_observer is not None:
                    try:
                        step_observer(model, data)
                    except Exception:
                        self.checked -= 1
                        raise

            result = self.motion.execute(plan, step_observer=observe)
        steps = result["physics_steps"]
        complete = (
            steps > 0 and steps == self.checked and result["status"] == "succeeded"
        )
        result.update(
            before=before,
            after=self.state(),
            direct_clearance=dict(
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
            ),
        )
        return result

    def observe_witness(self, index):
        self.preflight()
        if index not in (0, 1):
            raise ValueError("Choose an actual registered witness")
        m, d = self.model, self.data
        interface = self.session.equipment[EQUIPMENT]["inspection_interface"]
        site = m.site(interface["witness_sites"][index]).id
        geom = m.geom(interface["witness_geoms"][index]).id
        camera = m.camera(CAMERA).id
        point = d.site_xpos[site].copy()
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
        distance = float(np.linalg.norm(direction))
        direction /= distance
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
        visible = bool(
            pixel and 0 < pixel[0] < 1120 and 0 < pixel[1] < 700 and hit[0] == geom
        )
        error = float(np.linalg.norm(np.asarray(pixel) - [560, 350])) if pixel else None
        return dict(
            witness_index=index,
            camera=CAMERA,
            target_world_m=point.tolist(),
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[camera].tolist(),
            target_pixel=pixel,
            resolution=[1120, 700],
            fovy_deg=float(m.cam_fovy[camera]),
            centre_error_px=error,
            visible=visible,
            registered=visible and error < 1,
            first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            ray_distance_m=ray,
            optical_standoff_m=depth,
            estimated_witness_diameter_px=0.016 * focal / depth if depth > 0 else None,
            actual_state=self.state(),
            scope="Physical inert fiducial and uncalibrated pinhole visibility; no spectral acquisition",
        )

    def run(self, *, step_observer=None):
        self.preflight()
        if self.selected_controls != CONTROLS:
            raise ValueError("The complete task requires all three original axes")
        start = self.motion.measured().copy()
        if np.max(np.abs(start)) > 1e-4 or np.max(np.abs(self.data.qvel)) > 0.01:
            raise PlanningError("Start at the measured stationary authored home pose")
        initial = self.state()
        m, d = self.model, self.data
        interface = self.session.equipment[EQUIPMENT]["inspection_interface"]
        origin = d.cam_xpos[m.camera(CAMERA).id]
        sites = [d.site_xpos[m.site(n).id].copy() for n in interface["witness_sites"]]
        y = float(origin[1] - sites[0][1])
        heights = [float(p[2] - origin[2]) for p in sites]
        phases = []
        observations = []
        commands = [
            ("register_carrier", (y, 0, 0), None),
            ("inspect_lower_witness", (y, 0, heights[0]), 0),
            ("inspect_upper_witness", (y, 0, heights[1]), 1),
            ("return_camera", (y, 0, 0), None),
            ("return_carrier", start, None),
        ]
        for name, target, index in commands:
            result = self.move(target, step_observer=step_observer)
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
            if index is not None:
                observation = self.observe_witness(index)
                observations.append(dict(phase=name, **observation))
                if not observation["registered"]:
                    phases[-1].update(
                        status="failed",
                        failure="physical_witness_not_registered",
                        hold_requested=self.motion._request_hold(),
                    )
                    break
        error = np.abs(self.motion.measured() - start)
        success = len(phases) == len(commands) and all(
            p["status"] == "succeeded" for p in phases
        )
        if success and np.max(error) > 1e-4:
            success = False
            phases[-1].update(
                status="failed",
                failure="measured_return_error",
                hold_requested=self.motion._request_hold(),
            )
        return dict(
            status="succeeded" if success else "failed",
            scene_sha256=self.binding["scene_sha256"],
            task_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_state=initial,
            derived_carrier_translation_m=y,
            derived_camera_heights_m=heights,
            phases=phases,
            observations=observations,
            final_state=self.state(),
            executed_steps=sum(p["physics_steps"] for p in phases),
            checked_steps=sum(p["direct_clearance"]["checked_steps"] for p in phases),
            measured_return_error=error.tolist(),
            **combine_reset_reports(phases),
            limitations=[
                "Retained inert witnesses; no specimen loading, manipulation or release",
                "No calibrated spectral, biological, irrigation or environmental model",
                "Static roller/context chain geometry; ideal guided translation and original authored drives",
                "Completed-step/discrete planning coverage is not continuous collision or safety certification",
                "One exclusive session owner; read-only callbacks and no concurrent state writers",
                "Failure preserves measured state and requests hold; not a certified instantaneous stop",
            ],
        )


def inspect_purdue(session):
    return PurdueInspection(session).run()
