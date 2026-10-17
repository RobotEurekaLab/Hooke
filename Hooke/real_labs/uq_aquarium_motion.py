"""Measured inert probe positioning, observation and return over a visual aquarium.

This is an authored geometric access task. There is no fluid, collected sample,
water-quality reading or claim that UQ operates this robotic mechanism.
"""

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess

SCENE = "uq_moreton_aquarium"
EQUIPMENT = "water_sampling_bridge"
CONTROLS = ("probe_position", "probe_depth")


class AquariumAccess(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)
        self.interface = session.equipment[EQUIPMENT]["probe_interface"]

    def observe(self):
        self.preflight()
        m, d, i = self.model, self.data, self.interface
        surface = d.site_xpos[m.site(i["surface_site"]).id].copy()
        datum = d.site_xpos[m.site(i["datum_site"]).id].copy()
        cap = m.geom(i["cap_geom"]).id
        camera = m.camera(i["observation_camera"]).id
        origin = d.cam_xpos[camera].copy()
        local = d.cam_xmat[camera].reshape(3, 3).T @ (surface - origin)
        depth = -float(local[2])
        focal = 350 / math.tan(math.radians(m.cam_fovy[camera]) / 2)
        pixel = (
            [560 + focal * local[0] / depth, 350 - focal * local[1] / depth]
            if depth > 0
            else None
        )
        direction = surface - origin
        direction /= np.linalg.norm(direction)
        hit = np.array([-1], np.int32)
        distance = float(
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
        first_hit = int(hit[0])
        display = m.geom(i["display_surface_geom"]).id
        # Exclude only the separately grouped transparent presentation volume
        # from the second ray. This measures solid occlusion, not refraction,
        # transmission or rendered contrast. No contacts or state are changed.
        if first_hit == display:
            distance = float(
                mujoco.mj_ray(
                    m,
                    d,
                    origin,
                    direction,
                    np.array([1, 1, 0, 0, 0, 0], np.uint8),
                    1,
                    -1,
                    hit,
                )
            )
        return dict(
            measured_controls=self.motion.measured().tolist(),
            cap_front_surface_world_m=surface.tolist(),
            fixed_datum_surface_world_m=datum.tolist(),
            lateral_registration_error_m=float(surface[0] - datum[0]),
            vertical_registration_error_m=float(surface[2] - datum[2]),
            intentional_datum_offset_y_m=float(surface[1] - datum[1]),
            cap_bottom_world_m=float(d.geom_xpos[cap, 2] - m.geom_size[cap, 1]),
            camera=i["observation_camera"],
            camera_position_world_m=origin.tolist(),
            camera_rotation_matrix=d.cam_xmat[camera].tolist(),
            fovy_deg=float(m.cam_fovy[camera]),
            resolution=[1120, 700],
            target_pixel=pixel,
            estimated_cap_diameter_px=(
                i["cap_diameter_m"] * focal / depth if depth > 0 else None
            ),
            cap_solid_line_of_sight=bool(
                pixel and 0 < pixel[0] < 1120 and 0 < pixel[1] < 700 and hit[0] == cap
            ),
            camera_path_crosses_display_water=first_hit == display,
            camera_first_visible_geometry=(
                None if first_hit < 0 else m.geom(first_hit).name
            ),
            rendered_visibility_qualified=False,
            camera_first_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            camera_ray_distance_m=distance,
            scope="Actual inert cap surface and offset datum; solid line of sight explicitly ignores only the nonphysical water display layer, without refraction, transmission or image-contrast qualification",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.max(abs(start)) > 1e-4 or np.max(abs(self.data.qvel)) > 0.01:
            raise PlanningError("Begin inert probe inspection at the resting home pose")
        initial = self.observe()
        i = self.interface
        x, z = i["inspection_position_m"], i["inspection_depth_m"]
        phases, observations = [], []
        for name, target in (
            ("traverse_to_datum", (x, 0)),
            ("lower_to_sampling_position", (x, z)),
            ("retract_probe", (x, 0)),
            ("return_traverse", start),
        ):
            result = self.move(target)
            if result["status"] == "succeeded":
                observation = self.observe()
                observations.append(dict(phase=name, **observation))
                aligned_x = (
                    name == "return_traverse"
                    or abs(observation["lateral_registration_error_m"])
                    < i["registration_tolerance_m"]
                )
                aligned_z = (
                    name != "lower_to_sampling_position"
                    or abs(observation["vertical_registration_error_m"])
                    < i["registration_tolerance_m"]
                )
                if (
                    not observation["cap_solid_line_of_sight"]
                    or not aligned_x
                    or not aligned_z
                ):
                    result.update(
                        status="failed",
                        failure="probe_geometric_observation_failed",
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
            scope="Inert geometric sampling-position registration, visible observation and measured return",
            limitations=[
                "The fixed physical datum is offset by 344 mm in Y; only X and Z register. The task never claims touchdown or contact with the datum.",
                "All tanks retain static source-supported water appearance, but the probe collects nothing. No fluid forces, water chemistry, organisms, salinity, sensor response or environmental result.",
                "Original ideal drives and authored running clearances are not real bearing stiffness or verified UQ hardware calibration.",
                "All completed physics steps check enabled cross-weld geometry, including parent-filtered pairs. Finite stepping is not continuous-time safety certification.",
                "Solid geometric line of sight ignores only the explicitly identified transparent water display volume. Actual rendered visibility, transmission, contrast and refraction are not certified by that ray.",
                "Failure requests a hold without reset; that request is not an instantaneous or safety-rated stop.",
            ],
        )


def inspect_uq_aquarium(session):
    return AquariumAccess(session).run()
