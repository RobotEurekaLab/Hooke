"""Measured dry plate positioning in the photograph-informed Sydney tank."""

from copy import copy

import mujoco
import numpy as np

from real_labs.reset_reporting import combine_reset_reports
from real_labs.motion import PlanningError
from real_labs.rigid_access import RigidAccess
from real_labs.sydney_tank_geometry import CAMERA

SCENE = "sydney_fluids_convection"
EQUIPMENT = "side_heated_tank"
CONTROLS = ("probe_x", "probe_height")


class TankInspection(RigidAccess):
    def __init__(self, session):
        super().__init__(session, SCENE, EQUIPMENT, CONTROLS)

    def observe(self):
        self.preflight()
        m, d = self.model, self.data
        camera = m.camera(CAMERA).id
        target = m.geom(EQUIPMENT + "__sydney_plate_marker").id
        site = m.site(EQUIPMENT + "__site_calibration_plate").id
        point, origin = d.site_xpos[site], d.cam_xpos[camera]
        local = d.cam_xmat[camera].reshape(3, 3).T @ (point - origin)
        depth = -float(local[2])
        extent = depth * np.tan(np.deg2rad(m.cam_fovy[camera] / 2))
        normalized = (local[:2] / extent).tolist() if extent > 0 else None
        in_frame = bool(
            depth > 0 and abs(normalized[0]) < 1.6 and abs(normalized[1]) < 1
        )
        # Ignore only optically transparent context on a private ray model.
        # Glass/liquid refraction is absent; their live contact masks stay intact.
        ray_model = copy(m)
        alpha = m.geom_rgba[:, 3].copy()
        assigned = m.geom_matid >= 0
        alpha[assigned] *= m.mat_rgba[m.geom_matid[assigned], 3]
        ray_model.geom_group[:] = np.where(alpha >= 0.95, 0, 1)
        ray_data = mujoco.MjData(ray_model)
        ray_data.qpos[:] = d.qpos
        mujoco.mj_forward(ray_model, ray_data)
        direction = point - origin
        direction /= np.linalg.norm(direction)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            ray_model,
            ray_data,
            origin,
            direction,
            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            1,
            int(m.cam_bodyid[camera]),
            hit,
        )
        return dict(
            camera=CAMERA,
            target_geom=m.geom(target).name,
            target_world_m=point.tolist(),
            camera_world_m=origin.tolist(),
            frame_normalized_xy=normalized,
            frame_aspect=1.6,
            depth_m=depth,
            in_frame=in_frame,
            first_opaque_hit=None if hit[0] < 0 else m.geom(int(hit[0])).name,
            first_opaque_distance_m=float(distance),
            visible=bool(in_frame and hit[0] == target),
            ray_scope="Opaque geometric visibility through transparent context; no refraction or calibrated optics",
            **self.state(),
        )

    def run(self):
        self.preflight()
        start = self.motion.measured()
        if np.any(np.abs(start) > 0.0002):
            raise PlanningError("Tank round trip requires the measured home pose")
        phases, observations = [], []
        for name, target in (
            ("left_inspection", (-0.20, 0.035)),
            ("right_inspection", (0.20, -0.035)),
            ("return_home", start),
        ):
            result = self.move(target)
            if result["status"] == "succeeded":
                observation = self.observe()
                observations.append(dict(phase=name, **observation))
                if not observation["visible"] or np.any(
                    np.abs(self.motion.measured() - target) > 0.0002
                ):
                    result.update(
                        status="failed",
                        failure="plate_not_observed_at_target",
                        hold_requested=self.motion._request_hold(),
                    )
            phases.append(dict(phase=name, **result))
            if result["status"] != "succeeded":
                break
        return dict(
            status=(
                "succeeded"
                if len(phases) == 3 and all(p["status"] == "succeeded" for p in phases)
                else "failed"
            ),
            scene_sha256=self.binding["scene_sha256"],
            **combine_reset_reports(phases),
            phases=phases,
            observations=observations,
            measured_return_error_m=(self.motion.measured() - start).tolist(),
            scope="Measured dry positioning of an inert 30 mm plate with actual instrument-camera visibility",
            limitations=[
                "Clear liquid and blue illumination are visual context without fluid or thermal forces",
                "Camera pose, guides and tank dimensions are authored estimates",
                "No calibrated PIV/LIF, convection, fluid measurement or optical refraction",
                "Ideal guides and surrogate mass/gains; no bearing or hardware validation",
                "Finite planned paths and every completed physics-step clearance checks",
                "Failure requests a position hold without a reset; not a certified stop",
            ],
        )


def inspect_sydney_tank(session):
    return TankInspection(session).run()
