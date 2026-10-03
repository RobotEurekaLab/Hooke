"""Actual transformed plant geometry must fit the imaging enclosure's travel."""

import itertools
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.builder import build_scene


def _world_vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[first : first + count].astype(float)
    else:
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        local = center + np.array(list(itertools.product((-1, 1), repeat=3))) * half
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def specimen_fit_report(model, travel_override=None):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body("phenotyping_booth").id
    turn = model.body("phenotyping_booth__body_plant_turntable").id
    conveyor = model.joint("phenotyping_booth__joint_conveyor").id
    rotation = model.joint("phenotyping_booth__joint_plant_turntable").id
    specimen = [
        index
        for index in range(model.ngeom)
        if (model.geom(index).name or "").startswith("facility_imaging_specimen_")
    ]
    if not specimen:
        raise ValueError("Missing imaging specimen geometry")
    world_to_base = data.xmat[base].reshape(3, 3)
    origin = data.xpos[base].copy()

    def local_points(geom):
        return (_world_vertices(model, data, geom) - origin) @ world_to_base

    enclosure = [
        local_points(index)
        for index in range(model.ngeom)
        if model.geom_bodyid[index] == base
        and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
    ]
    overhead = min(
        points[:, 2].min() for points in enclosure if points[:, 2].min() > 2.20
    )
    backdrop = min(
        points[:, 1].min()
        for points in enclosure
        if points[:, 1].min() > 0.58
        and points[:, 0].min() < 0 < points[:, 0].max()
        and points[:, 2].min() < 1.4 < points[:, 2].max()
    )
    sides = min(
        abs(points[:, 0]).min()
        for points in enclosure
        if (points[:, 0].min() > 0.60 or points[:, 0].max() < -0.60)
        and points[:, 2].min() < 1.3 < points[:, 2].max()
    )
    limits = model.jnt_range[conveyor].copy()
    if travel_override is not None:
        limits[1] = travel_override
    lows, highs, radii, centers = [], [], [], []
    for travel, angle in itertools.product(
        np.linspace(*limits, 3), np.linspace(-np.pi, np.pi, 17)
    ):
        data.qpos[model.jnt_qposadr[conveyor]] = travel
        data.qpos[model.jnt_qposadr[rotation]] = angle
        mujoco.mj_forward(model, data)
        points = np.concatenate([local_points(geom) for geom in specimen])
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite transformed plant geometry")
        center = (data.xpos[turn] - origin) @ world_to_base
        radii.append(float(np.linalg.norm(points[:, :2] - center[:2], axis=1).max()))
        centers.append(center)
        lows.append(points.min(axis=0))
        highs.append(points.max(axis=0))
    maximum = np.max(highs, axis=0)
    radius = max(radii)
    # Translation is solely along Y and turntable rotation solely about Z.
    # The radial bound therefore covers every intermediate yaw, not just samples.
    rear_gap = backdrop - (np.max(centers, axis=0)[1] + radius)
    return dict(
        scope="Original young-maize visual bounds in authored surrogate enclosure; no plant contact mechanics or measured-source geometry",
        sampled_poses=len(highs),
        conveyor_range_m=limits.tolist(),
        specimen_minimum_m=np.min(lows, axis=0).tolist(),
        specimen_maximum_m=maximum.tolist(),
        rotational_radius_bound_m=radius,
        overhead_underside_m=float(overhead),
        rear_backdrop_front_m=float(backdrop),
        overhead_clearance_m=float(overhead - maximum[2]),
        rear_clearance_all_yaw_m=float(rear_gap),
        side_clearance_all_yaw_m=float(sides - radius),
        finite=True,
    )


class PurdueGeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, _ = build_scene("purdue_phenotyping")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def test_young_plant_fits_overhead_rear_and_side_through_all_travel(self):
        report = specimen_fit_report(self.model)
        self.assertTrue(report["finite"])
        for key in (
            "overhead_clearance_m",
            "rear_clearance_all_yaw_m",
            "side_clearance_all_yaw_m",
        ):
            self.assertGreaterEqual(report[key], 0.02, report)

    def test_original_long_travel_exposes_rear_intersection(self):
        report = specimen_fit_report(self.model, travel_override=1.08)
        self.assertLess(report["rear_clearance_all_yaw_m"], 0, report)


if __name__ == "__main__":
    unittest.main()
