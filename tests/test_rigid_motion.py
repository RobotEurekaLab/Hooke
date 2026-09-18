"""Rotational clearance checks must not certify an intersecting swing."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from microscopy.assembly_audit import resolve_motion_candidates
from microscopy.rigid_motion import RigidMotionBounds, body_displacement, geom_pose


def rotate_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class RigidBounds(unittest.TestCase):
    def test_rotating_offset_and_vertices_stay_inside_the_point_displacement_bound(self):
        positions = np.array([[0., 0., 0.]])
        rotations = np.array([np.eye(3)])
        motion = RigidMotionBounds(positions, rotations, segment_steps=10)
        point = np.array([.02, .03, .01])
        radius = np.linalg.norm(point)
        observed = []
        for angle in np.linspace(0., .7, 23):
            moved = positions + [angle * .001, 0., 0.]
            rotated = np.array([rotate_z(angle)])
            motion.observe(moved, rotated)
            observed.append(moved[0] + rotated[0] @ point)
        arrays = motion.arrays()
        bound = body_displacement(positions, arrays["rigid_positions_min"],
            arrays["rigid_positions_max"], arrays["rigid_angle_max"], [0], [radius])[0]
        self.assertGreaterEqual(bound, max(np.linalg.norm(p - point) for p in observed))
        self.assertEqual(len(arrays["rigid_segment_positions"]), 3)

    def test_part_offset_rotates_with_its_body_instead_of_remaining_in_world_coordinates(self):
        arrays = dict(rigid_geom_body_map=np.array([0]),
                      rigid_geom_local_positions=np.array([[.03, 0., 0.]]),
                      rigid_geom_local_rotations=np.array([np.eye(3)]))
        rotation, position = geom_pose(arrays, 0, np.array([[.1, .2, .3]]),
                                       np.array([rotate_z(np.pi / 2)]))
        np.testing.assert_allclose(position, [.1, .23, .3], atol=1e-15)
        np.testing.assert_allclose(rotation, rotate_z(np.pi / 2))


@unittest.skipUnless(importlib.util.find_spec("fcl"), "Optional offline surface inspection dependency")
class RotationalClearance(unittest.TestCase):
    def check_swing(self, obstacle_y):
        positions = np.array([[0., 0., 0.], [0., obstacle_y, 0.]])
        rotations = np.array([np.eye(3), np.eye(3)])
        motion = RigidMotionBounds(positions, rotations, segment_steps=1)
        for angle in np.linspace(0., np.pi / 2, 21):
            motion.observe(positions, np.array([rotate_z(angle), np.eye(3)]))
        sizes = np.array([[.05, .002, .002], [.004, .004, .004]])
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as folder:
            root = Path(folder)
            path = root / "geometry.npz"
            np.savez_compressed(path, names=np.array(["arm", "obstacle"]),
                geom_type=np.array([6, 6]), sizes=sizes, rotations=rotations,
                positions=positions, positions_min=positions, positions_max=positions,
                rigid_geom_body_map=np.array([0, 1]), rigid_geom_local_positions=np.zeros((2, 3)),
                rigid_geom_local_rotations=rotations,
                rigid_geom_radii=np.linalg.norm(sizes, axis=1), **motion.arrays())
            report = dict(geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                motion_steps=21, skipped=[], surface_clear=True,
                motion_envelopes={"arm vs obstacle": {"candidate_pairs": [["arm", "obstacle"]]}})
            (root / "report.json").write_text(json.dumps(report))
            return resolve_motion_candidates(root)

    def test_fixed_center_does_not_hide_a_rotating_arm_that_hits_an_obstacle(self):
        self.assertFalse(self.check_swing(.04)["recorded_motion_mesh_clearance_established"])

    def test_small_rotational_intervals_can_establish_a_clear_swing(self):
        self.assertTrue(self.check_swing(.07)["recorded_motion_mesh_clearance_established"])


if __name__ == "__main__":
    unittest.main()
