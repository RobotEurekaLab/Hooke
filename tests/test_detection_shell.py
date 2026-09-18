"""Check the original cavity, bores and preserved reference exterior."""

from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Hooke"))

from microscopy.detection_shell import hollow_shell_mesh

RINGS = [( .008, .0975, .2382, 0., .009), (.025, .0975, .2382, 0., .010),
         (.145, .089, .230, .003, .011), (.173, .083, .221, .011, .012)]


def segment_hits(mesh, start, end):
    """Finite segment intersections independent of optional ray libraries."""
    start, end = np.asarray(start), np.asarray(end)
    triangles, direction = mesh.triangles, end-start
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross(direction, e2)
    determinant = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(determinant) > 1e-14
    inverse = np.zeros_like(determinant); inverse[valid] = 1/determinant[valid]
    offset = start-triangles[:, 0]
    u = inverse*np.einsum("ij,ij->i", offset, h)
    q = np.cross(offset, e1)
    v = inverse*np.einsum("j,ij->i", direction, q)
    t = inverse*np.einsum("ij,ij->i", e2, q)
    return valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (t >= 0) & (t <= 1)


class DetectionShellTests(unittest.TestCase):
    def test_closed_cavity_keeps_reference_dimensions_and_both_ports_clear(self):
        vertices, faces = hollow_shell_mesh(RINGS)
        shape = trimesh.Trimesh(vertices, faces, process=False)
        self.assertTrue(shape.is_watertight)
        self.assertTrue(shape.is_winding_consistent)
        self.assertGreater(shape.volume, 0)
        np.testing.assert_allclose(np.ptp(vertices, axis=0), [.195, .4764, .165], atol=1e-12, rtol=0)
        # A finite 16 mm collection bundle fits the original 24/28 mm bores.
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x, y = .008*np.cos(angle), .008*np.sin(angle)
            self.assertFalse(segment_hits(shape, (x, y, .190), (x, y, .095)).any())
            self.assertFalse(segment_hits(shape, (0, x, .095+y), (-.120, x, .095+y)).any())
        # Keep the surrounding lid, left wall, opposite wall and floor closed.
        for start, end in (((.030, 0, .190), (.030, 0, .095)),
                           ((0, .040, .095), (-.120, .040, .095)),
                           ((0, 0, .095), (.120, 0, .095)),
                           ((0, 0, .095), (0, 0, 0))):
            self.assertTrue(segment_hits(shape, start, end).any())

    def test_invalid_insets_and_bores_cannot_silently_fill_or_cut_other_panels(self):
        for arguments in (dict(wall_m=.010), dict(roof_bore_radius_m=.090),
                          dict(side_bore_radius_m=.060), dict(side_bore_centre_yz_m=(0, .170)),
                          dict(wall_m=float('nan')), dict(side_bore_centre_yz_m=(0, float('inf')))):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                hollow_shell_mesh(RINGS, **arguments)
