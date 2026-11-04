"""The GCF visual web is a single planar-faced solid with thirteen true bores."""

from collections import Counter, defaultdict
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.perforated_web import web_surface
from real_labs.university_asia_wave2 import _gcf_web_cells, _gcf_web_sectors


class PerforatedWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = list(_gcf_web_cells())
        cls.sectors = [
            q for bounds, hole in cls.cells for q in _gcf_web_sectors(bounds, hole)
        ]
        vertices, cls.faces = web_surface(cls.sectors, half_thickness=0.010)
        cls.vertices = np.asarray(vertices)
        cls.triangles = cls.vertices[np.asarray(cls.faces)]
        cls.normals = np.cross(
            cls.triangles[:, 1] - cls.triangles[:, 0],
            cls.triangles[:, 2] - cls.triangles[:, 0],
        )

    def test_single_connected_oriented_manifold_has_thirteen_bores(self):
        self.assertEqual(len(set(map(tuple, self.vertices))), len(self.vertices))
        directed = Counter(
            edge for face in self.faces for edge in zip(face, face[1:] + face[:1])
        )
        adjacency = defaultdict(set)
        for (a, b), count in directed.items():
            self.assertEqual(count, 1)
            self.assertEqual(directed[(b, a)], 1)
            adjacency[a].add(b)
        visited, pending = set(), [0]
        while pending:
            vertex = pending.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            pending.extend(adjacency[vertex] - visited)
        self.assertEqual(len(visited), len(self.vertices))
        euler = len(self.vertices) - len(directed) // 2 + len(self.faces)
        self.assertEqual(euler, 2 - 2 * len(self.cells))

    def test_caps_are_exactly_flat_with_consistent_outward_normals(self):
        areas_twice = np.linalg.norm(self.normals, axis=1)
        self.assertGreater(areas_twice.min(), 1e-10)
        normal = self.normals / areas_twice[:, None]
        for side in (-1, 1):
            cap = np.all(np.isclose(self.triangles[:, :, 1], side * 0.010), axis=1)
            self.assertGreater(np.count_nonzero(cap), 100)
            np.testing.assert_allclose(
                normal[cap], np.tile([0, side, 0], (sum(cap), 1)), atol=1e-12
            )

    def test_side_walls_only_follow_rectangle_or_bore_boundaries(self):
        walls = np.ptp(self.triangles[:, :, 1], axis=1) > 0.019
        for triangle in self.triangles[walls]:
            edge = np.unique(triangle[:, (0, 2)], axis=0)
            self.assertEqual(len(edge), 2)
            outer = (
                np.allclose(edge[:, 0], -1.21)
                or np.allclose(edge[:, 0], 1.21)
                or np.allclose(edge[:, 1], 0.010)
                or np.allclose(edge[:, 1], 0.365)
            )
            bore = any(
                np.allclose(np.linalg.norm(edge - [cx, cz], axis=1), radius, atol=1e-11)
                for _, (cx, cz, radius) in self.cells
            )
            self.assertTrue(outer or bore, edge)

    def test_volume_matches_plate_minus_polygonal_holes(self):
        hole_area = 0
        for bounds, hole in self.cells:
            ring = np.asarray([q[0] for q in _gcf_web_sectors(bounds, hole)])
            following = np.roll(ring, -1, axis=0)
            hole_area += (
                np.sum(ring[:, 0] * following[:, 1] - following[:, 0] * ring[:, 1]) / 2
            )
        expected = (2.42 * (0.365 - 0.010) - hole_area) * 0.020
        volume = np.einsum("ij,ij->i", self.triangles[:, 0], self.normals).sum() / 6
        self.assertAlmostEqual(volume, expected, places=11)


if __name__ == "__main__":
    unittest.main()
