"""Terrain samples, mesh closure and top surface agree with source ray queries."""

from collections import Counter
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.heightfield import heightfield_mesh, heightfield_normals


class HeightfieldTests(unittest.TestCase):
    def test_visual_normals_preserve_top_slope_and_do_not_turn_bottom_upward(self):
        heights = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])
        fields = dict(
            hfield_nrow=[3],
            hfield_ncol=[3],
            hfield_size=[[1, 1, 1, 0.1]],
            hfield_adr=[0],
            hfield_data=heights.ravel(),
        )
        vertices, faces = heightfield_mesh(fields, 0)
        normals = heightfield_normals(vertices, faces, 3, 3).reshape(-1, 3, 3)
        expected = np.array([-0.1, -0.2, 1.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(
            normals[:8], np.broadcast_to(expected, (8, 3, 3)), atol=1e-12
        )
        self.assertTrue(np.all(normals[-8:, :, 2] < -0.99))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-12)

    def test_closed_terrain_retains_source_elevation(self):
        model = mujoco.MjModel.from_xml_string(
            """<mujoco><asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .2 .1"/></asset><worldbody><geom type="hfield" hfield="terrain"/></worldbody></mujoco>"""
        )
        model.hfield_data[:] = [0.0, 0.3, 0.4, 0.2, 1.0, 0.1, 0.4, 0.5, 0.6]
        fields = {
            name: getattr(model, name)
            for name in (
                "hfield_nrow",
                "hfield_ncol",
                "hfield_size",
                "hfield_adr",
                "hfield_data",
            )
        }
        vertices, faces = heightfield_mesh(fields, 0)
        np.testing.assert_allclose(vertices[:9, 2], model.hfield_data * 0.2, atol=1e-8)
        edges = Counter(
            tuple(sorted((int(a), int(b))))
            for triangle in faces
            for a, b in zip(triangle, np.roll(triangle, -1))
        )
        self.assertTrue(all(count == 2 for count in edges.values()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for xx, yy in [(-0.75, -0.3), (0.25, -0.4), (0.7, 0.2), (-0.3, 0.6)]:
            geom = np.array([-1], dtype=np.int32)
            hit = mujoco.mj_ray(
                model,
                data,
                np.array([xx, yy, 1.0]),
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                geom,
            )
            z = None
            for triangle in faces[:8]:
                points = vertices[triangle]
                weight = np.linalg.solve(
                    np.vstack((points[:, :2].T, np.ones(3))), [xx, yy, 1.0]
                )
                if np.all(weight >= -1e-10):
                    z = float(weight @ points[:, 2])
                    break
            self.assertIsNotNone(z)
            self.assertAlmostEqual(z, 1 - hit, places=7)


if __name__ == "__main__":
    unittest.main()
