"""Closed optical volume and grounded contact furniture in the ANU scene."""

from collections import Counter
import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class AnnulusContextTests(unittest.TestCase):
    def test_liquid_is_consistently_oriented_closed_annular_volume(self):
        root, _ = build_scene("anu_climate_fluid_physics")
        name = "large_rotating_annulus__annular_liquid"
        mesh = root.find(f"asset/mesh[@name='{name}']")
        vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
        faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        centers = triangles.mean(axis=1)
        self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 1e-8))
        bottom = np.all(np.isclose(triangles[:, :, 2], 0.054), axis=1)
        top = np.all(np.isclose(triangles[:, :, 2], 0.365), axis=1)
        sides = ~(bottom | top)
        inner = sides & (np.linalg.norm(centers[:, :2], axis=1) < 0.4)
        outer = sides & ~inner
        self.assertTrue(np.all(normals[bottom, 2] < 0))
        self.assertTrue(np.all(normals[top, 2] > 0))
        radial = np.einsum("ij,ij->i", normals[:, :2], centers[:, :2])
        self.assertTrue(np.all(radial[inner] < 0))
        self.assertTrue(np.all(radial[outer] > 0))
        edges = Counter(
            (int(face[i]), int(face[(i + 1) % 3])) for face in faces for i in range(3)
        )
        self.assertTrue(
            all(count == 1 and edges[b, a] == 1 for (a, b), count in edges.items())
        )
        volume = (
            np.einsum(
                "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
            ).sum()
            / 6
        )
        expected = (
            48 * math.sin(math.tau / 48) / 2 * (0.777**2 - 0.201**2) * (0.365 - 0.054)
        )
        self.assertAlmostEqual(volume, expected, delta=5e-9)
        geom = root.find(f".//geom[@name='{name}']")
        self.assertEqual((geom.get("contype"), geom.get("conaffinity")), ("0", "0"))

    def test_new_contact_furniture_has_continuous_support_to_floor(self):
        root, _ = build_scene("anu_climate_fluid_physics")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ids = [
            i
            for i in range(model.ngeom)
            if model.geom_contype[i] or model.geom_conaffinity[i]
            if model.geom(i).name == "arch_floor"
            or model.geom(i).name.startswith(("anu_context_", "acquisition_desk_"))
        ]
        targets = {i for i in ids if model.geom(i).name.startswith("anu_context_")}
        self.assertGreater(len(targets), 30)
        self.assertTrue(
            all(model.body_weldid[model.geom_bodyid[i]] == 0 for i in targets)
        )
        rotations = data.geom_xmat.reshape(-1, 3, 3)
        center = data.geom_xpos + np.einsum(
            "nij,nj->ni", rotations, model.geom_aabb[:, :3]
        )
        half = np.einsum("nij,nj->ni", abs(rotations), model.geom_aabb[:, 3:])
        adjacency = {i: set() for i in ids}
        for offset, a in enumerate(ids):
            for b in ids[offset + 1 :]:
                if np.any(abs(center[a] - center[b]) - half[a] - half[b] > 1e-7):
                    continue
                if mujoco.mj_geomDistance(model, data, a, b, 1e-6, None) <= 1e-7:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        floor = model.geom("arch_floor").id
        seen, todo = {floor}, [floor]
        while todo:
            for item in adjacency[todo.pop()] - seen:
                seen.add(item)
                todo.append(item)
        self.assertEqual({model.geom(i).name for i in targets - seen}, set())


if __name__ == "__main__":
    unittest.main()
