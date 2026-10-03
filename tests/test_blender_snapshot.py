"""Portable visual snapshots preserve observed geometry and appearance."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from microscopy.blender_export import export_snapshot


class BlenderSnapshotTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "temp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="snapshot-test-", dir=ROOT / "temp")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                                [[255, 255, 0], [0, 255, 255], [255, 0, 255]]], dtype=np.uint8)
        texture = self.output / "asymmetric.png"
        Image.fromarray(self.pixels).save(texture)
        self.model = mujoco.MjModel.from_xml_string(f"""
            <mujoco><asset>
              <texture name="asymmetric" type="2d" file="{texture}"/>
              <material name="surface" texture="asymmetric" rgba=".1 .4 .7 .8"/>
              <material name="hidden_surface" rgba="1 1 1 0"/>
              <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
                    face="0 2 1  0 1 3  0 3 2  1 2 3" texcoord="0 .1  1 .2  .3 1  .8 .7"/>
            </asset><worldbody>
              <body name="moving" pos=".1 .2 .3" euler="20 30 40">
                <joint name="rotation" axis="0 0 1"/>
                <geom name="mesh" type="mesh" mesh="tetra" material="surface" mass="1"/>
                <geom name="override" type="sphere" size=".03" material="surface" rgba=".8 .2 .1 .4"/>
                <geom name="hidden_geom" type="sphere" size=".03" material="surface" rgba="0 0 0 0"/>
                <geom name="hidden_material" type="sphere" size=".03" material="hidden_surface"/>
                <geom name="proxy" type="sphere" size=".03" group="3"/>
                <camera name="attached" pos="0 0 2" euler="15 25 35" fovy="37"/>
              </body>
            </worldbody></mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0], self.data.qvel[0], self.data.time = .2, .7, 3.25
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self, visuals=()):
        export_snapshot(self.model, self.data, self.output, visuals)
        scene = json.loads((self.output / "scene.json").read_text())
        self.assertEqual(scene["schema_version"], 1)
        self.assertEqual(scene["units"], "m")
        return scene

    def test_rotated_mesh_and_camera_roundtrip_in_world_coordinates(self):
        scene = self.snapshot()
        geom = next(item for item in scene["geoms"] if item["name"] == "mesh")
        mesh = next(item for item in scene["meshes"] if item["id"] == geom["mesh"])
        with np.load(self.output / scene["arrays"], allow_pickle=False) as arrays:
            world_vertices = (arrays[mesh["vertices"]] @ np.array(geom["mat"]).reshape(3, 3).T
                              + geom["pos"])
            geometry = self.model.geom("mesh")
            source = self.model.mesh(int(geometry.dataid[0]))
            start, count = int(source.vertadr[0]), int(source.vertnum[0])
            expected = (self.model.mesh_vert[start:start+count]
                        @ self.data.geom_xmat[geometry.id].reshape(3, 3).T
                        + self.data.geom_xpos[geometry.id])
            np.testing.assert_array_equal(world_vertices, expected)
        camera = next(item for item in scene["cameras"] if item["name"] == "attached")
        np.testing.assert_array_equal(camera["pos"], self.data.cam_xpos[0])
        np.testing.assert_array_equal(camera["mat"], self.data.cam_xmat[0])
        self.assertEqual(camera["fovy"], 37.)

    def test_material_override_and_hidden_geometry_match_the_observed_scene(self):
        scene = self.snapshot()
        geoms = {item["name"]: item for item in scene["geoms"]}
        self.assertEqual(set(geoms), {"mesh", "override"})
        np.testing.assert_allclose(geoms["mesh"]["rgba"], [.1, .4, .7, .8], atol=3e-8)
        np.testing.assert_allclose(geoms["override"]["rgba"], [.8, .2, .1, .4], atol=3e-8)

    def test_per_corner_uvs_and_asymmetric_texture_pixels_are_lossless(self):
        scene = self.snapshot()
        mesh = next(item for item in scene["meshes"] if item["name"] == "tetra")
        texture = next(item for item in scene["textures"] if item["name"] == "asymmetric")
        with np.load(self.output / scene["arrays"], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays[texture["array"]], self.pixels)
            restored_uvs = arrays[mesh["texcoords"]][arrays[mesh["face_texcoords"]]]
            source = self.model.mesh("tetra")
            coord = int(self.model.mesh_texcoordadr[source.id])
            count = int(self.model.mesh_texcoordnum[source.id])
            face, faces = int(source.faceadr[0]), int(source.facenum[0])
            expected_uvs = self.model.mesh_texcoord[coord:coord+count][
                self.model.mesh_facetexcoord[face:face+faces]]
            np.testing.assert_array_equal(restored_uvs, expected_uvs)

    def test_runtime_cell_shape_is_preserved_without_advancing_simulation(self):
        shape = dict(type=4, size=[18e-6, 14e-6, 4e-6], pos=[.3, .4, .5],
                     mat=np.eye(3).ravel().tolist(), rgba=[.8, .7, .6, .5], role="cell",
                     surface=dict(roughness=.6, specular_color=[.03]*3))
        qpos, qvel, time = self.data.qpos.copy(), self.data.qvel.copy(), self.data.time
        scene = self.snapshot([shape])
        runtime = [item for item in scene["geoms"] if item["source"] == "runtime"]
        self.assertEqual(len(runtime), 1)
        for field in ("type", "size", "pos", "mat", "rgba", "surface"):
            self.assertEqual(runtime[0][field], shape[field])
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.data.qvel, qvel)
        self.assertEqual(self.data.time, time)
        self.assertEqual(scene["time_s"], time)


if __name__ == "__main__":
    unittest.main()
