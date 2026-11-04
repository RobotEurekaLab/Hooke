"""Scene snapshots normalize only unused UVs on real compiled OBJ meshes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))

from microscopy.blender_export import export_snapshot
from real_labs.export import export_scene_snapshot


class SceneSnapshotExportTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "temp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="scene-snapshot-test-", dir=ROOT / "temp"
        )
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def model(self, *, textured=False, complete=False):
        obj = self.directory / "tetra.obj"
        faces = (
            "f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n"
            if complete
            else "f 1/1 3/3 2/2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        )
        obj.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n" + faces
        )
        # The same mesh has both flat and textured instances when requested.
        # An unused textured material must not prevent flat-colour normalization.
        extra_geom = (
            '<geom name="textured" type="mesh" mesh="tetra" material="pattern" '
            'pos="2 0 0"/>'
            if textured
            else ""
        )
        model = mujoco.MjModel.from_xml_string(
            f"""<mujoco><asset>
              <mesh name="tetra" file="{obj}"/>
              <texture name="checker" type="2d" builtin="checker"
                       width="2" height="2" rgb1="1 0 0" rgb2="0 0 1"/>
              <material name="pattern" texture="checker"/>
              <material name="flat" rgba=".2 .4 .6 1"/>
            </asset><worldbody><body pos=".1 .2 .3">
              <joint name="rotation" axis="0 0 1"/>
              <geom name="flat" type="mesh" mesh="tetra" material="flat"/>
              {extra_geom}
              <camera name="attached" pos="0 0 2"/>
            </body></worldbody></mujoco>"""
        )
        data = mujoco.MjData(model)
        data.qpos[0], data.qvel[0], data.time = 0.2, 0.7, 3.25
        mujoco.mj_forward(model, data)
        return model, data

    def test_incomplete_flat_colour_uvs_are_normalized_without_changing_state(self):
        model, data = self.model()
        self.assertTrue((model.mesh_facetexcoord < 0).any())
        raw_dir = self.directory / "raw"
        original = export_snapshot(model, data, raw_dir)
        before = (data.qpos.copy(), data.qvel.copy(), data.time)
        destination = self.directory / "normalized"
        snapshot = export_scene_snapshot(model, data, str(destination))
        self.assertEqual(snapshot, json.loads((destination / "scene.json").read_text()))
        self.assertEqual(snapshot["unused_incomplete_uv_channels_omitted"], ["tetra"])
        self.assertIsNone(snapshot["meshes"][0]["texcoords"])
        self.assertIsNone(snapshot["meshes"][0]["face_texcoords"])
        for field in ("geoms", "materials", "textures", "cameras", "time_s"):
            self.assertEqual(snapshot[field], original[field])
        with (
            np.load(raw_dir / "geometry.npz", allow_pickle=False) as raw,
            np.load(destination / "geometry.npz", allow_pickle=False) as normalized,
        ):
            self.assertEqual(raw.files, normalized.files)
            for name in raw.files:
                np.testing.assert_array_equal(raw[name], normalized[name])
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(data.time, before[2])

    def test_incomplete_uvs_are_rejected_if_any_mesh_instance_is_textured(self):
        model, data = self.model(textured=True)
        self.assertTrue((model.mesh_facetexcoord < 0).any())
        with self.assertRaisesRegex(
            ValueError, "Textured mesh has incomplete UVs: tetra"
        ):
            export_scene_snapshot(model, data, self.directory / "rejected")

    def test_complete_textured_uvs_and_pixels_are_preserved(self):
        model, data = self.model(textured=True, complete=True)
        self.assertFalse((model.mesh_facetexcoord < 0).any())
        raw_dir = self.directory / "raw"
        original = export_snapshot(model, data, raw_dir)
        destination = self.directory / "normalized"
        snapshot = export_scene_snapshot(model, data, destination)
        self.assertEqual(snapshot["unused_incomplete_uv_channels_omitted"], [])
        for field in ("meshes", "geoms", "materials", "textures"):
            self.assertEqual(snapshot[field], original[field])
        with (
            np.load(raw_dir / "geometry.npz", allow_pickle=False) as raw,
            np.load(destination / "geometry.npz", allow_pickle=False) as normalized,
        ):
            for name in raw.files:
                np.testing.assert_array_equal(raw[name], normalized[name])


if __name__ == "__main__":
    unittest.main()
