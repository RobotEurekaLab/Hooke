"""Portable scene archives, attributed geometry and usable cabin interiors."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from archetypes.task_catalog import CATALOG
from backends.baseline import write_snapshot
from backends.source_lights import point_light_specs
from worlds.asset_scenes import ASSET_ROOT
from worlds.profiles import WORLDS


class SpaceAssets(unittest.TestCase):
    def test_pinned_files_and_source_terms_survive_conversion(self):
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
        self.assertEqual(sum(row["source_faces"] for row in manifest["assets"]), 48225)
        for row in manifest["assets"]:
            self.assertEqual(len(row["commit"]), 40)
            self.assertTrue(row["terms_url"].startswith("https://"))
            self.assertIn(
                row["credit"], (ASSET_ROOT / row["id"] / "NOTICE.md").read_text()
            )
            for name, digest in row["files"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (ASSET_ROOT / row["id"] / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            mesh = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / row["path"]))
            self.assertEqual(mesh.nmesh, len(row["parts"]))
            self.assertEqual(mesh.ntex, row["diffuse_textures"])
            self.assertTrue(np.isfinite(mesh.mesh_vert).all())

    def test_asset_witnesses_follow_each_gravity_and_remain_unqualified(self):
        for profile in WORLDS.values():
            task = CATALOG[f"space_{profile.name}_assets_workstation"].make_expert()
            task.reset(0)
            joint = task.model.joint("free_cartridge_joint")
            address = int(joint.qposadr[0])
            initial = task.data.qpos[address : address + 3].copy()
            for _ in range(250):
                task.step()
            expected = initial + [0.03, 0, -0.5 * profile.gravity_m_s2 * 0.5**2]
            np.testing.assert_allclose(
                task.data.qpos[address : address + 3], expected, atol=0.0025, rtol=0
            )
            self.assertAlmostEqual(
                float(task.model.body("/external_free:asset_root").mass[0]), 0.1
            )
            self.assertFalse(task.check())
            self.assertTrue(task.task_info["display_only"])
            self.assertIn("asset_closeup", task.task_info["camera_mapping"].values())
            for sample in ("/apollo_display", "/tube_display"):
                np.testing.assert_allclose(
                    task.data.body(sample).xpos,
                    task.data.body(sample + "_mount").xpos,
                    atol=1e-12,
                )

    def test_cabin_visuals_do_not_fill_the_robot_workspace_with_a_convex_hull(self):
        task = CATALOG["space_orbital_assets_workstation"].make_expert()
        task.reset(0)
        model = task.model
        visual = [
            i for i in range(model.ngeom) if model.geom(i).name.startswith("/iss:")
        ]
        self.assertEqual(len(visual), 9)
        self.assertTrue(
            all(model.geom_contype[i] == model.geom_conaffinity[i] == 0 for i in visual)
        )
        points = point_light_specs(
            {
                name: getattr(model, name)
                for name in dir(model)
                if name.startswith("light_")
            }
        )
        self.assertEqual(len(points), 3)
        self.assertTrue(all(0 < row["position"][2] < 2.1 for row in points))
        self.assertTrue(all(row["intensity"] > 0 for row in points))

    def test_external_textures_resolve_after_independent_archive_reload(self):
        task = CATALOG["space_orbital_assets_workstation"].make_expert()
        task.reset(0)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            metadata = write_snapshot(task, folder)
            restored = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
            np.testing.assert_array_equal(restored.tex_data, task.model.tex_data)
            self.assertEqual(restored.nmesh, task.model.nmesh)
            self.assertGreaterEqual(len(metadata["source_assets"]), 16)


if __name__ == "__main__":
    unittest.main()
