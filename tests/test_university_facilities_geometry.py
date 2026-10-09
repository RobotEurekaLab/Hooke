"""Mechanical clearance and support regressions for four university facilities."""

import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.layout_audit import doorway_obstacle_warnings, model_obstacles


SCENES = (
    "harvard_cns_cambridge",
    "caltech_stoltz_schlinger",
    "ntu_smtc_membranes",
    "uq_moreton_aquarium",
)


class UniversityFacilityGeometryTests(unittest.TestCase):
    def load(self, scene_id):
        root, manifest = build_scene(scene_id)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, manifest

    def test_doorway_approaches_remain_clear_of_collision_bounds(self):
        for scene_id in SCENES:
            with self.subTest(scene=scene_id):
                model, data, manifest = self.load(scene_id)
                self.assertEqual(
                    doorway_obstacle_warnings(manifest, model_obstacles(model, data)),
                    [],
                )

    def test_mask_stage_sweep_does_not_collide_with_rear_columns(self):
        model, data, manifest = self.load("harvard_cns_cambridge")
        instrument = manifest["equipment"][0]
        indices = [
            model.joint(instrument["joints"][key]).qposadr[0]
            for key in ("wafer_x", "wafer_y", "alignment_head")
        ]
        for x, y, height in itertools.product(
            np.linspace(-0.065, 0.065, 5),
            np.linspace(-0.09, 0.09, 13),
            (0, 0.11),
        ):
            data.qpos[indices] = x, y, height
            mujoco.mj_forward(model, data)
            penetrations = []
            for contact in data.contact[: data.ncon]:
                names = [model.geom(int(index)).name for index in contact.geom]
                if any("lithography_aligner" in name for name in names):
                    if contact.dist < -1e-5:
                        penetrations.append((names, float(contact.dist)))
            self.assertEqual(penetrations, [], (x, y, height))

    def test_probe_stays_inside_tank_and_crosses_display_water_level(self):
        model, data, manifest = self.load("uq_moreton_aquarium")
        instrument = manifest["equipment"][0]
        site = model.site(instrument["sample_site"]).id
        traverse = model.joint(instrument["joints"]["probe_position"]).qposadr[0]
        immersion = model.joint(instrument["joints"]["probe_depth"]).qposadr[0]
        initial_z, immersed_z = [], []
        for x in np.linspace(-0.55, 0.55, 17):
            for depth in (0, 0.28):
                data.qpos[traverse], data.qpos[immersion] = x, depth
                mujoco.mj_forward(model, data)
                position = data.site_xpos[site] - np.asarray(instrument["pos"])
                self.assertLess(np.linalg.norm(position[:2]) + 0.022, 1.14 - 0.025)
                self.assertGreater(position[2] - 0.004, 0.18)
                (initial_z if depth == 0 else immersed_z).append(position[2])
        self.assertGreater(min(initial_z), 1.06)
        self.assertLess(max(immersed_z), 1.06)

    def test_hood_drawer_sample_is_visible_and_moves_with_carriage(self):
        model, data, manifest = self.load("caltech_stoltz_schlinger")
        instrument = manifest["equipment"][0]
        site = model.site(instrument["sample_site"]).id
        joint = model.joint(instrument["joints"]["supply_drawer"]).qposadr[0]
        before = data.site_xpos[site].copy()
        for extension in np.linspace(0, 0.23, 13):
            data.qpos[joint] = extension
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(
                data.site_xpos[site] - before, (0, -extension, 0), atol=1e-8
            )
            for contact in data.contact[: data.ncon]:
                names = [model.geom(int(index)).name for index in contact.geom]
                if any("hood_primary" in name for name in names):
                    self.assertGreaterEqual(contact.dist, -1e-5, names)
        body = int(model.site_bodyid[site])
        self.assertGreater(model.body_geomnum[body], 0)
        self.assertEqual(instrument["sample_attachment"], "clamped")


if __name__ == "__main__":
    unittest.main()
