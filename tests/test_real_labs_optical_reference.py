"""Reference-shaped optical housings retain supports and sample access."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.usability import camera_target_screening


class OpticalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("kyoto_yokokawa_nanometrics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_primary_housings_have_continuous_supports(self):
        attachments = (
            ("foot", "lower_body_shell"),
            ("lower_body_shell", "arm_hinge"),
            ("arm_hinge", "slanted_arm"),
            ("slanted_arm", "illumination_housing"),
            ("illumination_housing", "illumination_adapter"),
            ("slanted_arm", "condenser_crosshead"),
            ("condenser_crosshead", "condenser_reach"),
            ("condenser_reach", "condenser"),
            ("rear_support", "viewing_head_shell"),
            ("viewing_head_shell", "photo_tube"),
            ("photo_tube", "camera_collar"),
            ("camera_collar", "camera_body"),
            ("lower_body_shell", "port_flange"),
            ("port_flange", "port_barrel"),
            ("port_barrel", "port_adapter"),
            ("port_adapter", "side_camera"),
        )
        for first, second in attachments:
            with self.subTest(first=first, second=second):
                ids = [
                    self.model.geom("tirf_workspace__reference_" + key).id
                    for key in (first, second)
                ]
                self.assertTrue(np.all(self.model.geom_contype[ids] > 0))
                self.assertTrue(np.all(self.model.geom_conaffinity[ids] > 0))
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, *ids, 1.0, None
                )
                self.assertLessEqual(distance, 0.0001)

    def test_new_primary_geometry_stays_inside_the_authored_station_envelope(self):
        origin = self.data.xpos[self.model.body("tirf_workspace").id]
        checked = 0
        for geom in range(self.model.ngeom):
            if not self.model.geom(geom).name.startswith("tirf_workspace__reference_"):
                continue
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            center = (
                self.data.geom_xpos[geom]
                + rotation @ self.model.geom_aabb[geom, :3]
                - origin
            )
            extent = np.abs(rotation) @ self.model.geom_aabb[geom, 3:]
            self.assertTrue(np.all(center - extent >= [-1.1, -0.575, -1e-7]))
            self.assertTrue(np.all(center + extent <= [1.1, 0.575, 0.87]))
            checked += 1
        self.assertGreater(checked, 15)

    def test_existing_sample_camera_keeps_an_opaque_line_of_sight(self):
        report = camera_target_screening(self.model, self.data, self.manifest)
        item = next(row for row in report if row["equipment"] == "tirf_workspace")
        camera = next(row for row in item["cameras"] if row["camera"] == "sample")
        self.assertTrue(camera["in_frame"])
        self.assertTrue(camera["opaque_line_of_sight"])
        self.assertGreater(camera["estimated_target_diameter_px"], 10)


if __name__ == "__main__":
    unittest.main()
