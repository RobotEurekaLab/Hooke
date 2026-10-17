"""Stage windows must transmit a geometric ray while retaining slide supports."""

import itertools
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from real_labs.builder import build_scene
from real_labs.usability import audit_scene


class OpticalStageTests(unittest.TestCase):
    def test_clear_window_and_physical_end_support_through_xy_travel(self):
        for scene, axes, prefix in (
            ("kyoto_yokokawa_nanometrics", ("slide_x", "slide_y"), "slide"),
            ("tsinghua_rush3d", ("sample_x", "sample_y"), "sample"),
        ):
            with self.subTest(scene=scene):
                root, manifest = build_scene(scene)
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                data = mujoco.MjData(model)
                item = manifest["equipment"][0]
                selected = [
                    geom
                    for geom in range(model.ngeom)
                    if model.geom_contype[geom]
                    and model.geom(geom).name.startswith(
                        tuple(
                            item["id"] + "__" + prefix + suffix
                            for suffix in (
                                "_x_plate_",
                                "_y_plate_",
                                "_fixed_plate_",
                                "_holder_insert_",
                            )
                        )
                    )
                ]
                self.assertTrue(selected)
                self.assertTrue(np.all(model.geom_contype[selected] > 0))
                # Isolate these actual assembled surfaces for the aperture ray;
                # the microscope objective below them is intentionally opaque.
                model.geom_group[:] = 1
                model.geom_group[selected] = 0
                for fractions in itertools.product(np.linspace(0, 1, 5), repeat=2):
                    for axis, fraction in zip(axes, fractions):
                        joint = model.joint(item["joints"][axis]).id
                        low, high = model.jnt_range[joint]
                        data.qpos[model.jnt_qposadr[joint]] = low + fraction * (
                            high - low
                        )
                    mujoco.mj_forward(model, data)
                    target = data.site_xpos[model.site(item["sample_site"]).id]
                    for offset in (0, -0.033, 0.033):
                        hit = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(
                            model,
                            data,
                            target + [offset, 0, 0.005],
                            np.array([0.0, 0.0, -1.0]),
                            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                            1,
                            -1,
                            hit,
                        )
                        if offset == 0:
                            self.assertLess(distance, 0, (scene, fractions, hit))
                        else:
                            self.assertGreaterEqual(distance, 0, (scene, fractions))
                            self.assertLess(distance, 0.010)

    def test_shell_contacts_do_not_brake_or_block_stage_motion(self):
        for scene in ("kyoto_yokokawa_nanometrics", "tsinghua_rush3d"):
            with self.subTest(scene=scene):
                report = audit_scene(scene)
                self.assertEqual(report["home_pose_penetrations"], [])
                self.assertEqual(report["range_contacts"], [])
                self.assertEqual(report["transient_contacts"], [])
                self.assertEqual(report["tracking_failures"], [])


if __name__ == "__main__":
    unittest.main()
