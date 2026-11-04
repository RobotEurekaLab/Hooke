"""Physical assembly checks for the historical Imperial equipment exteriors."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class ImperialChambersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene("imperial_thinfilm_bessemer")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def geom(self, key):
        return self.model.geom("thinfilm_cluster__imperial_" + key).id

    def distance(self, first, second):
        return mujoco.mj_geomDistance(
            self.model, self.data, self.geom(first), self.geom(second), 10, None
        )

    def test_vessels_have_continuous_contact_support_to_floor(self):
        for key in ("left", "right"):
            contacts = [
                (key + "_lower_frame", key + "_cabinet"),
                (key + "_cabinet", key + "_table"),
            ]
            for i in range(2):
                contacts.extend(
                    (
                        (f"{key}_saddle_{i}", key + "_table"),
                        (f"{key}_saddle_{i}", key + "_vessel"),
                    )
                )
                for j in range(2):
                    foot = self.geom(f"{key}_foot_{i}_{j}")
                    self.assertAlmostEqual(
                        self.data.geom_xpos[foot, 2] - self.model.geom_size[foot, 1],
                        0,
                        places=7,
                    )
                    contacts.extend(
                        (
                            (f"{key}_foot_{i}_{j}", f"{key}_leg_{i}_{j}"),
                            (f"{key}_leg_{i}_{j}", key + "_table"),
                        )
                    )
            for first, second in contacts:
                with self.subTest(first=first, second=second):
                    self.assertTrue(
                        np.all(
                            self.model.geom_contype[
                                [self.geom(first), self.geom(second)]
                            ]
                            > 0
                        )
                    )
                    self.assertLessEqual(self.distance(first, second), 1e-6)

    def test_inclined_assemblies_clear_their_support_frames(self):
        # Fixed parts share a body and do not generate MuJoCo contacts. Explicit
        # surface queries include flange rims, bolts and tie rods, not just axes.
        prefix = "thinfilm_cluster__imperial_"
        for key in ("left", "right"):
            branches = [
                self.model.geom(i).name[len(prefix) :]
                for i in range(self.model.ngeom)
                if self.model.geom(i).name.startswith(prefix + key + "_branch_")
            ]
            supports = [
                key + "_" + name
                for name in (
                    "table",
                    "cabinet",
                    "leg_0_0",
                    "leg_0_1",
                    "leg_1_0",
                    "leg_1_1",
                    "saddle_0",
                    "saddle_1",
                    "lower_frame",
                )
            ]
            for branch in branches:
                for support in supports:
                    self.assertGreater(
                        self.distance(branch, support), 0.001, (branch, support)
                    )

    def test_service_leads_have_connected_ends_and_do_not_cross(self):
        for key in ("left", "right"):
            for i in range(2):
                self.assertLessEqual(
                    self.distance(f"{key}_lead_{i}_route_0", f"{key}_top_{i}_terminal"),
                    1e-6,
                )
                self.assertLessEqual(
                    self.distance(f"{key}_lead_{i}_route_31", f"{key}_socket_{i}"), 1e-6
                )
                self.assertLessEqual(
                    self.distance(f"{key}_socket_{i}", f"{key}_cabinet"), 1e-6
                )
            gap = min(
                self.distance(f"{key}_lead_0_route_{i}", f"{key}_lead_1_route_{j}")
                for i in range(32)
                for j in range(32)
            )
            self.assertGreater(gap, 0.001)

    def test_original_dry_controls_and_target_are_preserved(self):
        item = self.manifest["equipment"][0]
        self.assertEqual(set(item["joints"]), {"access_door", "coupon_x"})
        for key, pos, axis, limits in (
            ("access_door", (0.583, -0.993, 1.085), (0, 0, -1), (0, 1.2)),
            ("coupon_x", (0.94, -0.73, 1.015), (1, 0, 0), (-0.12, 0.12)),
        ):
            joint = self.model.joint(item["joints"][key]).id
            np.testing.assert_allclose(
                self.model.body_pos[self.model.jnt_bodyid[joint]], pos
            )
            np.testing.assert_allclose(self.model.jnt_axis[joint], axis)
            np.testing.assert_allclose(self.model.jnt_range[joint], limits)
        sample = self.model.site(item["sample_site"]).id
        np.testing.assert_allclose(self.model.site_pos[sample], (0, 0, 0.042))
        np.testing.assert_allclose(self.model.site_size[sample, 0], 0.004)


if __name__ == "__main__":
    unittest.main()
