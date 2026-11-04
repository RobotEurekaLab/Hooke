"""Reject false preinstalled carrier declarations without creating geometry."""

from copy import deepcopy
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from real_labs.builder import build_scene
from real_labs.specimens import _register_existing_free_specimen


class ExistingFreeSpecimenTests(unittest.TestCase):
    def setUp(self):
        root, self.manifest = build_scene("uw_isolab")
        self.xml = ET.tostring(root, encoding="unicode")
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.item = next(
            v for v in self.manifest["equipment"] if v["id"] == "isotope_analyzer"
        )

    def register(self, claimed=None):
        return _register_existing_free_specimen(
            self.m, self.d, self.item, set() if claimed is None else claimed
        )

    def test_registration_preserves_model_and_actual_state(self):
        state = self.d.qpos.copy()
        masks = self.m.geom_group.copy()
        process, record = self.register()
        self.assertEqual(record["body"], process["sample_body"])
        self.assertEqual(record["mass_kg"], 0.003)
        np.testing.assert_array_equal(self.d.qpos, state)
        np.testing.assert_array_equal(self.m.geom_group, masks)

    def test_missing_duplicate_mismatched_or_fixed_body_fails(self):
        original = deepcopy(self.item)
        for field, replacement in (
            ("body", "missing"),
            ("freejoint", "uw_cartesian_v3__joint_gantry_x"),
            ("collision_geom", "isotope_analyzer__oem_case_roof"),
            ("sample_site", "uw_cartesian_v3__site_physical_tip"),
            ("mass_kg", 0.005),
            ("dimensions_m", [0.012, 0.012, 0.05]),
            ("support_geom", "floor"),
        ):
            self.item = deepcopy(original)
            self.item["existing_free_specimen"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.register()
        self.item = deepcopy(original)
        claimed = set()
        self.register(claimed)
        with self.assertRaisesRegex(ValueError, "more than once"):
            self.register(claimed)

    def test_disabled_or_displaced_support_cannot_register(self):
        support = self.m.geom(self.item["existing_free_specimen"]["support_geom"]).id
        self.m.geom_contype[support] = 0
        with self.assertRaises(ValueError):
            self.register()
        self.m.geom_contype[support] = 1
        self.m.geom_pos[support, 2] -= 0.01
        mujoco.mj_forward(self.m, self.d)
        with self.assertRaises(ValueError):
            self.register()

    def test_tilted_collider_rejected_but_upright_yaw_is_valid(self):
        collider_name = self.item["existing_free_specimen"]["collision_geom"]
        for quaternion, accepted in (
            ([np.sqrt(0.5), 0, np.sqrt(0.5), 0], False),
            ([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], True),
        ):
            with self.subTest(accepted=accepted):
                # Compile the rotated pose so MuJoCo recomputes its same-frame
                # optimization instead of retaining the original upright one.
                root = ET.fromstring(self.xml)
                geom = root.find(f".//geom[@name='{collider_name}']")
                geom.set("quat", " ".join(map(str, quaternion)))
                self.m = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                self.d = mujoco.MjData(self.m)
                mujoco.mj_forward(self.m, self.d)
                before = self.d.qpos.copy()
                groups = self.m.geom_group.copy()
                if accepted:
                    self.register()
                else:
                    with self.assertRaisesRegex(ValueError, "upright collider"):
                        self.register()
                np.testing.assert_array_equal(self.d.qpos, before)
                np.testing.assert_array_equal(self.m.geom_group, groups)


if __name__ == "__main__":
    unittest.main()
