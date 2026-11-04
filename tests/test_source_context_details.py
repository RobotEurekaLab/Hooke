"""Reference context must preserve motion contracts and supported interfaces."""

import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.source_context_details import _BUILDERS, add_details


def compile_scene(identifier):
    root, manifest = build_scene(identifier)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest


class SourceContextTests(unittest.TestCase):
    def test_unselected_scene_is_an_exact_noop(self):
        root = ET.fromstring('<mujoco><worldbody><body name="retained"/></worldbody></mujoco>')
        before = ET.tostring(root)
        add_details(root, {"id": "nasa_planetary"})
        self.assertEqual(ET.tostring(root), before)

    def test_static_context_preserves_all_joint_and_drive_parameters(self):
        for identifier in _BUILDERS:
            with self.subTest(scene=identifier):
                current, manifest = compile_scene(identifier)
                with patch("real_labs.source_context_details.add_details"):
                    original, prior_manifest = compile_scene(identifier)
                self.assertEqual(manifest["equipment"], prior_manifest["equipment"])
                for field in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "qpos0",
                              "dof_damping", "dof_armature", "actuator_ctrlrange",
                              "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
                    np.testing.assert_array_equal(getattr(current, field), getattr(original, field))
                for joint in range(original.njnt):
                    old_body = original.jnt_bodyid[joint]
                    new_body = current.body(original.body(old_body).name).id
                    np.testing.assert_array_equal(current.body_inertia[new_body], original.body_inertia[old_body])
                    self.assertEqual(current.body_mass[new_body], original.body_mass[old_body])
                data = mujoco.MjData(current)
                mujoco.mj_forward(current, data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertFalse(any(contact.dist < -1e-5 for contact in data.contact))

    def test_ntu_carriers_have_floor_posts_and_hangers_share_roof_visibility(self):
        root, _ = build_scene("ntu_smtc_membranes")
        columns = root.findall(".//geom")
        columns = [geom for geom in columns if geom.get("name", "").startswith("smtc_photo_carrier_column")]
        self.assertEqual(len(columns), 6)
        for geom in columns:
            z = float(geom.get("pos").split()[2])
            half = float(geom.get("size").split()[2])
            self.assertAlmostEqual(z - half, .05)
            self.assertAlmostEqual(z + half, 2.78)
            self.assertEqual(geom.get("contype"), "1")
        hangers = [geom for geom in root.findall(".//geom") if "ntu_service_hanger" in geom.get("name", "")]
        self.assertEqual(len(hangers), 9)
        self.assertTrue(all(geom.get("name").startswith("arch_roof_") for geom in hangers))

    def test_epfl_secondary_hotplates_have_static_support_and_clear_primary_baths(self):
        model, _ = compile_scene("epfl_lspn_chemistry")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for owner, side in (("left_hood", "left"), ("right_hood", "right")):
            matches = [i for i in range(model.ngeom) if f"{side}_secondary_hotplate" in model.geom(i).name]
            self.assertEqual(len(matches), 1)
            hotplate = matches[0]
            self.assertEqual(model.body_dofnum[model.geom_bodyid[hotplate]], 0)
            plates = [i for i in range(model.ngeom) if f"{side}_secondary_jack_plate" in model.geom(i).name]
            top_plate = max(plates, key=lambda i: data.geom_xpos[i, 2])
            self.assertAlmostEqual(data.geom_xpos[hotplate, 2] - model.geom_size[hotplate, 2],
                                   data.geom_xpos[top_plate, 2] + model.geom_size[top_plate, 2])
            self.assertTrue(model.geom_contype[hotplate])
            self.assertTrue(model.geom_contype[top_plate])
            junction = next(i for i in range(model.ngeom)
                            if f"{side}_secondary_junction" in model.geom(i).name)
            wall = model.geom(owner + "__hood_context_back").id
            self.assertLessEqual(mujoco.mj_geomDistance(model, data, junction, wall, 1., None), 1e-8)
            self.assertTrue(model.geom_contype[junction])

    def test_new_context_has_real_support_and_preserves_the_ntu_aisle(self):
        for scene, pairs in (
            ("cmu_edl_electrochemical_testing", (
                ("edl_test_bench_top", "cmu_photo_blue_instrument_0"),
                ("cmu_photo_blue_instrument_0", "cmu_photo_blue_instrument_1"),
                ("cmu_overhead_utility_shelf", "cmu_photo_under_shelf_power_strip"),
            )),
            ("jhu_paradim_bulk_crystal_growth", (
                ("central_prep_island_top", "jhu_photo_island_computer_foot"),
                ("central_prep_island_top", "jhu_photo_balance_base"),
                ("jhu_photo_balance_base", "jhu_photo_balance_pan"),
            )),
        ):
            model, _ = compile_scene(scene)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for first, second in pairs:
                with self.subTest(scene=scene, pair=(first, second)):
                    a, b = model.geom(first), model.geom(second)
                    self.assertEqual(model.body_dofnum[b.bodyid], 0)
                    self.assertAlmostEqual(
                        mujoco.mj_geomDistance(model, data, a.id, b.id, .1, None),
                        0, places=7,
                    )
        model, _ = compile_scene("ntu_smtc_membranes")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name.startswith(("smtc_photo_auxiliary_", "smtc_photo_pressure_housing_")):
                rotation = data.geom_xmat[i].reshape(3, 3)
                center = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
                half = np.abs(rotation) @ model.geom_aabb[i, 3:]
                self.assertLess(center[0] + half[0], -1.5)


if __name__ == "__main__":
    unittest.main()
