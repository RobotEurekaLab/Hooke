"""Support continuity and physical access in the authored dry inspection pan."""

from collections import Counter
import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class NottinghamPanGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("nottingham_food_process_hall")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.session = InstrumentSession(cls.model, cls.manifest)

    def setUp(self):
        self.session.reset()

    def geoms(self, fragment):
        return [
            i for i in range(self.model.ngeom) if fragment in self.model.geom(i).name
        ]

    def distance(self, first, second):
        return mujoco.mj_geomDistance(
            self.model, self.session.data, first, second, 2, None
        )

    def test_all_five_trolleys_connect_floor_casters_legs_shelf_and_top(self):
        model = self.model
        legs = self.geoms("nottingham_trolley_leg_")
        tops = self.geoms("nottingham_mobile_table_")
        shelves = self.geoms("nottingham_lower_shelf_")
        castors = self.geoms("nottingham_castor_")
        self.assertEqual(len(legs), 20)
        self.assertEqual(len(tops), 5)
        self.assertEqual(len(shelves), 5)
        self.assertEqual(len(castors), 20)
        for leg in legs:
            top = min(
                tops, key=lambda i: abs(model.geom_pos[i, 0] - model.geom_pos[leg, 0])
            )
            shelf = min(
                shelves,
                key=lambda i: abs(model.geom_pos[i, 0] - model.geom_pos[leg, 0]),
            )
            upper = model.geom_pos[leg, 2] + model.geom_size[leg, 2]
            underside = model.geom_pos[top, 2] - model.geom_size[top, 2]
            self.assertAlmostEqual(upper, underside, places=10)
            self.assertAlmostEqual(
                model.geom_pos[leg, 2] - model.geom_size[leg, 2], 0.070, places=10
            )
            self.assertLessEqual(self.distance(leg, top), 1e-8)
            self.assertLessEqual(self.distance(leg, shelf), 1e-8)
        floor = model.geom("arch_floor").id
        for caster in castors:
            self.assertTrue(model.geom_contype[caster])
            self.assertTrue(model.geom_conaffinity[caster])
            self.assertAlmostEqual(self.distance(caster, floor), 0, places=8)
            self.assertLessEqual(
                min(self.distance(caster, leg) for leg in legs), 0
            )

    def test_cover_hinge_supports_remain_connected_and_coaxial_through_both_axes(self):
        model, data = self.model, self.session.data
        leaves = self.geoms("pan_cover_fixed_leaf")
        fixed = self.geoms("pan_cover_fixed_knuckle_contact")
        moving = self.geoms("pan_cover_moving_knuckle_contact")
        leaf = self.geoms("pan_cover_moving_leaf")[0]
        axle = self.geoms("pan_cover_axle_")[0]
        pan = model.body("dry_pan__body_pan_tilt").id
        cover = model.body("dry_pan__body_inspection_cover").id
        cover_cylinders = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == cover
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and abs(model.geom_size[i, 0] - 0.279) < 1e-9
        ]
        self.assertEqual(len(cover_cylinders), 1)
        lid = cover_cylinders[0]
        bowl_contacts = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == pan
            and model.geom_group[i] == 3
            and "knuckle" not in model.geom(i).name
        ]
        for tilt in np.linspace(0, 0.40, 7):
            for opening in np.linspace(0, 1.20, 13):
                data.qpos[
                    model.jnt_qposadr[model.joint("dry_pan__joint_pan_tilt").id]
                ] = tilt
                data.qpos[
                    model.jnt_qposadr[model.joint("dry_pan__joint_inspection_cover").id]
                ] = opening
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.xpos[cover], data.geom_xpos[axle], atol=1e-9
                )
                # Rotation about local X cannot detach the coaxial moving barrel.
                np.testing.assert_allclose(
                    data.xmat[pan].reshape(3, 3)[:, 0],
                    data.xmat[cover].reshape(3, 3)[:, 0],
                    atol=1e-9,
                )
                self.assertLessEqual(self.distance(leaf, lid), 1e-8)
                self.assertLessEqual(min(self.distance(leaf, i) for i in moving), 1e-8)
                for fixed_leaf in leaves:
                    self.assertLessEqual(
                        min(self.distance(fixed_leaf, i) for i in fixed), 1e-8
                    )
                    self.assertLessEqual(
                        min(self.distance(fixed_leaf, i) for i in bowl_contacts), 1e-8
                    )
                    # Direct distance checks cover pairs normally omitted by
                    # parent-child contact filtering; a passing contact scan
                    # alone cannot establish physical hinge clearance.
                    self.assertGreater(self.distance(fixed_leaf, lid), 0.0029)
                    self.assertGreater(
                        min(self.distance(fixed_leaf, i) for i in moving), 0.0019
                    )
                self.assertGreater(min(self.distance(lid, i) for i in fixed), 0.0003)
                self.assertGreater(
                    min(self.distance(i, j) for i in fixed for j in moving), 0.0039
                )
                # A true 6.8 mm bore surrounds the 6 mm axle; it is not a filled
                # cylinder relying on disabled contact flags to permit rotation.
                clearance = min(self.distance(axle, i) for i in (*fixed, *moving))
                self.assertGreater(clearance, 0.0007)
                self.assertLess(clearance, 0.001)
                self.assertFalse(any(c.dist < -1e-5 for c in data.contact))

    def test_tray_holes_pass_a_free_probe_and_wires_retain_collision(self):
        root = copy.deepcopy(self.root)
        body = ET.SubElement(root.find("worldbody"), "body", name="tray_test_probe")
        ET.SubElement(body, "freejoint", name="tray_test_probe_joint")
        ET.SubElement(
            body,
            "geom",
            name="tray_test_probe_geom",
            type="sphere",
            size=".008",
            mass=".01",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        address = model.jnt_qposadr[model.joint("tray_test_probe_joint").id]
        probe = model.geom("tray_test_probe_geom").id
        for x, blocked in ((-0.240, False), (-0.255, True)):
            data.qpos[address : address + 3] = (1.12 + x, -1.34 - 0.495, 0.547)
            mujoco.mj_forward(model, data)
            collisions = [
                c
                for c in data.contact
                if probe in (c.geom1, c.geom2) and c.dist < -1e-6
            ]
            self.assertEqual(bool(collisions), blocked)
            if blocked:
                self.assertTrue(
                    any(
                        "pan_tray_wire"
                        in model.geom(c.geom2 if c.geom1 == probe else c.geom1).name
                        for c in collisions
                    )
                )

    def test_trunnions_meet_outer_jacket_without_entering_the_inner_cavity(self):
        pan = self.root.find('.//body[@name="dry_pan__body_pan_tilt"]')
        mesh = next(
            m
            for m in self.root.findall("./asset/mesh")
            if "smooth_jacketed_bowl" in m.get("name", "")
        )
        vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
        # The closed lathe has two radii at each profile plane. The smaller is
        # the exposed inner surface. Its global maximum also bounds every
        # interpolated profile segment and every point within the cavity.
        inner_radii = [
            np.linalg.norm(vertices[np.isclose(vertices[:, 2], z), :2], axis=1).min()
            for z in np.unique(vertices[:, 2])
        ]
        cavity_bound = max(inner_radii)
        shafts = [
            g for g in pan.findall("geom") if "pan_trunnion_shaft" in g.get("name", "")
        ]
        self.assertEqual(len(shafts), 2)
        model, data = self.model, self.session.data
        base = model.body("dry_pan").id
        housings = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == base
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and abs(model.geom_size[i, 0] - 0.085) < 1e-9
        ]
        shell = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body("dry_pan__body_pan_tilt").id
            and model.geom_group[i] == 3
            and "knuckle" not in model.geom(i).name
        ]
        for shaft in shafts:
            ends = np.fromstring(shaft.get("fromto"), sep=" ").reshape(2, 3)
            radius = float(shaft.get("size"))
            # The capsule's complete surface is outside this radial bound,
            # including its rounded ends. Collision filtering is irrelevant.
            self.assertGreater(np.abs(ends[:, 0]).min() - radius - cavity_bound, 0.0149)
            index = model.geom(shaft.get("name")).id
            for tilt in np.linspace(0, 0.40, 25):
                data.qpos[
                    model.jnt_qposadr[model.joint("dry_pan__joint_pan_tilt").id]
                ] = tilt
                mujoco.mj_forward(model, data)
                self.assertLessEqual(min(self.distance(index, i) for i in shell), 1e-8)
                self.assertLessEqual(
                    min(self.distance(index, i) for i in housings), 1e-8
                )

    def test_side_webs_tray_rails_and_rear_apparatus_have_continuous_support(self):
        model = self.model
        webs = self.geoms("pan_tapered_support_")
        rails = self.geoms("pan_tray_support_")
        attachments = self.geoms("pan_tray_attachment_")
        self.assertEqual(len(webs), 2)
        for rail in rails:
            attachment = min(attachments, key=lambda i: self.distance(rail, i))
            self.assertLessEqual(self.distance(rail, attachment), 1e-8)
            self.assertLessEqual(min(self.distance(attachment, i) for i in webs), 1e-8)
        for web in webs:
            feet = [
                i
                for i in range(model.ngeom)
                if model.geom(i).name.startswith("dry_pan__")
                and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                and abs(model.geom_pos[i, 2] - 0.118) < 1e-9
            ]
            self.assertLessEqual(min(self.distance(web, i) for i in feet), 1e-8)
        for fragment in ("rear_grid_cabinet_", "rear_tall_unit_"):
            cabinet = self.geoms(fragment)[0]
            feet = self.geoms(
                "rear_grid_foot_" if "grid" in fragment else "rear_tall_foot_"
            )
            for foot in feet:
                self.assertLessEqual(self.distance(cabinet, foot), 1e-8)
                self.assertAlmostEqual(
                    model.geom_pos[foot, 2] - model.geom_size[foot, 2], 0, places=10
                )
        for post in self.geoms("rear_rack_post_"):
            self.assertAlmostEqual(
                model.geom_pos[post, 2] - model.geom_size[post, 2], 0, places=10
            )

    def test_visible_jacket_and_hinge_meshes_are_closed_consistently_wound_surfaces(
        self,
    ):
        keys = (
            "smooth_jacketed_bowl",
            "pan_cover_fixed_knuckle_mesh",
            "pan_cover_moving_knuckle_mesh",
        )
        meshes = [
            mesh
            for mesh in self.root.findall("./asset/mesh")
            if any(key in mesh.get("name", "") for key in keys)
        ]
        self.assertEqual(len(meshes), 4)
        for mesh in meshes:
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            self.assertEqual(len(vertices), len(np.unique(vertices.round(9), axis=0)))
            edges, winding = Counter(), Counter()
            for face in faces:
                for start, end in zip(face, np.roll(face, -1)):
                    edge = tuple(sorted((int(start), int(end))))
                    edges[edge] += 1
                    winding[edge] += 1 if start < end else -1
            self.assertTrue(all(value == 2 for value in edges.values()))
            self.assertTrue(all(value == 0 for value in winding.values()))

    def test_original_axes_target_limits_and_scientific_scope_remain(self):
        model = self.model
        for semantic, origin, axis, limits in (
            ("pan_tilt", (0, 0, 0.93), (1, 0, 0), (0, 0.40)),
            ("inspection_cover", (0, 0.294, 0.225), (-1, 0, 0), (0, 1.20)),
        ):
            joint = model.joint("dry_pan__joint_" + semantic)
            body = model.body("dry_pan__body_" + semantic)
            np.testing.assert_allclose(body.pos, origin, atol=1e-12)
            np.testing.assert_allclose(joint.axis, axis, atol=1e-12)
            np.testing.assert_allclose(joint.range, limits, atol=1e-12)
        np.testing.assert_allclose(
            model.site("dry_pan__site_inert_pan_coupon").pos, (0, 0, -0.070), atol=1e-12
        )
        self.assertEqual(model.njnt, 2)
        self.assertEqual(model.neq, 0)
        self.assertAlmostEqual(model.qpos0[1], 0.82)
        limitations = " ".join(self.manifest["equipment"][0]["limitations"])
        self.assertIn("25 L working capacity", limitations)
        self.assertIn("authored dry inspection interfaces", limitations)
        self.assertIn("No steam", limitations)
        self.assertIn("inspection", self.manifest["camera_names"])


if __name__ == "__main__":
    unittest.main()
