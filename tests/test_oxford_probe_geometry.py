"""Contact, guide and true-scale viewing checks for the noncontact probe task."""

from collections import Counter
import copy
import itertools
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.layout_audit import audit_layout
from real_labs.runtime import InstrumentSession


class OxfordProbeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("oxford_bonilla_semiconductor")
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
            self.model, self.session.data, first, second, 1, None
        )

    def test_original_frames_limits_tip_scale_and_lower_stop_gap_remain(self):
        model, data = self.model, self.session.data
        for semantic, origin, axis, limits in (
            ("sample_x", (0, 0, 0.067), (1, 0, 0), (-0.020, 0.020)),
            ("left_approach", (-0.155, 0.082, 0.13), (0, 0, 1), (-0.005, 0.022)),
            ("right_approach", (0.155, 0.082, 0.13), (0, 0, 1), (-0.005, 0.022)),
        ):
            np.testing.assert_allclose(
                model.body("probe_contact_fixture__body_" + semantic).pos,
                origin,
                atol=1e-12,
            )
            joint = model.joint("probe_contact_fixture__joint_" + semantic)
            np.testing.assert_allclose(joint.axis, axis, atol=1e-12)
            np.testing.assert_allclose(joint.range, limits, atol=1e-12)
        np.testing.assert_allclose(
            model.site("probe_contact_fixture__site_inert_die").pos,
            (0, 0, 0.0241),
            atol=1e-12,
        )
        self.assertEqual((model.njnt, model.neq), (3, 0))
        for side, semantic in ((-1, "left_approach"), (1, "right_approach")):
            body = self.root.find(
                f'.//body[@name="probe_contact_fixture__body_{semantic}"]'
            )
            tip = next(
                g for g in body.findall("geom") if "probe_fine_tip" in g.get("name", "")
            )
            np.testing.assert_allclose(
                np.fromstring(tip.get("fromto"), sep=" "),
                (-side * 0.142, -0.079, -0.026, -side * 0.151, -0.082, -0.030),
                atol=1e-12,
            )
            self.assertEqual(float(tip.get("size")), 0.00012)
            index = model.geom(tip.get("name")).id
            for q, expected in ((-0.005, 0.00378), (0, 0.00878), (0.022, 0.03078)):
                data.qpos[
                    model.jnt_qposadr[
                        model.joint("probe_contact_fixture__joint_" + semantic).id
                    ]
                ] = q
                mujoco.mj_forward(model, data)
                self.assertAlmostEqual(
                    min(
                        self.distance(index, p) for p in self.geoms("probe_contact_pad")
                    ),
                    expected,
                    places=10,
                )

    def test_moving_surfaces_pads_and_guides_have_effective_contact_masks(self):
        model = self.model
        for fragment in (
            "probe_fine_tip",
            "probe_fine_holder",
            "probe_broad_foreground_shaft",
            "probe_rear_brass_shaft",
            "probe_shaft_mount",
            "probe_clamp_ear",
            "probe_clamp_screw",
            "probe_guide_contact",
            "probe_guide_post",
            "probe_inert_die",
            "probe_contact_pad",
        ):
            ids = self.geoms(fragment)
            self.assertTrue(ids, fragment)
            for i in ids:
                self.assertEqual(model.geom_contype[i], 1)
                self.assertEqual(model.geom_conaffinity[i], 1)
        self.assertEqual(len(self.geoms("probe_guide_contact")), 80)
        self.assertGreater(
            model.geom_size[self.geoms("probe_broad_foreground_shaft")[0], 0],
            model.geom_size[self.geoms("probe_rear_brass_shaft")[0], 0],
        )

    def test_open_guides_clear_fixed_parts_and_support_chains_are_connected(self):
        model, data = self.model, self.session.data
        base = model.body("probe_contact_fixture").id
        fixed = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == base and model.geom_contype[i]
        ]
        for semantic in ("left_approach", "right_approach"):
            body = model.body("probe_contact_fixture__body_" + semantic).id
            moving = {
                i
                for i in range(model.ngeom)
                if model.geom_bodyid[i] == body and model.geom_contype[i]
            }
            guides = {i for i in moving if "probe_guide_contact" in model.geom(i).name}
            for q in np.linspace(-0.005, 0.022, 31):
                data.qpos[
                    model.jnt_qposadr[
                        model.joint("probe_contact_fixture__joint_" + semantic).id
                    ]
                ] = q
                mujoco.mj_forward(model, data)
                # Adjacent-body contacts are normally filtered. These direct
                # distances independently rule out a colliding solid guide.
                minimum = min(self.distance(i, j) for i in moving for j in fixed)
                self.assertGreater(minimum, 0.00059)
                self.assertLess(minimum, 0.00061)
            # In the rigid holder, a connected chain of actual contact surfaces
            # joins the open collar to the fine tip; no floating visual rods.
            connected = set(guides)
            while True:
                added = {
                    i
                    for i in moving - connected
                    if any(self.distance(i, j) <= 1e-8 for j in connected)
                }
                if not added:
                    break
                connected.update(added)
            self.assertEqual(connected, moving)

    def test_combined_travel_dynamic_reversals_reset_and_layout(self):
        model, data = self.model, self.session.data
        for state in itertools.product(
            np.linspace(-0.020, 0.020, 5),
            np.linspace(-0.005, 0.022, 9),
            np.linspace(-0.005, 0.022, 9),
        ):
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            self.assertFalse(any(c.dist < -1e-7 for c in data.contact), state)
        self.session.reset()
        initial = data.site_xpos.copy()
        for (item, key), control in self.session.controls.items():
            for fraction in (0.15, 0.85, 0.25):
                result = self.session.command(
                    item,
                    key,
                    control.lower + fraction * (control.upper - control.lower),
                    duration=2,
                )
                self.assertLessEqual(
                    abs(result["error"]), 0.025 * (control.upper - control.lower)
                )
                self.assertFalse(any(c.dist < -1e-7 for c in data.contact))
        self.session.reset()
        np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12)
        np.testing.assert_allclose(data.site_xpos, initial, atol=1e-12)
        self.assertTrue(audit_layout(self.manifest)["passed"])

    def test_free_test_particle_contacts_real_tip_and_pad_but_not_intentional_gap(self):
        root = copy.deepcopy(self.root)
        body = ET.SubElement(
            root.find("worldbody"), "body", name="probe_contact_test_particle"
        )
        ET.SubElement(body, "freejoint", name="probe_contact_test_joint")
        ET.SubElement(
            body,
            "geom",
            name="probe_contact_test_geom",
            type="sphere",
            size=".0003",
            mass=".0001",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        address = model.jnt_qposadr[model.joint("probe_contact_test_joint").id]
        particle = model.geom("probe_contact_test_geom").id
        for point, fragment in (
            ((-0.446, 0.5, 1.0), "probe_fine_tip"),
            ((-0.447, 0.5, 0.99125), "probe_contact_pad"),
            ((-0.447, 0.5, 0.993), "none"),
        ):
            data.qpos[address : address + 3] = point
            mujoco.mj_forward(model, data)
            touched = [
                model.geom(c.geom2 if c.geom1 == particle else c.geom1).name
                for c in data.contact
                if particle in (c.geom1, c.geom2) and c.dist < -1e-7
            ]
            if fragment == "none":
                self.assertEqual(touched, [])
            else:
                self.assertTrue(any(fragment in name for name in touched), touched)

    def test_tip_camera_rays_reach_die_and_tips_at_true_scale(self):
        model, data = self.model, self.session.data
        camera = model.camera("tip_inspection").id
        site = model.site("probe_contact_fixture__site_inert_die").id
        geomgroup = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        for sample, approach in itertools.product(
            (-0.020, 0, 0.020), (-0.005, 0, 0.022)
        ):
            data.qpos[:] = (sample, approach, approach)
            mujoco.mj_forward(model, data)
            origin = data.cam_xpos[camera]
            target = data.site_xpos[site]
            delta = target - origin
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                model,
                data,
                origin,
                delta / np.linalg.norm(delta),
                geomgroup,
                1,
                -1,
                hit,
            )
            self.assertGreater(distance, 0)
            self.assertLess(abs(distance - np.linalg.norm(delta)), 0.0005)
            self.assertIn("probe_inert_die", model.geom(hit[0]).name)
            rotation = data.cam_xmat[camera].reshape(3, 3)
            corners = np.array([target + (-0.006, 0, 0), target + (0.006, 0, 0)])
            local = (corners - origin) @ rotation
            pixels = (
                350
                * local[:, :2]
                / (-local[:, 2, None] * np.tan(np.deg2rad(model.cam_fovy[camera] / 2)))
            )
            self.assertGreater(np.linalg.norm(pixels[1] - pixels[0]), 60)
            for side in (-1, 1):
                endpoint = np.array((-0.45 + side * 0.004, 0.5, 1.0 + approach))
                ray = endpoint - origin
                distance = mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    ray / np.linalg.norm(ray),
                    geomgroup,
                    1,
                    -1,
                    hit,
                )
                self.assertLess(abs(distance - np.linalg.norm(ray)), 0.0005)
                self.assertIn("probe_fine_tip", model.geom(hit[0]).name)

    def test_open_collar_visual_meshes_are_closed_consistently_wound(self):
        meshes = [
            m
            for m in self.root.findall("./asset/mesh")
            if "probe_open_guide_mesh" in m.get("name", "")
        ]
        self.assertEqual(len(meshes), 2)
        for mesh in meshes:
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            self.assertEqual(len(vertices), len(np.unique(vertices.round(9), axis=0)))
            edges, winding = Counter(), Counter()
            for face in faces:
                for a, c in zip(face, np.roll(face, -1)):
                    key = tuple(sorted((int(a), int(c))))
                    edges[key] += 1
                    winding[key] += 1 if a < c else -1
            self.assertTrue(all(n == 2 for n in edges.values()))
            self.assertTrue(all(n == 0 for n in winding.values()))

    def test_holder_details_remain_inside_the_original_moving_envelopes(self):
        model = self.model
        for semantic, lower_x, upper_x in (
            ("left_approach", -0.019, 0.15112),
            ("right_approach", -0.15112, 0.019),
        ):
            body = model.body("probe_contact_fixture__body_" + semantic).id
            minimum, maximum = [], []
            for index in range(model.ngeom):
                if model.geom_bodyid[index] != body:
                    continue
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[index])
                rotation = rotation.reshape(3, 3)
                if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[index]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    points = (
                        model.mesh_vert[start : start + count] @ rotation.T
                        + model.geom_pos[index]
                    )
                    minimum.append(points.min(axis=0))
                    maximum.append(points.max(axis=0))
                elif model.geom_type[index] in (
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                ):
                    # Rotating the primitive's box would overestimate its
                    # footprint. Use the actual swept sphere/disc bounds.
                    axis = abs(rotation[:, 2])
                    radial = (
                        np.ones(3)
                        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
                        else np.sqrt(np.maximum(0, 1 - axis * axis))
                    )
                    half = (
                        model.geom_size[index, 0] * radial
                        + model.geom_size[index, 1] * axis
                    )
                    minimum.append(model.geom_pos[index] - half)
                    maximum.append(model.geom_pos[index] + half)
                else:
                    bounds = model.geom_aabb[index]
                    center = model.geom_pos[index] + rotation @ bounds[:3]
                    half = abs(rotation) @ bounds[3:]
                    minimum.append(center - half)
                    maximum.append(center + half)
            self.assertTrue(
                np.all(
                    np.min(minimum, axis=0)
                    >= np.array((lower_x, -0.08212, -0.03012)) - 1e-8
                )
            )
            self.assertTrue(
                np.all(
                    np.max(maximum, axis=0) <= np.array((upper_x, 0.023, 0.026)) + 1e-8
                )
            )


if __name__ == "__main__":
    unittest.main()
