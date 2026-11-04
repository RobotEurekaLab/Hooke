"""Supported counterweight and genuinely open curved UCSD inspection carrier."""

import copy
from collections import Counter
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.rigid_access import RigidAccess


class UcsdCentrifugeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("ucsd_mccartney_geotechnical_centrifuge")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.session = InstrumentSession(cls.model, cls.manifest)

    def setUp(self):
        self.session.reset()

    def _tagged(self, fragment):
        return {
            index
            for index in range(self.model.ngeom)
            if fragment in self.model.geom(index).name
        }

    def _connected_boxes(self, body, seeds):
        """Actual same-frame solid-box overlap graph, not body attachment alone."""
        model = self.model
        boxes = {
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_contype[i]
        }
        connected = set(seeds)
        pending = list(seeds)
        while pending:
            first = pending.pop()
            for other in boxes - connected:
                overlap = (
                    model.geom_size[first]
                    + model.geom_size[other]
                    - np.abs(model.geom_pos[first] - model.geom_pos[other])
                )
                if np.all(overlap >= -1e-9):
                    connected.add(other)
                    pending.append(other)
        return connected

    def test_each_counterweight_component_has_a_solid_path_to_the_arm(self):
        arm = self.model.body("centrifuge__body_arm_index").id
        components = self._tagged("__ucsd_counterweight_cheek_") | self._tagged(
            "__ucsd_counterweight_cap_"
        )
        self.assertGreaterEqual(len(components), 4)
        connected = self._connected_boxes(arm, self._tagged("__rotary_arm_"))
        self.assertTrue(components <= connected)
        for index in components:
            self.assertEqual(self.model.geom_bodyid[index], arm)
            self.assertTrue(self.model.geom_contype[index])

    def test_hangers_join_the_floor_and_leave_the_original_coupon_supported(self):
        basket = self.model.body("centrifuge__body_basket_tilt").id
        floors = self._tagged("__ucsd_basket_floor_")
        hangers = self._tagged("__ucsd_basket_hanger_")
        self.assertEqual(len(hangers), 2)
        self.assertTrue(hangers <= self._connected_boxes(basket, floors))
        floor = self.model.geom(next(iter(floors)))
        coupon = self.model.geom(next(iter(self._tagged("__container_max_footprint_"))))
        np.testing.assert_allclose(coupon.size[:2] * 2, (0.60, 0.70))
        self.assertAlmostEqual(
            floor.pos[2] + floor.size[2], coupon.pos[2] - coupon.size[2]
        )
        self.assertTrue(np.all(floor.size[:2] >= coupon.size[:2]))

    def test_physical_probe_enters_top_opening_but_collides_with_curved_wall(self):
        root = copy.deepcopy(self.root)
        body = ET.SubElement(root.find("worldbody"), "body", name="carrier_probe")
        ET.SubElement(body, "freejoint", name="carrier_probe_joint")
        ET.SubElement(
            body,
            "geom",
            name="carrier_probe_geom",
            type="sphere",
            size=".020",
            mass=".01",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        probe = model.geom("carrier_probe_geom").id
        address = model.jnt_qposadr[model.joint("carrier_probe_joint").id]
        basket = model.body("centrifuge__body_basket_tilt").id
        for angle in (0, 0.2, 0.4):
            data.qpos[
                model.jnt_qposadr[model.joint("centrifuge__joint_basket_tilt").id]
            ] = angle
            mujoco.mj_forward(model, data)
            rotation, origin = data.xmat[basket].reshape(3, 3), data.xpos[basket].copy()
            for local in ((0.18, -0.28, -0.30), (0.18, 0.28, -0.30)):
                data.qpos[address : address + 3] = origin + rotation @ local
                mujoco.mj_forward(model, data)
                self.assertFalse(
                    any(
                        probe in (contact.geom1, contact.geom2) and contact.dist < -1e-6
                        for contact in data.contact
                    )
                )
            for side in (-1, 1):
                data.qpos[address : address + 3] = origin + rotation @ np.array(
                    [0.18, side * 0.456, -0.455]
                )
                mujoco.mj_forward(model, data)
                self.assertTrue(
                    any(
                        probe in (contact.geom1, contact.geom2)
                        and contact.dist < -1e-4
                        and "ucsd_basket_shell_contact"
                        in model.geom(
                            contact.geom2 if contact.geom1 == probe else contact.geom1
                        ).name
                        for contact in data.contact
                    )
                )

    def test_original_frames_controls_and_authored_target_are_retained(self):
        model = self.model
        for semantic, limits, body_pos, mass, force in (
            ("arm_index", (-0.65, 0.65), (0, 0, 1.40), 12, 2000),
            ("basket_tilt", (0, 0.40), (1.45, 0, -0.05), 2, 500),
        ):
            control = self.session.controls["centrifuge", semantic]
            np.testing.assert_allclose((control.lower, control.upper), limits)
            body = model.body(f"centrifuge__body_{semantic}")
            np.testing.assert_allclose(body.pos, body_pos)
            self.assertEqual(body.mass[0], mass)
            actuator = model.actuator(f"centrifuge__actuator_{semantic}")
            np.testing.assert_allclose(actuator.forcerange, (-force, force))
        np.testing.assert_allclose(
            model.site("centrifuge__site_soil_analogue").pos, (0, 0, -0.373)
        )
        self.assertEqual(model.neq, 0)
        self.assertEqual(model.njnt, 2)
        self.assertEqual(self.manifest["equipment"][0]["sample_attachment"], "clamped")
        limitations = " ".join(self.manifest["equipment"][0]["limitations"])
        self.assertIn("not the real passive high-g dynamics", limitations)
        self.assertIn("estimated assembly details", limitations)

    def test_basket_clears_enabled_bearings_across_combined_travel(self):
        access = RigidAccess(
            self.session, self.manifest["id"], "centrifuge",
            ("arm_index", "basket_tilt"), minimum_m=0.0005,
        )
        for arm in np.linspace(-0.65, 0.65, 5):
            for basket in np.linspace(0, 0.4, 9):
                self.session.data.qpos[:] = [arm, basket]
                mujoco.mj_forward(self.model, self.session.data)
                access.check(self.model, self.session.data)
                self.assertFalse(any(c.dist < -1e-7 for c in self.session.data.contact))
        self.assertEqual(access.checked, 45)
        self.assertGreaterEqual(access.minimum, 0.0009999)

    def test_visible_curved_shell_has_shared_vertices_and_closed_surface_edges(self):
        geom = next(
            g
            for g in self.root.iter("geom")
            if "__ucsd_basket_curved_exterior_" in g.get("name", "")
        )
        mesh = self.root.find(f'./asset/mesh[@name="{geom.get("mesh")}"]')
        vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
        faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
        # Disconnected closed strips duplicate ring vertices and create false
        # surface-normal/bevel seams despite matching the collision silhouette.
        self.assertEqual(len(vertices), len(np.unique(vertices.round(9), axis=0)))
        edges, winding = Counter(), Counter()
        for face in faces:
            for first, second in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(first), int(second))))
                edges[key] += 1
                winding[key] += 1 if first < second else -1
        self.assertTrue(all(count == 2 for count in edges.values()))
        self.assertTrue(all(count == 0 for count in winding.values()))


if __name__ == "__main__":
    unittest.main()
