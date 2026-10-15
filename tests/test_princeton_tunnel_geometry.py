"""Princeton casing supports and the authored collision-enabled traverse slot."""

from collections import Counter
import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class PrincetonTunnelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("princeton_bam_closed_loop_wind_tunnel")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.session = InstrumentSession(cls.model, cls.manifest)

    def setUp(self):
        self.session.reset()

    def test_slot_passes_a_free_probe_across_full_travel_and_adjacent_roof_blocks_it(
        self,
    ):
        root = copy.deepcopy(self.root)
        body = ET.SubElement(root.find("worldbody"), "body", name="roof_test_probe")
        ET.SubElement(body, "freejoint", name="roof_test_probe_joint")
        ET.SubElement(
            body,
            "geom",
            name="roof_test_probe_geom",
            type="sphere",
            size=".006",
            mass=".01",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        probe = model.geom("roof_test_probe_geom").id
        address = model.jnt_qposadr[model.joint("roof_test_probe_joint").id]
        traverse = model.jnt_qposadr[
            model.joint("wind_tunnel__joint_probe_traverse").id
        ]
        for travel in np.linspace(-0.36, 0.36, 25):
            # The real shaft stays present and collidable, moved clear of the
            # independent test probe instead of masking its contact flags.
            data.qpos[traverse] = -0.36 if travel >= 0 else 0.36
            data.qpos[address : address + 3] = (0.9, -1.8 + travel, 2.7716)
            mujoco.mj_forward(model, data)
            self.assertFalse(
                any(
                    probe in (c.geom1, c.geom2) and c.dist < -1e-6 for c in data.contact
                )
            )
        for point in ((1.10, -1.8, 2.7716), (0.9, -2.22, 2.7716), (0.9, -1.38, 2.7716)):
            data.qpos[address : address + 3] = point
            mujoco.mj_forward(model, data)
            self.assertTrue(
                any(
                    probe in (c.geom1, c.geom2)
                    and c.dist < -1e-4
                    and "bam_roof_slot_contact"
                    in model.geom(c.geom2 if c.geom1 == probe else c.geom1).name
                    for c in data.contact
                )
            )

    def test_continuous_visible_shells_have_shared_vertices_and_manifold_edges(self):
        selected = [
            geom
            for geom in self.root.iter("geom")
            if "__bam_rounded_turn_" in geom.get("name", "")
            and "contact" not in geom.get("name", "")
        ]
        self.assertEqual(len(selected), 4)
        for geom in selected:
            mesh = self.root.find(f'./asset/mesh[@name="{geom.get("mesh")}"]')
            vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
            faces = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
            self.assertEqual(len(vertices), len(np.unique(vertices.round(9), axis=0)))
            edges, winding = Counter(), Counter()
            for face in faces:
                for first, second in zip(face, np.roll(face, -1)):
                    key = tuple(sorted((int(first), int(second))))
                    edges[key] += 1
                    winding[key] += 1 if first < second else -1
            self.assertTrue(all(value == 2 for value in edges.values()))
            self.assertTrue(all(value == 0 for value in winding.values()))
            # The old side blocks had an outer X boundary of 4.93 m. Curved
            # corners stay within that boundary and the old total loop bounds.
            world = vertices + np.fromstring(geom.get("pos"), sep=" ")
            self.assertLessEqual(abs(world[:, 0]).max(), 4.93 + 1e-9)
            self.assertGreaterEqual(world[:, 1].min(), -2.85 - 1e-9)
            self.assertLessEqual(world[:, 1].max(), 3.65 + 1e-9)

    def test_return_crossbars_connect_to_grounded_legs_and_reach_the_casing(self):
        model = self.model
        base = model.body("wind_tunnel").id
        boxes = {
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == base
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_contype[i]
        }
        grounded = {
            i for i in boxes if abs(model.geom_pos[i, 2] - model.geom_size[i, 2]) < 1e-9
        }
        connected, pending = set(grounded), list(grounded)
        while pending:
            current = pending.pop()
            for other in boxes - connected:
                overlap = (
                    model.geom_size[current]
                    + model.geom_size[other]
                    - abs(model.geom_pos[current] - model.geom_pos[other])
                )
                if np.all(overlap >= -1e-9):
                    connected.add(other)
                    pending.append(other)
        crossbars = {
            i for i in boxes if "bam_return_bearing_crossbar" in model.geom(i).name
        }
        self.assertEqual(len(crossbars), 4)
        self.assertTrue(crossbars <= connected)
        for index in crossbars:
            low = model.geom_pos[index, 2] - model.geom_size[index, 2]
            high = model.geom_pos[index, 2] + model.geom_size[index, 2]
            self.assertLessEqual(low, 1.10)
            self.assertGreaterEqual(high, 1.10)

    def test_original_control_frames_targets_and_collidable_shaft_remain(self):
        model = self.model
        for semantic, expected, origin in (
            ("model_pitch", (-0.32, 0.32), (-0.35, -1.8, 2.04)),
            ("probe_traverse", (-0.36, 0.36), (0.9, -1.8, 3.2)),
        ):
            control = self.session.controls["wind_tunnel", semantic]
            np.testing.assert_allclose((control.lower, control.upper), expected)
            np.testing.assert_allclose(
                model.body(f"wind_tunnel__body_{semantic}").pos, origin
            )
        np.testing.assert_allclose(
            model.site("wind_tunnel__site_airfoil_coupon").pos, (0, 0, 0.025)
        )
        np.testing.assert_allclose(
            model.site("wind_tunnel__site_probe_tip").pos, (0, 0, -0.695)
        )
        body = model.body("wind_tunnel__body_probe_traverse").id
        shafts = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body
            and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and np.allclose(model.geom_size[i, :2], (0.010, 0.30))
        ]
        self.assertEqual(len(shafts), 1)
        self.assertTrue(model.geom_contype[shafts[0]])
        self.assertEqual(model.njnt, 2)
        self.assertEqual(model.neq, 0)
        limitations = " ".join(self.manifest["equipment"][0]["limitations"])
        self.assertIn("Published 4 ft square test-section bore retained", limitations)
        self.assertIn("authored mechanical interface", limitations)
        self.assertIn("No airflow", limitations)

    def test_actual_equipment_geometry_stays_within_original_loop_envelope(self):
        model, data = self.model, self.session.data
        minimum, maximum = [], []
        for index in range(model.ngeom):
            if not model.geom(index).name.startswith("wind_tunnel__"):
                continue
            rotation = data.geom_xmat[index].reshape(3, 3)
            if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[index]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                points = (
                    model.mesh_vert[start : start + count] @ rotation.T
                    + data.geom_xpos[index]
                )
                minimum.append(points.min(axis=0))
                maximum.append(points.max(axis=0))
            else:
                bounds = model.geom_aabb[index]
                center = data.geom_xpos[index] + rotation @ bounds[:3]
                radius = abs(rotation) @ bounds[3:]
                minimum.append(center - radius)
                maximum.append(center + radius)
        # Mesh principal-axis AABBs can conservatively grow when rotated back;
        # actual mesh vertices establish the apparatus envelope without that artifact.
        self.assertTrue(
            np.all(np.min(minimum, axis=0) >= np.array([-4.95, -2.90, 0]) - 1e-6)
        )
        self.assertTrue(
            np.all(np.max(maximum, axis=0) <= np.array([4.95, 3.94, 3.245]) + 1e-6)
        )


if __name__ == "__main__":
    unittest.main()
