"""Perforated load-bearing bridge and backed diamond stiffeners for GCF."""

import copy
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.university_asia_wave2 import _gcf_web_cells


class HkustGcfGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("hkust_geotechnical_centrifuge")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.session = InstrumentSession(cls.model, cls.manifest)

    def setUp(self):
        self.session.reset()

    def test_open_bores_accept_physical_probes_but_solid_web_blocks_them(self):
        model, data = self.model, self.session.data
        bridge = model.body("gcf_model_testbed__body_bridge_y").id
        rotation, origin = data.xmat[bridge].reshape(3, 3), data.xpos[bridge]
        root = copy.deepcopy(self.root)
        holes = [hole for _, hole in _gcf_web_cells()]
        self.assertGreater(
            max(r for _, _, r in holes), 2.5 * min(r for _, _, r in holes)
        )
        for index, (x, z, radius) in enumerate(holes):
            pos = origin + rotation @ np.array([x, -0.15, z])
            body = ET.SubElement(
                root.find("worldbody"),
                "body",
                name=f"bore_probe_{index}",
                pos=" ".join(map(str, pos)),
            )
            ET.SubElement(body, "freejoint", name=f"bore_probe_joint_{index}")
            ET.SubElement(
                body,
                "geom",
                name=f"bore_probe_geom_{index}",
                type="sphere",
                size=str(radius * 0.5),
                mass=".01",
                contype="1",
                conaffinity="1",
            )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probes = {model.geom(f"bore_probe_geom_{i}").id for i in range(len(holes))}
        self.assertFalse(
            any(
                (c.geom1 in probes or c.geom2 in probes) and c.dist < -1e-6
                for c in data.contact
            )
        )
        # Move one probe into neighboring solid plate. The true opening must
        # not have been achieved by simply disabling the web's collisions.
        joint = model.joint("bore_probe_joint_0").id
        address = model.jnt_qposadr[joint]
        data.qpos[address : address + 3] += rotation @ np.array(
            [holes[0][2] + 0.02, 0, 0]
        )
        mujoco.mj_forward(model, data)
        probe = model.geom("bore_probe_geom_0").id
        self.assertTrue(
            any(probe in (c.geom1, c.geom2) and c.dist < -1e-4 for c in data.contact)
        )

    def test_top_carriage_is_supported_across_unchanged_travel(self):
        model, data = self.model, self.session.data
        carriage = next(
            i for i in range(model.ngeom) if "__gcf_top_carriage_" in model.geom(i).name
        )
        bridge = model.body("gcf_model_testbed__body_bridge_y").id
        deck = next(
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == bridge
            and np.allclose(model.geom_size[i], [1.28, 0.17, 0.025])
        )
        control = self.session.controls["gcf_model_testbed", "carriage_x"]
        for value in np.linspace(control.lower, control.upper, 21):
            data.qpos[model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(
                data.geom_xpos[carriage, 2] - model.geom_size[carriage, 2],
                data.geom_xpos[deck, 2] + model.geom_size[deck, 2],
                places=7,
            )
            self.assertLess(
                abs(data.geom_xpos[carriage, 0] - data.geom_xpos[deck, 0])
                + model.geom_size[carriage, 0],
                model.geom_size[deck, 0],
            )

    def test_dense_stiffeners_have_both_diagonals_and_contact_the_backing(self):
        model, data = self.model, self.session.data
        base = model.body("gcf_model_testbed").id
        origin = data.xpos[base]
        ribs = [
            i
            for i in range(model.ngeom)
            if "__gcf_diamond_stiffener_" in model.geom(i).name
        ]
        self.assertGreater(len(ribs), 100)
        families = {}
        for index in ribs:
            self.assertEqual(model.geom_type[index], mujoco.mjtGeom.mjGEOM_BOX)
            self.assertTrue(model.geom_contype[index])
            center = data.geom_xpos[index] - origin
            frame = data.geom_xmat[index].reshape(3, 3)
            axis = int(np.argmax(model.geom_size[index]))
            direction = frame[:, axis]
            if abs(abs(center[1]) - 0.825) < 1e-6:
                family, offset, skin = (
                    ("front_back", np.sign(center[1])),
                    abs(center[1]),
                    0.815,
                )
                self.assertLess(offset - model.geom_size[index, 1], skin)
            else:
                family, offset, skin = (
                    ("side", np.sign(center[0])),
                    abs(center[0]),
                    1.285,
                )
                self.assertLess(offset - model.geom_size[index, 0], skin)
            families.setdefault(family, set()).add(int(np.sign(direction[2])))
        self.assertEqual(len(families), 4)
        self.assertTrue(all(signs == {-1, 1} for signs in families.values()))

    def test_original_axis_limits_and_published_scope_are_preserved(self):
        bounds = {
            "bridge_y": (-0.48, 0.48),
            "carriage_x": (-0.76, 0.76),
            "probe_height": (-0.04, 0.09),
            "probe_rotation": (-0.7, 0.7),
        }
        for semantic, expected in bounds.items():
            control = self.session.controls["gcf_model_testbed", semantic]
            np.testing.assert_allclose([control.lower, control.upper], expected)
        limitations = " ".join(
            self.session.equipment["gcf_model_testbed"]["limitations"]
        )
        self.assertIn("does not spin", limitations)
        self.assertIn("estimated exterior reconstruction", limitations)


if __name__ == "__main__":
    unittest.main()
