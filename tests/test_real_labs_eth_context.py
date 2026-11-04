"""The photo-informed closed exterior must fit and provide physical contact."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene


class EthContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("eth_air_quality")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def bounds(self, index):
        rotation = self.data.geom_xmat[index].reshape(3, 3)
        if self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = self.model.geom_dataid[index]
            start = self.model.mesh_vertadr[mesh]
            vertices = self.model.mesh_vert[
                start : start + self.model.mesh_vertnum[mesh]
            ]
            world = vertices @ rotation.T + self.data.geom_xpos[index]
            return world.min(axis=0), world.max(axis=0)
        center = self.data.geom_xpos[index] + rotation @ self.model.geom_aabb[index, :3]
        half = np.abs(rotation) @ self.model.geom_aabb[index, 3:]
        return center - half, center + half

    def test_feet_meet_counter_and_published_envelope_fits(self):
        model = self.model
        bench_low, bench_high = self.bounds(model.geom("sampling_bench_top").id)
        indices = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith("eth_air_context__")
        ]
        feet = [
            i
            for i in indices
            if model.geom(i).name.startswith("eth_air_context__foot_")
        ]
        self.assertEqual(len(feet), 4)
        for i in feet:
            self.assertAlmostEqual(self.bounds(i)[0][2], bench_high[2], places=7)
            self.assertTrue(model.geom_contype[i] and model.geom_conaffinity[i])
        low = np.min([self.bounds(i)[0] for i in indices], axis=0)
        high = np.max([self.bounds(i)[1] for i in indices], axis=0)
        np.testing.assert_allclose(high - low, (0.33, 0.46, 0.36), atol=2e-6)
        self.assertTrue(np.all(low[:2] >= bench_low[:2]))
        self.assertTrue(np.all(high[:2] <= bench_high[:2]))

    def test_exterior_does_not_penetrate_existing_equipment_or_furniture(self):
        model = self.model
        contact = [
            i
            for i in range(model.ngeom)
            if model.geom_contype[i] or model.geom_conaffinity[i]
        ]
        new = [i for i in contact if model.geom(i).name.startswith("eth_air_context__")]
        existing = set(contact) - set(new)
        for a in new:
            for b in existing:
                distance = mujoco.mj_geomDistance(model, self.data, a, b, 0.01, None)
                self.assertGreaterEqual(
                    distance, -5e-5, (model.geom(a).name, model.geom(b).name, distance)
                )

    def test_visible_closed_hood_stops_falling_inert_probe(self):
        root = ET.fromstring(ET.tostring(self.root))
        body = ET.SubElement(
            root.find("worldbody"), "body", name="cover_probe", pos=".17 1.385 1.33"
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body, "geom", name="cover_probe", type="sphere", size=".005", mass=".005"
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data, nstep=750)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(
            data.xpos[model.body("cover_probe").id, 2], 1.265, delta=0.001
        )
        probe = model.geom("cover_probe").id
        self.assertTrue(any(probe in (c.geom1, c.geom2) for c in data.contact))

    def test_display_trim_and_feet_attach_to_closed_shell(self):
        pairs = [
            ("display_bezel", "base"),
            ("front_latch", "hood"),
            ("side_trim_-1", "hood"),
            ("side_trim_1", "hood"),
            ("base", "seam"),
            ("seam", "hood"),
        ]
        pairs.extend(
            (f"foot_{x}_{y}", "base") for x in (-0.115, 0.115) for y in (-0.175, 0.175)
        )
        for a, b in pairs:
            distance = mujoco.mj_geomDistance(
                self.model,
                self.data,
                self.model.geom("eth_air_context__" + a).id,
                self.model.geom("eth_air_context__" + b).id,
                0.01,
                None,
            )
            self.assertLessEqual(distance, 5e-5, (a, b, distance))
            self.assertGreaterEqual(distance, -5e-4, (a, b, distance))

    def test_static_context_is_separate_from_operable_filter(self):
        model = self.model
        bodies = {
            i
            for i in range(model.nbody)
            if model.body(i).name.startswith("eth_air_context__")
        }
        self.assertTrue(bodies)
        self.assertTrue(all(int(i) not in bodies for i in model.jnt_bodyid))
        evidence = self.manifest["reference_detail_evidence"]["eth_air_context"]
        self.assertTrue(evidence["capabilities"]["closed_exterior_collision"])
        self.assertFalse(evidence["capabilities"]["instrument_operation"])
        self.assertFalse(
            evidence["capabilities"]["scientific_or_electrical_simulation"]
        )
        self.assertIn(
            "aerosol_sampler__joint_filter_drawer",
            [model.joint(i).name for i in range(model.njnt)],
        )


if __name__ == "__main__":
    unittest.main()
