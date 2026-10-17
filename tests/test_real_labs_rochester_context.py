"""Photograph-informed room props must fit and have real exterior support."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.rochester_context import PROFILER_SIZE_M


class RochesterContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("rochester_genomics")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(cls.root, encoding="unicode")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def bounds(self, index):
        rotation = self.data.geom_xmat[index].reshape(3, 3)
        center = self.data.geom_xpos[index] + rotation @ self.model.geom_aabb[index, :3]
        half = np.abs(rotation) @ self.model.geom_aabb[index, 3:]
        return center - half, center + half

    def geoms(self, prefix, contacts=False):
        return [
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith(prefix)
            and (
                not contacts
                or self.model.geom_contype[i]
                or self.model.geom_conaffinity[i]
            )
        ]

    def test_profiler_feet_meet_counter_and_published_envelope_fits(self):
        model = self.model
        support_low, support_high = self.bounds(model.geom("reagent_bench_top").id)
        feet = self.geoms("rochester_context__profiler_foot_")
        self.assertEqual(len(feet), 4)
        for index in feet:
            low, high = self.bounds(index)
            self.assertAlmostEqual(low[2], support_high[2], places=7)
            self.assertTrue(np.all(low[:2] >= support_low[:2]))
            self.assertTrue(np.all(high[:2] <= support_high[:2]))
        body = model.body("rochester_context__historical_profiler").id
        bounds = [
            self.bounds(i) for i in range(model.ngeom) if model.geom_bodyid[i] == body
        ]
        low = np.min([pair[0] for pair in bounds], axis=0)
        high = np.max([pair[1] for pair in bounds], axis=0)
        # The instrument is turned 90 degrees in the room.
        np.testing.assert_allclose(
            high - low, np.array(PROFILER_SIZE_M)[[1, 0, 2]], atol=1e-6
        )
        self.assertTrue(np.all(low[:2] >= support_low[:2] - 1e-7))
        self.assertTrue(np.all(high[:2] <= support_high[:2] + 1e-7))

    def test_shelf_cartons_have_support_and_clear_the_posts_and_ceiling(self):
        model = self.model
        for level in (0, 1):
            low, high = self.bounds(model.geom(f"wetlab_genomics_shelf_{level}").id)
            cartons = self.geoms(f"rochester_context__stock_{level}_", contacts=True)
            self.assertTrue(cartons)
            for index in cartons:
                bottom, top = self.bounds(index)
                self.assertAlmostEqual(bottom[2], high[2], places=7)
                self.assertTrue(np.all(bottom[:2] >= low[:2]))
                self.assertTrue(np.all(top[:2] <= high[:2]))
                self.assertLess(top[2], self.manifest["room"]["size"][2])
            for index in (1, 6, 10) if level == 0 else ():
                _, base = self.bounds(
                    model.geom(f"rochester_context__stock_0_{index}_contact").id
                )
                bottom, _ = self.bounds(
                    model.geom(f"rochester_context__stack_{index}_contact").id
                )
                self.assertAlmostEqual(bottom[2], base[2], places=7)
        for level in (0, 1):
            shelf_low, _ = self.bounds(model.geom(f"wetlab_genomics_shelf_{level}").id)
            for column in range(3):
                bracket = model.geom(
                    f"rochester_context__shelf_bracket_{level}_{column}"
                ).id
                _, high = self.bounds(bracket)
                post_low, _ = self.bounds(
                    model.geom(f"wetlab_genomics_shelf_post_{column}").id
                )
                self.assertAlmostEqual(high[2], shelf_low[2], places=7)
                self.assertAlmostEqual(high[1], post_low[1], places=7)

    def test_new_exteriors_do_not_penetrate_existing_equipment_or_furniture(self):
        model = self.model
        new = self.geoms("rochester_context__", contacts=True)
        existing = [
            i
            for i in range(model.ngeom)
            if not model.geom(i).name.startswith("rochester_context__")
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        for a in new:
            for b in existing:
                distance = mujoco.mj_geomDistance(model, self.data, a, b, 0.01, None)
                self.assertGreaterEqual(
                    distance,
                    -5e-5,
                    (model.geom(a).name, model.geom(b).name, distance),
                )

    def test_closed_cover_stops_a_falling_inert_probe_at_its_visible_surface(self):
        root = ET.fromstring(ET.tostring(self.root))
        body = ET.SubElement(
            root.find("worldbody"), "body", name="cover_probe", pos=".64 -.77 1.26"
        )
        ET.SubElement(body, "freejoint", name="cover_probe_free")
        ET.SubElement(
            body, "geom", name="cover_probe", type="sphere", size=".005", mass=".005"
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data, nstep=750)
        # The visible cover is 0.24 m tall above the 0.92 m counter.
        self.assertAlmostEqual(
            data.xpos[model.body("cover_probe").id, 2], 1.165, delta=0.001
        )
        probe = model.geom("cover_probe").id
        self.assertTrue(
            any(probe in (c.geom1, c.geom2) for c in data.contact[: data.ncon])
        )

    def test_context_has_no_actuation_or_unclaimed_scientific_capability(self):
        model = self.model
        bodies = {
            i
            for i in range(model.nbody)
            if model.body(i).name.startswith("rochester_context__")
        }
        self.assertTrue(bodies)
        self.assertTrue(all(int(body) not in bodies for body in model.jnt_bodyid))
        evidence = self.manifest["reference_detail_evidence"]["rochester_context"]
        capabilities = evidence["capabilities"]
        self.assertTrue(capabilities["closed_exterior_collision"])
        for name in (
            "profiler_operation",
            "bench_device_operation",
            "fluid_or_electrical_simulation",
        ):
            self.assertFalse(capabilities[name])
        self.assertEqual(len(evidence["local_reference_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
