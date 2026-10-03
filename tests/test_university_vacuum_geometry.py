"""Continuous support and motion checks for the large vacuum instruments."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


def _bounds(model, data, index):
    local = model.geom_aabb[index]
    rotation = data.geom_xmat[index].reshape(3, 3)
    center = data.geom_xpos[index] + rotation @ local[:3]
    extent = np.abs(rotation) @ local[3:]
    return center - extent, center + extent


class UniversityVacuumGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = {}
        for identifier in ("cornell_schlom_mbe", "anu_shrimp_geochronology"):
            root, manifest = build_scene(identifier)
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            cls.sessions[identifier] = InstrumentSession(model, manifest)

    def test_cornell_chambers_meet_cradles_and_loadlock_meets_saddles(self):
        session = self.sessions["cornell_schlom_mbe"]
        model, data = session.model, session.data
        session.reset()
        cradles = [
            i for i in range(model.ngeom) if "__vessel_cradle_" in model.geom(i).name
        ]
        saddles = [
            i for i in range(model.ngeom) if "__loadlock_saddle_" in model.geom(i).name
        ]
        self.assertTrue(cradles)
        self.assertTrue(saddles)
        vessels = [
            i
            for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and model.geom_contype[i]
            and abs(model.geom_size[i, 1] - 0.29) < 1e-8
        ]
        self.assertEqual(len(vessels), 2)
        for vessel in vessels:
            bottom = _bounds(model, data, vessel)[0][2]
            center = data.geom_xpos[vessel]
            supports = [
                i
                for i in cradles
                if abs(_bounds(model, data, i)[1][2] - bottom) < 1e-6
                and abs(data.geom_xpos[i, 0] - center[0]) < model.geom_size[vessel, 0]
            ]
            self.assertGreaterEqual(len(supports), 2)
        # Saddle contact is needed at both the table and the horizontal vessel.
        for index in saddles:
            lower, upper = _bounds(model, data, index)
            self.assertAlmostEqual(lower[2], 0.965, places=6)
            self.assertAlmostEqual(upper[2], 1.065, places=6)
        rack_feet = [
            i for i in range(model.ngeom) if "__rack_foot_" in model.geom(i).name
        ]
        self.assertEqual(len(rack_feet), 4)
        for index in rack_feet:
            lower, upper = _bounds(model, data, index)
            self.assertAlmostEqual(lower[2], 0.0, places=7)
            self.assertAlmostEqual(upper[2], 0.04, places=7)

    def test_anu_carriage_retains_guide_support_through_full_travel(self):
        session = self.sessions["anu_shrimp_geochronology"]
        model, data = session.model, session.data
        guides = [
            i for i in range(model.ngeom) if "__carrier_guide_" in model.geom(i).name
        ]
        self.assertEqual(len(guides), 2)
        body = model.body("shrimp_rg__body_sample_transfer").id
        carrier = next(
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body and model.geom_contype[i]
        )
        control = session.controls["shrimp_rg", "sample_transfer"]
        for position in np.linspace(control.lower, control.upper, 25):
            session.reset()
            data.qpos[model.jnt_qposadr[control.joint]] = position
            mujoco.mj_forward(model, data)
            bottom, top = _bounds(model, data, carrier)
            for guide in guides:
                low, high = _bounds(model, data, guide)
                self.assertAlmostEqual(high[2], bottom[2], places=6)
                self.assertGreater(min(top[0], high[0]) - max(bottom[0], low[0]), 0)
                self.assertGreater(min(top[1], high[1]) - max(bottom[1], low[1]), 0.10)

    def test_controls_move_and_return_under_simulation(self):
        for identifier, session in self.sessions.items():
            for key, control in session.controls.items():
                with self.subTest(scene=identifier, control=key):
                    session.reset()
                    span = control.upper - control.lower
                    for fraction in (0.15, 0.85, 0.25):
                        result = session.command(
                            *key, control.lower + fraction * span, duration=2
                        )
                        self.assertLessEqual(
                            abs(result["error"]), max(1e-6, span * 0.025)
                        )


if __name__ == "__main__":
    unittest.main()
