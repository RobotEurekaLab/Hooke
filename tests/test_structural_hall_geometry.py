"""FSEL assembly continuity and actual travel, not a structural load validation."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


class StructuralHallGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, manifest = build_scene("texas_ferguson_structural_test_hall")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )
        cls.manifest = manifest

    def setUp(self):
        self.session = InstrumentSession(self.model, self.manifest)

    def tagged(self, tag):
        return [i for i in range(self.model.ngeom) if tag in self.model.geom(i).name]

    def bounds(self, geom):
        data = self.session.data
        rotation = data.geom_xmat[geom].reshape(3, 3)
        local = self.model.geom_aabb[geom]
        center = data.geom_xpos[geom] + rotation @ local[:3]
        radius = abs(rotation) @ local[3:]
        return center - radius, center + radius

    def assert_supported(self, upper, lower):
        low, high = self.bounds(upper)
        a, b = self.bounds(lower)
        self.assertAlmostEqual(low[2], b[2], places=7)
        self.assertTrue(
            np.all(np.minimum(high[:2], b[:2]) > np.maximum(low[:2], a[:2]))
        )

    def test_sensor_crossbar_remains_continuously_supported_during_full_travel(self):
        control = self.session.controls["beam_test", "sensor_traverse"]
        sled = self.tagged("__sensor_sled_")[0]
        riser = self.tagged("__sensor_riser_")[0]
        crossbar = self.tagged("__sensor_crossbar_")[0]
        for position in np.linspace(control.lower, control.upper, 57):
            self.session.data.qpos[self.model.jnt_qposadr[control.joint]] = position
            mujoco.mj_forward(self.model, self.session.data)
            self.assert_supported(riser, sled)
            self.assert_supported(crossbar, riser)
            self.assertTrue(self.model.geom_contype[riser])
            self.assertFalse(any(c.dist < -1e-5 for c in self.session.data.contact))

    def test_timber_support_chain_reaches_ground_and_cap(self):
        caps = self.tagged("__beam_crib_cap_")
        ties = self.tagged("__beam_crib_tie_")
        legs = self.tagged("__beam_crib_leg_")
        self.assertEqual(len(legs), 8)
        for leg in legs:
            low, high = self.bounds(leg)
            self.assertAlmostEqual(low[2], 0.0)
            supporters = []
            for tie in ties:
                a, b = self.bounds(tie)
                if abs(a[2] - high[2]) < 1e-8 and np.all(
                    np.minimum(high[:2], b[:2]) > np.maximum(low[:2], a[:2])
                ):
                    supporters.append(tie)
            self.assertEqual(len(supporters), 1)
            tie = supporters[0]
            matching_cap = min(
                caps, key=lambda g: np.linalg.norm(self.bounds(g)[0][:2] - low[:2])
            )
            self.assert_supported(matching_cap, tie)

    def test_dynamic_endpoint_commands_keep_assembly_clear_and_return(self):
        data = self.session.data
        for control in self.session.controls.values():
            for target in (control.upper, control.lower, 0.0):
                data.ctrl[control.actuator] = target
                for _ in range(round(2.0 / self.model.opt.timestep)):
                    mujoco.mj_step(self.model, data)
                    self.assertFalse(any(c.dist < -1e-5 for c in data.contact))
                measured = data.qpos[self.model.jnt_qposadr[control.joint]]
                self.assertLess(abs(measured - target), 0.001)
        self.assertFalse(any(w.number for w in data.warning))

    def test_side_windows_are_glazing_in_real_wall_openings(self):
        data = self.session.data
        for side, y, expected in (
            (-1, -4, "arch_opening_side_clerestory_1_glass"),
            (1, -5.5, "arch_cutaway_wall_fsel_right_glass_1"),
        ):
            hit = np.array([-1], dtype=np.int32)
            origin = np.array([side * 8.6, y, 7.2])
            direction = np.array([side, 0, 0], dtype=float)
            distance = mujoco.mj_ray(
                self.model, data, origin, direction, None, 1, -1, hit
            )
            self.assertGreater(distance, 0)
            self.assertEqual(self.model.geom(int(hit[0])).name, expected)
            original = self.model.geom_group.copy()
            try:
                self.model.geom_group[hit[0]] = 1
                empty = mujoco.mj_ray(
                    self.model,
                    data,
                    origin,
                    direction,
                    np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
                    1,
                    -1,
                    hit,
                )
                self.assertLess(empty, 0, "An opaque wall fills the window aperture")
            finally:
                self.model.geom_group[:] = original


if __name__ == "__main__":
    unittest.main()
