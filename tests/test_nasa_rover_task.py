"""Grounded contact task contracts and direct-distance failure sensitivity."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.nasa_rover_task import NasaRoverTask, box_separation, direct_gap
from real_labs.runtime import InstrumentSession


class GroundedNasaTests(unittest.TestCase):
    def session(self):
        root, manifest = build_scene("nasa_planetary")
        return InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            manifest,
        )

    def test_rover_and_sample_are_free_with_finite_ground_drive(self):
        s = self.session()
        m = s.model
        for name in ("rover__free_base", "nasa_task__sample_free"):
            self.assertEqual(m.jnt_type[m.joint(name).id], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.nmocap, 0)
        np.testing.assert_array_equal(m.body_gravcomp, 0)
        for side in ("left", "right"):
            for index in (1, 2, 3):
                wheel = m.actuator(f"rover__actuator_wheel_{side}_{index}").id
                steer = m.actuator(f"nasa_task__steer_wheel_{side}_{index}").id
                np.testing.assert_array_equal(m.actuator_forcerange[wheel], [-12, 12])
                np.testing.assert_array_equal(m.actuator_forcerange[steer], [-20, 20])
                self.assertTrue(m.actuator_forcelimited[wheel])
                self.assertTrue(m.actuator_forcelimited[steer])
        self.assertAlmostEqual(m.body_mass[m.body("nasa_task__sample").id], 0.025)
        task = NasaRoverTask(s, record=False)
        self.assertGreater(len(task.pairs), 100000)
        self.assertGreater(task.direct_queries, 0)
        self.assertEqual(s.data.time, 0)

    def test_displaced_sample_is_rejected_without_repositioning(self):
        s = self.session()
        q = s.model.jnt_qposadr[s.model.joint("nasa_task__sample_free").id]
        s.data.qpos[q] += 0.01
        before = s.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "Sample is not on the pickup fixture"):
            NasaRoverTask(s, record=False)
        np.testing.assert_array_equal(s.data.qpos, before)
        self.assertEqual(s.data.time, 0)

    def test_direct_boxes_distinguish_thin_clearance_from_penetration(self):
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            '<geom name="shelf" type="box" size=".03 .01 .003"/>'
            '<body pos="0 0 .016"><freejoint/>'
            '<geom name="sample" type="box" size=".012 .018 .012"/>'
            "</body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        for z, expected in ((0.016, 0.001), (0.015, 0), (0.014, -0.001)):
            data.qpos[2] = z
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(box_separation(model, data, 0, 1), expected)
            self.assertAlmostEqual(direct_gap(model, data, 0, 1), expected)


if __name__ == "__main__":
    unittest.main()
