"""Physical shaft clearance and load paths through the complete sample travel."""

from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene


class ShrimpStageSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, _ = build_scene("anu_shrimp_geochronology")
        cls.model = mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode")
        )

    def test_open_bearing_has_supported_puck_and_clear_shaft_over_coupled_travel(self):
        m = self.model
        d = mujoco.MjData(m)
        transfer = m.body("shrimp_rg__body_sample_transfer").id
        rotor = m.body("shrimp_rg__body_sample_rotation").id
        shaft = m.geom("shrimp_rg__stage_spindle").id
        puck = next(
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == rotor
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and m.geom_size[g, 0] > 0.04
        )
        flange = [
            g
            for g in range(m.ngeom)
            if "__stage_thrust_flange_proxy_" in m.geom(g).name
        ]
        sleeve = [
            g for g in range(m.ngeom) if "__stage_sleeve_proxy_" in m.geom(g).name
        ]
        deck = [g for g in range(m.ngeom) if "__stage_deck_" in m.geom(g).name]
        self.assertTrue(flange and sleeve and deck)
        self.assertTrue(
            all(
                m.geom_contype[g] and m.geom_conaffinity[g]
                for g in [shaft, *flange, *sleeve, *deck]
            )
        )
        self.assertTrue(
            all(m.geom_bodyid[g] == transfer for g in [*flange, *sleeve, *deck])
        )
        rotating = [
            g for g in range(m.ngeom) if m.geom_bodyid[g] == rotor and m.geom_contype[g]
        ]
        other = [
            g
            for g in range(m.ngeom)
            if m.geom_contype[g]
            and m.body_weldid[m.geom_bodyid[g]] != m.body_weldid[rotor]
        ]
        addresses = [
            m.jnt_qposadr[m.joint("shrimp_rg__joint_" + key).id]
            for key in ("load_lock_door", "sample_transfer", "sample_rotation")
        ]

        def gap(a, b):
            return mujoco.mj_geomDistance(m, d, a, b, 0.1, None)

        for pose in product(
            (0, 0.43, 1.20),
            (0, 0.023, 0.060, 0.097, 0.120),
            np.linspace(*m.joint("shrimp_rg__joint_sample_rotation").range, 7),
        ):
            d.qpos[addresses] = pose
            mujoco.mj_forward(m, d)
            # Explicit parent-child queries cover pairs suppressed by native filtering.
            for a, b in product(rotating, other):
                self.assertGreaterEqual(
                    gap(a, b), -1e-8, (pose, m.geom(a).name, m.geom(b).name)
                )
            self.assertAlmostEqual(min(gap(puck, g) for g in flange), 0, delta=1e-8)
            self.assertAlmostEqual(gap(puck, shaft), 0, delta=1e-8)
            self.assertGreater(min(gap(shaft, g) for g in sleeve), 0.00029)
            # The flange carries axial load to the deck, rather than floating in its bore.
            self.assertLessEqual(
                min(abs(gap(a, b)) for a, b in product(flange, deck)), 1e-8
            )

    def test_original_frames_ranges_and_rotating_inertia_are_preserved(self):
        m = self.model
        for key, position, mass, inertia in (
            ("sample_transfer", [1.27, -0.46, 1.205], 0.15, 0.0009),
            ("sample_rotation", [0, 0, 0.022], 0.06, 0.00036),
        ):
            body = m.body("shrimp_rg__body_" + key)
            np.testing.assert_array_equal(body.pos, position)
            self.assertAlmostEqual(float(body.mass[0]), mass)
            np.testing.assert_array_equal(body.inertia, np.full(3, inertia))
        np.testing.assert_array_equal(
            m.joint("shrimp_rg__joint_sample_transfer").range, [0, 0.12]
        )
        np.testing.assert_array_equal(
            m.joint("shrimp_rg__joint_sample_rotation").range, [-3.1415927, 3.1415927]
        )


if __name__ == "__main__":
    unittest.main()
