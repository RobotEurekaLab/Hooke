"""Physical registration-bore support, obstruction and source binding."""

import unittest

import mujoco
import numpy as np

from real_labs.annulus_probe_motion import ProbeRegistration, EQUIPMENT
from test_annulus_guides import session

PREFIX = EQUIPMENT + "__annulus_reference_"


class AnnulusRegistrationTests(unittest.TestCase):
    def test_full_coupled_domain_including_parent_filtered_pairs(self):
        s = session()
        task = ProbeRegistration(s)
        for angle in np.linspace(-0.5, 0.5, 5):
            for height in np.linspace(-0.07, 0.17, 7):
                s.data.qpos[:] = [angle, height]
                mujoco.mj_forward(s.model, s.data)
                task.check(s.model, s.data)
                self.assertTrue(task.observe()["registered"])
        self.assertGreaterEqual(task.minimum, 0.0005999)

    def test_contact_ring_is_supported_by_the_tank(self):
        s = session()
        m, d = s.model, s.data
        tank = m.body(EQUIPMENT + "__body_tank_rotation").id
        self.assertEqual(m.body_mass[tank], 18)
        ids = {
            i
            for i in range(m.ngeom)
            if m.geom(i).name.startswith(PREFIX)
            and (m.geom_contype[i] or m.geom_conaffinity[i])
        }
        self.assertTrue(ids)
        self.assertTrue(all(m.geom_bodyid[i] == tank for i in ids))
        stem = m.geom(PREFIX + "stem").id
        floor = next(
            i
            for i in range(m.ngeom)
            if m.geom_bodyid[i] == tank
            and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and np.allclose(m.geom_pos[i], [0, 0, 0.025])
        )
        self.assertLessEqual(mujoco.mj_geomDistance(m, d, stem, floor, 0.1, None), 0)
        remaining, reached = ids - {stem}, {stem}
        while remaining:
            linked = {
                a
                for a in remaining
                if any(
                    mujoco.mj_geomDistance(m, d, a, b, 1e-6, None) <= 1e-7
                    for b in reached
                )
            }
            self.assertTrue(linked, [m.geom(i).name for i in remaining])
            remaining -= linked
            reached |= linked

    def test_actual_stylus_passes_the_bore_but_hits_an_offset_wall(self):
        s = session()
        m, d = s.model, s.data
        body = m.body(EQUIPMENT + "__body_probe_height").id
        stylus = m.geom(EQUIPMENT + "__annulus_guide_stylus").id
        ring = [
            i
            for i in range(m.ngeom)
            if m.geom(i).name.startswith(PREFIX + "datum_proxy_")
        ]
        self.assertTrue(ring)
        self.assertTrue(all(m.geom_contype[i] and m.geom_conaffinity[i] for i in ring))

        def gap():
            return min(
                mujoco.mj_geomDistance(m, d, stylus, i, 0.02, None) for i in ring
            )

        self.assertAlmostEqual(gap(), 0.006, delta=2e-8)
        # A separate negative witness changes only this test's model. The
        # production task never repositions bodies or disables contacts.
        m.body_pos[body, 0] += 0.0065
        mujoco.mj_forward(m, d)
        self.assertLess(gap(), -0.0004)

    def test_observation_and_binding_reject_disabled_contacts(self):
        s = session()
        task = ProbeRegistration(s)
        observation = task.observe()
        self.assertTrue(observation["registered"])
        self.assertAlmostEqual(observation["tip_height_above_datum_m"], -0.1184)
        self.assertAlmostEqual(observation["nominal_radial_clearance_m"], 0.006)
        m, d = s.model, s.data
        before = d.qpos.copy(), d.ctrl.copy(), d.time
        m.geom_contype[m.geom(PREFIX + "datum_proxy_0").id] = 0
        with self.assertRaises(ValueError):
            task.run()
        np.testing.assert_array_equal(d.qpos, before[0])
        np.testing.assert_array_equal(d.ctrl, before[1])
        self.assertEqual(d.time, before[2])

    def test_actual_cycle_and_return(self):
        s = session()
        report = ProbeRegistration(s).run()
        self.assertEqual(report["status"], "succeeded", report)
        self.assertEqual(len(report["phases"]), 9)
        self.assertGreater(report["executed_steps"], 0)
        self.assertEqual(report["checked_steps"], report["executed_steps"])
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["state_reset"])
        self.assertAlmostEqual(
            s.data.time, report["executed_steps"] * s.model.opt.timestep
        )
        self.assertGreaterEqual(
            report["whole_route_clearance_lower_bound_m"], 0.0005999
        )
        for phase in report["phases"]:
            self.assertTrue(phase["observation"]["registered"])
            if phase["phase"].startswith("approach"):
                self.assertAlmostEqual(
                    phase["observation"]["tip_height_above_datum_m"],
                    0.008,
                    delta=0.00005,
                )
            elif phase["phase"].startswith("pass_"):
                self.assertAlmostEqual(
                    phase["observation"]["tip_height_above_datum_m"],
                    -0.015,
                    delta=0.00005,
                )
        np.testing.assert_allclose(s.data.qpos, [0, 0], atol=5e-5)


if __name__ == "__main__":
    unittest.main()
