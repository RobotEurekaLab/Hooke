"""Real sting passage, passive rolling contacts and source-bound towing motion."""

from copy import copy, deepcopy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.berkeley_tank_motion import (
    SCENE,
    EQUIPMENT,
    TankClearance,
    TankTransit,
    capsule_box_gap,
    cylinder_box_gap,
)
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession


class BerkeleyTankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, cls.manifest = build_scene(SCENE)
        cls.xml = ET.tostring(root, encoding="unicode")
        cls.model = mujoco.MjModel.from_xml_string(cls.xml)

    def session(self):
        return InstrumentSession(copy(self.model), deepcopy(self.manifest))

    def test_original_control_contract_and_explicit_passive_mass(self):
        m = self.model
        for key, position, axis, limits, mass, inertia, gain, force in (
            ("carriage_travel", [0, -22, 2.22], [0, 1, 0], [0, 8], 5, 0.03, 1100, 1200),
            (
                "model_immersion",
                [0, 0, -0.32],
                [0, 0, -1],
                [0, 0.35],
                0.4,
                0.0024,
                600,
                120,
            ),
        ):
            body = m.body(f"{EQUIPMENT}__body_{key}")
            joint = m.joint(f"{EQUIPMENT}__joint_{key}")
            actuator = m.actuator(f"{EQUIPMENT}__actuator_{key}")
            np.testing.assert_array_equal(body.pos, position)
            np.testing.assert_array_equal(joint.axis, axis)
            np.testing.assert_array_equal(joint.range, limits)
            self.assertEqual(float(body.mass[0]), mass)
            np.testing.assert_allclose(body.inertia, inertia)
            self.assertEqual(actuator.gainprm[0], gain)
            np.testing.assert_array_equal(actuator.forcerange, [-force, force])
        np.testing.assert_array_equal(
            m.geom(EQUIPMENT + "__berkeley_inert_hull").size, [0.14, 0.55, 0.12]
        )
        np.testing.assert_array_equal(
            m.site(EQUIPMENT + "__site_tow_model").pos, [0, 0, 0.12]
        )
        self.assertEqual(m.nu, 2)
        self.assertEqual(m.nv, 10)
        total = 0
        for side, end in product(range(2), repeat=2):
            name = f"{EQUIPMENT}__berkeley_wheel_{side}_{end}"
            body, spin, play = (
                m.body(name),
                m.joint(name + "_spin"),
                m.joint(name + "_radial_play"),
            )
            total += body.mass[0]
            self.assertEqual(m.body_gravcomp[body.id], 0)
            self.assertLess(
                play.id, spin.id
            )  # Play axis must not rotate with the wheel.
            np.testing.assert_array_equal(spin.axis, [1, 0, 0])
            np.testing.assert_array_equal(play.axis, [0, 0, 1])
            np.testing.assert_array_equal(play.range, [-0.0003, 0.0003])
            expected = [0.5 * (0.09 * 0.1**2 + 0.01 * 0.012**2)]
            expected += [
                (0.09 * (3 * 0.1**2 + 0.124**2) + 0.01 * (3 * 0.012**2 + 0.174**2)) / 12
            ] * 2
            np.testing.assert_allclose(body.inertia, expected, atol=1e-15)
        self.assertAlmostEqual(total, 0.4)

    def test_analytic_capsule_box_catches_centerline_through_solid(self):
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="b" type="box" size="1.49 .43 .05"/><geom name="s" type="capsule" fromto="0 0 -.21 0 0 .62" size=".018"/></worldbody></mujoco>'
        )
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertAlmostEqual(
            capsule_box_gap(m, d, m.geom("s").id, m.geom("b").id), -0.018
        )
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="b" type="box" pos="1.6 0 0" size="1.49 .43 .05"/>'
            '<geom name="s" type="capsule" fromto="0 0 -.21 0 0 .62" size=".018"/></worldbody></mujoco>'
        )
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertAlmostEqual(
            capsule_box_gap(m, d, m.geom("s").id, m.geom("b").id), 0.092
        )

    def test_open_platform_and_full_lift_support(self):
        s = self.session()
        m, d = s.model, s.data
        sting = m.geom(EQUIPMENT + "__berkeley_mounting_sting").id
        plate = [
            g
            for g in range(m.ngeom)
            if "__berkeley_platform" in m.geom(g).name and m.geom_contype[g]
        ]
        self.assertEqual(len(plate), 8)
        q = m.joint(EQUIPMENT + "__joint_model_immersion").qposadr[0]
        for height in np.linspace(0, 0.35, 19):
            d.qpos[q] = height
            mujoco.mj_forward(m, d)
            self.assertGreater(
                min(capsule_box_gap(m, d, sting, g) for g in plate), 0.0569
            )
            for side in range(2):
                guide = m.geom(f"{EQUIPMENT}__berkeley_lift_guide_{side}").id
                housing = m.geom(
                    f"{EQUIPMENT}__berkeley_guide_housing_{side}_surface"
                ).id
                gz, hz = d.geom_xpos[guide, 2], d.geom_xpos[housing, 2]
                self.assertLess(gz - m.geom_size[guide, 1], hz - 0.04)
                self.assertGreater(gz + m.geom_size[guide, 1], hz + 0.04)

    def test_all_moving_pairs_and_real_open_wheel_bearings(self):
        s = self.session()
        m, d = s.model, s.data
        task = TankTransit(s)
        ids = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        # Every dynamic body belongs to this instrument; this independent full
        # list includes parent-filtered shaft/bearing and wheel/deck pairs.
        expected = {
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
        }
        self.assertEqual(expected, {tuple(p) for p in task.clearances.pairs})
        rng = np.random.default_rng(1319)
        poses = [(0, 0), (8, 0.35), (8, 0), (0, 0.35)] + list(
            zip(rng.uniform(0, 8, 9), rng.uniform(0, 0.35, 9))
        )
        for travel, depth in poses:
            d.qpos[task.qadr] = travel, depth
            d.qpos[task.wheel_qadr] = rng.uniform(-90, 90, 4)
            mujoco.mj_forward(m, d)
            task.clearances.check(d)
            # Check every pair without the production AABB screen. Analytic
            # cylinder/box bounds supersede the documented bad CCD witnesses.
            for a, b in expected:
                gap = mujoco.mj_geomDistance(m, d, a, b, 0.002, None)
                ta, tb = m.geom_type[a], m.geom_type[b]
                if (
                    ta == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and tb == mujoco.mjtGeom.mjGEOM_BOX
                ):
                    analytic = cylinder_box_gap(m, d, a, b)
                    if analytic is not None:
                        gap = min(0.002, analytic)
                elif (
                    tb == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and ta == mujoco.mjtGeom.mjGEOM_BOX
                ):
                    analytic = cylinder_box_gap(m, d, b, a)
                    if analytic is not None:
                        gap = min(0.002, analytic)
                elif (
                    ta == mujoco.mjtGeom.mjGEOM_CAPSULE
                    and tb == mujoco.mjtGeom.mjGEOM_BOX
                ):
                    gap = min(gap, capsule_box_gap(m, d, a, b))
                elif (
                    tb == mujoco.mjtGeom.mjGEOM_CAPSULE
                    and ta == mujoco.mjtGeom.mjGEOM_BOX
                ):
                    gap = min(gap, capsule_box_gap(m, d, b, a))
                self.assertGreaterEqual(
                    gap,
                    task.clearances.limit(a, b),
                    (m.geom(a).name, m.geom(b).name, gap),
                )
        # Journal motion is vertical at any wheel angle, and its extreme still
        # leaves300 micrometres between the shaft and open bearing walls.
        for angle in (0, 1.7, 20):
            d.qpos[task.wheel_qadr] = angle
            d.qpos[task.play_qadr] = 0.0003
            mujoco.mj_forward(m, d)
            for side, end in product(range(2), repeat=2):
                name = f"{EQUIPMENT}__berkeley_wheel_{side}_{end}"
                body = m.body(name).id
                parent = m.body(EQUIPMENT + "__body_carriage_travel").id
                local = d.xmat[parent].reshape(3, 3).T @ (d.xpos[body] - d.xpos[parent])
                self.assertAlmostEqual(local[2], -0.1497)
                self.assertAlmostEqual(local[1], (-0.28, 0.28)[end])

    def test_retained_obstacle_is_not_screened_out(self):
        root = ET.fromstring(self.xml)
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="retained_obstacle",
            type="box",
            size=".04 .04 .04",
            pos="-.8 -22 2.12",
        )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        checker = TankClearance(m, d)
        with self.assertRaisesRegex(RuntimeError, "tank_.*(overlap|clearance)"):
            checker.check(d)
        obstacle = m.geom("retained_obstacle").id
        self.assertTrue(any(obstacle in pair for pair in checker.pairs))

    def test_binding_rejects_foreign_or_changed_contract_without_state_writes(self):
        changes = [
            lambda s: setattr(s, "data", mujoco.MjData(copy(s.model))),
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom(EQUIPMENT + "__berkeley_mounting_sting").id, 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom(EQUIPMENT + "__berkeley_lift_guide_0").id, 0), 0.5
            ),
            lambda s: s.model.actuator_gainprm.__setitem__((0, 0), 900),
            lambda s: setattr(s.model.opt, "timestep", 0.001),
        ]
        for change in changes:
            s = self.session()
            change(s)
            before = [
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            ]
            with self.assertRaises(ValueError):
                TankTransit(s)
            for a, b in zip(
                before, [s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time]
            ):
                np.testing.assert_array_equal(a, b)
        s = self.session()
        task = TankTransit(s)
        old = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            task.run()
        self.assertEqual(old.time, 0)
        self.assertEqual(s.data.time, 0)

    def test_force_failure_is_held_without_false_complete_coverage(self):
        s = self.session()
        task = TankTransit(s)
        s.data.qfrc_applied[task.dadr[1]] = 1e6
        result = task.run()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(result["executed_steps"], result["completed_checks"])
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(result["whole_route_clearance_lower_bound_m"])
        self.assertEqual(s.data.qfrc_applied[task.dadr[1]], 1e6)
        np.testing.assert_allclose(
            s.data.ctrl[task.actuators],
            np.clip(s.data.qpos[task.qadr], task.lower, task.upper),
        )

    def test_actual_full_transit_return_and_passive_rolling(self):
        s = self.session()
        signature = _model_signature(s.model)
        task = TankTransit(s)
        result = task.run()
        self.assertEqual(result["status"], "succeeded", result.get("failure"))
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["executed_steps"], result["completed_checks"])
        self.assertGreater(result["executed_steps"], 50000)
        self.assertFalse(result["reset_performed"])
        for phase in result["phases"]:
            self.assertTrue(phase["success"])
            self.assertEqual(phase["executed_steps"], phase["completed_checks"])
            self.assertLess(max(abs(np.array(phase["endpoint_error_m"]))), 0.0002)
        outward = result["phases"][1]
        self.assertLess(max(abs(np.array(outward["wheel_angle_rad"]) + 80)), 0.01)
        self.assertLess(result["maximum_permitted_wheel_contact_penetration_m"], 1e-5)
        rolling = result["passive_wheel_rotation"]
        self.assertLess(rolling["maximum_checked_accumulated_slip_m"], 0.001)
        self.assertGreater(rolling["minimum_checked_normal_load_N"], 0.1)
        self.assertEqual(_model_signature(s.model), signature)


if __name__ == "__main__":
    unittest.main()
