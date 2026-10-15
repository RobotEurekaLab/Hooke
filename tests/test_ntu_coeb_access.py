"""Supported COEB stages, full probe contacts and measured dry registration."""

from copy import copy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.motion import _model_signature
from real_labs.ntu_coeb_motion import CoebInspection, CONTROLS, EQUIPMENT, SCENE
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class CoebAccessTests(unittest.TestCase):
    def test_original_frames_controls_drives_specimen_and_world_views(self):
        s = session()
        m = s.model
        for name, position, limits, mass in (
            ("sample_x", (0, -0.04, 0.24), (-0.018, 0.018), 0.08),
            ("sample_y", (0, 0, 0.021), (-0.015, 0.015), 0.045),
            ("probe_clearance", (0, -0.04, 0.61), (-0.010, 0.014), 0.10),
        ):
            c = s.controls[EQUIPMENT, name]
            body = m.jnt_bodyid[c.joint]
            np.testing.assert_allclose(m.body_pos[body], position)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
            self.assertAlmostEqual(m.body_mass[body], mass)
        np.testing.assert_allclose(m.jnt_axis, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(m.actuator_gainprm[:, 0], [220, 180, 220])
        np.testing.assert_allclose(m.actuator_forcerange, [[-120, 120]] * 3)
        self.assertEqual(s.equipment[EQUIPMENT]["sample_size_m"], [0.012, 0.012, 0.001])
        np.testing.assert_allclose(
            m.site(EQUIPMENT + "__site_optical_coupon").pos, [0, 0, 0.009]
        )
        np.testing.assert_allclose(
            m.geom(EQUIPMENT + "__coeb_inert_coupon").size * 2, [0.012, 0.012, 0.001]
        )
        original = {
            "overview": ([6.5, -7.3, 5.2], [0, 0.2, 0.8], 42),
            "interior": ([1.15, -1.4, 1.65], [-0.3, 0.25, 1.3], 61),
            "workstation": ([0.7, -1.15, 1.75], [-0.35, 0.25, 1.2], 44),
        }
        for name, (position, target, fovy) in original.items():
            self.assertEqual(
                s.manifest["camera"][name],
                dict(position=position, target=target, fovy=fovy),
            )
        self.assertFalse(screen_contact_coverage(m)["issues"])

    def test_all_substantial_parts_have_contacts_and_continuous_attachment(self):
        s = session()
        m, d = s.model, s.data
        ids = [
            i
            for i in range(m.ngeom)
            if m.geom(i).name.startswith(EQUIPMENT + "__coeb_")
        ]
        self.assertGreater(len(ids), 200)
        self.assertTrue(all(m.geom_contype[i] and m.geom_conaffinity[i] for i in ids))
        for body in set(m.geom_bodyid[ids]):
            members = [i for i in ids if m.geom_bodyid[i] == body]
            attached, pending = {members[0]}, set(members[1:])
            while pending:
                new = {
                    a
                    for a in pending
                    if any(
                        mujoco.mj_geomDistance(m, d, a, b, 0.00001, None) <= 1e-7
                        for b in attached
                    )
                }
                if not new:
                    break
                attached |= new
                pending -= new
            self.assertFalse(pending, [m.geom(i).name for i in pending])
        # The separate fixed scope frame also actually seats on the common board.
        self.assertLessEqual(
            mujoco.mj_geomDistance(
                m,
                d,
                m.geom(EQUIPMENT + "__coeb_board").id,
                m.geom(EQUIPMENT + "__coeb_cast_base").id,
                0.001,
                None,
            ),
            1e-7,
        )

    def test_full_travel_and_parent_filtered_pair_coverage(self):
        s = session()
        m, d = s.model, s.data
        task = CoebInspection(s)
        ids = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        expected = {
            (int(a), int(b))
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if m.body_weldid[m.geom_bodyid[a]] != m.body_weldid[m.geom_bodyid[b]]
        }
        self.assertEqual(expected, set(map(tuple, task.pairs)))
        for pose in product(*[np.linspace(*limits, 5) for limits in m.jnt_range]):
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            task.check(m, d)
        self.assertEqual(task.checked, 125)
        self.assertGreaterEqual(task.minimum, 0.00059)
        # Unculled independent measurements include otherwise filtered moving/parent interfaces.
        for pose in product(*m.jnt_range):
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            direct = min(
                min(float(mujoco.mj_geomDistance(m, d, a, b, 0.002, None)), 0.002)
                for a, b in expected
            )
            task.reset_counts()
            task.check(m, d)
            self.assertAlmostEqual(direct, task.minimum, places=9)
        rail = m.geom(EQUIPMENT + "__coeb_y_rail_0.063").id
        shoe = m.geom(EQUIPMENT + "__coeb_y_shoe_0.063_0.021_0_1").id
        self.assertIn(tuple(sorted((rail, shoe))), expected)

    def test_guides_remain_engaged_and_coupon_surface_is_real(self):
        s = session()
        m, d = s.model, s.data
        for pose in product(*m.jnt_range):
            d.qpos[:] = pose
            mujoco.mj_forward(m, d)
            for rail_name, shoe_name, axis in (
                ("x_rail_0.055", "x_shoe_0.035_0.055_1_1", 0),
                ("y_rail_0.063", "y_shoe_0.063_0.021_0_1", 1),
                ("head_rail_0.073", "head_shoe_0.073_0_1", 2),
            ):
                rail, shoe = [
                    m.geom(EQUIPMENT + "__coeb_" + n).id for n in (rail_name, shoe_name)
                ]
                self.assertLess(
                    abs(d.geom_xpos[rail, axis] - d.geom_xpos[shoe, axis])
                    + m.geom_size[shoe, axis],
                    m.geom_size[rail, axis],
                )
                self.assertGreater(
                    mujoco.mj_geomDistance(m, d, rail, shoe, 0.002, None), 0.00059
                )
            marker = m.geom(EQUIPMENT + "__coeb_registration_mark").id
            coupon = m.geom(EQUIPMENT + "__coeb_inert_coupon").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, marker, coupon, 0.001, None), 1e-8
            )

    def test_binding_rejects_disabled_contacts_metadata_and_foreign_state(self):
        changes = (
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom(EQUIPMENT + "__coeb_terminal").id, 0
            ),
            lambda s: s.model.geom_conaffinity.__setitem__(
                s.model.geom(EQUIPMENT + "__coeb_probe_housing").id, 0
            ),
            lambda s: setattr(
                s.model.opt, "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda s: s.equipment[EQUIPMENT]["inspection_interface"].__setitem__(
                "guide_clearance_m", 0.01
            ),
            lambda s: s.manifest["camera"]["probe_registration"][
                "position"
            ].__setitem__(0, 0),
            lambda s: setattr(s, "data", mujoco.MjData(copy(s.model))),
        )
        for change in changes:
            s = session()
            task = CoebInspection(s)
            change(s)
            before = s.data.qpos.copy(), s.data.ctrl.copy()
            with self.assertRaises(ValueError):
                task.move((0, 0, 0.002))
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, 0)
        s = session()
        task = CoebInspection(s)
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            task.move((0, 0, 0.002))

    def test_real_registration_and_return_with_complete_step_coverage(self):
        s = session()
        before = _model_signature(s.model)
        result = CoebInspection(s).run()
        self.assertEqual(
            result["status"],
            "succeeded",
            [(p["phase"], p.get("failure")) for p in result["phases"]],
        )
        self.assertTrue(result["observation"]["registered"])
        self.assertGreater(result["observation"]["standoff_m"], 0.0015)
        self.assertLess(result["observation"]["projected_error_m"], 0.00002)
        self.assertEqual(result["checked_steps"], result["executed_steps"])
        self.assertGreater(result["executed_steps"], 1000)
        self.assertLess(max(result["measured_return_error_m"]), 0.00002)
        self.assertFalse(result["reset_performed"])
        self.assertEqual(_model_signature(s.model), before)
        for phase in result["phases"]:
            self.assertTrue(phase["direct_clearance"]["coverage_complete"])
            self.assertGreater(
                phase["direct_clearance"]["whole_leg_clearance_lower_bound_m"], 0.00059
            )

    def test_nonzero_unselected_axis_is_held_during_xy_registration(self):
        s = session()
        s.command(EQUIPMENT, "probe_clearance", 0.005, duration=2)
        before = s.data.qpos[2]
        task = CoebInspection(s, CONTROLS[:2])
        result = task.move((0.003, -0.002))
        self.assertEqual(result["status"], "succeeded")
        self.assertAlmostEqual(s.data.qpos[2], before, places=6)
        self.assertEqual(s.data.ctrl[2], 0.005)

    def test_force_failure_and_held_axis_failure_do_not_reset_or_fake_coverage(self):
        for controls, force_axis, target in (
            (CONTROLS, 2, (0, 0, 0.003)),
            (CONTROLS[:2], 2, (0.003, 0)),
        ):
            s = session()
            task = CoebInspection(s, controls)
            s.data.qfrc_applied[force_axis] = 500
            result = task.move(target)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["hold_requested"])
            self.assertFalse(result["reset_performed"])
            self.assertGreater(s.data.time, 0)
            coverage = result["direct_clearance"]
            self.assertLess(coverage["checked_steps"], result["physics_steps"])
            self.assertIsNone(coverage["whole_leg_clearance_lower_bound_m"])
            self.assertNotEqual(s.data.qpos[2], 0)

    def test_readonly_abort_records_failed_physical_state(self):
        s = session()
        task = CoebInspection(s)
        states = []

        def abort(model, data):
            states.append(data.qpos.copy())
            if len(states) == 5:
                raise RuntimeError("independent_readonly_abort")

        result = task.move((0.003, 0, 0), step_observer=abort)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertEqual(result["physics_steps"], 5)
        self.assertEqual(result["direct_clearance"]["checked_steps"], 5)
        np.testing.assert_array_equal(s.data.qpos, states[-1])


if __name__ == "__main__":
    unittest.main()
