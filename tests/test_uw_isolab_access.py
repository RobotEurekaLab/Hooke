"""Version 3 source-bound UW mechanics, explicit migration and guarded operation."""

from copy import deepcopy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import operations
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.uw_isolab_geometry import ID
from real_labs.uw_isolab_motion import CONTROLS, EQUIPMENT, SCENE, IsolabInspection


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return (
        s.data.qpos.copy(),
        s.data.qvel.copy(),
        s.data.ctrl.copy(),
        float(s.data.time),
    )


def assert_state(test, before, s):
    for a, b in zip(before[:3], state(s)[:3]):
        np.testing.assert_array_equal(a, b)
    test.assertEqual(before[-1], s.data.time)


class IsolabAccessTests(unittest.TestCase):
    def test_explicit_versioned_frames_ranges_and_true_free_vial(self):
        s = session()
        m = s.model
        expected = (
            ([-0.19, 0.19], [0, 0, 0], 0.7, 250, 30),
            ([0, 0.285], [0, -0.15, 0], 0.4, 250, 20),
            ([0, 0.18], [0, 0, 0.550], 0.1, 180, 120),
        )
        for key, (limits, origin, mass, gain, force) in zip(CONTROLS, expected):
            c = s.controls[EQUIPMENT, key]
            b = m.jnt_bodyid[c.joint]
            np.testing.assert_array_equal(m.jnt_range[c.joint], limits)
            np.testing.assert_array_equal(m.body_pos[b], origin)
            self.assertEqual(m.body_mass[b], mass)
            self.assertEqual(m.actuator_gainprm[c.actuator, 0], gain)
            np.testing.assert_array_equal(
                m.actuator_forcerange[c.actuator], [-force, force]
            )
        vial = m.geom(ID + "__vial_collision").id
        body = m.geom_bodyid[vial]
        self.assertEqual(m.jnt_type[m.body_jntadr[body]], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.body_mass[body], 0.003)
        np.testing.assert_array_equal(m.geom_size[vial, :2], [0.006, 0.016])
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.opt.timestep, 0.0005)
        self.assertEqual(len(s.manifest["specimens"]), 1)
        self.assertEqual(
            s.manifest["specimens"][0]["collision_geom"], m.geom(vial).name
        )

    def test_legacy_commands_are_rejected_without_mapping(self):
        s = session()
        before = state(s)
        for key in ("carousel", "gantry_x", "gantry_y", "injection_z"):
            for command in (s.command, lambda *args: operations.command(s, *args)):
                with self.assertRaises(KeyError) as caught:
                    command(EQUIPMENT, key, 0)
                self.assertEqual(
                    caught.exception.args, (s.equipment[EQUIPMENT]["retired_controls"][key],)
                )
                assert_state(self, before, s)
            self.assertFalse(
                s.equipment[EQUIPMENT]["retired_controls"][key]["automatic_mapping"]
            )
        with self.assertRaises(ValueError):
            IsolabInspection(s, ("gantry_x", "gantry_y", "injection_z"))
        assert_state(self, before, s)

    def test_cabinet_dimension_and_original_room_views(self):
        s = session()
        m, d = s.model, s.data
        cabinet = m.body("oem_analyzer_enclosure").id
        ids = np.flatnonzero(m.geom_bodyid == cabinet)
        ids = [
            i
            for i in ids
            if "__oem_case_" in m.geom(i).name and "screw" not in m.geom(i).name
        ]
        rot = d.geom_xmat[ids].reshape(-1, 3, 3)
        centers = d.geom_xpos[ids] + np.einsum("nij,nj->ni", rot, m.geom_aabb[ids, :3])
        half = np.einsum("nij,nj->ni", abs(rot), m.geom_aabb[ids, 3:])
        np.testing.assert_allclose(
            (centers + half).max(0) - (centers - half).min(0),
            [0.432, 0.432, 0.191],
            atol=1e-12,
        )
        self.assertEqual(
            s.manifest["camera"]["workstation"]["position"], [-0.35, -0.45, 2.5]
        )
        self.assertTrue(
            {"overview", "interior", "workstation"}.issubset(s.manifest["camera_names"])
        )
        self.assertEqual(s.manifest["room"]["size"], [10.8, 5.8, 3.05])

    def test_running_bores_remain_open_and_supported(self):
        s = session()
        m, d = s.model, s.data
        task = IsolabInspection(s)
        for q in ((0, 0, 0), (0.19, 0.285, 0), (-0.19, 0, 0.18)):
            d.qpos[task.addresses] = q
            mujoco.mj_forward(m, d)
            for axis, inner, rail_radius in [
                ("x", 0.0043, 0.004),
                ("y", 0.0053, 0.005),
                ("z", 0.0043, 0.004),
            ]:
                prefix = ID + "__guide_" + axis + "_bearing_"
                proxies = [
                    i
                    for i in range(m.ngeom)
                    if m.geom(i).name.startswith(prefix) and "_proxy_" in m.geom(i).name
                ]
                rails = [
                    m.geom(ID + "__" + axis + "_rail_" + str(i)).id for i in range(2)
                ]
                self.assertTrue(proxies)
                gap = min(
                    mujoco.mj_geomDistance(m, d, a, b, 0.002, None)
                    for a in rails
                    for b in proxies
                )
                self.assertAlmostEqual(gap, inner - rail_radius, delta=1e-8)
        self.assertTrue(
            np.all(
                m.geom_contype[
                    [m.geom(ID + "__dry_stylus").id, m.geom(ID + "__head").id]
                ]
            )
        )

    def test_model_state_masks_and_source_metadata_reject_before_mutation(self):
        mutations = (
            lambda s: setattr(
                s.model.opt, "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom(ID + "__x_rail_0").id, 0
            ),
            lambda s: s.model.geom_pos.__setitem__(
                (s.model.geom(ID + "__x_rail_0").id, 0), 0.04
            ),
            lambda s: s.equipment[EQUIPMENT]["inspection_interface"].__setitem__(
                "revision", 2
            ),
        )
        for mutation in mutations:
            s = session()
            mutation(s)
            before = state(s)
            with self.assertRaises(ValueError):
                IsolabInspection(s)
            assert_state(self, before, s)
        s = session()
        other = session()
        s.data = other.data
        before = state(s)
        with self.assertRaises(ValueError):
            IsolabInspection(s)
        assert_state(self, before, s)

    def test_replaced_state_and_modified_model_rejected_at_move(self):
        for replacement in (True, False):
            s = session()
            task = IsolabInspection(s)
            if replacement:
                s.data = mujoco.MjData(s.model)
            else:
                s.model.body_mass[s.model.body(ID + "__body_gantry_x").id] += 0.1
            before = state(s)
            with self.assertRaises(ValueError):
                task.move([0, 0, 0.001])
            assert_state(self, before, s)

    def test_unsafe_low_xy_and_out_of_range_targets_rejected_without_step(self):
        s = session()
        task = IsolabInspection(s)
        before = state(s)
        for target in ([0.06, 0.199, 0.137], [0, 0, 0.180001], [float("nan"), 0, 0]):
            with self.assertRaises(ValueError):
                task.move(target)
            assert_state(self, before, s)

    def test_known_wash_collision_is_detected_and_not_pruned(self):
        s = session()
        task = IsolabInspection(s)
        s.data.qpos[task.addresses] = [0.059, 0.199, 0.137]
        mujoco.mj_forward(s.model, s.data)
        task.held_other = []
        with self.assertRaisesRegex(RuntimeError, "Clearance lost"):
            task._observe_step(np.array([], int), task.measured())
        self.assertEqual(task.checked, 0)
        self.assertFalse(
            s.equipment[EQUIPMENT]["inspection_interface"][
                "full_joint_box_collision_free"
            ]
        )

    def test_actual_guarded_move_return_complete_coverage_without_reset(self):
        s = session()
        task = IsolabInspection(s)
        settle = task.leg("settle", settle_s=0.3)
        self.assertEqual(settle["status"], "succeeded")
        one = task.move([-0.003, 0, 0])
        two = task.move([0, 0, 0])
        for report in (one, two):
            self.assertEqual(report["status"], "succeeded")
            self.assertGreater(report["physics_steps"], 0)
            self.assertEqual(report["physics_steps"], report["checked_steps"])
            self.assertTrue(report["coverage_complete"])
            self.assertFalse(report["reset_performed"])
        self.assertEqual(one["after"], two["before"])
        self.assertLess(max(abs(task.measured())), 1e-5)

    def test_force_failure_holds_without_fabricating_complete_coverage(self):
        s = session()
        task = IsolabInspection(s)
        dof = s.model.jnt_dofadr[task.controls[1].joint]
        s.data.qfrc_applied[dof] = 85
        report = task.move([-0.001, 0, 0])
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["physics_steps"], report["checked_steps"])
        self.assertIsNone(report["whole_leg_clearance_lower_bound_m"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertGreater(s.data.time, 0)

    def test_other_instrument_is_held_and_failure_is_counted(self):
        s = session()
        task = IsolabInspection(s)
        control = s.controls["analytical_balance", "draft_door_joint"]
        s.data.qfrc_applied[s.model.jnt_dofadr[control.joint]] = 100
        report = task.move([-0.001, 0, 0])
        self.assertEqual(report["status"], "failed")
        self.assertIn("Other instrument", report["failure"])
        self.assertLess(report["checked_steps"], report["physics_steps"])
        self.assertIsNone(report["whole_leg_clearance_lower_bound_m"])

    def test_actual_cap_tie_state_registers_the_same_physical_top(self):
        # Reproduce the first production run's actual free-vial pose. Its
        # visible cap and solid collision surrogate share exactly one top.
        s = session()
        task = IsolabInspection(s)
        s.data.qpos[task.addresses] = [
            -0.10400557167477498,
            0.16900050539456066,
            0.14450196075282828,
        ]
        joint = s.model.joint(ID + "__inert_vial_free").id
        address = s.model.jnt_qposadr[joint]
        s.data.qpos[address : address + 7] = [
            -2.5539963410608095,
            1.9489849685788907,
            1.2049978776214183,
            0.999988857795647,
            6.173450578010268e-6,
            -9.540710069677021e-5,
            0.004719655064792434,
        ]
        mujoco.mj_forward(s.model, s.data)
        before = state(s)
        observed = task.observation("vial")
        self.assertTrue(observed["registered"])
        self.assertIn(observed["first_hit"], observed["equivalent_top_surface_geoms"])
        self.assertLess(observed["ray_surface_error_m"], 1e-6)
        self.assertAlmostEqual(observed["standoff_m"], 0.0045, delta=5e-5)
        assert_state(self, before, s)

    def test_free_vial_can_be_lifted_out_by_physical_force(self):
        s = session()
        m, d = s.model, s.data
        body = m.body(ID + "__inert_vial").id
        before = d.xpos[body, 2]
        d.xfrc_applied[body, 2] = m.body_mass[body] * 9.81 * 3
        mujoco.mj_step(m, d, nstep=400)
        mujoco.mj_forward(m, d)
        self.assertGreater(d.xpos[body, 2] - before, 0.06)
        self.assertIn("sample_not_at_target", s.process_readiness(EQUIPMENT)["reasons"])

    def test_observer_model_mutations_fail_with_honest_partial_coverage(self):
        for mutation in ("geometry_closure", "contact_flags"):
            with self.subTest(mutation=mutation):
                s = session()
                task = IsolabInspection(s)

                def corrupt(model, data):
                    if mutation == "geometry_closure":
                        # Also detect a write through the callback's closure.
                        s.model.geom_size[task.vial, 0] *= 0.5
                    else:
                        model.opt.disableflags |= int(
                            mujoco.mjtDisableBit.mjDSBL_CONTACT
                        )

                report = task.move([-0.001, 0, 0], step_observer=corrupt)
                self.assertEqual(report["status"], "failed")
                self.assertIn(
                    "observer mutated the simulation model", report["failure"]
                )
                self.assertEqual(report["physics_steps"], 1)
                self.assertEqual(report["checked_steps"], 0)
                self.assertIsNone(report["whole_leg_clearance_lower_bound_m"])
                self.assertTrue(report["hold_requested"])
                self.assertFalse(report["reset_performed"])
                self.assertEqual(s.data.time, s.model.opt.timestep)

    def test_neutral_observer_preserves_actual_move_dynamics_exactly(self):
        plain = IsolabInspection(session())
        observed = IsolabInspection(session())
        samples = []

        def observe(model, data):
            samples.append((float(data.time), data.qpos.copy()))
            self.assertEqual(model.opt.timestep, 0.0005)

        first = plain.move([-0.001, 0, 0])
        second = observed.move([-0.001, 0, 0], step_observer=observe)
        self.assertEqual(first, second)
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(len(samples), second["checked_steps"])
        self.assertGreater(len(samples), 0)
        np.testing.assert_array_equal(plain.d.qfrc_actuator, observed.d.qfrc_actuator)
        np.testing.assert_array_equal(plain.d.geom_xpos, observed.d.geom_xpos)


if __name__ == "__main__":
    unittest.main()
