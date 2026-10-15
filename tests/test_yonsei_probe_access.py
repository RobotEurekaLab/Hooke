"""Source pitch, open guide mechanics and actual dry pad inspection."""

from copy import copy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.contact_coverage import screen_contact_coverage
from real_labs.motion import _model_signature
from real_labs.runtime import InstrumentSession
from real_labs.yonsei_probe_motion import ProbeInspection


def session():
    root, manifest = build_scene("yonsei_mics_rf_probe")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class YonseiProbeTests(unittest.TestCase):
    def test_original_axes_frames_controls_and_sample_scale(self):
        s = session()
        m = s.model
        for name, frame, limits in [
            ("chuck_x", (0, 0, 0.058), (-0.003, 0.003)),
            ("right_probe_lift", (0, 0, 0), (0, 0.010)),
        ]:
            c = s.controls["rf_probe", name]
            np.testing.assert_allclose(m.body_pos[m.jnt_bodyid[c.joint]], frame)
            np.testing.assert_allclose(m.jnt_range[c.joint], limits)
        np.testing.assert_allclose(m.actuator_gainprm[:, 0], [250, 180])
        np.testing.assert_allclose(m.actuator_forcerange, [[-120, 120], [-120, 120]])
        self.assertEqual(
            s.equipment["rf_probe"]["sample_size_m"], [0.0048, 0.004, 0.0004]
        )
        site = m.site(s.equipment["rf_probe"]["sites"]["rf_coupon"]).id
        np.testing.assert_allclose(m.site_pos[site], (0, 0, 0.0234))
        self.assertEqual(screen_contact_coverage(m)["issues"], [])

    def test_three_real_fine_tips_and_pads_match_source_pitch(self):
        s = session()
        m, d = s.model, s.data
        for side in ("left", "right"):
            ids = [
                m.site(s.equipment["rf_probe"]["sites"][f"{side}_tip_{i}"]).id
                for i in range(3)
            ]
            np.testing.assert_allclose(
                np.diff(d.site_xpos[ids, 1]), [0.0001, 0.0001], atol=1e-12
            )
            for i in range(3):
                g = m.geom("rf_probe__mics_" + side + f"_tip_{i}").id
                self.assertAlmostEqual(m.geom_size[g, 0], 0.000025)
                self.assertEqual((m.geom_contype[g], m.geom_conaffinity[g]), (1, 1))
        for i in range(3):
            pad = m.geom(f"rf_probe__mics_pad_1_{i}").id
            site = m.site(s.equipment["rf_probe"]["sites"][f"pad_1_{i}"]).id
            self.assertAlmostEqual(
                d.site_xpos[site, 2], d.geom_xpos[pad, 2] + m.geom_size[pad, 2]
            )
            die = m.geom("rf_probe__mics_inert_die").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, pad, die, 0.001, None), 1e-9
            )

    def test_all_primary_parts_form_supported_same_body_assemblies(self):
        s = session()
        m, d = s.model, s.data
        for body in range(1, m.nbody):
            if not m.body(body).name.startswith("rf_probe"):
                continue
            ids = [
                i
                for i in range(m.ngeom)
                if m.geom_bodyid[i] == body
                and (m.geom_contype[i] or m.geom_conaffinity[i])
            ]
            reached = {ids[0]}
            pending = set(ids[1:])
            while pending:
                connected = {
                    a
                    for a in pending
                    if any(
                        mujoco.mj_geomDistance(m, d, a, b, 0.00001, None) <= 1e-7
                        for b in reached
                    )
                }
                if not connected:
                    break
                reached.update(connected)
                pending -= connected
            self.assertFalse(pending, [m.geom(i).name for i in pending])

    def test_whole_travel_with_enabled_parent_filtered_interfaces(self):
        s = session()
        task = ProbeInspection(s)
        for x, z in product(np.linspace(-0.003, 0.003, 9), np.linspace(0, 0.010, 9)):
            s.data.qpos[:] = (x, z)
            mujoco.mj_forward(s.model, s.data)
            task._observe(s.model, s.data)
        self.assertEqual(task.checked, 81)
        self.assertGreaterEqual(task.minimum, 0.00025)
        rail = s.model.geom("rf_probe__mics_right_z_rail_0").id
        shoe = s.model.geom("rf_probe__mics_right_z_shoe_0_0_-1").id
        self.assertIn(tuple(sorted((rail, shoe))), set(map(tuple, task.pairs)))

    def test_binding_rejects_essential_geometry_and_optical_metadata(self):
        cases = [
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom("rf_probe__mics_right_tip_1").id, 0
            ),
            lambda s: s.model.geom_conaffinity.__setitem__(
                s.model.geom("rf_probe__mics_right_z_rail_0").id, 0
            ),
            lambda s: setattr(
                s.model.opt, "enableflags", int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
            ),
            lambda s: s.equipment["rf_probe"]["inspection_interface"].__setitem__(
                "gsg_pitch_m", 0.001
            ),
        ]
        for change in cases:
            s = session()
            task = ProbeInspection(s)
            change(s)
            before = s.data.qpos.copy(), s.data.ctrl.copy()
            with self.assertRaises(ValueError):
                task.move((0, 0.001))
            np.testing.assert_array_equal(s.data.qpos, before[0])
            np.testing.assert_array_equal(s.data.ctrl, before[1])
            self.assertEqual(s.data.time, 0)

    def test_foreign_and_replaced_data_are_rejected_without_commands(self):
        s = session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaises(ValueError):
            ProbeInspection(s)
        s = session()
        task = ProbeInspection(s)
        s.data = mujoco.MjData(s.model)
        s.data.ctrl[:] = 0.001
        with self.assertRaises(ValueError):
            task.move((0, 0.001))
        np.testing.assert_array_equal(s.data.ctrl, [0.001, 0.001])
        self.assertEqual(s.data.time, 0)

    def test_real_alignment_and_return_with_every_step_observed(self):
        s = session()
        signature = _model_signature(s.model)
        r = ProbeInspection(s).inspect_and_return()
        self.assertEqual(
            r["status"],
            "succeeded",
            [(x["phase"], x.get("failure")) for x in r["legs"]],
        )
        self.assertTrue(r["observation"]["interface_observed"])
        self.assertEqual(r["physics_steps"], r["fully_checked_steps"])
        self.assertGreater(r["physics_steps"], 1000)
        self.assertLess(max(r["observation"]["projected_tip_pad_errors_m"]), 0.000010)
        self.assertGreater(min(r["observation"]["standoffs_m"]), 0.00075)
        self.assertLess(max(r["return_error_m"]), 0.000010)
        self.assertFalse(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertEqual(_model_signature(s.model), signature)

    def test_external_load_preserves_failed_state_with_partial_coverage(self):
        s = session()
        task = ProbeInspection(s)
        s.data.qfrc_applied[1] = 50
        before = _model_signature(s.model)
        r = task.move((0, 0.002))
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertGreater(
            r["physics_steps"], r["direct_checks"]["fully_checked_steps"]
        )
        self.assertIsNone(r["direct_checks"]["whole_leg_clearance_lower_bound_m"])
        self.assertEqual(s.data.qfrc_applied[1], 50)
        self.assertEqual(_model_signature(s.model), before)

    def test_readonly_abort_remains_failed_after_completed_geometry_step(self):
        s = session()
        states = []

        def abort(m, d):
            states.append((d.qpos.copy(), d.qvel.copy()))
            if len(states) == 5:
                raise RuntimeError("deliberate_readonly_abort")

        r = ProbeInspection(s).move((0, 0.001), step_observer=abort)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["physics_steps"], 5)
        self.assertEqual(r["direct_checks"]["fully_checked_steps"], 5)
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        np.testing.assert_array_equal(s.data.qpos, states[-1][0])
        np.testing.assert_array_equal(s.data.qvel, states[-1][1])


if __name__ == "__main__":
    unittest.main()
