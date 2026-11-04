"""Supported guides, retained sample scale and measured NTU dry inspection."""

from copy import copy
from itertools import product
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.motion import _model_signature
from real_labs.ntu_sgsr_motion import PrecisionInspection
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene("ntu_sgsr_characterization")
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class NtuPrecisionTests(unittest.TestCase):
    def test_original_frames_ranges_and_sites_preserved(self):
        s = session()
        expected = {
            ("mechanical_tester", "crosshead"): ((0, 0.14, 0.96), (-0.25, 0.30)),
            ("mechanical_tester", "upper_grip"): ((0, -0.19, -0.207), (-0.012, 0)),
            ("dimension_icon", "coarse_x"): ((0, -0.115, 0.125), (-0.08, 0.08)),
            ("dimension_icon", "coarse_y"): ((0, 0, 0.028), (-0.06, 0.06)),
            ("dimension_icon", "coarse_head"): ((0, 0.014, 0.38), (-0.025, 0.025)),
        }
        self.assertEqual(set(s.controls), set(expected))
        for key, (pose, limits) in expected.items():
            c = s.controls[key]
            np.testing.assert_allclose(
                s.model.body_pos[s.model.jnt_bodyid[c.joint]], pose
            )
            np.testing.assert_allclose(s.model.jnt_range[c.joint], limits)
            body = s.model.jnt_bodyid[c.joint]
            primary = np.flatnonzero(s.model.geom_bodyid == body)
            self.assertGreater(sum(bool(s.model.geom_contype[g]) for g in primary), 0)
        item = s.equipment["dimension_icon"]
        np.testing.assert_allclose(
            s.model.site_pos[s.model.site(item["sites"]["sample_coupon"]).id],
            (0, 0, 0.021),
        )

    def test_every_primary_assembly_has_a_connected_contact_graph(self):
        s = session()
        m, d = s.model, s.data
        for body in range(1, m.nbody):
            if not m.body(body).name.startswith(
                ("mechanical_tester", "dimension_icon")
            ):
                continue
            ids = [
                i
                for i in range(m.ngeom)
                if m.geom_bodyid[i] == body
                and (m.geom_contype[i] or m.geom_conaffinity[i])
            ]
            self.assertTrue(ids)
            graph = {i: set() for i in ids}
            for index, a in enumerate(ids):
                for b in ids[index + 1 :]:
                    if mujoco.mj_geomDistance(m, d, a, b, 1e-5, None) <= 1e-6:
                        graph[a].add(b)
                        graph[b].add(a)
            reached, todo = {ids[0]}, [ids[0]]
            while todo:
                for b in graph[todo.pop()] - reached:
                    reached.add(b)
                    todo.append(b)
            self.assertEqual(
                set(ids), reached, [m.geom(i).name for i in set(ids) - reached]
            )

    def test_true_probe_dimensions_and_surface_datum(self):
        s = session()
        m, d = s.model, s.data

        def geom(key):
            return m.geom("dimension_icon__ntu_" + key).id

        lever = geom("cantilever")
        np.testing.assert_allclose(
            2 * m.geom_size[lever], (0.000035, 0.000182, 0.000002)
        )
        self.assertAlmostEqual(2 * m.geom_size[geom("probe_tip"), 1], 0.000005)
        sample = geom("sample")
        site = m.site(s.equipment["dimension_icon"]["sites"]["sample_coupon"]).id
        self.assertAlmostEqual(
            d.site_xpos[site, 2], d.geom_xpos[sample, 2] + m.geom_size[sample, 1]
        )
        self.assertAlmostEqual(m.geom_size[geom("chuck"), 0], 0.105)
        c = s.controls["dimension_icon", "coarse_head"]
        d.qpos[m.jnt_qposadr[c.joint]] = c.lower
        mujoco.mj_forward(m, d)
        tip = m.site(s.equipment["dimension_icon"]["sites"]["probe_tip"]).id
        self.assertAlmostEqual(
            d.site_xpos[tip, 2] - d.site_xpos[site, 2], 0.001493, places=7
        )

    def test_filtered_interfaces_clear_through_combined_travel(self):
        s = session()
        for eid in ("dimension_icon", "mechanical_tester"):
            task = PrecisionInspection(s, eid)
            for values in product(
                *[
                    np.linspace(a, b, 3)
                    for a, b in zip(task.motion.lower, task.motion.upper)
                ]
            ):
                s.data.qpos[task.motion.qadr] = values
                mujoco.mj_forward(s.model, s.data)
                task._observe(s.model, s.data)
            s.reset()
            self.assertGreater(task.queries, 0)
            self.assertGreaterEqual(task.minimum, 0.00025)

    def test_changed_essential_contacts_flags_and_foreign_state_rejected(self):
        for name in ("x_plate", "head_chassis", "sample", "probe_tip", "head_rail_0"):
            s = session()
            g = s.model.geom("dimension_icon__ntu_" + name).id
            s.model.geom_contype[g] = s.model.geom_conaffinity[g] = 0
            with self.assertRaisesRegex(ValueError, "binding changed"):
                PrecisionInspection(s)
            self.assertEqual(s.data.time, 0)
        s = session()
        s.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            PrecisionInspection(s)
        s = session()
        s.data = mujoco.MjData(copy(s.model))
        with self.assertRaisesRegex(ValueError, "exact model"):
            PrecisionInspection(s)

    def test_replacement_after_construction_preserves_state_on_rejection(self):
        s = session()
        task = PrecisionInspection(s)
        s.data = mujoco.MjData(copy(s.model))
        s.data.ctrl[:] = 0.003
        before = s.data.ctrl.copy()
        with self.assertRaisesRegex(ValueError, "replaced"):
            task.move((0, 0, 0.005))
        np.testing.assert_array_equal(s.data.ctrl, before)
        self.assertEqual(s.data.time, 0)

    def test_actual_inspection_and_return_for_both_apparatus(self):
        for eid in ("dimension_icon", "mechanical_tester"):
            s = session()
            before = _model_signature(s.model)
            report = PrecisionInspection(s, eid).inspect_and_return()
            self.assertEqual(
                report["status"],
                "succeeded",
                [(x["phase"], x.get("failure")) for x in report["legs"]],
            )
            self.assertEqual(report["physics_steps"], report["fully_checked_steps"])
            self.assertGreater(report["physics_steps"], 1000)
            self.assertTrue(report["observation"]["interface_observed"])
            self.assertFalse(report["reset_performed"])
            self.assertFalse(report["hold_requested"])
            self.assertEqual(_model_signature(s.model), before)
            self.assertLess(max(report["return_error_m"]), 0.0002)
            for leg in report["legs"]:
                self.assertTrue(leg["direct_checks"]["coverage_complete"])
                self.assertGreaterEqual(
                    leg["direct_checks"]["whole_leg_clearance_lower_bound_m"], 0.00025
                )

    def test_external_head_load_fails_with_hold_partial_coverage_no_reset(self):
        s = session()
        task = PrecisionInspection(s)
        c = s.controls["dimension_icon", "coarse_head"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 500
        before = _model_signature(s.model)
        result = task.move((0, 0, 0.004))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["hold_requested"])
        self.assertFalse(result["reset_performed"])
        self.assertGreater(s.data.time, 0)
        self.assertFalse(result["direct_checks"]["coverage_complete"])
        self.assertIsNone(result["direct_checks"]["whole_leg_clearance_lower_bound_m"])
        self.assertEqual(_model_signature(s.model), before)
        self.assertEqual(s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]], 500)

    def test_direct_bearing_observer_rejects_intrusion_not_hidden_by_parent_filter(
        self,
    ):
        s = session()
        task = PrecisionInspection(s)
        g = s.model.geom("dimension_icon__ntu_head_backplate").id
        s.model.geom_pos[g, 1] -= 0.03
        with self.assertRaisesRegex(ValueError, "binding changed"):
            task.move((0, 0, 0.005))
        self.assertEqual(s.data.time, 0)


if __name__ == "__main__":
    unittest.main()
