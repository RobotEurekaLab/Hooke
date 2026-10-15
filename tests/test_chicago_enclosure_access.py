"""Versioned roof closure, connected open bearings, and actual inert inspection."""

from copy import copy
import itertools
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.chicago_enclosure_geometry import PREFIX
from real_labs.chicago_enclosure_motion import (
    SCENE,
    EQUIPMENT,
    CONTROLS,
    EnclosureInspection,
    inspect_chicago_enclosure,
)
from real_labs.runtime import InstrumentSession


def load(revision=2):
    definition = scene(SCENE)
    definition["equipment"][0]["params"]["enclosure_revision"] = revision
    with patch("real_labs.builder.scene", return_value=definition):
        root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return (
        s.data.time,
        *(
            getattr(s.data, n).tobytes()
            for n in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            )
        ),
    )


class ChicagoEnclosureTests(unittest.TestCase):
    def test_explicit_revision_preserves_frames_drives_ranges_and_specimen(self):
        old, new = load(1), load()
        for key in CONTROLS:
            a, b = [s.controls[EQUIPMENT, key] for s in (old, new)]
            for field in ("jnt_pos", "jnt_axis", "jnt_range"):
                np.testing.assert_array_equal(
                    getattr(old.model, field)[a.joint],
                    getattr(new.model, field)[b.joint],
                )
            for field in (
                "body_pos",
                "body_quat",
                "body_mass",
                "body_inertia",
                "body_gravcomp",
            ):
                np.testing.assert_array_equal(
                    getattr(old.model, field)[old.model.jnt_bodyid[a.joint]],
                    getattr(new.model, field)[new.model.jnt_bodyid[b.joint]],
                )
            for field in (
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
                "actuator_ctrlrange",
            ):
                np.testing.assert_array_equal(
                    getattr(old.model, field)[a.actuator],
                    getattr(new.model, field)[b.actuator],
                )
            self.assertEqual(
                old.model.dof_damping[old.model.jnt_dofadr[a.joint]],
                new.model.dof_damping[new.model.jnt_dofadr[b.joint]],
            )
        np.testing.assert_array_equal(
            new.model.geom(PREFIX + "inert_coupon").size, [0.020, 0.016, 0.010]
        )
        for field in ("site_pos", "site_size"):
            np.testing.assert_array_equal(
                getattr(old.model, field)[
                    old.model.site(EQUIPMENT + "__site_mineral_coupon").id
                ],
                getattr(new.model, field)[
                    new.model.site(EQUIPMENT + "__site_mineral_coupon").id
                ],
            )
        meta = new.equipment[EQUIPMENT]["mechanical_interface"]
        self.assertEqual(meta["version"], 2)
        self.assertIn("No automatic remapping", meta["migration"])
        self.assertNotEqual(old.model.ngeom, new.model.ngeom)

    def test_actual_closed_roof_and_real_open_access(self):
        s = load()
        t = EnclosureInspection(s)
        m, d = s.model, s.data
        self.assertFalse(t.observe()["coupon_visible"])
        self.assertEqual(t.observe()["camera_first_hit"], PREFIX + "lid_sheet")
        self.assertAlmostEqual(t.observe()["physical_open_angle_rad"], 0)
        self.assertTrue(m.geom(PREFIX + "retained_front_panel_1").contype)
        np.testing.assert_allclose(
            d.geom_xpos[m.geom(PREFIX + "lid_sheet").id, 2]
            + m.geom(PREFIX + "lid_sheet").size[2],
            1.410,
        )
        d.qpos[0] = -0.6
        mujoco.mj_forward(m, d)
        self.assertTrue(t.observe()["coupon_visible"])
        self.assertAlmostEqual(t.observe()["physical_open_angle_rad"], 0.6)
        self.assertGreater(t.observe()["cover_front_world_m"][2], 1.9)
        for x in (-0.07, 0, 0.07):
            d.qpos[1] = x
            mujoco.mj_forward(m, d)
            o = t.observe()
            self.assertTrue(o["coupon_visible"])
            self.assertAlmostEqual(o["coupon_index_plane_error_m"], x)

    def test_connected_supports_and_open_hinge_stage_bores(self):
        s = load()
        m, d = s.model, s.data

        def gap(a, b):
            return mujoco.mj_geomDistance(
                m, d, m.geom(PREFIX + a).id, m.geom(PREFIX + b).id, 0.1, None
            )

        for a, b in [
            ("stage_base", "deck"),
            ("datum_tab", "stage_base"),
            ("rear_header", "rear_overlap_lip"),
            ("rear_header", "header_crossbeam"),
            ("header_crossbeam", "side_wall_1"),
            ("header_crossbeam", "side_wall_-1"),
        ]:
            self.assertLessEqual(gap(a, b), 1e-8, (a, b))
        for sign in (-1, 1):
            for end in (-1, 1):
                self.assertLessEqual(
                    gap(f"rail_support_{sign}_{end}", "stage_base"), 1e-8
                )
                self.assertLessEqual(
                    gap(f"rail_support_{sign}_{end}", f"stage_rail_{sign}"), 1e-8
                )
                self.assertLessEqual(
                    gap(f"hinge_fixed_ear_{sign}_{end}", "rear_header"), 1e-8
                )
                self.assertLessEqual(
                    gap(f"hinge_fixed_ear_{sign}_{end}", f"hinge_pin_{sign}"), 1e-8
                )
            self.assertLessEqual(gap(f"bearing_crown_{sign}", "stage_plate"), 1e-8)
            for part, pin in (
                (f"hinge_knuckle_{sign}", f"hinge_pin_{sign}"),
                (f"stage_bearing_{sign}", f"stage_rail_{sign}"),
            ):
                proxies = [
                    g
                    for g in range(m.ngeom)
                    if m.geom(g).name.startswith(
                        EQUIPMENT + "__chicago_v2_" + part + "_proxy_"
                    )
                ]
                self.assertEqual(len(proxies), 48)
                self.assertTrue(
                    all(m.geom_contype[g] and m.geom_conaffinity[g] for g in proxies)
                )
                self.assertAlmostEqual(
                    min(
                        mujoco.mj_geomDistance(
                            m, d, g, m.geom(PREFIX + pin).id, 0.002, None
                        )
                        for g in proxies
                    ),
                    0.0006,
                    delta=1e-8,
                )
            # The entire moving sleeve remains inside the supported rail span.
            for x in (-0.07, 0.07):
                self.assertLessEqual(abs(x) + 0.040, 0.22)
            leaf = m.geom(PREFIX + f"hinge_moving_leaf_{sign}").id
            proxies = [
                g
                for g in range(m.ngeom)
                if f"chicago_v2_hinge_knuckle_{sign}_proxy_" in m.geom(g).name
            ]
            self.assertLessEqual(
                min(mujoco.mj_geomDistance(m, d, leaf, g, 0.01, None) for g in proxies),
                1e-8,
            )
        self.assertAlmostEqual(
            gap("lid_front_fold", "front_overlap_lip"), 0.0006, delta=1e-8
        )

    def test_full_coupled_range_retains_parent_filtered_pairs(self):
        s = load()
        t = EnclosureInspection(s)
        m, d = s.model, s.data
        pairnames = {frozenset((m.geom(a).name, m.geom(b).name)) for a, b in t.pairs}
        self.assertIn(
            frozenset((PREFIX + "hinge_pin_1", PREFIX + "hinge_knuckle_1_proxy_0")),
            pairnames,
        )
        self.assertIn(
            frozenset((PREFIX + "stage_rail_1", PREFIX + "stage_bearing_1_proxy_0")),
            pairnames,
        )
        for lid, x in itertools.product(
            np.linspace(-0.6, 0, 9), np.linspace(-0.07, 0.07, 9)
        ):
            d.qpos[:] = [lid, x]
            mujoco.mj_forward(m, d)
            t.check(m, d)
            self.assertFalse(any(c.dist < -1e-8 for c in d.contact))
        self.assertGreaterEqual(t.minimum, 0.0005999)

    def test_actual_inspection_has_complete_coverage_and_no_reset(self):
        s = load()
        report = inspect_chicago_enclosure(s)
        self.assertEqual(report["status"], "succeeded", report["phases"][-1])
        self.assertEqual(report["executed_steps"], report["checked_steps"])
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["reset_performed"])
        self.assertGreaterEqual(report["whole_run_clearance_lower_bound_m"], 0.000599)
        self.assertGreater(s.data.time, 10)
        np.testing.assert_allclose(s.data.qpos, [0, 0], atol=0.0001)
        for o in report["observations"]:
            self.assertEqual(len(o["qpos"]), s.model.nq)
            self.assertEqual(o["coupon_visible"], o["phase"] != "close_roof")
            if o["phase"] == "register_inert_coupon":
                self.assertLess(abs(o["coupon_index_plane_error_m"]), 0.0002)
        for p in report["phases"]:
            held = 1 if p["phase"] in ("open_roof", "close_roof") else 0
            drift = max(
                abs(v["measured"][held] - v["commanded"][held]) for v in p["telemetry"]
            )
            self.assertLess(drift, 1e-4)

    def test_changed_or_foreign_interface_rejects_without_mutation(self):
        changes = (
            lambda s: s.model.geom_size.__setitem__(
                (s.model.geom(PREFIX + "hinge_pin_1").id, 0), 0.01
            ),
            lambda s: s.model.geom_contype.__setitem__(
                s.model.geom(PREFIX + "stage_plate").id, 0
            ),
            lambda s: s.model.opt.__setattr__(
                "disableflags", int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            ),
            lambda s: s.equipment[EQUIPMENT]["mechanical_interface"].__setitem__(
                "version", 1
            ),
            lambda s: setattr(s, "data", mujoco.MjData(copy(s.model))),
        )
        for change in changes:
            s = load()
            change(s)
            before = state(s)
            with self.assertRaises(ValueError):
                EnclosureInspection(s)
            self.assertEqual(state(s), before)
        for replace_data in (False, True):
            s = load()
            t = EnclosureInspection(s)
            if replace_data:
                s.data = mujoco.MjData(s.model)
            else:
                s.model.geom_pos[s.model.geom(PREFIX + "stage_base").id, 0] += 0.1
            before = state(s)
            with self.assertRaises(ValueError):
                t.move([-0.1, 0])
            self.assertEqual(state(s), before)
        s = load()
        s.data.qpos[0] = -0.2
        mujoco.mj_forward(s.model, s.data)
        before = state(s)
        with self.assertRaises(ValueError):
            inspect_chicago_enclosure(s)
        self.assertEqual(state(s), before)

    def test_external_load_failure_retains_state_and_partial_coverage(self):
        s = load()
        t = EnclosureInspection(s)
        c = s.controls[EQUIPMENT, "access_lid_angle"]
        v = int(s.model.jnt_dofadr[c.joint])
        s.data.qfrc_applied[v] = -100
        start = s.data.qpos.copy()
        r = t.move([-0.1, 0])
        self.assertEqual(r["status"], "failed")
        self.assertTrue(r["hold_requested"])
        self.assertFalse(r["reset_performed"])
        self.assertGreater(r["physics_steps"], r["direct_clearance"]["checked_steps"])
        self.assertIsNone(r["direct_clearance"]["whole_leg_clearance_lower_bound_m"])
        self.assertFalse(np.array_equal(s.data.qpos, start))
        self.assertGreater(s.data.time, 0)
        self.assertAlmostEqual(
            s.data.ctrl[c.actuator],
            np.clip(s.data.qpos[s.model.jnt_qposadr[c.joint]], c.lower, c.upper),
        )


if __name__ == "__main__":
    unittest.main()
