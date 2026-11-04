"""Versioned saw access: supported geometry and actual inert contact/return."""

from collections import Counter
from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.copenhagen_saw_geometry import PREFIX
from real_labs.copenhagen_saw_motion import (
    SCENE,
    SawClearance,
    SawRegistration,
    core_blade_gap,
    register_copenhagen_core,
)
from real_labs.runtime import InstrumentSession


def load(revision=2):
    definition = scene(SCENE)
    definition["equipment"][0]["params"]["interface_revision"] = revision
    with patch("real_labs.builder.scene", return_value=definition):
        root, manifest = build_scene(SCENE)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return InstrumentSession(model, manifest), root


class CopenhagenSawInterfaceTests(unittest.TestCase):
    def test_explicit_migration_preserves_guard_drive_and_core_scale(self):
        old, _ = load(1)
        new, _ = load()
        self.assertNotIn(("ice_station", "inert_core_carriage"), new.controls)
        a, b = [s.controls["ice_station", "wheel_guard"] for s in (old, new)]
        for field in ("jnt_pos", "jnt_axis", "jnt_range"):
            np.testing.assert_array_equal(
                getattr(old.model, field)[a.joint], getattr(new.model, field)[b.joint]
            )
        for field in ("body_pos", "body_quat", "body_mass", "body_inertia"):
            np.testing.assert_array_equal(
                getattr(old.model, field)[old.model.jnt_bodyid[a.joint]],
                getattr(new.model, field)[new.model.jnt_bodyid[b.joint]],
            )
        for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
            np.testing.assert_array_equal(
                getattr(old.model, field)[a.actuator],
                getattr(new.model, field)[b.actuator],
            )
        m = new.model
        c = new.controls["ice_station", "core_feed_v2"]
        np.testing.assert_allclose(
            m.body_pos[m.jnt_bodyid[c.joint]], [0.205, -0.225, 0.92]
        )
        np.testing.assert_allclose(m.jnt_range[c.joint], [0, 0.152])
        np.testing.assert_allclose(
            m.geom(PREFIX + "inert_core").size[:2], [0.034, 0.066]
        )
        np.testing.assert_allclose(m.geom(PREFIX + "inert_core").pos, [0, 0, 0.048])
        self.assertEqual(m.body_mass[m.jnt_bodyid[c.joint]], 0.18)
        meta = new.equipment["ice_station"]["mechanical_interface"]
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["supersedes_control"], "inert_core_carriage")
        self.assertIn("No coordinate remapping", meta["migration"])
        self.assertEqual(new.equipment["ice_station"]["sample_attachment"], "clamped")

    def test_guard_and_carrier_have_connected_real_mounts(self):
        s, _ = load()
        m, d = s.model, s.data
        for suffix in ("guard", "feed_v2"):
            body = m.body(
                "ice_station__body_"
                + ("wheel_guard" if suffix == "guard" else "core_feed_v2")
            ).id
            ids = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == body and m.geom_contype[g]
            ]
            edges = {g: set() for g in ids}
            for a, b in itertools.combinations(ids, 2):
                if mujoco.mj_geomDistance(m, d, a, b, 1e-7, None) <= 1e-8:
                    edges[a].add(b)
                    edges[b].add(a)
            seen, todo = {ids[0]}, [ids[0]]
            while todo:
                for item in edges[todo.pop()] - seen:
                    seen.add(item)
                    todo.append(item)
            self.assertEqual(seen, set(ids), [m.geom(g).name for g in set(ids) - seen])
        pin = m.geom(PREFIX + "hinge_pin").id
        for sign in (-1, 1):
            first = m.geom(PREFIX + f"hinge_leaf_a_{sign}").id
            second = m.geom(PREFIX + f"hinge_leaf_b_{sign}").id
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, pin, first, 0.1, None), 1e-8
            )
            self.assertLessEqual(
                mujoco.mj_geomDistance(m, d, first, second, 0.1, None), 1e-8
            )
            # The rear leaf reaches the left fixed housing jamb.
            self.assertGreater(m.geom_pos[second, 0] + m.geom_size[second, 0], -0.271)
            pad = m.geom(PREFIX + f"retaining_pad_{sign}").id
            clamp = m.geom(PREFIX + f"retaining_clamp_{sign}").id
            self.assertAlmostEqual(m.geom_pos[pad, 2] - m.geom_size[pad, 2], 0.014)
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(m, d, pad, clamp, 0.01, None), 0, delta=1e-8
            )
            self.assertGreaterEqual(
                core_blade_gap(m, d, m.geom(PREFIX + "inert_core").id, clamp), -1e-8
            )
        for feed in (0, 0.151):
            self.assertGreater(-0.225 + feed - 0.024, -0.295 + 0.012)
            self.assertLess(-0.225 + feed + 0.024, -0.030 - 0.012)

    def test_actual_kerfs_are_manifold_and_not_filled_by_collision_hulls(self):
        s, root = load()
        m, d = s.model, s.data
        for key in ("table", "carrier"):
            mesh = root.find(f"./asset/mesh[@name='{PREFIX + key}_mesh']")
            faces = np.fromstring(mesh.get("face"), dtype=int, sep=" ").reshape(-1, 3)
            edges = Counter(
                tuple(sorted((a, b)))
                for face in faces
                for a, b in zip(face, np.roll(face, -1))
            )
            self.assertEqual(set(edges.values()), {2})
            geom = m.geom(PREFIX + key + "_surface").id
            self.assertEqual((m.geom_contype[geom], m.geom_conaffinity[geom]), (0, 0))
            pieces = [
                g
                for g in range(m.ngeom)
                if m.geom(g).name.startswith(PREFIX + key + "_contact_")
            ]
            self.assertEqual(len(pieces), 5)
            self.assertTrue(
                all(m.geom_contype[g] == m.geom_conaffinity[g] == 1 for g in pieces)
            )
        core, blade = [
            m.geom(PREFIX + key).id for key in ("inert_core", "stationary_blade")
        ]
        for feed, gap in ((0, 0.151), (0.15, 0.001), (0.151, 0)):
            d.qpos[:] = [0, feed]
            mujoco.mj_forward(m, d)
            self.assertAlmostEqual(core_blade_gap(m, d, core, blade), gap, delta=1e-8)
        # The core rests on the plate instead of penetrating the old top by 1 mm.
        self.assertAlmostEqual(m.geom_pos[core, 2] - m.geom_size[core, 0], 0.014)

    def test_screening_matches_unculled_direct_pairs_at_off_grid_poses(self):
        s, _ = load()
        m, d = s.model, s.data
        checker = SawClearance(m)
        self.assertGreater(len(checker.pairs), 30000)
        for angle, feed in ((0, 0), (1.1, 0.151), (0.367, 0.043), (0.861, 0.119)):
            d.qpos[:] = [angle, feed]
            mujoco.mj_forward(m, d)
            checker.check(d)
            minimum = checker.cap
            for a, b in checker.pairs:
                if frozenset((a, b)) != checker.allowed:
                    minimum = min(
                        minimum,
                        mujoco.mj_geomDistance(m, d, int(a), int(b), checker.cap, None),
                    )
            self.assertGreaterEqual(minimum, 0.0005999)
            self.assertAlmostEqual(minimum, checker.minimum_seen, delta=1e-8)

    def test_real_contact_registration_and_return_have_complete_coverage(self):
        s, _ = load()
        start = s.data.qpos.copy()
        report = register_copenhagen_core(s)
        self.assertEqual(report["status"], "succeeded", report["failure"])
        self.assertGreater(report["executed_steps"], 10000)
        self.assertEqual(report["executed_steps"], report["completed_checks"])
        self.assertTrue(report["coverage_complete"])
        contact = next(
            p for p in report["phases"] if p["phase"] == "register_stationary_blade"
        )
        self.assertGreater(contact["normal_force_N"], 0.1)
        self.assertLess(contact["normal_force_N"], 0.3)
        self.assertLess(contact["force_on_core_world_N"][1], -0.1)
        self.assertGreater(contact["core_blade_gap_m"], -0.0001)
        self.assertLess(contact["state"]["qpos"][1], 0.1511)
        self.assertEqual(contact["command_rad_m"][1], 0.152)
        self.assertTrue(
            all(p["success"] and p["coverage_complete"] for p in report["phases"])
        )
        self.assertLess(report["maximum_core_blade_penetration_m"], 0.0001)
        self.assertGreaterEqual(
            report["whole_route_unintended_clearance_lower_bound_m"], 0.0005999
        )
        self.assertGreater(len(report["telemetry"]), 1000)
        np.testing.assert_allclose(s.data.qpos, start, atol=1e-6)
        self.assertFalse(report["state_reset"])

    def test_force_failure_preserves_state_and_truthful_partial_coverage(self):
        s, _ = load()
        c = s.controls["ice_station", "core_feed_v2"]
        s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]] = 100
        report = register_copenhagen_core(s)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure"], "saw_tracking_limit")
        self.assertEqual(report["executed_steps"], report["completed_checks"] + 1)
        self.assertFalse(report["coverage_complete"])
        self.assertIsNone(report["whole_route_unintended_clearance_lower_bound_m"])
        self.assertTrue(report["failure_hold_requested"])
        self.assertFalse(report["state_reset"])
        self.assertGreater(s.data.time, 0)
        self.assertNotEqual(s.data.qpos[1], 0)
        self.assertEqual(s.data.qfrc_applied[s.model.jnt_dofadr[c.joint]], 100)
        self.assertFalse(report["phases"][-1]["success"])

    def test_foreign_state_changed_contract_and_replacement_reject_without_mutation(
        self,
    ):
        original, root = load()
        variants = []
        for key, attribute, value in (
            (PREFIX + "hinge_pin", "pos", "-.035 -.023 1.48"),
            (PREFIX + "stationary_blade", "contype", "0"),
        ):
            changed = deepcopy(root)
            changed.find(f".//geom[@name='{key}']").set(attribute, value)
            variants.append(
                InstrumentSession(
                    mujoco.MjModel.from_xml_string(
                        ET.tostring(changed, encoding="unicode")
                    ),
                    original.manifest,
                )
            )
        foreign, _ = load()
        foreign.data = original.data
        variants.append(foreign)
        altered, _ = load()
        altered.equipment["ice_station"]["mechanical_interface"][
            "nominal_contact_feed_m"
        ] = 0.075
        variants.append(altered)
        for s in variants:
            before = (
                s.data.qpos.copy(),
                s.data.qvel.copy(),
                s.data.ctrl.copy(),
                s.data.time,
            )
            with self.assertRaises(ValueError):
                SawRegistration(s)
            for expected, actual in zip(
                before, (s.data.qpos, s.data.qvel, s.data.ctrl, s.data.time)
            ):
                np.testing.assert_array_equal(expected, actual)
        s, _ = load()
        runner = SawRegistration(s)
        old = s.data
        s.data = mujoco.MjData(s.model)
        with self.assertRaises(ValueError):
            runner.run()
        self.assertEqual(old.time, 0)
        self.assertEqual(s.data.time, 0)

        s, _ = load()
        s.data.qpos[0] = 1.2
        before = s.data.qpos.copy(), s.data.ctrl.copy()
        with self.assertRaises(ValueError):
            register_copenhagen_core(s)
        np.testing.assert_array_equal(s.data.qpos, before[0])
        np.testing.assert_array_equal(s.data.ctrl, before[1])
        self.assertEqual(s.data.time, 0)

    def test_added_obstacle_is_not_removed_by_parent_filter_or_screening(self):
        s, root = load()
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            name="retained_obstacle",
            type="box",
            pos=".205 1.045 .968",
            size=".01 .01 .01",
            contype="1",
            conaffinity="1",
        )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(m)
        d.qpos[:] = [0, 0.07]
        mujoco.mj_forward(m, d)
        checker = SawClearance(m)
        pair = frozenset((checker.core, m.geom("retained_obstacle").id))
        self.assertIn(pair, {frozenset(p) for p in checker.pairs})
        with self.assertRaisesRegex(
            RuntimeError, "saw_forbidden_contact|saw_direct_clearance"
        ):
            checker.check(d)


if __name__ == "__main__":
    unittest.main()
