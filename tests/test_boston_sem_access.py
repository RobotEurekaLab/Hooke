"""Revision-three physical SEM access and honest checked-operation boundaries."""

import itertools
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import boston_sem_motion
from real_labs.boston_sem_motion import (
    CONTROLS,
    EQUIPMENT,
    SCENE,
    StageInspection,
    SpecimenAccess,
)
from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession


def session():
    root, manifest = build_scene(SCENE)
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


def state(s):
    return (
        float(s.data.time),
        *(
            getattr(s.data, f).tobytes()
            for f in (
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            )
        ),
    )


class BostonSemAccessTests(unittest.TestCase):
    def test_explicit_revision_and_retired_actions_with_true_stub_datum(self):
        s = session()
        item = s.equipment[EQUIPMENT]
        self.assertEqual(item["mechanism_revision"]["version"], 3)
        self.assertFalse(item["mechanism_revision"]["automatic_legacy_mapping"])
        self.assertEqual(set(s.controls), {(EQUIPMENT, key) for key in CONTROLS})
        before = state(s)
        for old in ("specimen_exchange", "chamber_access", "specimen_azimuth"):
            with self.assertRaises(KeyError):
                s.command(EQUIPMENT, old, 0.1)
            self.assertEqual(state(s), before)
        with self.assertRaisesRegex(ValueError, "retired"):
            SpecimenAccess(s)
        self.assertEqual(state(s), before)
        self.assertEqual(s.model.opt.timestep, 0.001)
        self.assertEqual(s.model.opt.iterations, 100)
        stub = s.model.geom(EQUIPMENT + "__retained_stub").id
        np.testing.assert_allclose(s.model.geom_size[stub, :2], [0.015, 0.006])
        site = s.model.site(item["sample_site"]).id
        np.testing.assert_allclose(
            s.data.site_xpos[site],
            s.data.geom_xpos[stub] + s.data.geom_xmat[stub].reshape(3, 3)[:, 2] * 0.006,
            atol=1e-12,
        )
        self.assertEqual(item["sample_attachment"], "clamped")
        np.testing.assert_array_equal(s.model.geom_solref[stub], [0.02, 1])

    def test_full_individual_strokes_and_blocked_coupled_workspace(self):
        s = session()
        task = StageInspection(s)
        for axis in range(7):
            for value in np.linspace(task.lower[axis], task.upper[axis], 13):
                s.reset()
                s.data.qpos[task.addresses[axis]] = value
                mujoco.mj_forward(s.model, s.data)
                self.assertGreaterEqual(task.geometry(s.data)[0], 0.00025)
        s.reset()
        target = np.array([0, 0, 0.065, 0.065, 0.05, 0, 0])
        result = task.leg("known_obstruction", target)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["physics_steps"], 0)
        self.assertTrue(result["hold_requested"])
        self.assertIsNone(result["whole_leg_clearance_lower_bound_m"])
        # The physical ranges remain unchanged despite the obstructed corner.
        np.testing.assert_allclose(task.upper[:5], [0.21, 0.21, 0.065, 0.065, 0.05])

    def test_unculled_pairs_and_physical_surface_in_world_placed_room(self):
        s = session()
        task = StageInspection(s)
        m, d = s.model, s.data
        enabled = np.flatnonzero(m.geom_contype | m.geom_conaffinity)
        weld = m.body_weldid[m.geom_bodyid]
        pairs = [
            (int(a), int(b))
            for a, b in itertools.combinations(enabled, 2)
            if weld[a] != weld[b]
        ]
        for pose in ((0.137, 0.071, 0, 0, 0, 0, 0), (0, 0, 0.011, -0.007, 0.045, 0, 0)):
            d.qpos[task.addresses] = pose
            mujoco.mj_forward(m, d)
            full = min(
                task.cap,
                min(
                    mujoco.mj_geomDistance(m, d, a, b, task.cap, None) for a, b in pairs
                ),
            )
            self.assertAlmostEqual(full, task.geometry(d)[0], places=12)
        d.qpos[task.addresses] = [0, 0, 0.003, -0.002, 0.045, 0, 0]
        mujoco.mj_forward(m, d)
        observed = task.observation((0.003, -0.002))
        self.assertTrue(observed["registered"])
        self.assertAlmostEqual(observed["physical_axial_gap_m"], 0.008, places=12)
        self.assertEqual(observed["first_hit"], EQUIPMENT + "__retained_stub")

    def test_connected_contact_groups_include_real_open_bearings(self):
        s = session()
        m, d = s.model, s.data
        ids = [
            int(g)
            for g in np.flatnonzero(m.geom_contype | m.geom_conaffinity)
            if (
                m.geom(int(g)).name.startswith(EQUIPMENT + "__")
                or m.geom(int(g)).name.startswith(EQUIPMENT + "_exterior__")
                or m.geom(int(g)).name.startswith(EQUIPMENT + "_chamber_observation_")
            )
            or m.geom(int(g)).name == "arch_floor"
        ]
        weld = m.body_weldid[m.geom_bodyid]
        for group in set(weld[ids]):
            parts = [g for g in ids if weld[g] == group]
            adjacency = {g: set() for g in parts}
            for a, b in itertools.combinations(parts, 2):
                if mujoco.mj_geomDistance(m, d, a, b, 0.003, None) <= 1e-7:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            reached, pending = {parts[0]}, [parts[0]]
            while pending:
                extra = adjacency[pending.pop()] - reached
                reached.update(extra)
                pending.extend(extra)
            self.assertEqual(
                reached, set(parts), [m.geom(g).name for g in set(parts) - reached]
            )
        a, b = (
            m.geom(EQUIPMENT + "__retained_stub").id,
            m.geom(EQUIPMENT + "__pole_lower").id,
        )
        self.assertGreater(
            min(mujoco.mj_geomDistance(m, d, a, b, 0.003, None), 0.003), 1e-7
        )

    def test_strict_limits_and_changed_bindings_reject_unchanged(self):
        s = session()
        task = StageInspection(s)
        for axis in range(7):
            for boundary, direction in (
                (task.lower[axis], -np.inf),
                (task.upper[axis], np.inf),
            ):
                target = np.zeros(7)
                target[axis] = np.nextafter(boundary, direction)
                before = state(s)
                with self.assertRaises(ValueError):
                    task.leg("invalid", target)
                self.assertEqual(state(s), before)
        for change in ("foreign", "mask", "drive", "revision"):
            s = session()
            task = StageInspection(s)
            if change == "foreign":
                s.data = mujoco.MjData(session().model)
            elif change == "mask":
                g = s.model.geom(EQUIPMENT + "__retained_stub")
                g.contype = g.conaffinity = 0
            elif change == "drive":
                s.model.actuator_gainprm[0, 0] *= 2
            else:
                s.equipment[EQUIPMENT]["mechanism_revision"]["version"] = 2
            before = state(s)
            with self.assertRaises(ValueError):
                task.leg("invalid_binding", np.zeros(7))
            self.assertEqual(state(s), before)

    def test_changed_chamber_observation_source_rejects_before_motion(self):
        s = session()
        task = StageInspection(s)
        before = state(s)
        original_sha = boston_sem_motion.sha

        def changed_sha(path):
            if Path(path).name == "boston_sem_observation.py":
                return "changed_observation_source"
            return original_sha(path)

        with patch.object(boston_sem_motion, "sha", side_effect=changed_sha):
            with self.assertRaises(ValueError):
                task.leg("changed_observation_source", np.zeros(7))
        self.assertEqual(state(s), before)

    def test_native_failure_and_ambiguous_clock_propagate_through_public_task(self):
        for native in (False, True):
            s = session()
            task = StageInspection(s)
            s.data.time = 0.7 if native else 1e30
            if native:
                s.data.qfrc_applied[task.dofs[0]] = 1e25
            report = task.run()
            phase = report["phases"][0]
            self.assertEqual(report["status"], "failed")
            self.assertEqual((phase["physics_steps"], phase["checked_steps"]), (1, 0))
            self.assertIs(report["reset_performed"], True if native else None)
            self.assertFalse(report["explicit_reset_performed"])
            self.assertTrue(phase["hold_requested"])
            self.assertIsNone(report["whole_run_clearance_lower_bound_m"])

    def test_invalid_native_history_and_nonfinite_clock_reject_before_motion(self):
        for label in ("history", "nan", "inf"):
            s = session()
            task = StageInspection(s)
            if label == "history":
                s.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
            else:
                s.data.time = float(label)
            before = state(s)
            with self.assertRaises(ValueError):
                task.run()
            self.assertEqual(state(s)[1:], before[1:])
            self.assertEqual(task.executed, 0)

    def test_held_force_has_partial_honest_coverage_and_no_reset(self):
        s = session()
        task = StageInspection(s)
        s.data.qfrc_applied[task.dofs[1]] = -160
        report = task.leg("held_axis_disturbance", np.zeros(7))
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["physics_steps"], report["checked_steps"])
        self.assertTrue(report["hold_requested"])
        self.assertFalse(report["reset_performed"])
        self.assertIsNone(report["whole_leg_clearance_lower_bound_m"])

    def test_actual_short_stage_movement_returns_without_reset(self):
        s = session()
        task = StageInspection(s)
        target = np.zeros(7)
        target[4] = 0.001
        for q in (target, np.zeros(7)):
            report = task.leg("short_observed_motion", q)
            self.assertEqual(report["status"], "succeeded", report["failure"])
            self.assertEqual(report["physics_steps"], report["checked_steps"])
            self.assertFalse(report["reset_performed"])
        self.assertLess(abs(task.measured()[4]), 1e-5)
        self.assertGreater(s.data.time, 4)


if __name__ == "__main__":
    unittest.main()
