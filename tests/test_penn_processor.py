"""Physical cassette routing, real access obstruction and rejection contracts."""

import copy
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs.penn_processor_motion import PennProcessor


class PennProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.manifest = build_scene("penn_pathology")
        cls.xml = ET.tostring(cls.root, encoding="unicode")

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.session = InstrumentSession(self.model, self.manifest)

    def test_free_cassette_and_both_contact_enabled_powered_groups(self):
        model = self.model
        joint = model.joint("sample_processor__cassette__free").id
        body = int(model.jnt_bodyid[joint])
        self.assertEqual(model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.body_jntnum[body], 1)
        self.assertAlmostEqual(model.body_mass[body], 0.06)
        for kind, first, second in zip(model.eq_type, model.eq_obj1id, model.eq_obj2id):
            if kind in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
                self.assertNotIn(body, (first, second))
        for key in ("access_door", "carrier_translation"):
            owner = model.body("sample_processor__body_" + key).id
            ids = np.flatnonzero(model.geom_bodyid == owner)
            self.assertGreater(len(ids), 10)
            self.assertTrue(np.all(model.geom_contype[ids] != 0))
            self.assertTrue(np.all(model.geom_conaffinity[ids] != 0))

    def test_fixed_and_moving_solids_are_connected_and_feet_touch_worktop(self):
        model, data = self.model, self.session.data
        for name in (
            "sample_processor",
            "sample_processor__body_carrier_translation",
            "sample_processor__body_access_door",
        ):
            owner = model.body(name).id
            ids = np.flatnonzero(model.geom_bodyid == owner)
            connected = {int(ids[0])}
            pending = list(connected)
            while pending:
                first = pending.pop()
                for second in ids:
                    if (
                        second not in connected
                        and mujoco.mj_geomDistance(
                            model, data, first, int(second), 1e-7, None
                        )
                        <= 1e-8
                    ):
                        connected.add(int(second))
                        pending.append(int(second))
            self.assertEqual(connected, set(ids), name)
        worktop = model.geom("processing_bench_top").id
        feet = [
            i
            for i in range(model.ngeom)
            if model.geom(i).name.startswith("sample_processor__foot_")
        ]
        self.assertEqual(len(feet), 4)
        for foot in feet:
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(model, data, foot, worktop, 0.001, None),
                0,
                places=9,
            )

    def test_closed_physical_window_blocks_probe_and_open_window_passes(self):
        root = copy.deepcopy(self.root)
        base = self.model.body("sample_processor").id
        d = self.session.data
        rotation = d.xmat[base].reshape(3, 3)
        origin = d.xpos[base].copy()
        pos = origin + rotation @ np.array([0, -0.20, 0.22])
        body = ET.SubElement(
            root.find("worldbody"),
            "body",
            name="access_probe",
            pos=" ".join(map(str, pos)),
            gravcomp="1",
        )
        ET.SubElement(body, "freejoint", name="access_probe_free")
        ET.SubElement(
            body,
            "geom",
            name="access_probe_geom",
            type="sphere",
            size=".005",
            mass=".01",
            contype="1",
            conaffinity="1",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        final = []
        hits = []
        for opened in (False, True):
            data = mujoco.MjData(model)
            door = model.joint("sample_processor__joint_access_door").id
            value = 0.255 if opened else 0.0
            data.qpos[model.jnt_qposadr[door]] = value
            data.ctrl[model.actuator("sample_processor__actuator_access_door").id] = (
                value
            )
            j = model.joint("access_probe_free").id
            data.qvel[model.jnt_dofadr[j] : model.jnt_dofadr[j] + 3] = (
                rotation @ np.array([0, 0.30, 0])
            )
            probe = model.geom("access_probe_geom").id
            contact = False
            for _ in range(175):
                mujoco.mj_step(model, data)
                contact |= any(probe in (c.geom1, c.geom2) for c in data.contact)
            mujoco.mj_forward(model, data)
            final.append(
                float(
                    (rotation.T @ (data.xpos[model.body("access_probe").id] - origin))[
                        1
                    ]
                )
            )
            hits.append(contact)
            self.assertFalse(np.any(data.warning.number))
        self.assertTrue(hits[0])
        self.assertFalse(hits[1])
        self.assertLess(final[0], -0.148)
        self.assertGreater(final[1], -0.120)

    def test_closed_door_and_changed_model_reject_without_state_mutation(self):
        route = PennProcessor(self.session)
        before = [
            a.copy()
            for a in (
                self.session.data.qpos,
                self.session.data.qvel,
                self.session.data.ctrl,
            )
        ]
        with self.assertRaisesRegex(ValueError, "Open access"):
            route.move("carrier_translation", 0.38)
        for old, new in zip(
            before,
            (self.session.data.qpos, self.session.data.qvel, self.session.data.ctrl),
        ):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(self.session.data.time, 0)
        geom = self.model.geom("sample_processor__door_window").id
        self.model.geom_contype[geom] = 0
        with self.assertRaisesRegex(ValueError, "binding changed"):
            route.move("access_door", 0.255)
        for old, new in zip(
            before,
            (self.session.data.qpos, self.session.data.qvel, self.session.data.ctrl),
        ):
            np.testing.assert_array_equal(old, new)

    def test_legacy_microtome_controls_are_explicitly_retired(self):
        state = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(self.model, state))
        mujoco.mj_getState(self.model, self.session.data, before, state)
        for key in ("handwheel", "specimen_feed", "section_stroke", "blade_guard"):
            with self.assertRaisesRegex(KeyError, "Retired.*PennProcessor loading route"):
                self.session.command("sample_processor", key, 0, duration=0.01)
            after = np.empty_like(before)
            mujoco.mj_getState(self.model, self.session.data, after, state)
            np.testing.assert_array_equal(after, before)
        self.assertEqual(self.session.data.time, 0)

    def test_native_reset_is_reported_and_failed_leg_has_no_complete_clearance(self):
        route = PennProcessor(self.session)
        advance = self.session.advance

        def inject_bad_state(steps):
            self.session.data.qpos[route.qadr[0]] = np.nan
            advance(steps)

        self.session.advance = inject_bad_state
        result = route.move("access_door", 0.255)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["reset_performed"])
        self.assertTrue(result["automatic_reset_detected"])
        self.assertFalse(result["explicit_reset_performed"])
        self.assertFalse(result["direct_clearance"]["coverage_complete"])
        self.assertIsNone(result["direct_clearance"]["whole_leg_minimum_m"])

    def test_actual_complete_route_has_physical_loading_and_explicit_separate_reset(
        self,
    ):
        route = PennProcessor(self.session)
        initial = self.session.data.qpos.copy()
        result = route.run()
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["physics_steps"], 9000)
        self.assertFalse(result["reset_performed"])
        self.assertFalse(result["explicit_reset_performed"])
        self.assertFalse(result["automatic_reset_detected"])
        self.assertTrue(result["phases"][2]["specimen"]["inside_chamber"])
        self.assertTrue(result["phases"][2]["specimen"]["access_closed"])
        self.assertTrue(result["phases"][-1]["specimen"]["outside_chamber"])
        for phase in result["phases"]:
            self.assertTrue(phase["direct_clearance"]["coverage_complete"])
            self.assertGreaterEqual(
                phase["direct_clearance"]["whole_leg_minimum_m"], 0.00025
            )
            self.assertGreater(phase["specimen"]["support_contacts"], 0)
            self.assertLess(phase["specimen"]["seat_error_m"], 0.002)
        np.testing.assert_allclose(
            result["measured_axis_return_error_m"], 0, atol=0.0001
        )
        self.assertAlmostEqual(self.session.data.time, 18, places=8)
        self.session.reset()
        self.assertEqual(self.session.data.time, 0)
        np.testing.assert_array_equal(self.session.data.qpos, initial)


if __name__ == "__main__":
    unittest.main()
