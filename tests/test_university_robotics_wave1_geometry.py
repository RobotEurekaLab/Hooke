"""CPU articulation and sampled collision checks for the licensed Gen3 substitute."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs import university_extensions
from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.layout_audit import audit_layout
from real_labs.motion import JointMotion
from real_labs.manchester_operations import OperationRequired
from real_labs.runtime import InstrumentSession


SCENES = ("manchester_collaborative_robotics",)


def make_session(identifier):
    with patch.object(university_extensions, "MODULES", ("university_robotics_wave1",)):
        university_extensions.instrument_specs.cache_clear()
        try:
            root, manifest = build_scene(identifier)
        finally:
            university_extensions.instrument_specs.cache_clear()
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class UniversityRoboticsWave1GeometryTests(unittest.TestCase):
    def test_extension_kinds_register_alongside_existing_instruments(self):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_robotics_wave1"))
        )
        with patch.object(university_extensions, "MODULES", modules):
            university_extensions.instrument_specs.cache_clear()
            try:
                specs = university_extensions.instrument_specs()
                self.assertIn("manchester_gen3_inspection_workspace", specs)
                for identifier in SCENES:
                    root, manifest = build_scene(identifier)
                    model = mujoco.MjModel.from_xml_string(
                        ET.tostring(root, encoding="unicode")
                    )
                    self.assertGreaterEqual(
                        len(InstrumentSession(model, manifest).controls), 2
                    )
            finally:
                university_extensions.instrument_specs.cache_clear()

    def test_private_axis_response_and_public_operation_interlock(self):
        # Private servo diagnostics retain both signs and original targets.
        # Public requests need the separate collision-checked inspection route.
        for identifier in SCENES:
            session = make_session(identifier)
            self.assertEqual(len(session.controls), 7)
            np.testing.assert_allclose(session.model.opt.gravity, [0, 0, -9.81])
            for key, control in session.controls.items():
                for target in (0.04, -0.04, 0.02):
                    state = mujoco.mjtState.mjSTATE_INTEGRATION
                    before = np.empty(mujoco.mj_stateSize(session.model, state))
                    mujoco.mj_getState(session.model, session.data, before, state)
                    with self.assertRaises(OperationRequired):
                        session.command(*key, target, duration=2)
                    after = np.empty_like(before)
                    mujoco.mj_getState(session.model, session.data, after, state)
                    np.testing.assert_array_equal(after, before)
                    result = session._command_unchecked(*key, target, duration=2)
                    self.assertLessEqual(
                        abs(result["error"]), 0.003, (identifier, result)
                    )
                session.reset()

    def test_source_range_sweeps_retain_obstacles_requiring_motion_planning(self):
        for identifier in SCENES:
            session = make_session(identifier)
            motion = JointMotion(
                session,
                "collaborative_arm",
                tuple(f"joint_{index}_offset" for index in range(1, 8)),
            )
            self.assertFalse(motion.configuration_contacts(motion.measured()))
            blocked, clear = [], 0
            for axis in range(7):
                for value in np.linspace(motion.lower[axis], motion.upper[axis], 21):
                    pose = motion.measured()
                    pose[axis] = value
                    contacts = motion.configuration_contacts(pose)
                    if contacts:
                        blocked.extend(contacts)
                    else:
                        clear += 1
            self.assertGreater(clear, 0)
            self.assertTrue(any("robot_table_top" in row["geoms"] for row in blocked))
            self.assertTrue(
                any(
                    any("gen3_collision" in name for name in row["geoms"])
                    for row in blocked
                )
            )
            # Private screening must not teleport or advance the live model.
            np.testing.assert_array_equal(session.data.qpos, session.model.qpos0)
            self.assertEqual(session.data.time, 0)

    def test_clamped_sites_have_visible_same_body_specimens(self):
        for identifier in SCENES:
            session = make_session(identifier)
            self.assertTrue(audit_layout(scene(identifier))["passed"])
            for item in session.equipment.values():
                self.assertEqual(item["sample_attachment"], "clamped")
                site = session.model.site(item["sample_site"]).id
                body = session.model.site_bodyid[site]
                nearby = [
                    index
                    for index in range(session.model.ngeom)
                    if session.model.geom_bodyid[index] == body
                    and session.model.geom_rgba[index, 3] > 0
                    and np.linalg.norm(
                        session.data.geom_xpos[index] - session.data.site_xpos[site]
                    )
                    <= np.linalg.norm(session.model.geom_size[index]) + 0.002
                ]
                self.assertTrue(nearby, (identifier, item["id"]))


if __name__ == "__main__":
    unittest.main()
