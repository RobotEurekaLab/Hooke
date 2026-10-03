"""Mechanical reach, support continuity and published facility-scale regressions."""

import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from real_labs import university_extensions
from real_labs.builder import build_scene
from real_labs.layout_audit import (
    audit_layout,
    doorway_obstacle_warnings,
    model_obstacles,
)
from real_labs.runtime import InstrumentSession


def _bounds(session, index):
    model, data = session.model, session.data
    local = model.geom_aabb[index]
    rotation = data.geom_xmat[index].reshape(3, 3)
    center = data.geom_xpos[index] + rotation @ local[:3]
    radius = np.abs(rotation) @ local[3:]
    return center - radius, center + radius


def _tagged(session, tag):
    return [i for i in range(session.model.ngeom) if tag in session.model.geom(i).name]


class AmericasWaveFiveGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_americas_wave5",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_americas_wave5.json"
        )
        cls.sessions = {}
        for row in json.loads(path.read_text())["scenes"]:
            root, manifest = build_scene(row["id"])
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            cls.sessions[row["id"]] = InstrumentSession(model, manifest)

    @classmethod
    def tearDownClass(cls):
        cls.extension_patch.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_layouts_classify_support_and_keep_door_approaches_clear(self):
        for identifier, session in self.sessions.items():
            with self.subTest(scene=identifier):
                session.reset()
                report = audit_layout(session.manifest)
                self.assertTrue(report["passed"], report)
                self.assertTrue(report["all_equipment_origins_classified"], report)
                self.assertFalse(
                    doorway_obstacle_warnings(
                        session.manifest, model_obstacles(session.model, session.data)
                    )
                )
                for item in session.manifest["equipment"]:
                    self.assertEqual(item["sample_attachment"], "clamped")
                    self.assertGreater(session.model.site(item["sample_site"]).id, -1)

    def test_all_controls_reach_targets_and_return(self):
        for identifier, session in self.sessions.items():
            self.assertGreaterEqual(len(session.controls), 2)
            for key, control in session.controls.items():
                with self.subTest(scene=identifier, control=key):
                    session.reset()
                    span = control.upper - control.lower
                    for fraction in (0.15, 0.85, 0.25):
                        result = session.command(
                            *key, control.lower + span * fraction, duration=2
                        )
                        self.assertLessEqual(
                            abs(result["error"]), max(1e-6, span * 0.025)
                        )
                    session.reset()
                    np.testing.assert_allclose(
                        session.data.qpos, session.model.qpos0, atol=1e-10
                    )

    def test_joint_endpoints_and_intermediate_combinations_do_not_penetrate(self):
        for identifier, session in self.sessions.items():
            controls = list(session.controls.values())
            configurations = list(itertools.product((0, 0.5, 1), repeat=len(controls)))
            # Sweep each axis against both endpoint configurations of its peers.
            for axis in range(len(controls)):
                for side in (0, 1):
                    for fraction in np.linspace(0, 1, 25):
                        row = [side] * len(controls)
                        row[axis] = fraction
                        configurations.append(row)
            for fractions in configurations:
                session.reset()
                for control, fraction in zip(controls, fractions):
                    session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                        control.lower + fraction * (control.upper - control.lower)
                    )
                mujoco.mj_forward(session.model, session.data)
                for contact in session.data.contact:
                    self.assertGreaterEqual(
                        contact.dist,
                        -1e-5,
                        (
                            identifier,
                            fractions,
                            session.model.geom(contact.geom1).name,
                            session.model.geom(contact.geom2).name,
                            contact.dist,
                        ),
                    )

    def test_brown_published_clear_section_dimensions(self):
        session = self.sessions["brown_breuer_afam_tunnel"]
        session.reset()
        floor = _bounds(session, _tagged(session, "__test_floor_")[0])
        roof = _bounds(session, _tagged(session, "__test_roof_")[0])
        self.assertAlmostEqual(floor[1][0] - floor[0][0], 4.0)
        self.assertAlmostEqual(roof[0][2] - floor[1][2], 1.2)
        sides = [_bounds(session, i) for i in _tagged(session, "__test_side_")]
        self.assertEqual(len(sides), 8)
        for front, back in zip(sides[:4], sides[4:]):
            self.assertAlmostEqual(back[0][1] - front[1][1], 1.2)

    def test_ubc_camera_wheels_remain_on_rails(self):
        session = self.sessions["ubc_hassan_legoflume"]
        control = session.controls[("river_flume", "camera_traverse")]
        for fraction in np.linspace(0, 1, 17):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                control.lower + fraction * (control.upper - control.lower)
            )
            mujoco.mj_forward(session.model, session.data)
            for wheel in _tagged(session, "__camera_wheel_"):
                low, high = _bounds(session, wheel)
                supports = []
                for rail in _tagged(session, "__camera_runway_"):
                    rail_low, rail_high = _bounds(session, rail)
                    if np.all(rail_high[:2] >= low[:2]) and np.all(
                        rail_low[:2] <= high[:2]
                    ):
                        self.assertAlmostEqual(low[2], rail_high[2])
                        supports.append(rail)
                self.assertEqual(len(supports), 1)

    def test_boston_exchange_platform_remains_supported(self):
        session = self.sessions["boston_photonics_pml_sem"]
        control = session.controls[("sem_station", "specimen_exchange")]
        body = session.model.jnt_bodyid[control.joint]
        plate = next(
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == body and session.model.geom_contype[i]
        )
        for fraction in np.linspace(0, 1, 13):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                control.lower + fraction * (control.upper - control.lower)
            )
            mujoco.mj_forward(session.model, session.data)
            low, high = _bounds(session, plate)
            for index in _tagged(session, "__sem_drawer_guide_"):
                rail_low, rail_high = _bounds(session, index)
                self.assertAlmostEqual(low[2], rail_high[2])
                self.assertTrue(
                    np.all(rail_high[:2] >= low[:2])
                    and np.all(rail_low[:2] <= high[:2])
                )

    def test_chicago_enclosed_table_rests_on_four_isolators(self):
        session = self.sessions["chicago_mineralphysics_laser_bay"]
        session.reset()
        table_low, table_high = _bounds(
            session, _tagged(session, "__covered_table_")[0]
        )
        isolators = _tagged(session, "__optical_isolator_")
        self.assertEqual(len(isolators), 4)
        for index in isolators:
            low, high = _bounds(session, index)
            self.assertAlmostEqual(high[2], table_low[2])
            self.assertTrue(
                np.all(low[:2] >= table_low[:2]) and np.all(high[:2] <= table_high[:2])
            )
        # The bounded bay deliberately does not invent the whole 1006 ft² room.
        room = session.manifest["room"]["size"]
        self.assertLess(room[0] * room[1], 93.46045824)


if __name__ == "__main__":
    unittest.main()
