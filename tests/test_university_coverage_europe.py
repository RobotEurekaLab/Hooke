"""Mechanical coverage of six reference-informed European workstations."""

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


class EuropeCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = patch.object(
            university_extensions, "MODULES", ("university_coverage_europe",)
        )
        cls.registration.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_coverage_europe.json"
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
        cls.registration.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_bidirectional_servo_targets_and_reset(self):
        for identifier, session in self.sessions.items():
            self.assertGreaterEqual(len(session.controls), 2)
            for key, control in session.controls.items():
                with self.subTest(scene=identifier, axis=key):
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

    def test_swept_control_clearance(self):
        for identifier, session in self.sessions.items():
            controls = list(session.controls.values())
            states = list(itertools.product((0, 0.5, 1), repeat=len(controls)))
            for axis in range(len(controls)):
                for side in (0, 1):
                    for fraction in np.linspace(0, 1, 25):
                        row = [side] * len(controls)
                        row[axis] = fraction
                        states.append(row)
            for fractions in states:
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

    def test_supported_equipment_and_entry_routes(self):
        for identifier, session in self.sessions.items():
            with self.subTest(scene=identifier):
                session.reset()
                self.assertTrue(audit_layout(session.manifest)["passed"])
                self.assertEqual(
                    doorway_obstacle_warnings(
                        session.manifest, model_obstacles(session.model, session.data)
                    ),
                    [],
                )
                equipment = session.manifest["equipment"][0]
                self.assertEqual(equipment["sample_attachment"], "clamped")
                site = session.model.site(equipment["sample_site"]).id
                self.assertGreater(
                    session.model.body_geomnum[session.model.site_bodyid[site]], 0
                )

    def test_controls_move_actual_interfaces(self):
        for identifier, session in self.sessions.items():
            session.reset()
            initial = session.data.site_xpos.copy()
            for key, control in session.controls.items():
                session.command(
                    *key,
                    control.lower + 0.75 * (control.upper - control.lower),
                    duration=2
                )
            self.assertGreater(
                np.max(np.linalg.norm(session.data.site_xpos - initial, axis=1)),
                0.005,
                identifier,
            )

    def test_published_durham_turntable_diameter(self):
        session = self.sessions["durham_2m_wind_tunnel"]
        geom = session.model.geom("primary_apparatus__published_1650mm_turntable").id
        self.assertAlmostEqual(session.model.geom_size[geom, 0] * 2, 1.65)

    def test_published_durham_section_and_nozzle_offset(self):
        session = self.sessions["durham_2m_wind_tunnel"]
        m = session.model
        left = m.geom("primary_apparatus__nozzle_left").id
        right = m.geom("primary_apparatus__nozzle_right").id
        top = m.geom("primary_apparatus__nozzle_top").id
        floor = m.geom("primary_apparatus__test_floor").id
        self.assertAlmostEqual(
            (m.geom_pos[right, 0] - m.geom_size[right, 0])
            - (m.geom_pos[left, 0] + m.geom_size[left, 0]),
            1.75,
        )
        self.assertAlmostEqual(
            (m.geom_pos[top, 2] - m.geom_size[top, 2])
            - (m.geom_pos[floor, 2] + m.geom_size[floor, 2]),
            1.15,
        )
        self.assertAlmostEqual(m.geom_size[floor, 1] * 2, 5.5)
        self.assertAlmostEqual(m.geom_pos[left, 1] - m.geom_size[left, 1], 1.8)


if __name__ == "__main__":
    unittest.main()
