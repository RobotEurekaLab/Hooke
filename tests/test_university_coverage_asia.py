"""Support, full-stroke clearance and bidirectional operation of six new workcells."""

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
from real_labs.inspection_contacts import InspectionAccess


class UniversityAsiaCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = patch.object(
            university_extensions, "MODULES", ("university_coverage_asia",)
        )
        cls.patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_coverage_asia.json"
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
        cls.patch.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_support_and_operator_approaches(self):
        for name, s in self.sessions.items():
            with self.subTest(scene=name):
                s.reset()
                report = audit_layout(s.manifest)
                self.assertTrue(report["passed"], report)
                self.assertTrue(report["all_equipment_origins_classified"], report)
                self.assertFalse(
                    doorway_obstacle_warnings(
                        s.manifest, model_obstacles(s.model, s.data)
                    )
                )
                item = s.manifest["equipment"][0]
                self.assertEqual(item["sample_attachment"], "clamped")
                self.assertGreaterEqual(s.model.site(item["sample_site"]).id, 0)

    def test_two_controls_move_both_ways_and_reset(self):
        for name, s in self.sessions.items():
            self.assertEqual(len(s.controls), 2)
            for key, c in s.controls.items():
                with self.subTest(scene=name, control=key):
                    s.reset()
                    for f in (0.15, 0.85, 0.25):
                        r = s.command(
                            *key, c.lower + f * (c.upper - c.lower), duration=2
                        )
                        self.assertLessEqual(
                            abs(r["error"]), max(1e-6, 0.025 * (c.upper - c.lower))
                        )
                        self.assertTrue(np.isfinite(s.data.qpos).all())
                    s.reset()
                    np.testing.assert_allclose(s.data.qpos, s.model.qpos0, atol=1e-10)

    def test_full_range_combinations_and_sweeps_are_collision_clear(self):
        for name, s in self.sessions.items():
            controls = list(s.controls.values())
            access = InspectionAccess(
                s,
                name,
                "primary_apparatus",
                [key[1] for key in s.controls],
                minimum_m=2.5e-5,
            )
            configurations = list(itertools.product((0, 0.5, 1), repeat=2))
            for axis in range(2):
                for side in (0, 1):
                    for fraction in np.linspace(0, 1, 21):
                        row = [side, side]
                        row[axis] = fraction
                        configurations.append(row)
            for fractions in configurations:
                s.reset()
                for c, f in zip(controls, fractions):
                    s.data.qpos[s.model.jnt_qposadr[c.joint]] = c.lower + f * (
                        c.upper - c.lower
                    )
                mujoco.mj_forward(s.model, s.data)
                access.check(s.model, s.data)
                for contact in s.data.contact:
                    self.assertGreaterEqual(
                        contact.dist,
                        -1e-5,
                        (
                            name,
                            fractions,
                            s.model.geom(contact.geom1).name,
                            s.model.geom(contact.geom2).name,
                            contact.dist,
                        ),
                    )

    def test_published_platform_footprint_and_fixed_support(self):
        session = self.sessions["monash_geoextremes_seismic_platform"]
        deck = next(
            i
            for i in range(session.model.ngeom)
            if "fixed_seismic_deck" in session.model.geom(i).name
        )
        np.testing.assert_allclose(
            session.model.geom_size[deck, :2] * 2, (2.4, 1.2), atol=1e-9
        )
        body = session.model.geom_bodyid[deck]
        self.assertEqual(session.model.body_jntnum[body], 0)
        session.reset()
        start = session.data.geom_xpos[deck].copy()
        session.command("primary_apparatus", "specimen_translation", 0.06, duration=2)
        np.testing.assert_allclose(session.data.geom_xpos[deck], start, atol=1e-12)

    def test_six_distinct_apparatus_and_scientific_limits(self):
        kinds = set()
        for s in self.sessions.values():
            item = s.manifest["equipment"][0]
            kinds.add(item["kind"])
            self.assertTrue(item["limitations"])
        self.assertEqual(len(kinds), 6)


if __name__ == "__main__":
    unittest.main()
