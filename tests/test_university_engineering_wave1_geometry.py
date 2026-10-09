"""Dry-mechanism reach, clearance and source-scale checks for four test bays."""

from itertools import product
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


class EngineeringWaveOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = patch.object(
            university_extensions, "MODULES", ("university_engineering_wave1",)
        )
        cls.registry.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_engineering_wave1.json"
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
        cls.registry.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_motion_reaches_targets_and_resets(self):
        for identifier, session in self.sessions.items():
            for key, control in session.controls.items():
                session.reset()
                span = control.upper - control.lower
                for fraction in (0.15, 0.85, 0.25):
                    result = session.command(
                        *key, control.lower + fraction * span, duration=2
                    )
                    self.assertLessEqual(
                        abs(result["error"]),
                        max(1e-6, span * 0.025),
                        (identifier, result),
                    )
                session.reset()
                np.testing.assert_allclose(
                    session.data.qpos, session.model.qpos0, atol=1e-10
                )

    def test_swept_and_combined_motion_clearances(self):
        for identifier, session in self.sessions.items():
            controls = list(session.controls.values())
            poses = list(product(*(np.linspace(c.lower, c.upper, 5) for c in controls)))
            for axis, control in enumerate(controls):
                for value in np.linspace(control.lower, control.upper, 21):
                    for side in ("lower", "upper"):
                        pose = [getattr(c, side) for c in controls]
                        pose[axis] = value
                        poses.append(pose)
            for pose in poses:
                session.reset()
                for c, value in zip(controls, pose):
                    session.data.qpos[session.model.jnt_qposadr[c.joint]] = value
                mujoco.mj_forward(session.model, session.data)
                bad = [
                    (
                        session.model.geom(c.geom1).name,
                        session.model.geom(c.geom2).name,
                        float(c.dist),
                    )
                    for c in session.data.contact
                    if c.dist < -0.0001
                ]
                self.assertEqual(bad, [], (identifier, list(pose), bad))

    def test_door_approaches_and_retained_targets(self):
        for identifier, session in self.sessions.items():
            session.reset()
            self.assertTrue(audit_layout(session.manifest)["passed"], identifier)
            self.assertEqual(
                doorway_obstacle_warnings(
                    session.manifest, model_obstacles(session.model, session.data)
                ),
                [],
                identifier,
            )
            for item in session.manifest["equipment"]:
                self.assertEqual(item["sample_attachment"], "clamped")
                site = session.model.site(item["sample_site"])
                self.assertTrue(
                    any(
                        session.model.geom_contype[i]
                        for i in np.flatnonzero(
                            session.model.geom_bodyid == site.bodyid
                        )
                    )
                )

    def test_auckland_published_cross_section_and_local_scope(self):
        session = self.sessions["auckland_aerodynamics_tunnel"]
        geoms = {
            tag: [
                session.model.geom(i)
                for i in range(session.model.ngeom)
                if "__" + tag + "_" in session.model.geom(i).name
            ]
            for tag in ("test_floor", "test_roof", "test_side")
        }
        floor, roof = geoms["test_floor"][0], geoms["test_roof"][0]
        self.assertAlmostEqual(
            roof.pos[2] - roof.size[2] - floor.pos[2] - floor.size[2], 2.5
        )
        sides = sorted(geoms["test_side"], key=lambda g: g.pos[1])
        self.assertAlmostEqual(
            sides[1].pos[1] - sides[1].size[1] - sides[0].pos[1] - sides[0].size[1], 3.6
        )
        self.assertAlmostEqual(floor.size[0] * 2, 6.0)


if __name__ == "__main__":
    unittest.main()
