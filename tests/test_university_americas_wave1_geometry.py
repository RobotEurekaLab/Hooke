"""Mechanical reach, support continuity and published tank-scale regressions."""

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


class AmericasWaveOneGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_americas_wave1",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_americas_wave1.json"
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

    def test_richmond_published_clear_tank_dimensions(self):
        session = self.sessions["berkeley_richmond_towing_tank"]
        session.reset()
        sides = sorted(
            (_bounds(session, i) for i in _tagged(session, "__tank_sidewall_")),
            key=lambda pair: pair[0][0],
        )
        ends = sorted(
            (_bounds(session, i) for i in _tagged(session, "__tank_endwall_")),
            key=lambda pair: pair[0][1],
        )
        foundation = _bounds(session, _tagged(session, "__tank_foundation_")[0])
        self.assertAlmostEqual(sides[1][0][0] - sides[0][1][0], 2.40)
        self.assertAlmostEqual(ends[1][0][1] - ends[0][1][1], 64.0)
        self.assertAlmostEqual(sides[0][1][2] - foundation[1][2], 1.80)
        self.assertAlmostEqual(foundation[0][2], 0.0)

    def test_small_carriages_retain_support_through_full_travel(self):
        cases = (
            (
                "yale_becton_cleanroom_metrology",
                "wafer_inspection",
                "stage_x",
                "__stage_bearing_",
            ),
            (
                "yale_becton_cleanroom_metrology",
                "wafer_inspection",
                "stage_y",
                "__stage_cross_guide_",
            ),
            (
                "cmu_edl_electrochemical_testing",
                "electrochemical_test",
                "specimen_drawer",
                "__drawer_guide_",
            ),
        )
        for identifier, equipment, axis, tag in cases:
            session = self.sessions[identifier]
            control = session.controls[equipment, axis]
            body = session.model.body(f"{equipment}__body_{axis}").id
            plate = next(
                i
                for i in range(session.model.ngeom)
                if session.model.geom_bodyid[i] == body
                and session.model.geom_contype[i]
            )
            for fraction in np.linspace(0, 1, 25):
                session.reset()
                session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                    control.lower + fraction * (control.upper - control.lower)
                )
                mujoco.mj_forward(session.model, session.data)
                low, high = _bounds(session, plate)
                guides = [_bounds(session, i) for i in _tagged(session, tag)]
                self.assertEqual(len(guides), 2)
                for a, b in guides:
                    self.assertAlmostEqual(low[2], b[2], places=7)
                    self.assertTrue(
                        np.all(np.minimum(high[:2], b[:2]) > np.maximum(low[:2], a[:2]))
                    )

    def test_richmond_wheels_remain_on_rails(self):
        session = self.sessions["berkeley_richmond_towing_tank"]
        control = session.controls["towing_facility", "carriage_travel"]
        for fraction in np.linspace(0, 1, 25):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                control.upper * fraction
            )
            mujoco.mj_forward(session.model, session.data)
            rails = [_bounds(session, i) for i in _tagged(session, "__carriage_rail_")]
            for index in _tagged(session, "__carriage_wheel_"):
                low, high = _bounds(session, index)
                supports = [
                    (a, b)
                    for a, b in rails
                    if min(high[0], b[0]) > max(low[0], a[0])
                    and min(high[1], b[1]) > max(low[1], a[1])
                ]
                self.assertEqual(len(supports), 1)
                self.assertAlmostEqual(low[2], supports[0][1][2], places=6)

    def test_ferguson_beam_meets_two_crib_caps_and_ram_stays_above_it(self):
        session = self.sessions["texas_ferguson_structural_test_hall"]
        session.reset()
        low, high = _bounds(session, _tagged(session, "__inert_concrete_beam_")[0])
        caps = [_bounds(session, i) for i in _tagged(session, "__beam_crib_cap_")]
        self.assertEqual(len(caps), 2)
        for a, b in caps:
            self.assertAlmostEqual(b[2], low[2], places=6)
            self.assertGreater(min(high[1], b[1]) - max(low[1], a[1]), 0.8)
        control = session.controls["beam_test", "load_ram"]
        session.data.qpos[session.model.jnt_qposadr[control.joint]] = control.upper
        mujoco.mj_forward(session.model, session.data)
        platen = session.model.site("beam_test__site_loading_platen").id
        self.assertGreater(session.data.site_xpos[platen][2], high[2])


if __name__ == "__main__":
    unittest.main()
