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


class AmericasWaveTwoGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_americas_wave2",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_americas_wave2.json"
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

    def test_published_tunnel_clear_bore(self):
        session = self.sessions["princeton_bam_closed_loop_wind_tunnel"]
        session.reset()
        side = [_bounds(session, i) for i in _tagged(session, "__test_side_panel_")]
        horizontal = [
            _bounds(session, i) for i in _tagged(session, "__test_horizontal_panel_")
        ]
        self.assertEqual(len(side), 6)
        self.assertEqual(len(horizontal), 6)
        for lo, hi in zip(side[::2], side[1::2]):
            self.assertAlmostEqual(hi[0][1] - lo[1][1], 1.2192)
        for lo, hi in zip(horizontal[::2], horizontal[1::2]):
            self.assertAlmostEqual(hi[0][2] - lo[1][2], 1.2192)

    def test_published_container_footprint_and_continuous_support(self):
        session = self.sessions["ucsd_mccartney_geotechnical_centrifuge"]
        session.reset()
        container = _tagged(session, "__container_max_footprint_")[0]
        low, high = _bounds(session, container)
        np.testing.assert_allclose(high[:2] - low[:2], (0.6, 0.7), atol=1e-8)
        body = session.model.geom_bodyid[container]
        supports = [
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == body
            and session.model.geom_contype[i]
            and i != container
        ]
        self.assertTrue(
            any(abs(_bounds(session, i)[1][2] - low[2]) < 1e-8 for i in supports)
        )
        bearing = _bounds(session, _tagged(session, "__rotary_bearing_")[0])
        arm = _bounds(session, _tagged(session, "__rotary_arm_")[0])
        self.assertAlmostEqual(arm[0][2], bearing[1][2])

    def test_rotating_table_platter_seated_on_bearing(self):
        session = self.sessions["mit_paoc_rotating_fluids"]
        control = session.controls["rotating_table", "table_rotation"]
        for fraction in np.linspace(0, 1, 25):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                control.lower + fraction * (control.upper - control.lower)
            )
            mujoco.mj_forward(session.model, session.data)
            bearing = _bounds(session, _tagged(session, "__turntable_bearing_")[0])
            platter = _bounds(session, _tagged(session, "__rotating_platter_")[0])
            self.assertAlmostEqual(platter[0][2], bearing[1][2])

    def test_tundish_floor_meets_continuous_crossbars(self):
        session = self.sessions["mcgill_mmpc_water_modelling"]
        session.reset()
        floor = _tagged(session, "__tank_base_")[0]
        low, high = _bounds(session, floor)
        supports = []
        for i in range(session.model.ngeom):
            if (
                i == floor
                or session.model.geom_bodyid[i] != session.model.geom_bodyid[floor]
                or not session.model.geom_contype[i]
            ):
                continue
            a, b = _bounds(session, i)
            if (
                a[2] < low[2]
                and low[2] <= b[2] <= high[2]
                and min(b[0], high[0]) > max(a[0], low[0])
                and min(b[1], high[1]) > max(a[1], low[1])
            ):
                supports.append(i)
        self.assertGreaterEqual(len(supports), 2)
