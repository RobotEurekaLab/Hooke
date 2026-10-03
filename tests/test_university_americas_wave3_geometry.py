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


class AmericasWaveThreeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_americas_wave3",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_americas_wave3.json"
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

    def test_diagnostic_line_has_continuous_saddles(self):
        session = self.sessions["columbia_fusion_hbt_ep"]
        saddles = [
            _bounds(session, i) for i in _tagged(session, "__diagnostic_saddle_")
        ]
        self.assertEqual(len(saddles), 2)
        static = [
            i
            for i in range(session.model.ngeom)
            if session.model.body_dofnum[session.model.geom_bodyid[i]] == 0
        ]
        for low, high in saddles:
            self.assertAlmostEqual(low[2], 1.015)
            self.assertAlmostEqual(high[2], 1.22)
            # The horizontal stand rail reaches the bottom of each saddle.
            touching = []
            for index in static:
                candidate_low, candidate_high = _bounds(session, index)
                if (
                    abs(candidate_high[2] - low[2]) < 1e-7
                    and np.all(candidate_high[:2] >= low[:2])
                    and np.all(candidate_low[:2] <= high[:2])
                ):
                    touching.append(index)
            self.assertTrue(touching, (low, high))
        # Support beams touch the lower toroidal vessel envelope.
        for index in _tagged(session, "__vessel_saddle_beam_"):
            self.assertAlmostEqual(_bounds(session, index)[1][2], 1.20)

    def test_optical_decks_are_supported_from_below(self):
        for identifier in (
            "harvard_doyle_cold_molecule_optics",
            "illinois_bil_nonlinear_optics",
        ):
            session = self.sessions[identifier]
            for index in _tagged(session, "__optical_breadboard_"):
                low, high = _bounds(session, index)
                supports = []
                for other in range(session.model.ngeom):
                    if other == index or not session.model.geom_contype[other]:
                        continue
                    other_low, other_high = _bounds(session, other)
                    if (
                        abs(other_high[2] - low[2]) < 1e-7
                        and other_high[2] - other_low[2] > 0.10
                        and np.all(other_high[:2] >= low[:2])
                        and np.all(other_low[:2] <= high[:2])
                    ):
                        supports.append(other)
                self.assertGreaterEqual(
                    len(supports), 4, (identifier, session.model.geom(index).name)
                )

    def test_suspended_shelf_rods_reach_ceiling_anchors(self):
        session = self.sessions["illinois_bil_nonlinear_optics"]
        shelf = _tagged(session, "__suspended_utility_shelf_")
        rods = _tagged(session, "__shelf_suspension_")
        self.assertEqual((len(shelf), len(rods)), (1, 4))
        shelf_top = _bounds(session, shelf[0])[1][2]
        for index in rods:
            low, high = _bounds(session, index)
            self.assertAlmostEqual(low[2], shelf_top)
            self.assertAlmostEqual(high[2], 3.068)
        self.assertEqual(session.manifest["room"]["size"][2], 3.1)

    def test_wafer_carriages_keep_bearing_contact_during_translation(self):
        session = self.sessions["michigan_lurie_lithography_bay"]
        for name, tag in (
            ("wafer_x", "__lurie_x_guide_"),
            ("wafer_y", "__lurie_y_guide_"),
        ):
            control = session.controls[("wafer_inspection", name)]
            body = session.model.jnt_bodyid[control.joint]
            # The first collider on each carriage is its load-bearing plate.
            plate = next(
                i
                for i in range(session.model.ngeom)
                if session.model.geom_bodyid[i] == body
                and session.model.geom_contype[i]
            )
            for fraction in np.linspace(0, 1, 9):
                session.reset()
                session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                    control.lower + fraction * (control.upper - control.lower)
                )
                mujoco.mj_forward(session.model, session.data)
                plate_low, plate_high = _bounds(session, plate)
                for index in _tagged(session, tag):
                    low, high = _bounds(session, index)
                    self.assertAlmostEqual(plate_low[2], high[2])
                    self.assertTrue(
                        np.all(high[:2] >= plate_low[:2])
                        and np.all(low[:2] <= plate_high[:2])
                    )


if __name__ == "__main__":
    unittest.main()
