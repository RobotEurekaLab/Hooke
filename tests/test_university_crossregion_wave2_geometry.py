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


class CrossregionWaveTwoGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_crossregion_wave2",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_crossregion_wave2.json"
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

    def test_spectrometer_lift_keeps_telescoping_support(self):
        session = self.sessions["kfupm_superconductivity_spectroscopy"]
        shaft = _tagged(session, "spectrometer_lift_shaft")[0]
        sleeve = _tagged(session, "spectrometer_lift_sleeve")[0]
        control = next(v for k, v in session.controls.items() if k[1] == "sample_lift")
        for value in (control.lower, control.upper):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(session.model, session.data)
            self.assertLess(
                _bounds(session, shaft)[0][2], _bounds(session, sleeve)[1][2]
            )
            self.assertGreater(
                _bounds(session, shaft)[1][2], _bounds(session, sleeve)[0][2]
            )

    def test_thermobalance_inspection_tray_remains_supported(self):
        session = self.sessions["science_tokyo_thermal_storage_metrology"]
        plate = _tagged(session, "thermobalance_tray_plate")[0]
        rails = _tagged(session, "thermobalance_access_rail")
        control = next(
            v for k, v in session.controls.items() if k[1] == "dry_tray_extension"
        )
        for value in (control.lower, control.upper):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(session.model, session.data)
            lower, upper = _bounds(session, plate)
            for rail in rails:
                low, high = _bounds(session, rail)
                self.assertAlmostEqual(lower[2], high[2])
                self.assertGreater(high[1], upper[1])
                self.assertLess(low[1], lower[1])

    def test_deenergized_fixture_uses_grounded_rails(self):
        session = self.sessions["kfupm_high_voltage_insulation"]
        plate = _tagged(session, "insulation_fixture_plate")[0]
        rails = _tagged(session, "insulation_fixture_rail")
        control = next(v for k, v in session.controls.items() if k[1] == "fixture_x")
        for value in (control.lower, control.upper):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(session.model, session.data)
            lower, upper = _bounds(session, plate)
            for rail in rails:
                low, high = _bounds(session, rail)
                self.assertAlmostEqual(lower[2], high[2])
                self.assertLess(low[0], lower[0])
                self.assertGreater(high[0], upper[0])


if __name__ == "__main__":
    unittest.main()
