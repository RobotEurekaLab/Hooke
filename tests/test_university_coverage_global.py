"""Bounds, support, truthful provenance and commands for the six added schools."""

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
from real_labs.inspection_contacts import InspectionAccess
from real_labs.layout_audit import (
    audit_layout,
    doorway_obstacle_warnings,
    model_obstacles,
)
from real_labs.runtime import InstrumentSession

PATH = (
    Path(__file__).resolve().parents[1]
    / "Hooke/real_labs/scenes/university_coverage_global.json"
)


class UniversityCoverageGlobalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_patch = patch.object(
            university_extensions, "MODULES", ("university_coverage_global",)
        )
        cls.module_patch.start()
        university_extensions.instrument_specs.cache_clear()
        cls.rows = json.loads(PATH.read_text())["scenes"]
        cls.sessions = {}
        for row in cls.rows:
            root, manifest = build_scene(row["id"])
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            # Include parent-child contacts: guide bores must be genuinely clear.
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
            cls.sessions[row["id"]] = InstrumentSession(model, manifest)

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_sources_and_scope_remain_explicit(self):
        self.assertEqual(len(self.rows), 6)
        for row in self.rows:
            with self.subTest(scene=row["id"]):
                self.assertEqual(row["gravity"], [0, 0, -9.81])
                self.assertEqual(
                    row["layout_fidelity"], "estimated_from_public_reference"
                )
                self.assertFalse(row["verified_spatial_dimensions"])
                self.assertTrue(row["layout_evidence"])
                for evidence in row["layout_evidence"]:
                    self.assertTrue(evidence["reviewed"])
                    self.assertEqual(len(evidence["sha256"]), 64)
                self.assertGreaterEqual(len(self.sessions[row["id"]].controls), 2)
        uba = next(row for row in self.rows if row["id"] == "uba_fiuba_chemistry_2008")
        self.assertIn("2008", uba["title"])
        self.assertIn("Historical", " ".join(uba["limitations"]))

    def test_layout_support_and_entry_approaches(self):
        for identifier, session in self.sessions.items():
            with self.subTest(scene=identifier):
                report = audit_layout(session.manifest)
                self.assertTrue(report["passed"], report)
                self.assertTrue(report["all_equipment_origins_classified"], report)
                self.assertFalse(
                    doorway_obstacle_warnings(
                        session.manifest, model_obstacles(session.model, session.data)
                    )
                )
                for equipment in session.manifest["equipment"]:
                    self.assertEqual(equipment["sample_attachment"], "clamped")
                    self.assertGreaterEqual(
                        session.model.site(equipment["sample_site"]).id, 0
                    )

    def test_full_bounded_motion_has_no_moving_body_penetrations(self):
        for identifier, session in self.sessions.items():
            controls = list(session.controls.values())
            fractions = list(itertools.product((0, 0.5, 1), repeat=len(controls)))
            for axis in range(len(controls)):
                for other in (0, 1):
                    for fraction in np.linspace(0, 1, 21):
                        row = [other] * len(controls)
                        row[axis] = fraction
                        fractions.append(row)
            for row in fractions:
                session.reset()
                for control, fraction in zip(controls, row):
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
                            row,
                            session.model.geom(contact.geom1).name,
                            session.model.geom(contact.geom2).name,
                            contact.dist,
                        ),
                    )

    def test_controls_track_and_reset_without_teleporting(self):
        for identifier, session in self.sessions.items():
            for key, control in session.controls.items():
                with self.subTest(scene=identifier, control=key):
                    session.reset()
                    span = control.upper - control.lower
                    for fraction in (0.2, 0.8, 0.4):
                        result = session.command(
                            *key, control.lower + span * fraction, duration=2
                        )
                        registration = 0.0002 if control.units == "m" else 0.003
                        self.assertLessEqual(abs(result["error"]), registration, result)
                        self.assertTrue(np.isfinite(session.data.qpos).all())
                    session.reset()
                    np.testing.assert_allclose(
                        session.data.qpos, session.model.qpos0, atol=1e-10
                    )

    def test_direct_geometry_preserves_bearing_clearance(self):
        for row in self.rows:
            with self.subTest(scene=row["id"]):
                root, manifest = build_scene(row["id"])
                model = mujoco.MjModel.from_xml_string(
                    ET.tostring(root, encoding="unicode")
                )
                session = InstrumentSession(model, manifest)
                equipment = manifest["equipment"][0]["id"]
                keys = [key for owner, key in session.controls if owner == equipment]
                access = InspectionAccess(
                    session, row["id"], equipment, keys, minimum_m=0.000025
                )
                for fractions in itertools.product((0, 0.5, 1), repeat=len(keys)):
                    session.reset()
                    for key, fraction in zip(keys, fractions):
                        control = session.controls[(equipment, key)]
                        session.data.qpos[model.jnt_qposadr[control.joint]] = (
                            control.lower + fraction * (control.upper - control.lower)
                        )
                    mujoco.mj_forward(model, session.data)
                    # Direct queries include parent-child pairs and reject
                    # touching faces, even when the solver filters them out.
                    access.check(model, session.data)
                self.assertEqual(access.checked, 3 ** len(keys))


if __name__ == "__main__":
    unittest.main()
