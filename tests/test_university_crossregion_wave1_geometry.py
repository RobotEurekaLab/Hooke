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


class CrossregionWaveOneGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_crossregion_wave1",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_crossregion_wave1.json"
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

    def test_optical_tables_have_contiguous_isolator_support(self):
        for identifier in ("lmu_hybrid_snom", "heidelberg_jochim_quantum_architecture"):
            session = self.sessions[identifier]
            session.reset()
            table = _tagged(session, "optical_tabletop")[0]
            supports = _tagged(session, "table_isolator")
            self.assertEqual(len(supports), 4)
            for support in supports:
                self.assertAlmostEqual(
                    _bounds(session, support)[1][2], _bounds(session, table)[0][2]
                )

    def test_lmu_dry_probe_stays_above_calibration_coupon(self):
        session = self.sessions["lmu_hybrid_snom"]
        session.reset()
        control = next(
            v for k, v in session.controls.items() if k[1] == "inspection_head_height"
        )
        session.data.qpos[session.model.jnt_qposadr[control.joint]] = control.lower
        mujoco.mj_forward(session.model, session.data)
        tip = _tagged(session, "snom_probe_tip")[0]
        coupon = _tagged(session, "snom_inert_coupon")[0]
        self.assertGreater(
            _bounds(session, tip)[0][2] - _bounds(session, coupon)[1][2], 0.06
        )

    def test_profiler_coupon_is_below_all_objectives(self):
        session = self.sessions["copenhagen_nbi_surface_metrology"]
        session.reset()
        coupon = _tagged(session, "profiler_inert_coupon")[0]
        objectives = _tagged(session, "profiler_objective_tip")
        self.assertEqual(len(objectives), 3)
        for objective in objectives:
            self.assertGreater(
                _bounds(session, objective)[0][2] - _bounds(session, coupon)[1][2], 0.25
            )

    def test_franke_access_plate_remains_on_both_rails(self):
        session = self.sessions["fuberlin_franke_surface_physics"]
        control = next(
            v for k, v in session.controls.items() if k[1] == "holder_extension"
        )
        plate = _tagged(session, "franke_slider_plate")[0]
        rails = _tagged(session, "franke_access_rail")
        self.assertEqual(len(rails), 2)
        for value in (control.lower, control.upper):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(session.model, session.data)
            lower, upper = _bounds(session, plate)
            for rail in rails:
                low, high = _bounds(session, rail)
                self.assertAlmostEqual(lower[2], high[2])
                self.assertLess(low[1], lower[1])
                self.assertGreater(high[1], upper[1])


if __name__ == "__main__":
    unittest.main()
