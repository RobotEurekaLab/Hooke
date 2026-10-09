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


class AmericasWaveFourGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_americas_wave4",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_americas_wave4.json"
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

    def test_printer_bed_retains_guide_contact_and_nozzle_clearance(self):
        session = self.sessions["nyu_epp_detector_assembly"]
        control = session.controls[("desktop_prototyping", "buildplate_y")]
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
            for index in _tagged(session, "__bed_y_guide_"):
                support_low, support_high = _bounds(session, index)
                self.assertAlmostEqual(support_high[2], low[2])
                self.assertTrue(
                    np.all(support_high[:2] >= low[:2])
                    and np.all(support_low[:2] <= high[:2])
                )
        head = session.controls[("desktop_prototyping", "printhead_x")]
        head_body = session.model.jnt_bodyid[head.joint]
        nozzle_low = min(
            _bounds(session, i)[0][2]
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == head_body
            and session.model.geom_contype[i]
        )
        target_high = max(
            _bounds(session, i)[1][2]
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == body and session.model.geom_contype[i]
        )
        self.assertGreaterEqual(nozzle_low - target_high, 0.023)

    def test_axial_chuck_shaft_stays_in_housing_and_clear_of_coupon(self):
        session = self.sessions["duke_joint_mechanical_testing"]
        upper = session.controls[("biaxial_frame", "axial_chuck_position")]
        lower = session.controls[("biaxial_frame", "lower_chuck_rotation")]
        upper_body = session.model.jnt_bodyid[upper.joint]
        lower_body = session.model.jnt_bodyid[lower.joint]
        upper_geoms = [
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == upper_body
        ]
        lower_geoms = [
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == lower_body
        ]
        for fraction in np.linspace(0, 1, 13):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[upper.joint]] = (
                upper.lower + fraction * (upper.upper - upper.lower)
            )
            mujoco.mj_forward(session.model, session.data)
            self.assertGreaterEqual(
                max(_bounds(session, i)[1][2] for i in upper_geoms), 1.79
            )
            clearance = min(_bounds(session, i)[0][2] for i in upper_geoms) - max(
                _bounds(session, i)[1][2] for i in lower_geoms
            )
            self.assertGreaterEqual(clearance, 0.05)
        bearing = _tagged(session, "__torsion_bearing_")[0]
        chuck = _tagged(session, "__lower_chuck_")[0]
        self.assertAlmostEqual(
            _bounds(session, bearing)[1][2], _bounds(session, chuck)[0][2]
        )

    def test_rotor_load_paths_are_continuous(self):
        session = self.sessions["pennstate_aeroacoustics_rotor_chamber"]
        session.reset()
        # Along each shaft/gantry column, union collider intervals must cover
        # the complete load path; this catches millimetre-sized static gaps.
        for x, y, lo, hi in (
            (-0.70, 0.08, 0, 1.42),
            (0.70, 0.35, 1.59, 2.79),
            (-1.65, 1.22, 0, 2.79),
            (1.65, 1.22, 0, 2.79),
        ):
            intervals = []
            for i in range(session.model.ngeom):
                if (
                    not session.model.geom(i).name.startswith("tandem_rotors__")
                    or not session.model.geom_contype[i]
                ):
                    continue
                low, high = _bounds(session, i)
                if (
                    low[0] <= x <= high[0]
                    and low[1] <= y <= high[1]
                    and high[2] >= lo
                    and low[2] <= hi
                ):
                    intervals.append((max(lo, low[2]), min(hi, high[2])))
            end = lo
            for start, finish in sorted(intervals):
                self.assertLessEqual(start, end + 1e-7, (x, y, intervals))
                end = max(end, finish)
            self.assertGreaterEqual(end, hi - 1e-7)

    def test_charpy_inspection_sector_stays_above_specimen(self):
        session = self.sessions["northwestern_clammp_charpy"]
        control = session.controls[("charpy_frame", "pendulum_inspection_angle")]
        body = session.model.jnt_bodyid[control.joint]
        for fraction in np.linspace(0, 1, 17):
            session.reset()
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = (
                control.lower + fraction * (control.upper - control.lower)
            )
            mujoco.mj_forward(session.model, session.data)
            low = min(
                _bounds(session, i)[0][2]
                for i in range(session.model.ngeom)
                if session.model.geom_bodyid[i] == body
            )
            self.assertGreater(low, 1.80)
        session.reset()


if __name__ == "__main__":
    unittest.main()
