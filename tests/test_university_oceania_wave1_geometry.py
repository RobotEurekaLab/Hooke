"""CPU motion, collision and specimen-scale checks for three Oceania university apparatus bays."""

from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs import university_extensions
from real_labs.builder import build_scene
from real_labs.catalog import scene
from real_labs.layout_audit import audit_layout
from real_labs.runtime import InstrumentSession


SCENES = (
    "anu_climate_fluid_physics",
    "sydney_fluids_convection",
    "unsw_tyree_xray",
)


def make_session(identifier):
    with patch.object(university_extensions, "MODULES", ("university_oceania_wave1",)):
        university_extensions.instrument_specs.cache_clear()
        try:
            root, manifest = build_scene(identifier)
        finally:
            university_extensions.instrument_specs.cache_clear()
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class UniversityOceaniaWave1GeometryTests(unittest.TestCase):
    def test_extension_kinds_register_alongside_existing_instruments(self):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_oceania_wave1"))
        )
        with patch.object(university_extensions, "MODULES", modules):
            university_extensions.instrument_specs.cache_clear()
            try:
                specs = university_extensions.instrument_specs()
                self.assertIn("coretom_positioning_station", specs)
                for identifier in SCENES:
                    root, manifest = build_scene(identifier)
                    model = mujoco.MjModel.from_xml_string(
                        ET.tostring(root, encoding="unicode")
                    )
                    self.assertGreaterEqual(
                        len(InstrumentSession(model, manifest).controls), 2
                    )
            finally:
                university_extensions.instrument_specs.cache_clear()

    def test_axes_reach_three_targets_in_both_directions(self):
        for identifier in SCENES:
            session = make_session(identifier)
            self.assertGreaterEqual(len(session.controls), 2)
            np.testing.assert_allclose(session.model.opt.gravity, [0, 0, -9.81])
            for key, control in session.controls.items():
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

    def test_full_stroke_grid_and_individual_sweeps_do_not_penetrate(self):
        for identifier in SCENES:
            session = make_session(identifier)
            controls = list(session.controls.values())
            # Combined interior poses catch interactions that endpoint-only checks miss.
            poses = list(product(*[np.linspace(c.lower, c.upper, 5) for c in controls]))
            for axis, control in enumerate(controls):
                for value in np.linspace(control.lower, control.upper, 21):
                    pose = [0.0] * len(controls)
                    pose[axis] = value
                    poses.append(pose)
            for pose in poses:
                session.reset()
                for control, value in zip(controls, pose):
                    session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
                mujoco.mj_forward(session.model, session.data)
                penetration = [
                    (
                        session.model.geom(contact.geom1).name,
                        session.model.geom(contact.geom2).name,
                        float(contact.dist),
                    )
                    for contact in session.data.contact
                    if contact.dist < -0.0001
                ]
                self.assertEqual(penetration, [], (identifier, pose, penetration))

    def test_clamped_sites_have_visible_same_body_specimens(self):
        for identifier in SCENES:
            session = make_session(identifier)
            self.assertTrue(audit_layout(scene(identifier))["passed"])
            for item in session.equipment.values():
                self.assertEqual(item["sample_attachment"], "clamped")
                site = session.model.site(item["sample_site"]).id
                body = session.model.site_bodyid[site]
                nearby = [
                    index
                    for index in range(session.model.ngeom)
                    if session.model.geom_bodyid[index] == body
                    and session.model.geom_rgba[index, 3] > 0
                    and np.linalg.norm(
                        session.data.geom_xpos[index] - session.data.site_xpos[site]
                    )
                    <= np.linalg.norm(session.model.geom_size[index]) + 0.002
                ]
                self.assertTrue(nearby, (identifier, item["id"]))

    def test_published_coretom_separation_and_visible_rotation(self):
        session = make_session("unsw_tyree_xray")
        source = session.model.site("coretom_sample_station__site_source_aperture").id
        detector = session.model.site("coretom_sample_station__site_detector_plane").id
        self.assertAlmostEqual(
            np.linalg.norm(
                session.data.site_xpos[source] - session.data.site_xpos[detector]
            ),
            0.970,
        )
        body = session.model.body("coretom_sample_station__body_sample_rotation").id
        seam = [
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == body
            and np.allclose(session.model.geom_size[i], [0.003, 0.0005, 0.10])
        ]
        self.assertEqual(len(seam), 1)
        control = session.controls["coretom_sample_station", "sample_rotation"]
        positions = []
        for value in (control.lower, control.upper):
            session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
            mujoco.mj_forward(session.model, session.data)
            positions.append(session.data.geom_xpos[seam[0]].copy())
        self.assertGreater(np.linalg.norm(positions[1] - positions[0]), 0.030)

    def test_annulus_remains_hollow_with_published_depth(self):
        session = make_session("anu_climate_fluid_physics")
        body = session.model.body("large_rotating_annulus__body_tank_rotation").id
        panels = [
            i
            for i in range(session.model.ngeom)
            if session.model.geom_bodyid[i] == body
            and np.allclose(session.model.geom_size[i][[0, 2]], [0.01, 0.20])
        ]
        self.assertEqual(len(panels), 40)
        for geom in panels:
            self.assertAlmostEqual(
                np.linalg.norm(session.model.geom_pos[geom, :2])
                + session.model.geom_size[geom, 0],
                0.8,
            )
            self.assertAlmostEqual(session.model.geom_size[geom, 2] * 2, 0.4)
        # The non-colliding water is explicitly a visual mesh, not a solid obstacle.
        liquid = session.model.geom("large_rotating_annulus__annular_liquid").id
        self.assertEqual(session.model.geom_contype[liquid], 0)
        self.assertEqual(session.model.geom_conaffinity[liquid], 0)


if __name__ == "__main__":
    unittest.main()
