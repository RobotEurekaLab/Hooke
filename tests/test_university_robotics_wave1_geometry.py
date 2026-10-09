"""CPU articulation and sampled collision checks for the licensed Gen3 substitute."""

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


SCENES = ("manchester_collaborative_robotics",)


def make_session(identifier):
    with patch.object(university_extensions, "MODULES", ("university_robotics_wave1",)):
        university_extensions.instrument_specs.cache_clear()
        try:
            root, manifest = build_scene(identifier)
        finally:
            university_extensions.instrument_specs.cache_clear()
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class UniversityRoboticsWave1GeometryTests(unittest.TestCase):
    def test_extension_kinds_register_alongside_existing_instruments(self):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_robotics_wave1"))
        )
        with patch.object(university_extensions, "MODULES", modules):
            university_extensions.instrument_specs.cache_clear()
            try:
                specs = university_extensions.instrument_specs()
                self.assertIn("manchester_gen3_inspection_workspace", specs)
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
            # Seven-joint corners plus pairwise grids and full individual strokes.
            # This is a sampled collision check, not a continuous workspace proof.
            poses = list(product(*[(c.lower, c.upper) for c in controls]))
            for first in range(len(controls)):
                for second in range(first + 1, len(controls)):
                    for a, b in product(
                        np.linspace(controls[first].lower, controls[first].upper, 5),
                        np.linspace(controls[second].lower, controls[second].upper, 5),
                    ):
                        pose = [0.0] * len(controls)
                        pose[first], pose[second] = a, b
                        poses.append(pose)
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


if __name__ == "__main__":
    unittest.main()
