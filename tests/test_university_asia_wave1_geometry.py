"""CPU motion, collision and specimen-scale checks for four university labs."""

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
    "kyoto_yokokawa_nanometrics",
    "sjtu_deepwater_offshore_basin",
    "kaist_usdl_magnetic_probe",
    "hkust_aaf_wind_tunnel",
)


def make_session(identifier):
    with patch.object(university_extensions, "MODULES", ("university_asia_wave1",)):
        university_extensions.instrument_specs.cache_clear()
        try:
            root, manifest = build_scene(identifier)
        finally:
            university_extensions.instrument_specs.cache_clear()
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class UniversityAsiaWave1GeometryTests(unittest.TestCase):
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

    def test_kyoto_slide_and_target_keep_microscopy_scale(self):
        session = make_session("kyoto_yokokawa_nanometrics")
        item = session.equipment["tirf_workspace"]
        self.assertEqual(item["sample_size_m"], [0.002, 0.002, 0.00016])
        body = session.model.body("tirf_workspace__body_slide_y").id
        sizes = session.model.geom_size[session.model.geom_bodyid == body]
        self.assertTrue(any(np.allclose(s, [0.0375, 0.0125, 0.0005]) for s in sizes))

    def test_probe_platform_posts_meet_the_breadboard_and_plate(self):
        session = make_session("kaist_usdl_magnetic_probe")
        model, data = session.model, session.data
        origin = data.xpos[model.body("moke_probe_workspace").id]
        for x in (-0.34, 0.34):
            for y in (-0.28, 0.28):
                supporting = [
                    index
                    for index in range(model.ngeom)
                    if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and np.allclose(data.geom_xpos[index, :2], origin[:2] + (x, y))
                ]
                self.assertEqual(len(supporting), 1)
                index = supporting[0]
                center = data.geom_xpos[index, 2] - origin[2]
                half = model.geom_size[index, 1]
                self.assertAlmostEqual(center - half, 0.036)
                self.assertGreaterEqual(center + half, 0.346)


if __name__ == "__main__":
    unittest.main()
