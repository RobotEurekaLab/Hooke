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
    "kyoto_kawayama_terahertz",
    "cityu_rogach_photoluminescence",
    "polyu_materials_tribology",
    "cuhk_mpil_ultrafast_optics",
)


def make_session(identifier):
    with patch.object(university_extensions, "MODULES", ("university_asia_wave3",)):
        university_extensions.instrument_specs.cache_clear()
        try:
            root, manifest = build_scene(identifier)
        finally:
            university_extensions.instrument_specs.cache_clear()
    return InstrumentSession(
        mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), manifest
    )


class UniversityAsiaWave3GeometryTests(unittest.TestCase):
    def test_extension_kinds_register_alongside_existing_instruments(self):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_asia_wave3"))
        )
        with patch.object(university_extensions, "MODULES", modules):
            university_extensions.instrument_specs.cache_clear()
            try:
                specs = university_extensions.instrument_specs()
                self.assertIn("terahertz_alignment_workspace", specs)
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
                targets = [control.lower + f * span for f in (0.15, 0.85, 0.25)]
                if (
                    identifier == "polyu_materials_tribology"
                    and key[1] == "probe_height"
                ):
                    # Revision3 has real contact within the unchanged physical
                    # travel. This is a free-motion servo smoke, not load testing.
                    targets = [0.02, 0.065, -0.02]
                for target in targets:
                    result = session.command(*key, target, duration=2)
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
            if identifier == "polyu_materials_tribology":
                # The lower joint region now physically reaches the specimen.
                # Its deliberate obstruction is tested below and in the actual
                # loaded task; the free-motion grid keeps5 mm tip clearance.
                poses = list(
                    (x, z, 0, 0, 0, 0)
                    for x, z in product(
                        np.linspace(-0.065, 0.065, 5), np.linspace(-0.035, 0.08, 5)
                    )
                )
            for pose in poses:
                session.reset()
                for control, value in zip(controls, pose):
                    session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
                if identifier == "polyu_materials_tribology":
                    # Private clearance grid: place the free coupon on the
                    # translated support. Actual retention is tested by stepping.
                    address = session.model.joint(
                        "umt_access__free_coupon_joint"
                    ).qposadr[0]
                    session.data.qpos[address] += pose[0]
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

    def test_sample_sites_have_visible_specimens_with_declared_attachment(self):
        for identifier in SCENES:
            session = make_session(identifier)
            self.assertTrue(audit_layout(scene(identifier))["passed"])
            for item in session.equipment.values():
                site = session.model.site(item["sample_site"]).id
                if identifier == "polyu_materials_tribology":
                    self.assertEqual(item["sample_attachment"], "supported")
                    spec = item["existing_free_specimen"]
                    body = session.model.body(spec["body"]).id
                    self.assertEqual(
                        session.model.jnt_type[
                            session.model.joint(spec["freejoint"]).id
                        ],
                        mujoco.mjtJoint.mjJNT_FREE,
                    )
                else:
                    self.assertEqual(item["sample_attachment"], "clamped")
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

    def test_tribometer_revision3_reaches_contact_before_lower_joint_limit(self):
        session = make_session("polyu_materials_tribology")
        model, data = session.model, session.data
        height = session.controls["umt_station", "probe_height"]
        np.testing.assert_array_equal(model.jnt_range[height.joint], [-0.10, 0.08])
        tip = model.geom("umt_station__polyu_counterface").id
        coupon = model.geom("umt_station__polyu_coupon").id
        for value, expected in ((0, 0.04), (-0.04, 0), (-0.043, -0.003)):
            data.qpos[model.jnt_qposadr[height.joint]] = value
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(
                mujoco.mj_geomDistance(model, data, tip, coupon, 0.05, None),
                expected,
                delta=1e-8,
            )
        self.assertTrue(
            any(
                {c.geom1, c.geom2} == {tip, coupon} and c.dist < -0.0029
                for c in data.contact
            )
        )


if __name__ == "__main__":
    unittest.main()
