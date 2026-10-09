"""Mechanical clearances and specimen scales in three reference-informed labs."""

from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))

from real_labs.builder import build_scene
from real_labs.runtime import InstrumentSession
from real_labs import university_extensions


class UniversityPrecisionGeometryTests(unittest.TestCase):
    def make_session(self, identifier):
        with patch.object(university_extensions, "MODULES", ("university_precision",)):
            university_extensions.instrument_specs.cache_clear()
            root, manifest = build_scene(identifier)
            university_extensions.instrument_specs.cache_clear()
        return InstrumentSession(
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")),
            manifest,
        )

    def test_focus_and_specimen_carriers_clear_at_all_axis_corners(self):
        for identifier in ("ntu_sgsr_characterization", "tsinghua_rush3d"):
            session = self.make_session(identifier)
            controls = list(session.controls.values())
            # Combined endpoint configurations catch interactions missed by
            # checking each moving axis while all other axes stay at zero.
            for values in product(*[(c.lower, c.upper) for c in controls]):
                session.reset()
                for control, value in zip(controls, values):
                    session.data.qpos[session.model.jnt_qposadr[control.joint]] = value
                mujoco.mj_forward(session.model, session.data)
                penetrations = [
                    (
                        session.model.geom(contact.geom1).name,
                        session.model.geom(contact.geom2).name,
                        float(contact.dist),
                    )
                    for contact in session.data.contact
                    if contact.dist < -0.0001
                ]
                self.assertEqual(penetrations, [], (identifier, values, penetrations))

    def test_independent_treadmill_drives_track_both_directions(self):
        session = self.make_session("stanford_biomechatronics_gait")
        for key, control in session.controls.items():
            span = control.upper - control.lower
            for fraction in (0.15, 0.85, 0.25):
                target = control.lower + fraction * span
                session.command(*key, target, duration=2)
                measured = (
                    session.data.qvel[session.model.jnt_dofadr[control.joint]]
                    if control.mode == "velocity"
                    else session.data.qpos[session.model.jnt_qposadr[control.joint]]
                )
                self.assertLessEqual(abs(measured - target), max(1e-6, span * 0.025))
            session.reset()

    def test_microscope_sample_is_millimetre_target_on_real_slide(self):
        session = self.make_session("tsinghua_rush3d")
        item = session.equipment["rush3d_microscope"]
        self.assertEqual(item["sample_size_m"], [0.002, 0.002, 0.00016])
        carrier = session.model.body("rush3d_microscope__body_sample_y").id
        sizes = [
            session.model.geom_size[index]
            for index in range(session.model.ngeom)
            if session.model.geom_bodyid[index] == carrier
        ]
        self.assertTrue(any(np.allclose(s, (0.0375, 0.0125, 0.0005)) for s in sizes))
        self.assertTrue(any(np.allclose(s, (0.001, 0.001, 0.00008)) for s in sizes))

    def test_afm_has_the_published_210mm_chuck(self):
        session = self.make_session("ntu_sgsr_characterization")
        stage = session.model.body("dimension_icon__body_coarse_y").id
        chuck = [
            index
            for index in range(session.model.ngeom)
            if session.model.geom_bodyid[index] == stage
            and session.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and np.isclose(session.model.geom_size[index, 0], 0.105)
        ]
        self.assertEqual(len(chuck), 1)
        # Metrology identity must not become a false nanoscale actuator claim.
        self.assertEqual(
            set(session.equipment["dimension_icon"]["actuators"]),
            {"coarse_x", "coarse_y", "coarse_head"},
        )

    def test_rush_right_camera_has_continuous_solid_support_to_table(self):
        session = self.make_session("tsinghua_rush3d")
        base = session.model.body("rush3d_microscope").id
        origin = session.data.xpos[base]
        sample_xy = origin[:2] + (0.51, 0.13)
        intervals = []
        for index in range(session.model.ngeom):
            if (
                session.model.geom_bodyid[index] != base
                or session.model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX
            ):
                continue
            center = session.data.geom_xpos[index]
            half = session.model.geom_size[index]
            if np.all(abs(sample_xy - center[:2]) <= half[:2]):
                intervals.append((center[2] - half[2], center[2] + half[2]))
        top = origin[2]
        for lower, upper in sorted(intervals):
            if lower <= top + 1e-5:
                top = max(top, upper)
        self.assertGreaterEqual(top, origin[2] + 1.50)


if __name__ == "__main__":
    unittest.main()
