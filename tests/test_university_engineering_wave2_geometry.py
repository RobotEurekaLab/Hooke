"""Gantry clearances, genuine panel openings and acoustic-room construction."""

from itertools import product
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


class EngineeringWaveTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = patch.object(
            university_extensions, "MODULES", ("university_engineering_wave2",)
        )
        cls.registry.start()
        university_extensions.instrument_specs.cache_clear()
        cls.sessions = {}
        for identifier in ("sorbonne_isir_covr", "epfl_lwe_anechoic_acoustics"):
            root, manifest = build_scene(identifier)
            model = mujoco.MjModel.from_xml_string(
                ET.tostring(root, encoding="unicode")
            )
            cls.sessions[identifier] = InstrumentSession(model, manifest)

    @classmethod
    def tearDownClass(cls):
        cls.registry.stop()
        university_extensions.instrument_specs.cache_clear()

    def test_all_controls_reach_targets_and_return(self):
        for identifier, session in self.sessions.items():
            for key, control in session.controls.items():
                session.reset()
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
                np.testing.assert_allclose(
                    session.data.qpos, session.model.qpos0, atol=1e-10
                )

    def test_full_stroke_combinations_and_door_approach(self):
        for identifier, session in self.sessions.items():
            controls = list(session.controls.values())
            for pose in product(*(np.linspace(c.lower, c.upper, 11) for c in controls)):
                session.reset()
                for c, value in zip(controls, pose):
                    session.data.qpos[session.model.jnt_qposadr[c.joint]] = value
                mujoco.mj_forward(session.model, session.data)
                bad = [
                    (
                        session.model.geom(c.geom1).name,
                        session.model.geom(c.geom2).name,
                        c.dist,
                    )
                    for c in session.data.contact
                    if c.dist < -0.0001
                ]
                self.assertEqual(bad, [], (identifier, pose))
            session.reset()
            self.assertTrue(audit_layout(session.manifest)["passed"])
            self.assertEqual(
                doorway_obstacle_warnings(
                    session.manifest, model_obstacles(session.model, session.data)
                ),
                [],
                identifier,
            )

    def test_panel_ports_are_open_through_the_mesh(self):
        s = self.sessions["sorbonne_isir_covr"]
        s.reset()
        body = s.model.body("primary_apparatus__body_panel_x").id
        center = s.data.xpos[body]
        geom = np.zeros(1, dtype=np.int32)
        for offset in (-1.04, -1.69):
            start = center + np.array([0, -0.30, offset])
            distance = mujoco.mj_ray(
                s.model, s.data, start, np.array([0.0, 1.0, 0.0]), None, 1, -1, geom
            )
            self.assertTrue(
                distance < 0 or distance > 0.6,
                (offset, distance, s.model.geom(geom[0]).name),
            )
            start[0] += 0.25
            distance = mujoco.mj_ray(
                s.model, s.data, start, np.array([0.0, 1.0, 0.0]), None, 1, -1, geom
            )
            self.assertAlmostEqual(distance, 0.285, places=5)

    def test_absorber_depth_and_floor_void(self):
        s = self.sessions["epfl_lwe_anechoic_acoustics"]
        s.reset()
        # Compiled mesh coordinates are recentered/rotated, so measure in world space.
        g = s.model.geom("epfl_wedge_back_0_0_-1")
        mesh = g.dataid.item()
        vertices = s.model.mesh_vert[
            s.model.mesh_vertadr[mesh] : s.model.mesh_vertadr[mesh]
            + s.model.mesh_vertnum[mesh]
        ]
        world = (
            vertices @ s.data.geom_xmat[g.id].reshape(3, 3).T + s.data.geom_xpos[g.id]
        )
        self.assertAlmostEqual(np.ptp(world[:, 1]), 1.0, places=5)
        slab = s.model.geom("arch_floor_epfl_lower_slab")
        self.assertLess(slab.pos[2] + slab.size[2], -1.0)
        self.assertNotIn(
            "arch_floor", [s.model.geom(i).name for i in range(s.model.ngeom)]
        )
        self.assertTrue(
            any("floor_grate" in s.model.geom(i).name for i in range(s.model.ngeom))
        )

    def test_sheet_and_absorber_faces_retain_flat_normals(self):
        for session in self.sessions.values():
            model = session.model
            for i in range(model.nmesh):
                if not any(
                    key in model.mesh(i).name for key in ("plywood_mesh", "__wedge")
                ):
                    continue
                start = model.mesh_normaladr[i]
                normals = model.mesh_normal[start : start + model.mesh_normalnum[i]]
                start = model.mesh_faceadr[i]
                indices = model.mesh_facenormal[start : start + model.mesh_facenum[i]]
                corners = normals[indices]
                self.assertLess(np.max(np.abs(corners[:, 1:] - corners[:, :1])), 1e-5)


if __name__ == "__main__":
    unittest.main()
