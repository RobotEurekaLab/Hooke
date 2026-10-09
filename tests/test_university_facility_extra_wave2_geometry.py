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


class FacilityExtraWaveTwoGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit-test this module independently of other in-progress extensions;
        # shared registry uniqueness belongs to the full catalogue regression.
        modules = ("university_facility_extra_wave2",)
        cls.extension_patch = patch.object(university_extensions, "MODULES", modules)
        cls.extension_patch.start()
        university_extensions.instrument_specs.cache_clear()
        path = (
            Path(__file__).resolve().parents[1]
            / "Hooke/real_labs/scenes/university_facility_extra_wave2.json"
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

    def test_glasgow_retained_disks_rest_on_carrier_seats(self):
        session = self.sessions["glasgow_jwnc_wafer_loading"]
        session.reset()
        wafers = _tagged(session, "retained_100mm_estimated_wafer")
        self.assertEqual(len(wafers), 2)
        for wafer in wafers:
            self.assertAlmostEqual(session.model.geom_size[wafer, 0] * 2, 0.10)
            body = session.model.geom_bodyid[wafer]
            seats = [
                i
                for i in range(session.model.ngeom)
                if session.model.geom_bodyid[i] == body
                and session.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
                and 0.060 < session.model.geom_size[i, 0] < 0.065
            ]
            self.assertEqual(len(seats), 1)
            self.assertAlmostEqual(
                _bounds(session, wafer)[0][2], _bounds(session, seats[0])[1][2]
            )

    def test_kth_documented_two_metre_clear_octagonal_section(self):
        session = self.sessions["kth_l2000_wind_tunnel"]
        session.reset()
        top = _tagged(session, "octagon_face_0")[0]
        bottom = _tagged(session, "octagon_face_4")[0]
        right = _tagged(session, "octagon_face_2")[0]
        left = _tagged(session, "octagon_face_6")[0]
        self.assertAlmostEqual(
            _bounds(session, top)[0][2] - _bounds(session, bottom)[1][2], 2.0, places=6
        )
        self.assertAlmostEqual(
            _bounds(session, right)[0][0] - _bounds(session, left)[1][0], 2.0, places=6
        )

    def test_eth_access_drawer_is_supported_at_full_extension(self):
        session = self.sessions["eth_experimental_petrology_multianvil"]
        control = next(
            v for k, v in session.controls.items() if k[1] == "specimen_drawer"
        )
        session.reset()
        session.data.qpos[session.model.jnt_qposadr[control.joint]] = control.upper
        mujoco.mj_forward(session.model, session.data)
        lower, upper = _bounds(session, _tagged(session, "multianvil_drawer_plate")[0])
        for rail in _tagged(session, "multianvil_access_rail"):
            low, high = _bounds(session, rail)
            self.assertAlmostEqual(high[2], lower[2])
            self.assertLess(low[1], lower[1])
            self.assertGreater(high[1], upper[1])

    def test_eth_stored_disks_share_a_continuous_supported_spindle(self):
        session = self.sessions["eth_experimental_petrology_multianvil"]
        session.reset()
        spindle = session.model.geom("eth_right_inferred_storage_spindle").id
        platen = session.model.geom("eth_right_platen_1.22").id
        lower, upper = _bounds(session, spindle)
        self.assertAlmostEqual(lower[2], _bounds(session, platen)[1][2])
        disks = _tagged(session, "eth_right_static_disc_")
        self.assertEqual(len(disks), 5)
        for disk in disks:
            low, high = _bounds(session, disk)
            self.assertLess(lower[2], high[2])
            self.assertGreaterEqual(upper[2] + 1e-9, high[2])
            np.testing.assert_allclose(
                session.data.geom_xpos[spindle, :2], session.data.geom_xpos[disk, :2]
            )

    def test_epfl_probe_clears_dry_bed_and_bridge_retains_contact(self):
        session = self.sessions["epfl_lch_hydraulic_flume"]
        session.reset()
        lift = next(v for k, v in session.controls.items() if k[1] == "probe_height")
        session.data.qpos[session.model.jnt_qposadr[lift.joint]] = lift.lower
        mujoco.mj_forward(session.model, session.data)
        probe = _tagged(session, "lch_dry_probe")[0]
        bed = _tagged(session, "lch_bed_support")[0]
        self.assertGreater(
            _bounds(session, probe)[0][2] - _bounds(session, bed)[1][2], 0.05
        )
        bridge = _tagged(session, "lch_traverse_bridge")[0]
        for rail in _tagged(session, "lch_traverse_rail"):
            self.assertAlmostEqual(
                _bounds(session, bridge)[0][2], _bounds(session, rail)[1][2]
            )
        # Contact remains active; only uncalibrated tangential bearing friction is omitted.
        self.assertTrue(any(bridge in (c.geom1, c.geom2) for c in session.data.contact))


if __name__ == "__main__":
    unittest.main()
