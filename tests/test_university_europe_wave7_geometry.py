"""Physical access and carriage regression for the seventh European lab wave."""

import itertools
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np

from real_labs import university_extensions
from real_labs.builder import build_scene
from real_labs.layout_audit import (
    audit_layout,
    doorway_obstacle_warnings,
    model_obstacles,
)


SCENES = (
    "leeds_environmental_tribology",
    "copenhagen_pice_ice_workroom",
    "uppsala_freia_gersemi",
    "lund_maxiv_nanomax_eh1",
)


class EuropeWaveSevenGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_europe_wave7"))
        )
        cls.registration = patch.object(university_extensions, "MODULES", modules)
        cls.registration.start()
        university_extensions.instrument_specs.cache_clear()

    @classmethod
    def tearDownClass(cls):
        cls.registration.stop()
        university_extensions.instrument_specs.cache_clear()

    def load(self, identifier):
        root, manifest = build_scene(identifier)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, manifest

    def controls(self, model, manifest):
        for item in manifest["equipment"]:
            for key, joint in item["joints"].items():
                index = model.joint(joint).id
                yield (
                    item["id"] + "/" + key,
                    model.jnt_qposadr[index],
                    model.actuator(item["actuators"][key]).id,
                    model.jnt_range[index].copy(),
                )

    def test_bidirectional_targets_and_reset(self):
        for identifier in SCENES:
            model, data, manifest = self.load(identifier)
            initial_sites = data.site_xpos.copy()
            for name, address, actuator, (low, high) in self.controls(model, manifest):
                for fraction in (0.15, 0.85, 0.25):
                    with self.subTest(scene=identifier, axis=name, fraction=fraction):
                        target = low + fraction * (high - low)
                        data.ctrl[actuator] = target
                        mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
                        if (
                            identifier == "leeds_environmental_tribology"
                            and name.endswith("/contact_approach")
                            and target > 0.020
                        ):
                            # A compliant load command must stop at real contact,
                            # not teleport through the coupon to reach its target.
                            self.assertLess(data.qpos[address], target)
                            self.assertGreater(data.qpos[address], 0.020)
                            self.assertTrue(
                                any(
                                    c.dist < 0
                                    and {
                                        model.geom(c.geom1).name,
                                        model.geom(c.geom2).name,
                                    }
                                    == {
                                        "tribology__leeds_counterface",
                                        "tribology__leeds_coupon",
                                    }
                                    for c in data.contact
                                )
                            )
                        else:
                            self.assertLessEqual(
                                abs(data.qpos[address] - target),
                                max(1e-6, 0.025 * (high - low)),
                            )
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12)
            np.testing.assert_allclose(data.site_xpos, initial_sites, atol=1e-12)
            self.assertTrue(np.all(data.ctrl == 0))

    def test_full_strokes_have_no_unexpected_penetration(self):
        for identifier in SCENES:
            model, data, manifest = self.load(identifier)
            controls = list(self.controls(model, manifest))
            states = [
                {address: value}
                for _, address, _, (low, high) in controls
                for value in np.linspace(low, high, 25)
            ]
            states.extend(
                {control[1]: control[3][edge] for control, edge in zip(controls, edges)}
                for edges in itertools.product((0, 1), repeat=len(controls))
            )
            for state in states:
                mujoco.mj_resetData(model, data)
                if identifier in (
                    "leeds_environmental_tribology",
                    "copenhagen_pice_ice_workroom",
                ):
                    # Load-axis extrema are force-bearing servo commands. Raw
                    # qpos assignment beyond the surface would force penetration.
                    # Exercise full declared commands through actual dynamics.
                    for _, address, actuator, _ in controls:
                        data.ctrl[actuator] = state.get(address, data.qpos[address])
                    mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
                else:
                    for address, value in state.items():
                        data.qpos[address] = value
                mujoco.mj_forward(model, data)
                penetrations = [
                    (
                        [model.geom(int(g)).name for g in contact.geom],
                        float(contact.dist),
                    )
                    for contact in data.contact[: data.ncon]
                    if contact.dist < -1e-5
                    and not (
                        identifier == "leeds_environmental_tribology"
                        and {
                            model.geom(contact.geom1).name,
                            model.geom(contact.geom2).name,
                        }
                        == {"tribology__leeds_counterface", "tribology__leeds_coupon"}
                        and contact.dist >= -0.0005
                    )
                ]
                self.assertEqual(penetrations, [], (identifier, state))

    def test_layout_and_actual_door_approaches(self):
        for identifier in SCENES:
            with self.subTest(scene=identifier):
                model, data, manifest = self.load(identifier)
                report = audit_layout(manifest)
                self.assertTrue(report["passed"])
                self.assertEqual(report["manual_inspection_count"], 0)
                self.assertEqual(
                    doorway_obstacle_warnings(manifest, model_obstacles(model, data)),
                    [],
                )

    def test_retained_samples_and_carriage_displacement(self):
        for identifier in SCENES:
            model, data, manifest = self.load(identifier)
            item = manifest["equipment"][0]
            site = model.site(item["sample_site"]).id
            self.assertEqual(item["sample_attachment"], "clamped")
            self.assertGreater(model.body_geomnum[model.site_bodyid[site]], 0)
        for identifier, key, value, expected in (
            ("copenhagen_pice_ice_workroom", "core_feed_v2", 0.04, [0, 0.04, 0]),
            ("uppsala_freia_gersemi", "dry_insert_lift", 0.12, [0, 0, 0.12]),
            ("lund_maxiv_nanomax_eh1", "dry_specimen_carriage", 0.05, [0, 0.05, 0]),
        ):
            model, data, manifest = self.load(identifier)
            item = manifest["equipment"][0]
            site = model.site(item["sample_site"]).id
            before = data.site_xpos[site].copy()
            joint = model.joint(item["joints"][key]).id
            data.qpos[model.jnt_qposadr[joint]] = value
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(
                data.site_xpos[site] - before, expected, atol=1e-9
            )

    def test_inspection_starts_open_and_leeds_has_reachable_contact(self):
        for identifier, key in (
            ("leeds_environmental_tribology", "access_door"),
            ("copenhagen_pice_ice_workroom", "wheel_guard"),
            ("uppsala_freia_gersemi", "inspection_port_cover"),
            ("lund_maxiv_nanomax_eh1", "sample_access_door"),
        ):
            model, data, manifest = self.load(identifier)
            item = manifest["equipment"][0]
            joint = model.joint(item["joints"][key]).id
            self.assertGreater(data.qpos[model.jnt_qposadr[joint]], 0.5)
        model, data, manifest = self.load("leeds_environmental_tribology")
        item = manifest["equipment"][0]
        joint = model.joint(item["joints"]["contact_approach"]).id
        tip = model.site("tribology__site_counterface_tip").id
        specimen = model.site(item["sample_site"]).id
        self.assertAlmostEqual(
            data.site_xpos[tip, 2] - data.site_xpos[specimen, 2], 0.020
        )
        data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 1]
        mujoco.mj_forward(model, data)
        self.assertLess(data.site_xpos[tip, 2], data.site_xpos[specimen, 2])
        # This geometric reach check is not a valid loaded pose. The dedicated
        # Leeds regression verifies actual compliant contact and sliding.


if __name__ == "__main__":
    unittest.main()
