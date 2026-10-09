"""Physical access and carriage regression for the sixth European lab wave."""

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
    "imperial_thinfilm_bessemer",
    "nottingham_food_process_hall",
    "warwick_highfield_nmr_1ghz",
)


class EuropeWaveSixGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_europe_wave6"))
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
                        self.assertLessEqual(
                            abs(data.qpos[address] - target),
                            max(1e-6, 0.025 * (high - low)),
                        )
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12)
            np.testing.assert_allclose(data.site_xpos, initial_sites, atol=1e-12)
            self.assertTrue(np.all(data.ctrl == 0))

    def test_sampled_full_strokes_and_combined_extrema_have_no_penetration(self):
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

    def test_visible_retained_samples_follow_carriers(self):
        for identifier, key, value, expected in (
            ("imperial_thinfilm_bessemer", "coupon_x", 0.08, [0.08, 0, 0]),
            ("warwick_highfield_nmr_1ghz", "inspection_lift", 0.12, [0, 0, 0.12]),
        ):
            with self.subTest(scene=identifier):
                model, data, manifest = self.load(identifier)
                item = manifest["equipment"][0]
                joint = model.joint(item["joints"][key]).id
                site = model.site(item["sample_site"]).id
                body = int(model.site_bodyid[site])
                self.assertGreater(model.body_geomnum[body], 0)
                self.assertEqual(item["sample_attachment"], "clamped")
                before = data.site_xpos[site].copy()
                data.qpos[model.jnt_qposadr[joint]] = value
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.site_xpos[site] - before, expected, atol=1e-9
                )
        for identifier, key, axis in (
            ("nottingham_food_process_hall", "pan_tilt", "x"),
            ("warwick_highfield_nmr_1ghz", "holder_rotation", "z"),
        ):
            model, data, manifest = self.load(identifier)
            item = manifest["equipment"][0]
            site = model.site(item["sample_site"]).id
            body = int(model.site_bodyid[site])
            self.assertGreater(model.body_geomnum[body], 0)
            before = data.xmat[body].copy().reshape(3, 3)
            joint = model.joint(item["joints"][key]).id
            data.qpos[model.jnt_qposadr[joint]] = 0.2
            mujoco.mj_forward(model, data)
            delta = before.T @ data.xmat[body].reshape(3, 3)
            angle = (
                np.arctan2(delta[2, 1], delta[1, 1])
                if axis == "x"
                else np.arctan2(delta[1, 0], delta[0, 0])
            )
            self.assertAlmostEqual(angle, 0.2, places=8)

    def test_access_starts_open_and_dummy_stays_below_cryostat(self):
        for identifier, key in (
            ("imperial_thinfilm_bessemer", "access_door"),
            ("nottingham_food_process_hall", "inspection_cover"),
        ):
            model, data, manifest = self.load(identifier)
            joint = model.joint(manifest["equipment"][0]["joints"][key]).id
            self.assertGreater(data.qpos[model.jnt_qposadr[joint]], 0.5)
        model, data, manifest = self.load("warwick_highfield_nmr_1ghz")
        item = manifest["equipment"][0]
        joint = model.joint(item["joints"]["inspection_lift"]).id
        data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 1]
        mujoco.mj_forward(model, data)
        sample_z = data.site_xpos[model.site(item["sample_site"]).id, 2]
        self.assertLess(sample_z + 0.060, 1.30)


if __name__ == "__main__":
    unittest.main()
