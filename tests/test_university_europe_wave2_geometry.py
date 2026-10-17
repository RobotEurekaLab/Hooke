"""Physical access and carriage regression for the second European lab wave."""

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
    "cambridge_schofield_turner",
    "bristol_equals_shaking_table",
    "tum_wind_tunnel_a",
    "leeds_litac_yarn_spinning",
)


class EuropeWaveTwoGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_europe_wave2"))
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
            if identifier == "bristol_equals_shaking_table":
                # Version 2 is a redundantly actuated parallel mechanism.
                # Independent leg targets are not compatible platform poses.
                from real_labs.bristol_table_geometry import CONTROLS, ID, leg_lengths

                controls = list(self.controls(model, manifest))
                for axis in range(3):
                    for target in (-0.105, 0.105, -0.075):
                        pose = np.zeros(3)
                        pose[axis] = target
                        values = leg_lengths(pose)
                        for value, (_, _, actuator, _) in zip(values, controls):
                            data.ctrl[actuator] = value
                        mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
                        for value, (_, address, _, (low, high)) in zip(values, controls):
                            self.assertLessEqual(abs(data.qpos[address]-value), .025*(high-low))
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, model.qpos0, atol=1e-12)
                np.testing.assert_allclose(data.site_xpos, initial_sites, atol=1e-12)
                continue
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
            if identifier == "bristol_equals_shaking_table":
                from real_labs.bristol_table_motion import TableInspection, diagnostic_pose
                from real_labs.runtime import InstrumentSession

                session = InstrumentSession(model, manifest)
                checker = TableInspection(session)
                for axis in range(3):
                    for value in np.linspace(-.15, .15, 25):
                        pose = np.zeros(6)
                        pose[axis] = value
                        diagnostic_pose(model, data, pose)
                        self.assertGreater(checker.direct_clearance(data)[0], .00025)
                with self.assertRaises(ValueError):
                    diagnostic_pose(model, data, [.15, .15, 0, 0, 0, 0])
                continue
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

    def test_shake_table_has_an_actual_service_pit(self):
        model, data, _ = self.load("bristol_equals_shaking_table")
        collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_group[:] = collides.astype(np.int32)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([2.0, 2.0, 0.3]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
            True,
            -1,
            hit,
        )
        # The version-2 strong block is at -1.72 m inside the estimated pit.
        # A false full-room
        # collision floor at zero would instead return a distance of 0.30 m.
        self.assertAlmostEqual(distance, 2.02, places=6)
        self.assertGreaterEqual(int(hit[0]), 0)

    def test_clamped_samples_follow_supported_carriages(self):
        for identifier, key, displacement in (
            ("cambridge_schofield_turner", "specimen_carriage", 0.10),
            ("bristol_equals_shaking_table", None, 0.04),
            ("leeds_litac_yarn_spinning", "bobbin_carriage", 0.10),
        ):
            with self.subTest(scene=identifier):
                model, data, manifest = self.load(identifier)
                item = manifest["equipment"][0]
                site = model.site(item["sample_site"]).id
                body = int(model.site_bodyid[site])
                self.assertGreater(model.body_geomnum[body], 0)
                self.assertEqual(item["sample_attachment"], "clamped")
                before = data.site_xpos[site].copy()
                if identifier == "bristol_equals_shaking_table":
                    from real_labs.bristol_table_motion import diagnostic_pose

                    diagnostic_pose(model, data, [displacement, 0, 0, 0, 0, 0])
                else:
                    joint = model.joint(item["joints"][key]).id
                    data.qpos[model.jnt_qposadr[joint]] = displacement
                    mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.site_xpos[site] - before, [displacement, 0, 0], atol=1e-9
                )

    def test_vehicle_orientation_follows_yaw_stage(self):
        model, data, manifest = self.load("tum_wind_tunnel_a")
        item = manifest["equipment"][0]
        site = model.site(item["sample_site"]).id
        body = int(model.site_bodyid[site])
        joint = model.joint(item["joints"]["model_yaw"]).id
        before = data.xmat[body].copy().reshape(3, 3)
        data.qpos[model.jnt_qposadr[joint]] = 0.30
        mujoco.mj_forward(model, data)
        delta = before.T @ data.xmat[body].reshape(3, 3)
        self.assertAlmostEqual(np.arctan2(delta[1, 0], delta[0, 0]), 0.30, places=7)
        self.assertEqual(item["sample_attachment"], "clamped")
        self.assertGreater(model.body_geomnum[body], 10)


if __name__ == "__main__":
    unittest.main()
