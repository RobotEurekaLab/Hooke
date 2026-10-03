"""Physical access and carriage regression for the third European lab wave."""

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
    "epfl_lspn_chemistry",
    "kth_robot_design_prototyping",
    "manchester_hv_dielectric_fluids",
    "birmingham_chbh_mri",
)


class EuropeWaveThreeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_europe_wave3"))
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

    def test_scanner_bore_is_physically_open(self):
        model, data, _ = self.load("birmingham_chbh_mri")
        collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_group[:] = collides.astype(np.int32)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([0.0, 0.45, 1.25]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8),
            True,
            -1,
            hit,
        )
        # A hollow bore permits the ray to reach the room wall behind it.
        self.assertGreater(distance, 1.7)
        self.assertTrue(model.geom(int(hit[0])).name.startswith("arch_wall_back"))

    def test_visible_inert_samples_follow_their_carriers(self):
        for identifier, key, value, expected in (
            ("epfl_lspn_chemistry", "flask_lift", 0.09, [0, 0, 0.09]),
            ("kth_robot_design_prototyping", "build_plate_z", 0.13, [0, 0, 0.13]),
            ("manchester_hv_dielectric_fluids", "specimen_height", 0.04, [0, 0, 0.04]),
            ("birmingham_chbh_mri", "table_insert", 0.50, [0, 0.50, 0]),
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


if __name__ == "__main__":
    unittest.main()
