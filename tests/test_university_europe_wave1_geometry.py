"""Physical access and carriage regression for the first European lab wave."""

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
    "delft_stevin_structures",
    "eth_robotic_fabrication",
    "edinburgh_flowave_basin",
    "eth_mougel_inorganic",
)


class EuropeWaveOneGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = tuple(
            dict.fromkeys((*university_extensions.MODULES, "university_europe_wave1"))
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

    def test_loading_head_stops_above_rigid_beam(self):
        model, data, manifest = self.load("delft_stevin_structures")
        item = manifest["equipment"][0]
        address = model.joint(item["joints"]["load_head_approach"]).qposadr[0]
        pad = model.site(item["sites"]["load_contact_plane"]).id
        specimen = model.site(item["sample_site"]).id
        for approach in np.linspace(0, 0.22, 25):
            data.qpos[address] = approach
            mujoco.mj_forward(model, data)
            specimen_top = data.site_xpos[specimen, 2] + item["sample_size_m"][2] / 2
            self.assertGreater(data.site_xpos[pad, 2] - specimen_top, 0.06)

    def test_basin_is_recessed_and_not_filled_by_a_collision_floor(self):
        model, data, manifest = self.load("edinburgh_flowave_basin")
        self.assertFalse(
            any(model.geom(i).name == "arch_floor" for i in range(model.ngeom))
        )
        # Raycast only against actual collision geometry. At this clear position
        # inside the basin, the first downward hit must be its two-metre-deep floor.
        collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_group[:] = collides.astype(np.int32)
        group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([3.0, 2.0, 0.30]),
            np.array([0.0, 0.0, -1.0]),
            group,
            True,
            -1,
            hit,
        )
        self.assertAlmostEqual(distance, 2.30, places=6)
        self.assertGreaterEqual(int(hit[0]), 0)
        self.assertEqual(manifest["equipment"][0]["pos"][2], 0)

    def test_samples_follow_their_visible_carriers(self):
        for identifier, joint_key in (
            ("eth_mougel_inorganic", "vial_carriage"),
            ("eth_robotic_fabrication", "gantry_traverse"),
        ):
            model, data, manifest = self.load(identifier)
            item = manifest["equipment"][0]
            joint = model.joint(item["joints"][joint_key]).id
            site = model.site(item["sample_site"]).id
            body = int(model.site_bodyid[site])
            self.assertGreater(model.body_geomnum[body], 0)
            self.assertEqual(item["sample_attachment"], "clamped")
            initial = data.site_xpos[site].copy()
            data.qpos[model.jnt_qposadr[joint]] = 0.20
            mujoco.mj_forward(model, data)
            displacement = data.site_xpos[site] - initial
            self.assertAlmostEqual(np.linalg.norm(displacement), 0.20, places=7)
            self.assertGreater(data.site_xpos[site, 2], 0.5)


if __name__ == "__main__":
    unittest.main()
