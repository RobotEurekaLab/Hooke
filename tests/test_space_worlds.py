"""Environmental distinctions and actual unrestrained rigid-body motion."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from archetypes.task_catalog import CATALOG
from worlds.build import build_scene
from worlds.environment import ThermalWitness
from worlds.profiles import Environment, WORLDS, exposure_issues


class SpaceWorlds(unittest.TestCase):
    def test_cabin_and_exterior_are_separate_and_mars_is_not_vacuum(self):
        self.assertGreater(WORLDS["orbital"].workspace.pressure_pa, 100000)
        self.assertEqual(WORLDS["orbital"].exterior.pressure_pa, 0)
        self.assertEqual(WORLDS["lunar"].workspace.pressure_pa, 0)
        self.assertGreater(WORLDS["martian"].workspace.pressure_pa, 0)
        self.assertIn("pressure_loads", WORLDS["martian"].report()["disabled"])

    def test_unretained_samples_and_unqualified_wet_experiments_are_flagged(self):
        self.assertEqual(
            exposure_issues(WORLDS["orbital"], sealed=True, retained=True), []
        )
        self.assertIn(
            "unrestrained_sample_in_microgravity",
            exposure_issues(WORLDS["orbital"], sealed=True, retained=False),
        )
        for name in ("lunar", "martian"):
            self.assertIn(
                "unsealed_sample_in_low_pressure_environment",
                exposure_issues(WORLDS[name], sealed=False, retained=True),
            )
            self.assertIn(
                "liquid_phase_stability_and_containment_not_qualified",
                exposure_issues(
                    WORLDS[name], sealed=True, retained=True, requires_liquid=True
                ),
            )
        with self.assertRaises(ValueError):
            exposure_issues(
                WORLDS["orbital"], sealed=True, retained=True, zone="unknown"
            )

    def test_nonphysical_boundaries_are_rejected(self):
        for pressure, temperature in (
            (-1, 293),
            (float("nan"), 293),
            (0, 0),
            (0, float("inf")),
        ):
            with self.assertRaises(ValueError):
                Environment(pressure, temperature, "vacuum")
        with self.assertRaises(ValueError):
            ThermalWitness(emissivity=float("nan"))

    def test_generated_scenes_and_shared_asset_hashes_are_reproducible(self):
        manifest = json.loads((ROOT / "Hooke/worlds/assets_manifest.json").read_text())
        for record in manifest["scenes"]:
            profile = WORLDS[record["world"]]
            root, _ = build_scene(profile)
            self.assertEqual(
                ET.tostring(root, encoding="unicode"),
                (ROOT / "Hooke" / record["scene"]).read_text(),
            )
        for asset in manifest["assets"].values():
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / "Hooke" / asset["path"]).read_bytes()
                ).hexdigest(),
                asset["sha256"],
            )

    def test_all_worlds_follow_gravity_without_changing_sample_mass(self):
        for profile in WORLDS.values():
            with self.subTest(world=profile.name):
                task = CATALOG[profile.task_name].make_expert()
                task.reset(0)
                joint = task.model.joint("free_cartridge_joint")
                address = int(joint.qposadr[0])
                origin = task.data.qpos[address : address + 3].copy()
                retained = task.data.body("/retained_0:base").xpos.copy()
                for _ in range(250):
                    task.step()
                elapsed = float(task.data.time)
                expected = origin + [
                    0.06 * elapsed,
                    0,
                    -0.5 * profile.gravity_m_s2 * elapsed**2,
                ]
                np.testing.assert_allclose(
                    task.data.qpos[address : address + 3], expected, atol=0.0025, rtol=0
                )
                np.testing.assert_allclose(
                    task.data.body("/retained_0:base").xpos, retained, atol=1e-12
                )
                self.assertAlmostEqual(
                    float(task.model.body("/free:base").mass[0]), 0.1
                )
                self.assertAlmostEqual(task.environment.witness.elapsed_s, elapsed)
                self.assertLess(
                    abs(task.environment.witness.report()["energy_residual_j"]), 1e-8
                )
                self.assertFalse(task.check())
                self.assertEqual(task.model.opt.density, 0)

    def test_radiative_equilibrium_and_cooling_are_distinct_from_air_temperature(self):
        environment = Environment(0, 293.15, "vacuum")
        equilibrium = ThermalWitness()
        for _ in range(100):
            equilibrium.step(0.01, environment, 0)
        self.assertEqual(equilibrium.temperature_k, 293.15)
        hot = ThermalWitness(temperature_k=350)
        for _ in range(1000):
            hot.step(0.01, environment, 0)
        self.assertGreater(hot.temperature_k, 293.15)
        self.assertLess(hot.temperature_k, 350)
        self.assertLess(abs(hot.report()["energy_residual_j"]), 1e-8)
        with self.assertRaises(ValueError):
            hot.step(float("nan"), environment, 0)


if __name__ == "__main__":
    unittest.main()
