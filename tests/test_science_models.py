"""Numerical laws and validity boundaries of the declared reduced models."""

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from science.thermal import ThermalBath, ThermalParameters
from science.flow import CapillaryFlow, CapillaryParameters
from liquid_transfer import VolumeLedger, VolumeState


class ScienceModels(unittest.TestCase):
    def test_two_node_heating_and_cooling_conserve_energy(self):
        bath = ThermalBath()
        for _ in range(45000):
            bath.step(0.002, 33.0)
        r = bath.report()
        self.assertGreater(bath.sample_c, 32.0)
        self.assertLess(bath.block_c, 33.0)
        self.assertLess(abs(r["energy_residual_j"]), 1e-8)
        for _ in range(45000):
            bath.step(0.002, 15.0)
        self.assertLess(bath.sample_c, 16.0)
        self.assertLess(abs(bath.report()["energy_residual_j"]), 1e-8)

    def test_ambient_equilibrium_and_invalid_lumped_conditions(self):
        bath = ThermalBath()
        for _ in range(100):
            bath.step(0.002, 25.0)
        self.assertEqual(bath.sample_c, 25.0)
        self.assertEqual(bath.report()["stored_energy_j"], 0.0)
        with self.assertRaises(ValueError):
            ThermalParameters(biot_number=0.2)
        with self.assertRaises(ValueError):
            bath.step(10.0, 33.0)
        with self.assertRaises(ValueError):
            bath.step(0.002, float("nan"))

    def flow(self, parameters=None):
        ledger = VolumeLedger(
            {"a": VolumeState(1e-6, 2e-6), "b": VolumeState(0.0, 2e-6)}
        )
        return CapillaryFlow(ledger, "a", "b", parameters)

    def test_poiseuille_law_reverse_flow_and_capacity_conservation(self):
        flow = self.flow()
        expected = math.pi * 0.0001**4 * 5000 / (8 * 0.001 * 0.02)
        self.assertAlmostEqual(flow.step(1.0, 5000.0), expected, places=18)
        self.assertAlmostEqual(flow.step(1.0, -5000.0), -expected, places=18)
        self.assertAlmostEqual(flow.transferred_m3, 0.0, places=18)
        flow.step(10000.0, 5000.0)
        self.assertAlmostEqual(flow.ledger.state("a").volume_m3, 0.0, places=18)
        self.assertLess(abs(flow.report()["conservation_error_m3"]), 1e-18)

    def test_radius_fourth_power_and_outside_laminar_bound_rejected(self):
        small, big = self.flow(), self.flow(CapillaryParameters(radius_m=0.0002))
        self.assertAlmostEqual(big.step(0.1, 100.0) / small.step(0.1, 100.0), 16.0)
        before = small.ledger.snapshot()
        with self.assertRaises(ValueError):
            small.step(0.002, 1e8)
        self.assertEqual(small.ledger.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
