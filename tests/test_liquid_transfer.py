"""Constrained transfers and real piston strokes must conserve liquid volume."""
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from liquid_transfer import PistonPipette, VolumeLedger, VolumeState


class LiquidTransferTests(unittest.TestCase):
    def ledger(self, source=1e-6, tip=0.):
        return VolumeLedger({'source': VolumeState(source, 2e-6),
                             'tip': VolumeState(tip, 200e-9), 'environment': VolumeState(0., None)})

    def test_transfer_is_limited_by_supply_and_capacity(self):
        ledger = self.ledger(source=100e-9)
        self.assertAlmostEqual(ledger.transfer('source', 'tip', 300e-9), 100e-9)
        self.assertEqual(ledger.state('source').volume_m3, 0.)
        self.assertEqual(ledger.total_m3, ledger.initial_total_m3)
        ledger = self.ledger()
        self.assertEqual(ledger.transfer('source', 'tip', 300e-9), 200e-9)
        self.assertEqual(ledger.transfer('source', 'tip', 1e-9), 0.)

    def test_press_in_air_then_release_under_liquid_aspirates_one_stroke(self):
        ledger = self.ledger()
        pipette = PistonPipette(ledger, 'tip')
        pipette.update(1., None, 'environment')
        pipette.update(.75, 'source', 'source')
        pipette.update(0., 'source', 'source')
        self.assertAlmostEqual(ledger.state('tip').volume_m3, 200e-9)
        self.assertAlmostEqual(ledger.state('source').volume_m3, 800e-9)
        self.assertAlmostEqual(ledger.total_m3, ledger.initial_total_m3)
        json.dumps(pipette.snapshot(), allow_nan=False)

    def test_release_in_air_is_not_credited_on_reentry(self):
        ledger = self.ledger()
        pipette = PistonPipette(ledger, 'tip', pressed_fraction=1.)
        pipette.update(0., None, 'environment')
        pipette.update(0., 'source', 'source')
        self.assertEqual(ledger.state('tip').volume_m3, 0.)
        self.assertAlmostEqual(pipette.air_stroke_m3, 200e-9)

    def test_partial_dispense_and_environment_loss_remain_in_total(self):
        ledger = self.ledger(tip=200e-9)
        pipette = PistonPipette(ledger, 'tip')
        pipette.update(.5, 'source', 'source')
        pipette.update(1., None, 'environment')
        self.assertAlmostEqual(ledger.state('source').volume_m3, 1.1e-6)
        self.assertAlmostEqual(ledger.state('environment').volume_m3, 100e-9)
        self.assertEqual(ledger.state('tip').volume_m3, 0.)
        self.assertAlmostEqual(ledger.total_m3, ledger.initial_total_m3)

    def test_invalid_transfer_cannot_partially_mutate_the_ledger(self):
        ledger = self.ledger()
        initial = ledger.snapshot()
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):ledger.transfer('source', 'tip', value)
        with self.assertRaises(ValueError):ledger.transfer('source', 'source', 1e-9)
        with self.assertRaises(KeyError):ledger.transfer('source', 'missing', 1e-9)
        self.assertEqual(ledger.snapshot(), initial)

    def test_repeated_partial_transfers_do_not_create_volume(self):
        ledger = self.ledger()
        for _ in range(1000):
            ledger.transfer('source', 'tip', 13e-9)
            ledger.transfer('tip', 'source', 7e-9)
        self.assertAlmostEqual(ledger.total_m3, ledger.initial_total_m3, places=18)
        self.assertLessEqual(ledger.state('tip').volume_m3, 200e-9)
