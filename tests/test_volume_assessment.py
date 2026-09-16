"""A motion pass cannot hide absent liquid or inconsistent rendered volume."""
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Hooke'))
from backends.volume_assessment import PipetteVolumeAssessment
from liquid_transfer import PistonPipette, VolumeLedger, VolumeState


class VolumeAssessmentTests(unittest.TestCase):
    def transfer(self):
        ledger = VolumeLedger({'source': VolumeState(1e-6, 2e-6),
                               'tip': VolumeState(0., 200e-9), 'environment': VolumeState(0., None)})
        piston = PistonPipette(ledger, 'tip', 1.)
        container = SimpleNamespace(volume=1e-6)
        container.liquid = SimpleNamespace(surface=SimpleNamespace(distance=0.),
            meshplane=SimpleNamespace(calculate_volume=lambda distance: (container.volume, 1.)))
        source = SimpleNamespace(container=container)
        return SimpleNamespace(snapshot=piston.snapshot, pipette=piston, tip_capacity_m3=200e-9,
                               source=source, containers={'source': source}, target_reservoir='tip')

    def test_unexecuted_piston_is_not_a_volume_success(self):
        observer = PipetteVolumeAssessment(self.transfer())
        observer.update()
        self.assertFalse(observer.report()['success'])

    def test_accounting_detects_outdated_surface_and_environment_loss(self):
        transfer = self.transfer()
        transfer.pipette.update(0., 'source', 'source')
        observer = PipetteVolumeAssessment(transfer)
        observer.update()
        self.assertFalse(observer.report()['checks']['surface_volume_matches_ledger'])
        transfer.source.container.volume = transfer.pipette.ledger.state('source').volume_m3
        observer = PipetteVolumeAssessment(transfer)
        observer.update()
        self.assertTrue(observer.report()['success'])
        transfer.pipette.update(.5, None, 'environment')
        observer.update()
        self.assertFalse(observer.report()['checks']['no_liquid_expelled_to_environment'])

    def test_cached_volume_agreement_cannot_hide_wrong_surface_geometry(self):
        transfer = self.transfer()
        transfer.pipette.update(0., 'source', 'source')
        volume = transfer.pipette.ledger.state('source').volume_m3
        transfer.source.container.volume = volume
        transfer.source.container.liquid.meshplane.calculate_volume = lambda distance: (volume+30e-9, 1.)
        observer = PipetteVolumeAssessment(transfer)
        observer.update()
        report = observer.report()
        self.assertTrue(report['checks']['surface_volume_matches_ledger'])
        self.assertFalse(report['checks']['geometric_surface_volume_matches_ledger'])
        self.assertFalse(report['success'])

    def test_nonfinite_surface_is_a_serializable_failure(self):
        transfer = self.transfer()
        transfer.source.container.liquid.meshplane.calculate_volume = lambda distance: (float('nan'), 1.)
        observer = PipetteVolumeAssessment(transfer)
        observer.update()
        report = observer.report()
        self.assertFalse(report['checks']['geometric_surface_volume_matches_ledger'])
        self.assertIsNone(report['metrics']['max_geometric_volume_error_m3'])
        json.dumps(report, allow_nan=False)
