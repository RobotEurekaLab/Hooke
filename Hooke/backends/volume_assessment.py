"""Check the ideal-volume model separately from motion and fluid dynamics."""
import numpy as np


class PipetteVolumeAssessment:
    def __init__(self, transfer):
        self.transfer = transfer
        self.samples = 0
        self.max_conservation_error_m3 = 0.
        self.max_surface_volume_error_m3 = 0.
        self.max_geometric_volume_error_m3 = 0.

    def update(self):
        state = self.transfer.snapshot()
        self.samples += 1
        self.max_conservation_error_m3 = max(self.max_conservation_error_m3,
                                              abs(state['conservation_error_m3']))
        error = abs(self.transfer.source.container.volume-state['reservoirs']['source']['volume_m3'])
        self.max_surface_volume_error_m3 = max(self.max_surface_volume_error_m3, error)
        liquid = self.transfer.source.container.liquid
        geometric_volume = 0. if liquid is None else liquid.meshplane.calculate_volume(liquid.surface.distance)[0]
        geometric_error = abs(geometric_volume-state['reservoirs']['source']['volume_m3'])
        if not np.isfinite(geometric_error):
            self.max_geometric_volume_error_m3 = float('inf')
        else:
            self.max_geometric_volume_error_m3 = max(self.max_geometric_volume_error_m3, geometric_error)

    def report(self):
        state = self.transfer.snapshot()
        reservoirs = state['reservoirs']
        tip = reservoirs['tip']['volume_m3']
        checks = {'aspirated_target_within_5pct': abs(tip-self.transfer.tip_capacity_m3) <= self.transfer.tip_capacity_m3*.05,
                  'volume_conserved': self.max_conservation_error_m3 <= 1e-12,
                  'surface_volume_matches_ledger': self.max_surface_volume_error_m3 <= 1e-12,
                  'geometric_surface_volume_matches_ledger': self.max_geometric_volume_error_m3 <= 1e-12,
                  'reservoirs_within_capacity': all(np.isfinite(r['volume_m3']) and r['volume_m3'] >= 0 and
                      (r['capacity_m3'] is None or r['volume_m3'] <= r['capacity_m3']) for r in reservoirs.values()),
                  'no_liquid_expelled_to_environment': reservoirs['environment']['volume_m3'] <= 1e-12}
        return dict(version='hooke-ideal-volume-v2', scope='ideal_volume_accounting', samples=self.samples,
                    checks=checks, success=bool(self.samples and all(checks.values())),
                    failure_reasons=[key for key, value in checks.items() if not value],
                    metrics=dict(state, max_conservation_error_m3=self.max_conservation_error_m3,
                                 max_surface_volume_error_m3=self.max_surface_volume_error_m3,
                                 max_geometric_volume_error_m3=self.max_geometric_volume_error_m3 if np.isfinite(self.max_geometric_volume_error_m3) else None),
                    scientific_process_validated=False)
