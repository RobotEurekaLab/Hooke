"""Check the ideal-volume model separately from motion and fluid dynamics."""
import numpy as np


class PipetteVolumeAssessment:
    def __init__(self, transfer):
        self.transfer = transfer
        self.samples = 0
        self.max_conservation_error_m3 = 0.
        self.max_surface_volume_error_m3 = 0.
        self.max_geometric_volume_error_m3 = 0.
        self.reservoir_errors = {name: dict(surface_m3=0., geometric_m3=0.)
                                 for name in transfer.containers}

    def update(self):
        state = self.transfer.snapshot()
        self.samples += 1
        self.max_conservation_error_m3 = max(self.max_conservation_error_m3,
                                              abs(state['conservation_error_m3']))
        for name, system in self.transfer.containers.items():
            volume = state['reservoirs'][name]['volume_m3']
            error = abs(system.container.volume-volume)
            liquid = system.container.liquid
            geometric_volume = 0. if liquid is None else liquid.meshplane.calculate_volume(liquid.surface.distance)[0]
            geometric_error = abs(geometric_volume-volume)
            if not np.isfinite(geometric_error):
                geometric_error = float('inf')
            errors = self.reservoir_errors[name]
            errors['surface_m3'] = max(errors['surface_m3'], error)
            errors['geometric_m3'] = max(errors['geometric_m3'], geometric_error)
        self.max_surface_volume_error_m3 = max(e['surface_m3'] for e in self.reservoir_errors.values())
        self.max_geometric_volume_error_m3 = max(e['geometric_m3'] for e in self.reservoir_errors.values())

    def report(self):
        state = self.transfer.snapshot()
        reservoirs = state['reservoirs']
        tip = reservoirs['tip']['volume_m3']
        target = self.transfer.target_reservoir
        if target == 'tip':
            target_checks = {'aspirated_target_within_5pct': abs(tip-self.transfer.tip_capacity_m3) <= self.transfer.tip_capacity_m3*.05}
        else:
            delivered = reservoirs[target]['volume_m3']-state['initial_volumes_m3'][target]
            target_checks = {'delivered_target_within_5pct': abs(delivered-self.transfer.tip_capacity_m3) <= self.transfer.tip_capacity_m3*.05,
                             'tip_residual_within_1nl': tip <= 1e-12}
        checks = {**target_checks, 'volume_conserved': self.max_conservation_error_m3 <= 1e-12,
                  'surface_volume_matches_ledger': self.max_surface_volume_error_m3 <= 1e-12,
                  'geometric_surface_volume_matches_ledger': self.max_geometric_volume_error_m3 <= 1e-12,
                  'reservoirs_within_capacity': all(np.isfinite(r['volume_m3']) and r['volume_m3'] >= 0 and
                      (r['capacity_m3'] is None or r['volume_m3'] <= r['capacity_m3']) for r in reservoirs.values()),
                  'no_liquid_expelled_to_environment': reservoirs['environment']['volume_m3'] <= 1e-12}
        return dict(version='hooke-ideal-volume-v2' if target == 'tip' else 'hooke-ideal-volume-v3', scope='ideal_volume_accounting', samples=self.samples,
                    checks=checks, success=bool(self.samples and all(checks.values())),
                    failure_reasons=[key for key, value in checks.items() if not value],
                    metrics=dict(state, reservoir_errors_m3={name: {key: value if np.isfinite(value) else None
                                 for key, value in errors.items()} for name, errors in self.reservoir_errors.items()},
                                 max_conservation_error_m3=self.max_conservation_error_m3,
                                 max_surface_volume_error_m3=self.max_surface_volume_error_m3,
                                 max_geometric_volume_error_m3=self.max_geometric_volume_error_m3 if np.isfinite(self.max_geometric_volume_error_m3) else None),
                    scientific_process_validated=False)
