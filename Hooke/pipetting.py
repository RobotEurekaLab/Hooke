"""Connect actual pipette motion and container geometry to volume accounting."""
import mujoco
import numpy as np

from liquid import ContainerSystem
from liquid_transfer import PistonPipette, VolumeLedger, VolumeState
from simulation import System


class PipetteTransferSystem(System):
    def _configure(self, *, source: ContainerSystem, tip_site: str,
                   plunger_joint: str, tip_capacity_m3: float,
                   destinations: dict[str, ContainerSystem] | None = None,
                   target_reservoir: str = 'tip', endpoint_tolerance_m: float = 0.):
        if not np.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
            raise ValueError('Plunger endpoint tolerance must be finite and nonnegative')
        self.endpoint_tolerance_m = float(endpoint_tolerance_m)
        self.source = source
        if destinations and {'source', 'tip', 'environment'} & destinations.keys():
            raise ValueError('Destination names conflict with the pipette reservoirs')
        self.containers = {'source': source, **(destinations or {})}
        if len({id(system) for system in self.containers.values()}) != len(self.containers):
            raise ValueError('Each reservoir requires a distinct container')
        self.tip_site = tip_site
        self.plunger_joint = plunger_joint
        self.tip_capacity_m3 = tip_capacity_m3
        if target_reservoir not in {'tip', *self.containers} or target_reservoir == 'source':
            raise ValueError('Target must be the tip or a destination container')
        self.target_reservoir = target_reservoir

    def _reload(self, model: mujoco.MjModel):
        self.site_id = model.site(self.tip_site).id
        joint = model.joint(self.plunger_joint)
        self.plunger_adr = int(joint.qposadr[0])
        self.stroke_low, self.stroke_high = model.jnt_range[joint.id]
        if (model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_SLIDE
                or not model.jnt_limited[joint.id] or self.stroke_high <= self.stroke_low):
            raise ValueError('Pipette plunger requires a finite, positive travel range')
        if self.endpoint_tolerance_m >= (self.stroke_high-self.stroke_low)/2:
            raise ValueError('Plunger endpoint tolerance must be less than half the stroke')
        self.boundaries = {}
        for name, system in self.containers.items():
            interior = system.definition._interior
            normals = interior.face_normals.copy()
            self.boundaries[name] = normals, np.einsum('ij,ij->i', normals, interior.triangles_center)

    def _fraction(self, data: mujoco.MjData) -> float:
        return float(np.clip((self.stroke_high-data.qpos[self.plunger_adr]) /
                             (self.stroke_high-self.stroke_low), 0., 1.))

    def _resolved_fraction(self, fraction: float) -> float:
        """Resolve only endpoint motion within the declared numeric tolerance.

        Interior strokes retain their full displacement, including gradual
        motion in increments smaller than the tolerance. Native state is kept
        unchanged; snapshots expose both raw and resolved fractions.
        """
        tolerance = self.endpoint_tolerance_m/(self.stroke_high-self.stroke_low)
        if fraction <= tolerance:
            return 0.
        if fraction >= 1.-tolerance:
            return 1.
        return fraction

    def _reset(self, data: mujoco.MjData):
        reservoirs = {name: VolumeState(float(system.container.volume), float(system.definition.interior.volume*.9))
                      for name, system in self.containers.items()}
        ledger = VolumeLedger({**reservoirs, 'tip': VolumeState(0., self.tip_capacity_m3),
                               'environment': VolumeState(0., None)})
        self.raw_pressed_fraction = self._fraction(data)
        self.pipette = PistonPipette(ledger, 'tip', self._resolved_fraction(self.raw_pressed_fraction))
        self.initial_volumes_m3 = {name: state.volume_m3 for name, state in reservoirs.items()}
        self.tip_reservoir = None
        self.tip_submerged = False
        self.immersed_ticks = 0

    def _update(self, data: mujoco.MjData):
        matches = []
        for name, system in self.containers.items():
            container = system.container
            local = container.rotation_matrix.T @ (data.site_xpos[self.site_id]-container.position)
            normals, offsets = self.boundaries[name]
            if np.all(normals @ local <= offsets+1e-9):
                submerged = (container.liquid is not None and
                             local @ container.liquid.surface_normal < container.liquid.surface.distance)
                matches.append((name, bool(submerged)))
        if len(matches) > 1:
            raise RuntimeError('Pipette tip is inside overlapping containers')
        destination, submerged = matches[0] if matches else ('environment', False)
        self.tip_reservoir = destination if matches else None
        self.tip_submerged = submerged
        self.immersed_ticks += submerged
        self.raw_pressed_fraction = self._fraction(data)
        self.pipette.update(self._resolved_fraction(self.raw_pressed_fraction),
                            destination if submerged else None, destination)
        for name, system in self.containers.items():
            volume = self.pipette.ledger.state(name).volume_m3
            if volume != system.container.volume:
                system.set_volume(data, volume)

    def snapshot(self) -> dict:
        return dict(self.pipette.snapshot(), immersed_ticks=self.immersed_ticks,
                    input='actual_plunger_joint', tip_capacity_m3=self.tip_capacity_m3,
                    target_reservoir=self.target_reservoir,
                    initial_volumes_m3=dict(self.initial_volumes_m3),
                    tip_reservoir=self.tip_reservoir, tip_submerged=self.tip_submerged,
                    plunger_input=dict(mapping='endpoint_tolerance_v1',
                        raw_pressed_fraction=self.raw_pressed_fraction,
                        endpoint_tolerance_m=self.endpoint_tolerance_m,
                        endpoint_tolerance_m3=self.tip_capacity_m3*self.endpoint_tolerance_m /
                                              (self.stroke_high-self.stroke_low)))
