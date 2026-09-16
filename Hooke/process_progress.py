"""Shared manipulation history, updated after instrument and liquid systems.

Predicates and reports only read this history; they never advance simulation.
"""
from dataclasses import dataclass, field

import numpy as np

from contact_state import body_geoms, touching
from simulation import System


@dataclass
class PipetteSequence:
    phase: str = 'await_submersion'
    events: dict = field(default_factory=dict)
    clearance_m: float = 0.
    max_depth_m: float = 0.
    liquid_available: bool = True

    def update(self, time, depth, radial, above_bottom, thumb, liquid_available=True):
        self.liquid_available = liquid_available
        if not liquid_available:
            self.phase = 'await_submersion'
            self.events.clear()
            self.clearance_m = 0.
            return
        inside = radial < .0065 and above_bottom
        self.clearance_m = -depth
        if inside:
            self.max_depth_m = max(self.max_depth_m, depth)
        if self.phase == 'await_submersion' and inside and depth > .005 and thumb > .70:
            self.phase = 'await_release'
            self.events['submerged_pressed_s'] = time
        elif self.phase == 'await_release':
            if not inside or depth <= 0:
                # A release in air cannot be credited on subsequent re-entry.
                self.phase = 'await_submersion'
                self.events.clear()
            elif thumb < .45:
                self.phase = 'await_withdrawal'
                self.events['released_under_liquid_s'] = time
        elif self.phase == 'await_withdrawal' and depth < -.05:
            self.phase = 'complete'
            self.events['withdrawn_s'] = time

    def checks(self):
        return {'ordered_submerge_release_withdraw': self.phase == 'complete',
                'final_clearance_50mm': self.clearance_m > .05,
                'liquid_available': self.liquid_available}


@dataclass
class VortexSequence:
    lifted: bool = False
    active_run_s: float = 0.
    longest_active_s: float = 0.
    contact_s: float = 0.
    peak_speed_rad_s: float = 0.
    returned: bool = False
    released: bool = False
    stopped: bool = False

    def update(self, dt, height_gain, contact, speed, return_distance, held):
        self.lifted |= height_gain > .03
        self.peak_speed_rad_s = max(self.peak_speed_rad_s, abs(speed))
        if contact:
            self.contact_s += dt
        active = self.lifted and contact and abs(speed) >= 1.
        self.active_run_s = self.active_run_s + dt if active else 0.
        self.longest_active_s = max(self.longest_active_s, self.active_run_s)
        self.returned = return_distance < .03
        self.released = not held
        self.stopped = abs(speed) < 1.

    def checks(self):
        return {'tube_lifted_30mm': self.lifted,
                'continuous_rotating_tube_contact_500ms': self.longest_active_s >= .5,
                'returned_within_30mm': self.returned,
                'tube_released': self.released, 'platform_stopped': self.stopped}


@dataclass
class PipetteTransferSequence:
    target_m3: float
    aspiration: PipetteSequence = field(default_factory=PipetteSequence)
    phase: str = 'aspirating'
    events: dict = field(default_factory=dict)
    delivered_m3: float = 0.
    tip_volume_m3: float = 0.
    environment_volume_m3: float = 0.
    tip_outside: bool = False
    source_return_distance_m: float | None = None
    source_tilt_rad: float | None = None
    source_released: bool = False
    source_stable_s: float = 0.
    destination_distance_m: float | None = None
    destination_tilt_rad: float | None = None

    def update(self, time, state, target):
        reservoirs = state['reservoirs']
        self.delivered_m3 = reservoirs[target]['volume_m3'] - state['initial_volumes_m3'][target]
        self.tip_volume_m3 = reservoirs['tip']['volume_m3']
        self.environment_volume_m3 = reservoirs['environment']['volume_m3']
        self.tip_outside = state['tip_reservoir'] is None
        if (self.phase == 'aspirating' and all(self.aspiration.checks().values())
                and abs(self.tip_volume_m3-self.target_m3) <= self.target_m3*.05):
            self.phase = 'await_dispense'
            self.events['aspirated_and_withdrawn_s'] = time
        elif (self.phase == 'await_dispense' and state['tip_reservoir'] == target
              and self.delivered_m3 >= self.target_m3*.95 and self.tip_volume_m3 <= 1e-12):
            self.phase = 'await_destination_withdrawal'
            self.events['dispensed_in_destination_s'] = time
        elif self.phase == 'await_destination_withdrawal' and self.tip_outside:
            self.phase = 'complete'
            self.events['withdrawn_from_destination_s'] = time

    def checks(self):
        return {'ordered_aspirate_transfer_dispense_withdraw': self.phase == 'complete',
                'delivered_target_within_5pct': abs(self.delivered_m3-self.target_m3) <= self.target_m3*.05,
                'tip_residual_within_1nl': self.tip_volume_m3 <= 1e-12,
                'no_liquid_expelled_to_environment': self.environment_volume_m3 <= 1e-12,
                'tip_outside_containers': self.tip_outside,
                'source_returned_within_5mm': self.source_return_distance_m is not None and self.source_return_distance_m < .005,
                'source_upright_within_5deg': self.source_tilt_rad is not None and self.source_tilt_rad < float(np.deg2rad(5.)),
                'source_released': self.source_released,
                'source_settled_500ms': self.source_stable_s >= .5,
                'destination_within_5mm': self.destination_distance_m is not None and self.destination_distance_m < .005,
                'destination_upright_within_5deg': self.destination_tilt_rad is not None and self.destination_tilt_rad < float(np.deg2rad(5.))}


class ProcessProgress:
    """One observer per consumer, using actual joints, contacts and geometry."""

    def __init__(self, task, name):
        self.task, self.name = task, name
        self.time = float(task.data.time)
        self.samples = 0
        self.metrics = {}
        model, data = task.model, task.data
        if name == 'pipette':
            self.state = PipetteSequence()
        elif name == 'pipette_transfer':
            self.state = PipetteTransferSequence(task.liquid_transfer.tip_capacity_m3)
            self.source_geoms = body_geoms(model, task.object.body_id)
            self.robot = {i for i in range(model.ngeom)
                          if '/ur:' in model.body(int(model.geom_bodyid[i])).name}
            self.destination_body = model.body('6/centrifuge_50ml_screw_body').id
            self.source_anchor = None
        elif name == 'vortex_mixer':
            self.state = VortexSequence()
            self.tube_body = model.body('1/centrifuge_15ml_body').id
            self.tube = body_geoms(model, self.tube_body)
            self.held_assembly = self.tube | body_geoms(model, model.body('2/centrifuge_15ml_cap').id)
            self.platform = body_geoms(model, model.body('/vortex_mixer_genie_2:platform').id)
            self.speed_adr = int(model.joint('/vortex_mixer_genie_2:platform/pivot').dofadr[0])
            self.initial_height = float(data.xpos[self.tube_body, 2])
            self.return_position = data.site_xpos[model.site('origin1').id].copy()
            self.robot = {i for i in range(model.ngeom)
                          if '/aloha:' in model.body(int(model.geom_bodyid[i])).name}
            self.arms = tuple({i for i in self.robot
                               if f'{arm}/aloha:' in model.body(int(model.geom_bodyid[i])).name}
                              for arm in (1, 2))
            self.inter_arm_contact_s = 0.
        else:
            raise ValueError(f'Unknown process: {name}')

    def update(self):
        data = self.task.data
        now = float(data.time)
        dt = now - self.time
        if dt <= 0:
            return
        self.time = now
        self.samples += 1
        if isinstance(self.state, VortexSequence):
            position = data.xpos[self.tube_body]
            self.inter_arm_contact_s += dt if touching(data, *self.arms) else 0.
            axis = data.xmat[self.tube_body].reshape(3, 3)[:, 2]
            distance = float(np.linalg.norm(position-self.return_position))
            self.metrics.update(inter_arm_contact_s=self.inter_arm_contact_s,
                                final_tube_tilt_rad=float(np.arccos(np.clip(axis[2], -1., 1.))),
                                return_distance_m=distance)
            self.state.update(dt, float(position[2]-self.initial_height),
                              touching(data, self.tube, self.platform), float(data.qvel[self.speed_adr]),
                              distance, touching(data, self.held_assembly, self.robot))
        else:
            self._update_pipette(now)
            if isinstance(self.state, PipetteTransferSequence):
                self._update_transfer_return(dt)

    def _update_pipette(self, now):
        task, data = self.task, self.task.data
        sequence = self.state.aspiration if isinstance(self.state, PipetteTransferSequence) else self.state
        container = task.container.container
        liquid = container.liquid
        tip = data.site_xpos[task.arm1.site_id]
        local = container.rotation_matrix.T @ (tip-container.position)
        depth = float(liquid.surface.distance-local @ liquid.surface_normal) if liquid is not None else 0.
        tube_local = data.xmat[task.object.body_id].reshape(3, 3).T @ (tip-data.xpos[task.object.body_id])
        thumb = float(data.qpos[task.arm1.thj3_qposadr])
        radial = float(np.linalg.norm(tube_local[:2]))
        sequence.update(now, depth, radial, bool(tube_local[2] > 0), thumb, liquid is not None)
        self.metrics.update(tip_depth_m=depth, radial_offset_m=radial, thumb_joint_rad=thumb,
                            thumb_command=float(data.ctrl[task.arm1.thj3_id]))
        if isinstance(self.state, PipetteTransferSequence):
            transfer = task.liquid_transfer
            self.state.update(now, transfer.snapshot(), transfer.target_reservoir)

    def _update_transfer_return(self, dt):
        task, data, state = self.task, self.task.data, self.state
        source_body = task.object.body_id
        state.source_return_distance_m = float(np.linalg.norm(data.xpos[source_body]-task.source_return_position))
        state.source_tilt_rad = float(np.arccos(np.clip(data.xmat[source_body].reshape(3, 3)[2, 2], -1., 1.)))
        state.source_released = not touching(data, self.source_geoms, self.robot)
        settled = (state.phase == 'complete' and state.source_return_distance_m < .005
                   and state.source_tilt_rad < float(np.deg2rad(5.)) and state.source_released)
        if not settled:
            state.source_stable_s, self.source_anchor = 0., None
        elif self.source_anchor is None or np.linalg.norm(data.xpos[source_body]-self.source_anchor) > .0005:
            state.source_stable_s = 0.
            self.source_anchor = data.xpos[source_body].copy()
        else:
            state.source_stable_s += dt
        state.destination_distance_m = float(np.linalg.norm(data.xpos[self.destination_body]-task.destination_position))
        state.destination_tilt_rad = float(np.arccos(np.clip(data.xmat[self.destination_body].reshape(3, 3)[2, 2], -1., 1.)))

    @property
    def success(self):
        return bool(self.samples and all(self.state.checks().values()))


class ProcessProgressSystem(System):
    """Place last in the manager so progress sees updated liquid/instruments."""

    def _configure(self):
        self.task = None
        self.observer = None

    def bind(self, task):
        self.task = task

    def _reset(self, data):
        self.observer = ProcessProgress(self.task, self.task.default_task)

    def _update(self, data):
        self.observer.update()

    @property
    def success(self):
        return self.observer is not None and self.observer.success
