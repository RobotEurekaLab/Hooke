"""Versioned episode evidence, separate from the historical task.check().

Observers consume post-system-update state. Reading a report never advances
history or invokes a task predicate. These are manipulation checks, not
validation of transferred liquid volume or chemical mixing.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field

import numpy as np

VERSION = 'hooke-manipulation-v2'


def literal_predicate(check):
    """Recognize only an unconditional literal return; never execute check."""
    try:
        body = ast.parse(textwrap.dedent(inspect.getsource(check))).body[0].body
        body = [x for x in body if not isinstance(x, ast.Expr)
                or not isinstance(x.value, ast.Constant) or not isinstance(x.value.value, str)]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            if isinstance(value, ast.Constant) and type(value.value) is bool:
                return value.value
    except (OSError, TypeError, IndentationError, SyntaxError):
        pass
    return None


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
class SashManipulation:
    initial: float
    target: float
    previous: float = field(init=False)
    position: float = field(init=False)
    contact_s: float = 0.
    travel_m: float = 0.
    previous_contact: bool = False

    def __post_init__(self):
        self.previous = self.position = self.initial

    def update(self, position, contact, dt):
        direction = 1 if self.target > self.initial else -1
        if contact and self.previous_contact:
            self.contact_s += dt
            # Signed displacement: jitter/repeated back-and-forth cannot accrue
            # artificial progress. Both interval endpoints must have contact.
            self.travel_m += direction * (position - self.previous)
        self.position = self.previous = position
        self.previous_contact = contact

    def checks(self):
        return {'target_within_20mm': abs(self.position - self.target) < .02,
                'handle_contact_50ms': self.contact_s >= .05,
                'travel_in_contact_50mm': self.travel_m >= .05}


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


def body_geoms(model, root):
    bodies = {int(root)}
    for i in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    return {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies}


def touching(data, first, second):
    return any((int(c.geom[0]) in first and int(c.geom[1]) in second)
               or (int(c.geom[1]) in first and int(c.geom[0]) in second)
               for c in data.contact)


class EpisodeAssessment:
    def __init__(self, task, name):
        self.task, self.name = task, name
        self.time = float(task.data.time)
        self.samples = 0
        self.metrics = {}
        self.state = None
        model, data = task.model, task.data
        self.constant = literal_predicate(task.check)
        if name == 'pipette':
            self.state = PipetteSequence()
        elif name in ('close_fume_hood', 'open_fume_hood'):
            self.state = SashManipulation(float(data.qpos[task.sash_jnt_adr]),
                                         0. if name == 'close_fume_hood' else .18)
            self.handle = {model.geom('/fume_hood:handle_bar').id}
            self.gripper = {i for i in range(model.ngeom)
                            if '/ur:2f85:' in model.body(int(model.geom_bodyid[i])).name}
            if not self.gripper:
                raise ValueError('Fume hood assessment could not locate gripper geoms')
        elif name == 'vortex_mixer':
            self.state = VortexSequence()
            self.tube_body = model.body('1/centrifuge_15ml_body').id
            self.tube = body_geoms(model, self.tube_body)
            self.platform = body_geoms(model, model.body('/vortex_mixer_genie_2:platform').id)
            joint = model.joint('/vortex_mixer_genie_2:platform/pivot')
            self.speed_adr = int(joint.dofadr[0])
            self.initial_height = float(data.xpos[self.tube_body, 2])
            self.return_position = data.site_xpos[model.site('origin1').id].copy()
            self.robot = {i for i in range(model.ngeom)
                          if '/aloha:' in model.body(int(model.geom_bodyid[i])).name}

    def update(self):
        task = self.task
        data = task.data
        now = float(data.time)
        dt = now - self.time
        if dt <= 0:
            return
        self.time = now
        self.samples += 1
        if isinstance(self.state, PipetteSequence):
            container = task.container.container
            if container.liquid is None:
                self.state.update(now, 0., 0., False, 0., liquid_available=False)
                return
            tip = data.site_xpos[task.arm1.site_id]
            local = container.rotation_matrix.T @ (tip - container.position)
            depth = float(container.liquid.surface.distance - local @ container.liquid.surface_normal)
            tube_local = data.xmat[task.object.body_id].reshape(3, 3).T @ (tip - data.xpos[task.object.body_id])
            thumb = float(data.qpos[task.arm1.thj3_qposadr])
            radial = float(np.linalg.norm(tube_local[:2]))
            self.state.update(now, depth, radial, bool(tube_local[2] > 0), thumb)
            self.metrics.update(tip_depth_m=depth, radial_offset_m=radial, thumb_joint_rad=thumb,
                                thumb_command=float(data.ctrl[task.arm1.thj3_id]))
        elif isinstance(self.state, SashManipulation):
            self.state.update(float(data.qpos[task.sash_jnt_adr]),
                              touching(data, self.handle, self.gripper), dt)
        elif isinstance(self.state, VortexSequence):
            position = data.xpos[self.tube_body]
            self.state.update(dt, float(position[2] - self.initial_height),
                              touching(data, self.tube, self.platform), float(data.qvel[self.speed_adr]),
                              float(np.linalg.norm(position - self.return_position)),
                              touching(data, self.tube, self.robot))
        elif self.name in ('insert_centrifuge_5430', 'composite_insert_centrifuge_5430'):
            position = task.tube.get_body_pose(data).pos
            self.metrics.update(tube_height_m=float(position[2]),
                                target_distance_m=float(np.linalg.norm(position - task.final_tar_tubepose.pos)))

    def report(self):
        from dataclasses import asdict
        scope = 'unaudited_legacy'
        checks = {}
        metrics = dict(self.metrics)
        if self.state is not None:
            checks = self.state.checks()
            metrics.update(asdict(self.state))
            scope = 'motion_sequence_proxy' if self.name == 'pipette' else 'manipulation'
        elif self.name in ('insert_centrifuge_5430', 'composite_insert_centrifuge_5430') and metrics:
            scope = 'legacy_insertion_geometry'
            checks = {'height_955_to_961mm': .955 < metrics['tube_height_m'] < .961,
                      'target_within_5mm': metrics['target_distance_m'] < .005}
        audited = bool(checks)
        failures = [key for key, value in checks.items() if not value]
        within_limit = self.time <= self.task.time_limit
        return {'version': VERSION, 'scope': scope, 'samples': self.samples,
                'legacy_constant': self.constant, 'checks': checks, 'metrics': metrics,
                'failure_reasons': failures,
                'success': bool(not failures and self.samples) if audited else None,
                'within_declared_time_limit': bool(within_limit),
                'success_within_time_limit': bool(not failures and self.samples and within_limit) if audited else None,
                'scientific_process_validated': False}
