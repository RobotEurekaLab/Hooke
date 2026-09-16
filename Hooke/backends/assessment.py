"""Versioned episode evidence without invoking task.check().

Observers consume post-system-update state. Reading a report never advances
history or invokes a task predicate. Pipette and vortex tasks share the same
observation rules; ideal volume geometry is audited separately. These checks
do not validate fluid dynamics or chemical mixing.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field

import numpy as np

from contact_state import body_geoms, touching
from archetypes.centrifuge_insertion import insertion_target
from grasp.quat import quatapply, quatinv

from process_progress import PROCESS_NAMES, ProcessProgress, PipetteSequence, VortexSequence

VERSION = 'hooke-manipulation-v4'


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
class InsertionSequence:
    height_m: float = 0.
    target_distance_m: float = 0.
    tilt_error_rad: float = 0.
    released: bool = False
    stable_s: float = 0.
    anchor: tuple | None = None

    def update(self, dt, height, relative_position, tilt, held):
        self.height_m = float(height)
        self.target_distance_m = float(np.linalg.norm(relative_position))
        self.tilt_error_rad = float(tilt)
        self.released = not held
        seated = (.955 < self.height_m < .961 and self.target_distance_m < .005
                  and self.tilt_error_rad < float(np.deg2rad(5.)) and self.released)
        if not seated:
            self.stable_s, self.anchor = 0., None
        elif self.anchor is None or np.linalg.norm(relative_position - self.anchor) > .0005:
            self.stable_s = 0.
            self.anchor = tuple(float(x) for x in relative_position)
        else:
            self.stable_s += dt

    def checks(self):
        return {'height_955_to_961mm': .955 < self.height_m < .961,
                'target_within_5mm': self.target_distance_m < .005,
                'tube_axis_within_5deg': self.tilt_error_rad < float(np.deg2rad(5.)),
                'tube_released': self.released, 'seated_stable_500ms': self.stable_s >= .5}


class EpisodeAssessment:
    def __init__(self, task, name):
        self.task, self.name = task, name
        self.time = float(task.data.time)
        self.samples = 0
        self.metrics = {}
        self.state = None
        model, data = task.model, task.data
        self.constant = literal_predicate(task.check)
        self.progress = None
        if name in PROCESS_NAMES:
            self.progress = ProcessProgress(task, name)
            self.state = self.progress.state
        elif name in ('close_fume_hood', 'open_fume_hood'):
            self.state = SashManipulation(float(data.qpos[task.sash_jnt_adr]),
                                         0. if name == 'close_fume_hood' else .18)
            self.handle = {model.geom('/fume_hood:handle_bar').id}
            self.gripper = {i for i in range(model.ngeom)
                            if '/ur:2f85:' in model.body(int(model.geom_bodyid[i])).name}
            if not self.gripper:
                raise ValueError('Fume hood assessment could not locate gripper geoms')
        elif name in ('insert_centrifuge_5430', 'composite_insert_centrifuge_5430'):
            self.state = InsertionSequence()
            self.tube = body_geoms(model, int(model.body_weldid[task.tube.body_id]))
            self.robot = {i for i in range(model.ngeom)
                          if '/ur:' in model.body(int(model.geom_bodyid[i])).name}

    def update(self):
        task = self.task
        data = task.data
        now = float(data.time)
        dt = now - self.time
        if dt <= 0:
            return
        self.time = now
        self.samples += 1
        if self.progress is not None:
            self.progress.update()
            self.metrics = self.progress.metrics
        elif isinstance(self.state, SashManipulation):
            self.state.update(float(data.qpos[task.sash_jnt_adr]),
                              touching(data, self.handle, self.gripper), dt)
        elif isinstance(self.state, InsertionSequence):
            actual = task.tube.get_body_pose(data)
            target = insertion_target(task)
            relative = quatapply(quatinv(target.quat), actual.pos - target.pos)
            axis = np.array([0., 0., 1.])
            cosine = np.dot(quatapply(actual.quat, axis), quatapply(target.quat, axis))
            tilt = float(np.arccos(np.clip(cosine, -1., 1.)))
            self.state.update(dt, actual.pos[2], relative, tilt, touching(data, self.tube, self.robot))
            self.metrics.update(target_frame='current_rotor_slot', slot_id=int(task.slot_id),
                                legacy_world_target_distance_m=float(np.linalg.norm(actual.pos - task.final_tar_tubepose.pos)))

    def report(self):
        from dataclasses import asdict
        scope = 'unaudited_legacy'
        checks = {}
        metrics = dict(self.metrics)
        if self.state is not None:
            checks = self.state.checks()
            metrics.update(asdict(self.state))
            scope = ('motion_sequence_proxy' if self.name == 'pipette' else
                     'ideal_liquid_transfer_manipulation' if self.name == 'pipette_transfer' else 'manipulation')
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
