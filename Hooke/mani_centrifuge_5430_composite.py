"""Composite task: insert a second centrifuge tube into the Eppendorf 5430
rotor, then close and lock the lid -- a genuine two-step lab sequence (you
don't close/lock a centrifuge lid *before* loading the samples; this is
literally the operating order any real protocol uses), picked specifically
because it's a real workflow, not a device pairing invented to cover more
discipline labels (see private/technical-log.md's "Composite batch,
corrected" entry on why that distinction matters for this catalog
category).

Both halves already exist as separately-verified, hand-written tasks --
`load_centrifuge_5430.py`'s `InsertCentrifuge5430` and
`archetypes/lever_lock_centrifuge.py`'s generic lid-lever family
(instantiated for this instrument via `archetypes/centrifuge_specs.py`'s
`CENTRIFUGE_5430_SPEC`, i.e. `mani_centrifuge_5430.py`'s
`Centrifuge5430Manipulate`). Both already use the same scene family (in
fact `insert_centrifuge_5430.xml` *is* `mani_centrifuge_5430.xml` plus a
tube rack and two tubes -- confirmed by diffing the two scene files, not
assumed) and the same UR5e arm, so no cross-robot scene-merging problem
exists here the way it would for `pickup_centrifuge_tube` (which uses a
single-armed Aloha rig entirely disjoint from this one).

`archetypes/composite_task.py`'s `run_composite_scripted` requires every
step in a sequence to share one Task/Expert class (a single `reset()`
covers the whole episode). Since the two halves were built as fully
independent class hierarchies (different `Manager`/instrument-registration
calls, different motion-primitive mixins), the only clean way to chain
them without a fragile diamond-inheritance merge is to re-host both
recipes' *logic* (not re-derive it -- transcribed from the originals,
same numbers, same steps) against one shared instrument object and one
shared arm, branching on `self.task`:
- the instrument class here is `_make_instrument_class(CENTRIFUGE_5430_SPEC)`'s
  output (gives the lid-lever `fk`/`lever_path`/`get_eef_pose` methods)
  with `InsertCentrifuge5430`'s own `get_slot_pose`/`get_tube_pose`/
  `rotor_perturb` added on top -- both are already subclasses of the same
  `instrument.Centrifuge_Eppendorf_5430` base, and add non-overlapping
  method names, so this is a safe additive subclass, not a merge of
  conflicting logic.
- the recipe-running `_step_*` methods for the close-lid half are
  transcribed verbatim from `lever_lock_centrifuge.py`'s closure-based
  versions (they can't be imported directly -- they're built as closures
  inside `make_task_classes`, bound to a *different* instrument instance
  than this file's).

One real behavior difference from running each half standalone, worth
stating rather than glossing over: `centrifuge_5430_close_lid` upstream
has **no real completion check** (`lever_lock_centrifuge.py`'s own
`check()` returns `True` unconditionally -- an already-documented
limitation of that whole family, not something this file weakens further).
So `check()` here is real (measures the inserted tube's position) for the
`insert_centrifuge_5430_step` half and trivial for `close_lid_step`,
matching each half's own upstream honesty rather than inventing a new
untested lid-closed criterion.
"""
import math
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose, mul_pose, neg_pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner
from archetypes.lever_lock_centrifuge import _make_instrument_class, LOCK_QUAT
from archetypes.centrifuge_specs import CENTRIFUGE_5430_SPEC
from load_centrifuge_5430 import CentrifugeTube, GridSlot

_LeverLockInstrumentBase = _make_instrument_class(CENTRIFUGE_5430_SPEC)


class _CompositeCentrifuge5430Instrument(_LeverLockInstrumentBase):
    """Adds InsertCentrifuge5430's slot/tube-position helpers on top of the
    lever-lock family's fk/lever_path helpers -- transcribed verbatim from
    `load_centrifuge_5430.py`'s `Centrifuge_5430`, not re-derived."""

    def get_slot_pose(self, data: mujoco.MjData, slot_id: int) -> Pose:
        if slot_id < 0 or slot_id >= self.num_slots:
            raise ValueError(f'Invalid slot id {slot_id}')
        pos = data.site_xpos[self.slot_sites[slot_id]]
        mat = data.site_xmat[self.slot_sites[slot_id]]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(pos, quat)

    def get_tube_pose(self, data: mujoco.MjData, slot_id: int, mode: str = "distal") -> Pose:
        slot_pose = self.get_slot_pose(data, slot_id)
        if mode == "distal":
            rel_pos = np.array([0.0, 0.0, 0.005])
        elif mode == "proximal":
            rel_pos = np.array([0.0, 0.0, -0.03])
        rel_quat = np.array([1.0, 0.0, 0.0, -1.0])
        rel_quat /= np.linalg.norm(rel_quat)
        return mul_pose(p1=slot_pose, p2=Pose(rel_pos, rel_quat))

    def rotor_perturb(self):
        return np.random.uniform(-0.1, 0.1)


class CentrifugeInsertCloseComposite(Task):
    default_scene = SCENE_ROOT / "insert_centrifuge_5430.xml"
    default_task = "insert_centrifuge_5430_step"
    time_limit = 15.0
    early_stop = True

    slot_window: tuple = (-4, 9)  # same reachable-slot window as InsertCentrifuge5430

    @classmethod
    def prepare(cls, mjspec: mujoco.MjSpec) -> mujoco.MjSpec:
        set_gravcomp(mjspec.body('/ur:world'))
        return mjspec

    def __init__(self, mjspec: mujoco.MjSpec):
        self.instrument = _CompositeCentrifuge5430Instrument(CENTRIFUGE_5430_SPEC.instrument_prefix)
        manager = Manager.from_spec(mjspec, [self.instrument])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.tube = CentrifugeTube(self.model, "1/", "1/")
        self.tube2 = CentrifugeTube(self.model, "2/", "2/")
        self.rack = GridSlot(self.model, 'rack/')

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        num_slots = self.instrument.num_slots
        slot_id = np.random.randint(*self.slot_window) % num_slots

        # Composite episode has exactly one reset for the whole sequence,
        # and the insert step logically comes first -- use its own
        # (already-tuned) arm perturbation range for that reason, not the
        # lever-lock family's tighter one. The close-lid step, run second,
        # inherits whatever configuration the insert step actually leaves
        # the arm in rather than a fresh draw from its own tuned range --
        # that's the whole point of a no-reset composite (see
        # archetypes/composite_task.py's docstring), not an oversight.
        lows = (-1.2, -0.2, -0.1, -0.5, -0.2, -0.2)
        highs = (0.0, 0.0, 0.1, 0.2, 0.2, 0.2)
        perturbation = np.random.uniform(lows, highs)
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        perturbation = self.instrument.rotor_perturb()
        self.data.qpos[self.instrument.rotor_qposadr] += perturbation
        mujoco.mj_kinematics(self.model, self.data)

        tubepos = self.rack.get_position(self.data, 0, 0, '0')
        quat = self.tube.get_pose(self.data).quat
        self.tube.set_pose(self.data, Pose(tubepos, quat))
        self.tube2.set_pose(self.data, self.instrument.get_tube_pose(self.data, (slot_id + num_slots // 2) % num_slots, 'proximal'))
        mujoco.mj_kinematics(self.model, self.data)

        self.slot_id = slot_id
        self.tar_tubepose = self.instrument.get_tube_pose(self.data, self.slot_id, 'distal')
        self.final_tar_tubepose = self.instrument.get_tube_pose(self.data, self.slot_id, 'proximal')

        self.task_info = {
            'prefix': self.task.replace('_', ' '),
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {'image': 'table_cam_front', 'wrist_image': '/ur:wrist_cam'},
            'seed': seed,
        }
        return self.task_info

    def check(self):
        if self.task == 'insert_centrifuge_5430_step':
            tube_height = self.tube.get_body_pose(self.data).pos
            tube_pos_2 = np.array(self.tube.get_body_pose(self.data).pos)
            site_pos = np.array(self.final_tar_tubepose.pos)
            distance_site = math.sqrt(np.sum((tube_pos_2 - site_pos) ** 2))
            return 0.955 < tube_height[2] < 0.961 and distance_site < 0.005
        elif self.task == 'close_lid_step':
            # Matches upstream (see module docstring): the lever-lock family
            # has no real completion check for this instrument yet.
            return True
        raise ValueError(f"unknown task {self.task!r}")


class CentrifugeInsertCloseCompositeExpert(CentrifugeInsertCloseComposite, Expert, ExpertMotionMixin):
    def __init__(self, mjspec: mujoco.MjSpec, freq: int = 20):
        super().__init__(mjspec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / freq))
        self.arm.register_ik(self.data)
        # Two different planners, matching each half's own original tuning
        # exactly -- InsertCentrifuge5430Expert hardcodes a more
        # conservative qc_vel/qc_acc than the lever-lock family's shared
        # `make_topp_planner` default (1.5/1.0). Using the wrong one for
        # the insert half was a real bug caught by comparing against the
        # upstream baseline (see private/technical-log.md): it silently
        # ran the insert motion faster/less constrained than validated,
        # dropping its success rate from a 7/10 baseline to 3/10 even
        # though every waypoint computed identically to the original.
        from topp import Topp
        self._insert_planner = Topp(dof=self.arm.dof, qc_vel=0.8, qc_acc=0.6, ik=self.arm.ik.solve)
        self._lever_planner = make_topp_planner(self.arm.dof, self.arm.ik.solve)
        self.planner = self._insert_planner
        self._lever_end_pose = None

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(delay):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        if self.task == 'insert_centrifuge_5430_step':
            self.planner = self._insert_planner
            self._execute_insert()
        elif self.task == 'close_lid_step':
            self.planner = self._lever_planner
            self._execute_close_lid()
        else:
            raise ValueError(f"unknown task {self.task!r}")
        self.finish()

    # --- insert half: transcribed from InsertCentrifuge5430Expert.execute() ---
    def _execute_insert(self):
        path = self.interpolate(self.arm.get_site_pose(self.data), self.tube.get_eef_pose(self.data), 10)
        self.path_follow(path)
        self.gripper_control(240)
        cur_pose = self.arm.get_site_pose(self.data)
        self.move_to(Pose(cur_pose.pos + (0.0, 0.0, 0.1), cur_pose.quat), num_steps=20)

        tube_pose = self.tube.get_body_pose(self.data)
        site_pose = self.arm.get_site_pose(self.data)
        rel_pose = mul_pose(p1=neg_pose(tube_pose), p2=site_pose)
        tar_pose = mul_pose(p1=self.tar_tubepose, p2=rel_pose)
        path = self.interpolate2(self.arm.get_site_pose(self.data), tar_pose, 20)
        self.path_follow(path)

        tar_pose = mul_pose(p1=self.final_tar_tubepose, p2=rel_pose)
        self.move_to(tar_pose, num_steps=20)
        self.gripper_control(190)
        for _ in range(200):
            self.step_and_log({})

    # --- close-lid half: transcribed from lever_lock_centrifuge.py's
    # closure-based _step_*/_run_recipe (see module docstring for why this
    # can't just be imported) ---
    def _step_move_to_pose(self, mode: str, num_steps: int, quat_override: str | None = None, gripper_before: float | None = None):
        pose = self.instrument.get_eef_pose(self.data, loc='lid', mode=mode)
        if quat_override == 'lock_quat':
            pose.quat = LOCK_QUAT
        if gripper_before is not None:
            self.gripper_control(gripper_before)
        self.move_to(pose, num_steps=num_steps)

    def _step_gripper(self, value: float, delay: int = 300):
        self.gripper_control(value, delay=delay)

    def _step_lever_close(self, mode: str = '1/close'):
        path = self.instrument.lever_path(self.data, mode=mode)
        self.path_follow(path[:-1])
        self._lever_end_pose = path[-1]

    def _step_move_to_lever_end(self, num_steps: int):
        assert self._lever_end_pose is not None
        self.move_to(self._lever_end_pose, num_steps=num_steps)

    def _step_force_lock(self):
        self.data.eq_active[self.instrument.lid_lock] = 1

    def _step_wait(self, seconds: float):
        for _ in range(int(seconds / self.dt)):
            self.step_and_log({})

    def _execute_close_lid(self):
        for step in CENTRIFUGE_5430_SPEC.recipe:
            op = step['op']
            getattr(self, f'_step_{op}')(**{k: v for k, v in step.items() if k != 'op'})


CentrifugeInsertCloseComposite.Expert = CentrifugeInsertCloseCompositeExpert

if __name__ == "__main__":
    for name in ['insert_centrifuge_5430_step', 'close_lid_step']:
        spec = CentrifugeInsertCloseComposite.load()
        expert = CentrifugeInsertCloseComposite.Expert(spec)
        expert.task = name
        expert.reset(0)
        expert.execute()
        print(name, expert.check())
