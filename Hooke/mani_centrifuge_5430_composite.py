"""Insert a tube, then optionally close the lid using the existing lid recipe.

Insertion targets and controls are shared with the atomic insertion task.
The close-lid step uses the lever-lock recipe and actual lid/lock feedback.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose, mul_pose, neg_pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, ExpertMotionMixin, set_gravcomp, make_topp_planner
from archetypes.lever_lock_centrifuge import _make_instrument_class, LOCK_QUAT
from archetypes.centrifuge_specs import CENTRIFUGE_5430_SPEC
from archetypes.lid_lock import lid_lock_passes
from load_centrifuge_5430 import CentrifugeTube, GridSlot
from archetypes.centrifuge_insertion import RotorSlotPoses, insertion_geometry_passes, execute_insertion

_LeverLockInstrumentBase = _make_instrument_class(CENTRIFUGE_5430_SPEC)


class _CompositeCentrifuge5430Instrument(RotorSlotPoses, _LeverLockInstrumentBase):
    pass


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
            return insertion_geometry_passes(self)
        elif self.task == 'close_lid_step':
            return lid_lock_passes(self.data, self.instrument)
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

    def _execute_insert(self):
        execute_insertion(self)

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
