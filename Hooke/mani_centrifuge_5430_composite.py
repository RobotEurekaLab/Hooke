"""Insert a tube, then optionally close the lid using the existing lid recipe.

Insertion targets and controls are shared with the atomic insertion task.
The close-lid step uses the lever-lock recipe and actual lid/lock feedback.
"""
import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose
from task import Task, Expert, Manager, SCENE_ROOT
from expert_common import UR5eArm, set_gravcomp, make_topp_planner
from archetypes.lever_lock_centrifuge import _make_instrument_class, LeverLockMotionMixin
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


class CentrifugeInsertCloseCompositeExpert(CentrifugeInsertCloseComposite, Expert, LeverLockMotionMixin):
    lever_spec = CENTRIFUGE_5430_SPEC

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


    def _step_wait(self, seconds: float):
        for _ in range(int(seconds / self.dt)):
            self.step_and_log({})

    def _execute_close_lid(self):
        # Clear the insertion arm configuration through real actuator motion.
        # Starting the lid recipe from the insertion IK branch jams a follower
        # against the open lid before the finger pads can grasp its edge.
        self.gripper_control(0)
        approach = self.model.key_qpos[0, self.arm.jnt_span].copy()
        self.move_joints(approach)
        if np.max(abs(self.data.qpos[self.arm.jnt_span] - approach)) > .01:
            raise RuntimeError('Arm did not reach the lid approach configuration')
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span].copy()
        self._run_recipe()


CentrifugeInsertCloseComposite.Expert = CentrifugeInsertCloseCompositeExpert

if __name__ == "__main__":
    for name in ['insert_centrifuge_5430_step', 'close_lid_step']:
        spec = CentrifugeInsertCloseComposite.load()
        expert = CentrifugeInsertCloseComposite.Expert(spec)
        expert.task = name
        expert.reset(0)
        expert.execute()
        print(name, expert.check())
