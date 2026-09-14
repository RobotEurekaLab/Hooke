"""Roll out a served policy (real π0/π0.5 checkpoint, or
`mock_policy_server.py`'s dummy one) against a Hooke `Task`, in place of a
hand-written scripted `Expert` -- the missing half of reproducing
AutoBio's Table 4 concatenation finding (private/proposal.tex Phase II):
`archetypes/composite_task.py` already chains scripted experts with no
reset between steps; this chains a *policy* the same way, so the same
composite scenes can eventually show whether a trained policy composes
sub-skills or collapses on the seam the way AutoBio's own baselines did.

Deliberately does not train or load a model itself -- it only knows how
to talk to whatever's listening on the other end of a
`policy_client.WebsocketClientPolicy` connection. Point it at
`mock_policy_server.py` to validate the harness itself (rendering,
wire protocol, action application) with no model or GPU involved; point
it at a real `openpi` `WebsocketPolicyServer` serving a checkpoint when
one exists. Building this harness is infrastructure work, not a training
run, and doesn't need per-run permission the way starting an actual
fine-tune does -- but nothing in this file starts one.

Observation/action shape follows `openpi`'s own UR5 example
(`openpi/examples/ur5/README.md`'s `UR5Inputs`/`UR5Outputs`): state is
`concat(joints, gripper)`, images are `base_rgb` (the task's own
`camera_mapping['image']`) and `wrist_rgb` (`camera_mapping['wrist_image']`),
and the action returned is applied as an absolute target on the arm's
6 joints + gripper -- matching how every scripted `Expert` in this
codebase already drives `data.ctrl` (see `expert_common.py`), so a served
policy is a drop-in replacement for the scripted waypoint sequence, not a
different control convention.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from archetypes.policy_client import WebsocketClientPolicy, ActionChunkBroker, CameraRenderer
from archetypes.task_catalog import CatalogEntry


@dataclasses.dataclass
class RolloutResult:
    steps: list[str]
    completed_steps: int
    per_step_success: list[bool]
    full_sequence_success: bool
    task_infos: list[dict]


class PolicyRolloutExpert:
    """Drives one or more `Task`s (sharing one scene, one `reset()`, the
    same shape `composite_task.py`'s scripted chaining uses) via a served
    policy instead of `Expert.execute()`."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        action_horizon: int = 10,
        img_size: int = 224,
        control_freq: int = 20,
        connect_timeout: float | None = 30.0,
    ):
        raw_policy = WebsocketClientPolicy(host=host, port=port, connect_timeout=connect_timeout)
        self.server_metadata = raw_policy.get_server_metadata()
        self.policy = ActionChunkBroker(raw_policy, action_horizon)
        self.control_freq = control_freq
        self.camera = CameraRenderer(img_size=img_size)

    def get_observation(self, task) -> dict:
        return self.camera.get_observation(task)

    def _apply_action(self, task, action: np.ndarray):
        arm = task.arm
        action = np.asarray(action, dtype=np.float64)
        task.data.ctrl[arm.act_span] = action[:6]
        task.data.ctrl[arm.gripper_id] = action[6]

    def _step_policy_controlled(self, task, seconds: float):
        period = max(1, int(round(1.0 / task.dt / self.control_freq)))
        n_control_steps = int(seconds * self.control_freq)
        for _ in range(n_control_steps):
            obs = self.get_observation(task)
            action = self.policy.infer(obs)["actions"]
            self._apply_action(task, action)
            for _ in range(period):
                mujoco.mj_step(task.model, task.data)
            if task.early_stop and task.check():
                break

    def rollout(self, entry: CatalogEntry, seed: int, seconds: float | None = None) -> RolloutResult:
        """Single-task rollout -- the policy-driven equivalent of
        `entry.make_expert()` + `expert.execute()`."""
        task_cls, _ = entry.load_classes()
        task = task_cls(task_cls.load())
        if entry.task_override:
            task.task = entry.task_override
        task.task_info = task.reset(seed=seed)
        self.policy.reset()
        self._step_policy_controlled(task, seconds or task.time_limit)
        success = bool(task.check())
        return RolloutResult(
            steps=[entry.name], completed_steps=1, per_step_success=[success],
            full_sequence_success=success, task_infos=[task.task_info],
        )

    def rollout_composite(self, steps: list[tuple[CatalogEntry, str]], seed: int, seconds_per_step: float | None = None) -> RolloutResult:
        """Chains `steps` in one continuous episode (one `reset()`, one
        shared `task.model`/`task.data`) the same way
        `composite_task.run_composite_scripted` chains scripted experts --
        see that module's docstring for why every step must share one
        Task class. `steps` is `(entry, task_override)` pairs, matching
        `composite_task.CompositeStep`."""
        if not steps:
            raise ValueError("steps must be non-empty")
        task_cls, _ = steps[0][0].load_classes()
        for entry, _ in steps[1:]:
            other_cls, _ = entry.load_classes()
            if other_cls is not task_cls:
                raise ValueError(
                    "rollout_composite only supports steps that share one Task class -- "
                    f"got both {task_cls.__name__} and {other_cls.__name__}."
                )

        task = task_cls(task_cls.load())
        task.task = steps[0][1]
        task.task_info = task.reset(seed=seed)
        self.policy.reset()

        per_step_success = []
        task_infos = []
        for entry, task_override in steps:
            task.task = task_override
            # Re-derive task_info's prompt/camera mapping for the new step
            # without a real reset -- mirrors composite_task.py's own
            # comment about task_override needing to take effect before
            # any per-task branching runs.
            task.task_info = dict(task.task_info, prefix=task_override.replace('_', ' '))
            self._step_policy_controlled(task, seconds_per_step or task.time_limit)
            success = bool(task.check())
            per_step_success.append(success)
            task_infos.append(dict(task.task_info))
            if not success:
                break

        completed = len(per_step_success)
        return RolloutResult(
            steps=[s[1] for s in steps], completed_steps=completed,
            per_step_success=per_step_success,
            full_sequence_success=completed == len(steps) and all(per_step_success),
            task_infos=task_infos,
        )
