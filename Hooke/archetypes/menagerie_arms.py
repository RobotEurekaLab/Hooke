"""Real (IK-driving, gripper-controlling) arm classes for the vendored
single-arm Menagerie robots (Franka Panda, UFACTORY xArm7), mirroring
`expert_common.UR5eArm`'s interface so existing task recipes (the
lever-lock archetype, etc.) can run against them unchanged.

This is the "usable kinematics" half of the robot picker -- as opposed to
`webui/robot_scene.py`'s visualization-only placement, which bypasses IK
entirely. It only covers fixed-base, single serial-chain arms: making the
humanoid/mobile robots (Unitree G1, PAL Tiago Dual) actually executable is
a separate, larger effort (whole-body or fixed-base-plus-one-arm control),
deliberately out of scope here -- see private/TODO.md.

Both arms reuse `kinematics.IK` directly. That class was already generic
(build_hierarchy works from any root body + site name, for any chain
length) except for one bug: it hardcoded `bounds[:6]`, silently correct
only because the only arm it had ever driven was UR5e's 6-DOF chain. Fixed
in kinematics.py (`bounds[:self.dof]`) as part of this work -- verified via
the existing lever-lock equivalence fixture that this is a no-op for UR5e
before relying on it for these 7-DOF arms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from kinematics import IK, Pose

MODEL_ROOT = Path(__file__).parent.parent / "model"

# Franka Panda ships with no site at its gripper's pinch point (unlike
# xArm7, which already has one -- see XArm7Arm below) -- add one, at the
# conventional ~0.1m offset along the hand's local +z from the flange,
# roughly where the fingertips meet. Approximate, like every such
# hand-picked offset in this codebase; tune against real grasp success if
# it matters later.
_PANDA_SRC = MODEL_ROOT / "robot_menagerie" / "franka_emika_panda" / "panda.xml"
_PANDA_TCP_MARKER = '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'


def panda_with_tcp_site() -> Path:
    """Returns a generated copy of panda.xml with a `tcp` site added at the
    gripper's pinch point, caching the result across calls.

    Panda's own <equality><joint joint1="finger_joint1" joint2="finger_joint2".../>
    (finger-coupling, redundant given the "split" tendon that already moves
    both fingers together) is left in place -- it used to be stripped out
    here to dodge an over-strict assertion in grasp/equality.py's
    build_equality() that assumed a UR5e-specific joint1/joint2 id
    ordering; that assertion was a bug, not a real requirement, and has
    since been fixed at the root (grasp/equality.py), so this copy no
    longer needs to touch the constraint at all."""
    out = _PANDA_SRC.with_name("panda_with_tcp.gen.xml")
    if not out.exists():
        text = _PANDA_SRC.read_text()
        if _PANDA_TCP_MARKER not in text:
            raise RuntimeError(f"Expected hand-body marker not found in {_PANDA_SRC}; it may have changed upstream.")
        new_text = text.replace(
            _PANDA_TCP_MARKER,
            _PANDA_TCP_MARKER + '\n                      <site name="tcp" pos="0 0 0.1" quat="0.7071068 0 0 -0.7071068" group="4"/>',
            1,
        )
        out.write_text(new_text)
    return out


class PandaArm:
    """7-DOF Franka Panda + its own parallel gripper, addressed via an MJCF
    name prefix -- same shape as UR5eArm, so it's a drop-in for any recipe
    written against that interface."""

    dof = 7

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}link0"
        self.site_name = f"{prefix}tcp"
        self.jnt_adr = model.joint(f"{prefix}joint1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}actuator1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}actuator8").id
        self.gripper_jnt_adr = model.joint(f"{prefix}finger_joint1").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self, lows=None, highs=None):
        # Smaller default perturbation than UR5e's: Panda's joint4 has a
        # narrow range (-3.0718, -0.0698) that a UR5e-scale perturbation can
        # push out of bounds.
        lows = lows if lows is not None else (-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1)
        highs = highs if highs is not None else (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        return np.random.uniform(lows, highs)


class XArm7Arm:
    """7-DOF UFACTORY xArm7 + its own gripper. Already ships a `link_tcp`
    site and a 0-255 gripper ctrlrange matching the Robotiq/Panda
    convention used elsewhere in this codebase, so no asset changes were
    needed for this one (unlike Panda)."""

    dof = 7

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}link_base"
        self.site_name = f"{prefix}link_tcp"
        self.jnt_adr = model.joint(f"{prefix}joint1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}act1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}gripper").id
        self.gripper_jnt_adr = model.joint(f"{prefix}left_finger_joint").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self, lows=None, highs=None):
        lows = lows if lows is not None else (-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1)
        highs = highs if highs is not None else (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        return np.random.uniform(lows, highs)
