"""Real (IK-driving, gripper-controlling) arm classes for the vendored
single-arm Menagerie robots, mirroring `expert_common.UR5eArm`'s interface
so existing task recipes (the lever-lock archetype, etc.) can run against
them unchanged.

This is the "usable kinematics" half of the robot picker -- as opposed to
`webui/robot_scene.py`'s visualization-only placement, which bypasses IK
entirely. It only covers fixed-base, single serial-chain arms: making the
humanoid/mobile robots (Unitree G1, PAL Tiago Dual) actually executable is
a separate, larger effort (whole-body or fixed-base-plus-one-arm control),
deliberately out of scope here -- see private/TODO.md.

Every arm reuses `kinematics.IK` directly. That class was already generic
(build_hierarchy works from any root body + site name, for any chain
length) except for one bug: it hardcoded `bounds[:6]`, silently correct
only because the only arm it had ever driven was UR5e's 6-DOF chain. Fixed
in kinematics.py (`bounds[:self.dof]`) as part of this work -- verified via
the existing lever-lock equivalence fixture that this is a no-op for UR5e
before relying on it for 7-DOF arms.

Franka Panda and UFACTORY xArm7 ship with their own gripper already
attached. Kinova Gen3, KUKA iiwa14, and Flexiv Rizon4 don't (Menagerie
vendors the bare arm only), so this module attaches the same Robotiq
2F-85 gripper `ur5e_gripper.xml` already uses (`model/robot/2f85.xml`) at
each arm's own flange site, via the identical `<attach>` pattern that file
uses for UR5e. UFACTORY Lite6 ships with its own two-finger gripper, but
driven by a force-controlled `<motor>` rather than this codebase's
established 0-255 position-servo-on-a-coupling-tendon convention (see
`ExpertMotionMixin.gripper_control`), so its generator rewrites just that
one actuator to match, on the theory that a shared interface should mean a
shared *control* convention too, not just shared method names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from kinematics import IK, Pose

MODEL_ROOT = Path(__file__).parent.parent / "model"
_2F85_ASSET = '<model name="2f85" file="../../robot/2f85.xml" content_type="text/xml"/>'


class _MenagerieArm:
    """Shared plumbing for every arm class in this module: identical
    `register_ik`/`get_site_pose`/`qpos_perturb` implementations that used
    to be copy-pasted per class (PandaArm, XArm7Arm) -- now that there are
    several more of these, factoring them out actually pays for itself.
    Subclasses just set `dof`, `base_name`, `site_name` and the various
    `*_id`/`*_adr` fields in their own `__init__`."""

    dof: int
    base_name: str
    site_name: str

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self, lows=None, highs=None):
        lows = lows if lows is not None else (-0.1,) * self.dof
        highs = highs if highs is not None else (0.1,) * self.dof
        return np.random.uniform(lows, highs)


def _attach_2f85_gripper(src: Path, out_name: str, site_marker: str, frame_pos: str, frame_quat: str) -> Path:
    """Returns a generated copy of `src` with a Robotiq 2F-85 gripper
    attached at the site declared by `site_marker` (that site's own exact
    `<site .../>` text, used both to locate the insertion point and,
    implicitly, confirm the site hasn't moved upstream). `frame_pos`/
    `frame_quat` place the gripper there -- the same site the marker names,
    since MuJoCo has no "attach a model at this existing site" primitive,
    only "attach at this body/frame pose" (mirrors `ur5e_gripper.xml`'s own
    `<frame><attach .../></frame>` pattern for the exact same gripper)."""
    out = src.with_name(out_name)
    if not out.exists():
        text = src.read_text()
        if site_marker not in text:
            raise RuntimeError(f"Expected site marker not found in {src}; it may have changed upstream.")
        if "</asset>" not in text:
            raise RuntimeError(f"Expected </asset> in {src}")
        new_text = text.replace("</asset>", f"  {_2F85_ASSET}\n  </asset>", 1)
        attach = (
            f'\n<frame pos="{frame_pos}" quat="{frame_quat}">'
            f'<attach model="2f85" body="world" prefix="2f85:"/></frame>'
        )
        new_text = new_text.replace(site_marker, site_marker + attach, 1)
        out.write_text(new_text)
    return out

# Franka Panda ships with no site at its gripper's pinch point (unlike
# xArm7, which already has one -- see XArm7Arm below) -- add one, at the
# conventional ~0.1m offset along the hand's local +z from the flange,
# roughly where the fingertips meet. The offset magnitude is approximate
# (hand-picked, like every such offset in this codebase), but the site's
# *orientation* is deliberately identity, not a hand-picked rotation: an
# earlier version of this rotated it -90deg/+90deg about Z, reasoning from
# the raw joint axis attributes (Panda's fingers: axis="0 1 0"; Robotiq's:
# axis="1 0 0") that the two conventions were 90deg apart. That reasoning
# skipped over how each site's own orientation composes with those joints.
# Measuring the actual open/close axis in each TCP site's own local frame
# (finger-body world positions, projected through the site's rotation
# matrix) shows they already agree on the Y axis with an *identity* Panda
# site -- the earlier Z-rotation was actively wrong, not merely
# unnecessary: it remapped the true open axis onto the wrong local axis,
# which is what made Panda's open gripper crash into the (thin, flat)
# grasp target at the "grip" pose in every recipe (confirmed via a direct
# test: solving IK for that exact pose and checking contacts with the
# gripper held open showed 0 contacts for UR5e/2f85 and ~19mm of
# penetration on both fingers for Panda with the old rotated site -- 0
# contacts once the site went back to identity). See
# private/technical-log.md for the full measurement.
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
            _PANDA_TCP_MARKER + '\n                      <site name="tcp" pos="0 0 0.1" quat="1 0 0 0" group="4"/>',
            1,
        )
        out.write_text(new_text)
    return out


class PandaArm(_MenagerieArm):
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

    def qpos_perturb(self, lows=None, highs=None):
        # Smaller default perturbation than UR5e's: Panda's joint4 has a
        # narrow range (-3.0718, -0.0698) that a UR5e-scale perturbation can
        # push out of bounds.
        lows = lows if lows is not None else (-0.1,) * self.dof
        highs = highs if highs is not None else (0.1,) * self.dof
        return np.random.uniform(lows, highs)


class XArm7Arm(_MenagerieArm):
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


# --- Kinova Gen3 (7-DOF, bare arm -- gets a 2F-85 like ur5e_gripper.xml's UR5e) ---

_GEN3_SRC = MODEL_ROOT / "robot_menagerie" / "kinova_gen3" / "gen3.xml"
_GEN3_PINCH_SITE = '<site name="pinch_site" pos="0 0 -0.061525" quat="0 1 0 0"/>'


def gen3_with_2f85() -> Path:
    """Gen3 ships a `pinch_site` on its wrist body marking where a gripper
    is meant to mount (pos/quat below are that site's own, reused verbatim
    so the gripper sits exactly where Kinova's own file says it should,
    not a guessed offset) but no gripper -- attach the same 2F-85
    `ur5e_gripper.xml` already uses."""
    return _attach_2f85_gripper(
        _GEN3_SRC, "gen3_with_2f85.gen.xml", _GEN3_PINCH_SITE,
        frame_pos="0 0 -0.061525", frame_quat="0 1 0 0",
    )


class KinovaGen3Arm(_MenagerieArm):
    """7-DOF Kinova Gen3 + an attached Robotiq 2F-85 (see gen3_with_2f85)."""

    dof = 7

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}base_link"
        self.site_name = f"{prefix}2f85:pinch"
        self.jnt_adr = model.joint(f"{prefix}joint_1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}joint_1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}2f85:fingers_actuator").id
        self.gripper_jnt_adr = model.joint(f"{prefix}2f85:right_driver_joint").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None


# --- KUKA iiwa14 (7-DOF, bare arm -- gets a 2F-85) ---

_IIWA14_SRC = MODEL_ROOT / "robot_menagerie" / "kuka_iiwa_14" / "iiwa14.xml"
_IIWA14_ATTACHMENT_SITE = '<site pos="0 0 0.045" name="attachment_site"/>'


def iiwa14_with_2f85() -> Path:
    """Same treatment as Gen3: iiwa14 ships its own `attachment_site` on
    link7 (reused verbatim below), no gripper."""
    return _attach_2f85_gripper(
        _IIWA14_SRC, "iiwa14_with_2f85.gen.xml", _IIWA14_ATTACHMENT_SITE,
        frame_pos="0 0 0.045", frame_quat="1 0 0 0",
    )


class KukaIiwa14Arm(_MenagerieArm):
    """7-DOF KUKA iiwa14 + an attached Robotiq 2F-85 (see iiwa14_with_2f85)."""

    dof = 7

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}base"
        self.site_name = f"{prefix}2f85:pinch"
        self.jnt_adr = model.joint(f"{prefix}joint1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}actuator1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}2f85:fingers_actuator").id
        self.gripper_jnt_adr = model.joint(f"{prefix}2f85:right_driver_joint").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None


# --- Flexiv Rizon4 (7-DOF, bare arm, *no* flange site at all -- gets one, then a 2F-85) ---

_RIZON4_SRC = MODEL_ROOT / "robot_menagerie" / "flexiv_rizon4" / "flexiv_rizon4.xml"
_RIZON4_LINK7_MARKER = '<joint name="joint7" axis="0 0 1" range="-3.0543 3.0543" class="joint3"/>'
# Unlike Gen3/iiwa14, Rizon4's vendored file has no flange/attachment site
# at all to reuse -- link7's own geom bounding radius is ~0.082m, so 0.08m
# along its local +z (joint7's own axis, the natural "outward" direction
# every arm in this file uses) is an order-of-magnitude-correct approximate
# offset, hand-picked the same way Panda's tcp site was; there's no vendor
# spec value here to reuse instead. Tune against real grasp success if it
# matters later.
_RIZON4_FLANGE_SITE = '<site name="attachment_site" pos="0 0 0.08"/>'


def rizon4_with_2f85() -> Path:
    out = _RIZON4_SRC.with_name("flexiv_rizon4_with_2f85.gen.xml")
    if not out.exists():
        text = _RIZON4_SRC.read_text()
        if _RIZON4_LINK7_MARKER not in text:
            raise RuntimeError(f"Expected joint7 marker not found in {_RIZON4_SRC}; it may have changed upstream.")
        text = text.replace(_RIZON4_LINK7_MARKER, _RIZON4_LINK7_MARKER + "\n" + _RIZON4_FLANGE_SITE, 1)
        _RIZON4_SRC.with_name("_rizon4_staged.tmp.xml").write_text(text)
        staged = _RIZON4_SRC.with_name("_rizon4_staged.tmp.xml")
        try:
            out = _attach_2f85_gripper(
                staged, "flexiv_rizon4_with_2f85.gen.xml", _RIZON4_FLANGE_SITE,
                frame_pos="0 0 0", frame_quat="1 0 0 0",
            )
        finally:
            staged.unlink(missing_ok=True)
    return out


class FlexivRizon4Arm(_MenagerieArm):
    """7-DOF Flexiv Rizon4 + an attached Robotiq 2F-85 (see rizon4_with_2f85)."""

    dof = 7

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}base"
        self.site_name = f"{prefix}2f85:pinch"
        self.jnt_adr = model.joint(f"{prefix}joint1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}joint1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}2f85:fingers_actuator").id
        self.gripper_jnt_adr = model.joint(f"{prefix}2f85:right_driver_joint").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None


# --- UFACTORY Lite6 (6-DOF, own gripper, but force- not position-controlled) ---

_LITE6_SRC = MODEL_ROOT / "robot_menagerie" / "ufactory_lite6" / "lite6_gripper_narrow.xml"
_LITE6_MOTOR = (
    '<motor name="gripper" joint="gripper_left_finger" forcerange="-10 10" ctrlrange="-10 10"/>'
)
# Lite6's own finger joints have opposite-signed ranges (left: -0.0081..-1e-5,
# right: 1e-5..0.0081) and an <equality> that ties them as left = -right, so
# a "split" tendon needs opposite-signed coefficients (unlike every other
# gripper in this codebase, whose two finger joints share the same sign and
# so use +0.5/+0.5) to combine them into one monotonic "openness" scalar.
# gainprm/biasprm follow 2f85.xml's own derivation
# (gainprm[0] = kp * joint_range / 255, biasprm = [0, -kp, -kv]), just
# recomputed for this joint's own (much smaller, ~8mm) travel instead of
# 2f85's 0.8 rad -- same method, different physical scale.


def lite6_with_position_gripper() -> Path:
    """Returns a generated copy of lite6_gripper_narrow.xml with its native
    force-controlled `<motor>` gripper actuator replaced by a position
    servo on a coupling tendon, matching every other gripper in this
    codebase's 0-255 convention (see module docstring). Also names the
    6 arm actuators (the vendored file leaves them nameless, addressable
    only by declaration order in MJCF terms -- but this codebase's arm
    classes address everything by name, e.g. `model.actuator(f"{prefix}joint1")`,
    so nameless actuators would be unreachable once attached)."""
    out = _LITE6_SRC.with_name("lite6_gripper_narrow_position.gen.xml")
    if not out.exists():
        text = _LITE6_SRC.read_text()
        for i in range(1, 7):
            marker = f'<position joint="joint{i}"'
            if marker not in text:
                raise RuntimeError(f"Expected {marker!r} not found in {_LITE6_SRC}; it may have changed upstream.")
            text = text.replace(marker, f'<position name="joint{i}" joint="joint{i}"', 1)
        if _LITE6_MOTOR not in text:
            raise RuntimeError(f"Expected gripper motor actuator not found in {_LITE6_SRC}; it may have changed upstream.")
        kp = 100.0
        joint_range = 0.0081 - 1e-5
        gain = kp * joint_range / 255
        new_actuator = (
            f'<general class="lite6" name="gripper" tendon="split" forcerange="-10 10" ctrlrange="0 255" '
            f'gainprm="{gain:.8f} 0 0" biasprm="0 -{kp:g} -10"/>'
        )
        text = text.replace(_LITE6_MOTOR, new_actuator, 1)
        if "</worldbody>" not in text:
            raise RuntimeError(f"Expected </worldbody> in {_LITE6_SRC}")
        tendon_block = (
            '<tendon><fixed name="split">'
            '<joint joint="gripper_right_finger" coef="0.5"/>'
            '<joint joint="gripper_left_finger" coef="-0.5"/>'
            '</fixed></tendon>'
        )
        text = text.replace("</worldbody>", f"</worldbody>\n  {tendon_block}", 1)
        out.write_text(text)
    return out


class Lite6Arm(_MenagerieArm):
    """6-DOF UFACTORY Lite6 + its own narrow parallel gripper, rewritten to
    the 0-255 position-servo convention (see lite6_with_position_gripper)."""

    dof = 6

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.base_name = f"{prefix}link_base"
        self.site_name = f"{prefix}end_effector"
        self.jnt_adr = model.joint(f"{prefix}joint1").qposadr.item()
        self.act_id = model.actuator(f"{prefix}joint1").id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f"{prefix}gripper").id
        self.gripper_jnt_adr = model.joint(f"{prefix}gripper_right_finger").qposadr.item()
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None
