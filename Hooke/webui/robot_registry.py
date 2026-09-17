"""Robots available in the Step 5 UI's robot picker, beyond the one robot
each task's scene already has baked in.

Two placement strategies, since not every robot mounts the same way (see
private/technical-log.md for the reasoning):

- "arm" (arm_mount): a fixed-base tabletop arm that can take over the exact
  mount point of the UR5e already in most task scenes (same scale, same
  kind of attachment). Swappable 1:1.
- "floor": a floor-standing or mobile-base robot (humanoid, mobile
  manipulator) that doesn't fit a tabletop bracket. Rather than replacing
  the task's own robot, it's added standing near the table -- this is a
  visualization of "what would this robot look like near this task", not a
  functional substitution (no IK/control adaptation is attempted; see
  private/TODO.md).

Models under model/robot_menagerie/ are vendored, unmodified, from
google-deepmind/mujoco_menagerie (each subdirectory keeps its own LICENSE
file from that repo -- Apache-2.0 for franka_emika_panda and pal_tiago_dual,
BSD-3-Clause-style for ufactory_xarm7 and unitree_g1; see each LICENSE).
"""
import dataclasses

from archetypes.menagerie_arms import (
    MODEL_ROOT, PandaArm, XArm7Arm, panda_with_tcp_site,
    KinovaGen3Arm, KukaIiwa14Arm, FlexivRizon4Arm, Lite6Arm,
    gen3_with_2f85, iiwa14_with_2f85, rizon4_with_2f85, lite6_with_position_gripper,
)
from expert_common import UR5eArm


@dataclasses.dataclass(frozen=True)
class RobotEntry:
    name: str
    display_name: str
    category: str  # single_arm | dual_arm | humanoid | mobile_manipulator
    mount: str  # "arm" | "floor" | "native_only"
    mjcf_path: str | None  # relative to Hooke/model/
    source: str
    freejoint_name: str | None = None  # unprefixed name of the root free joint, for "floor" robots
    # For "arm"-mount robots: the real, IK-driving arm class (see
    # archetypes/menagerie_arms.py) that makes this robot actually usable,
    # not just visually placeable -- archetypes/lever_lock_centrifuge.py's
    # LeverLockSpec.arm_cls is how a task recipe is pointed at one of these.
    arm_cls: type | None = None


# franka_emika_panda/panda.xml ships with no gripper-TCP site (unlike
# xArm7, which already has one); panda_with_tcp_site() generates a copy
# with one added, once, and every reference to Panda's MJCF below uses
# that copy rather than the raw vendored file.
_PANDA_MJCF = str(panda_with_tcp_site().relative_to(MODEL_ROOT))
_GEN3_MJCF = str(gen3_with_2f85().relative_to(MODEL_ROOT))
_IIWA14_MJCF = str(iiwa14_with_2f85().relative_to(MODEL_ROOT))
_RIZON4_MJCF = str(rizon4_with_2f85().relative_to(MODEL_ROOT))
_LITE6_MJCF = str(lite6_with_position_gripper().relative_to(MODEL_ROOT))

ROBOTS: dict[str, RobotEntry] = {
    entry.name: entry for entry in [
        RobotEntry(
            name="g1_rover_team",
            display_name="G1 人形机器人 + 六轮采样车",
            category="mobile_manipulator",
            mount="native_only",
            mjcf_path=None,
            source="Unitree G1 and Hooke sampling rover",
            arm_cls=UR5eArm,
        ),
        RobotEntry(
            name="surface_rover",
            display_name="六轮采样车 + UR5e",
            category="mobile_manipulator",
            mount="native_only",
            mjcf_path=None,
            source="Hooke rover with existing UR5e / Robotiq assets",
            arm_cls=UR5eArm,
        ),
        RobotEntry(
            name="ur5e", display_name="UR5e + Robotiq 2F-85", category="single_arm",
            mount="arm", mjcf_path="robot/ur5e_gripper.xml",
            source="Hooke/AutoBio original asset", arm_cls=UR5eArm,
        ),
        RobotEntry(
            name="franka_panda", display_name="Franka Emika Panda", category="single_arm",
            mount="arm", mjcf_path=_PANDA_MJCF,
            source="MuJoCo Menagerie (Apache-2.0)", arm_cls=PandaArm,
        ),
        RobotEntry(
            name="xarm7", display_name="UFACTORY xArm7", category="single_arm",
            mount="arm", mjcf_path="robot_menagerie/ufactory_xarm7/xarm7.xml",
            source="MuJoCo Menagerie (UFACTORY, BSD-style)", arm_cls=XArm7Arm,
        ),
        RobotEntry(
            name="kinova_gen3", display_name="Kinova Gen3 + Robotiq 2F-85", category="single_arm",
            mount="arm", mjcf_path=_GEN3_MJCF,
            source="MuJoCo Menagerie (Kinova, BSD-style)", arm_cls=KinovaGen3Arm,
        ),
        RobotEntry(
            name="kuka_iiwa14", display_name="KUKA iiwa14 + Robotiq 2F-85", category="single_arm",
            mount="arm", mjcf_path=_IIWA14_MJCF,
            source="MuJoCo Menagerie (KUKA/Drake, BSD-style)", arm_cls=KukaIiwa14Arm,
        ),
        RobotEntry(
            name="flexiv_rizon4", display_name="Flexiv Rizon4 + Robotiq 2F-85", category="single_arm",
            mount="arm", mjcf_path=_RIZON4_MJCF,
            source="MuJoCo Menagerie (Flexiv, Apache-2.0)", arm_cls=FlexivRizon4Arm,
        ),
        RobotEntry(
            name="ufactory_lite6", display_name="UFACTORY Lite6", category="single_arm",
            mount="arm", mjcf_path=_LITE6_MJCF,
            source="MuJoCo Menagerie (UFACTORY, BSD-style)", arm_cls=Lite6Arm,
        ),
        RobotEntry(
            name="aloha", display_name="Aloha (dual-arm)", category="dual_arm",
            mount="native_only", mjcf_path=None,
            source="Hooke/AutoBio original asset",
        ),
        RobotEntry(
            name="dual_ur5e", display_name="Dual UR5e (pipetting rig)", category="dual_arm",
            mount="native_only", mjcf_path=None,
            source="Hooke/AutoBio original asset",
        ),
        RobotEntry(
            name="unitree_g1", display_name="Unitree G1 (humanoid)", category="humanoid",
            mount="floor", mjcf_path="robot_menagerie/unitree_g1/g1.xml",
            source="MuJoCo Menagerie (Unitree Robotics, BSD-style)",
            freejoint_name="floating_base_joint",
        ),
        RobotEntry(
            name="tiago_dual", display_name="PAL Tiago Dual (mobile, dual-arm)", category="mobile_manipulator",
            mount="floor", mjcf_path="robot_menagerie/pal_tiago_dual/tiago_dual.xml",
            source="MuJoCo Menagerie (PAL Robotics, Apache-2.0)",
            freejoint_name="reference",
        ),
    ]
}

# Robots offered as swap-in replacements for a task's native arm, keyed by
# the native robot they can replace. Only "arm"-mount robots participate --
# there is no dual-arm or floor robot that cleanly substitutes for the
# existing ur5e/aloha mount points.
ARM_ALTERNATIVES: dict[str, list[str]] = {
    "ur5e": ["franka_panda", "xarm7", "kinova_gen3", "kuka_iiwa14", "flexiv_rizon4", "ufactory_lite6"],
}

# Floor-standing robots offered as an *addition* next to any task's scene
# (not a replacement), since they don't compete for the same mount point.
FLOOR_BYSTANDERS = ["unitree_g1", "tiago_dual"]


def robot_options_for(native_robot: str) -> list[str]:
    """All robot names selectable for a task whose native robot is
    `native_robot`: itself, any arm-mount alternative, and the floor
    bystanders."""
    return [native_robot] + ARM_ALTERNATIVES.get(native_robot, []) + FLOOR_BYSTANDERS
