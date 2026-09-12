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


@dataclasses.dataclass(frozen=True)
class RobotEntry:
    name: str
    display_name: str
    category: str  # single_arm | dual_arm | humanoid | mobile_manipulator
    mount: str  # "arm" | "floor" | "native_only"
    mjcf_path: str | None  # relative to Hooke/model/
    source: str
    freejoint_name: str | None = None  # unprefixed name of the root free joint, for "floor" robots


ROBOTS: dict[str, RobotEntry] = {
    entry.name: entry for entry in [
        RobotEntry(
            name="ur5e", display_name="UR5e + Robotiq 2F-85", category="single_arm",
            mount="arm", mjcf_path="robot/ur5e_gripper.xml",
            source="Hooke/AutoBio original asset",
        ),
        RobotEntry(
            name="franka_panda", display_name="Franka Emika Panda", category="single_arm",
            mount="arm", mjcf_path="robot_menagerie/franka_emika_panda/panda.xml",
            source="MuJoCo Menagerie (Apache-2.0)",
        ),
        RobotEntry(
            name="xarm7", display_name="UFACTORY xArm7", category="single_arm",
            mount="arm", mjcf_path="robot_menagerie/ufactory_xarm7/xarm7.xml",
            source="MuJoCo Menagerie (UFACTORY, BSD-style)",
        ),
        RobotEntry(
            name="aloha", display_name="Aloha (dual-arm)", category="dual_arm",
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
    "ur5e": ["franka_panda", "xarm7"],
}

# Floor-standing robots offered as an *addition* next to any task's scene
# (not a replacement), since they don't compete for the same mount point.
FLOOR_BYSTANDERS = ["unitree_g1", "tiago_dual"]


def robot_options_for(native_robot: str) -> list[str]:
    """All robot names selectable for a task whose native robot is
    `native_robot`: itself, any arm-mount alternative, and the floor
    bystanders."""
    return [native_robot] + ARM_ALTERNATIVES.get(native_robot, []) + FLOOR_BYSTANDERS
