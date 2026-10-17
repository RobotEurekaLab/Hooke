"""Explicit inspection drives for legacy assets with non-position semantics.

These parameters make the copied geometry inspectable. They are neither OEM
motor specifications nor centrifugation, sample-separation or safety models.
The source MJCF assets and their instrument-specific task logic stay unchanged.
"""

import math
import xml.etree.ElementTree as ET


MINI_CENTRIFUGE = "instrument/centrifuge_tiangen_tgear_mini.xml"


def add_inspection_drive(actuators, filename, semantic, joint, drive_name):
    """Add a known continuous-rotation drive, or leave a manual joint unchanged.

    The mini rotor's legacy +/-180 range is in radians, not degrees. Treating
    it as a position target requests dozens of turns on each reversal. Its
    copied inspection model instead uses a signed speed and an unlimited angle.
    """
    if filename != MINI_CENTRIFUGE or semantic != "rotor":
        return None
    if joint.get("type", "hinge") != "hinge":
        raise ValueError("The mini centrifuge inspection rotor must be a hinge")
    original_range = [float(v) for v in joint.get("range", "").split()]
    joint.set("limited", "false")
    # Restore the source's zero passive rotor damping instead of inheriting a
    # manual handle's 1 Nm s/rad drag. Velocity feedback supplies braking. The
    # existing mesh inertia and inherited armature are deliberately unchanged.
    joint.set("damping", "0")
    speed = math.tau
    ET.SubElement(
        actuators,
        "velocity",
        name=drive_name,
        joint=joint.get("name"),
        kv="0.05",
        ctrllimited="true",
        ctrlrange=f"{-speed:.16g} {speed:.16g}",
        forcelimited="true",
        forcerange="-0.05 0.05",
    )
    return {
        "mode": "velocity",
        "units": "rad/s",
        "range": [-speed, speed],
        "continuous_angle": True,
        "source_joint_range_rad": original_range,
        "torque_limit_nm": 0.05,
        "velocity_gain_nm_s_per_rad": 0.05,
        "scope": "slow geometric rotor inspection",
        "parameter_basis": (
            "Authored one-revolution-per-second inspection limit; 0.05 Nm "
            "torque cap and velocity feedback give a visible start/stop with "
            "the retained scene armature. These are not manufacturer ratings."
        ),
        "limitations": (
            "The copied asset has separate detailed contact geometry and optional "
            "inert gauge fixtures. This inspection drive does not add calibrated "
            "balance sensing, a lid interlock, a manufacturer motor model or "
            "scientific centrifugation. Original instrument-specific process "
            "logic remains absent."
        ),
    }
