"""SI kinematics of the licensed Open Micro-Manipulator v4.

Derived from 0x23/MicroManipulatorStepper, revision
63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94, kinematic_model_delta3d.cpp.
Project: MicroManipulatorStepper; Author: M. S. (diffraction limited).
The exact author-modified MIT notice is in licenses/MicroManipulatorStepper.txt.
This describes the fixed-orientation, three-motor parallel mechanism. It does
not infer hardware accuracy or replace a physical closed-loop assembly.
"""

from dataclasses import dataclass

import numpy as np


SOURCE_REVISION = "63d960f8a5b218a67436f5a96b0c1bfa7e2d0f94"
ARM_LENGTH_M = .0725
ROTOR_RADIUS_M = .015
ROTOR_OFFSET_RAD = np.deg2rad(42.)
ACTUATOR_ORIGINS_M = np.array([[-69., -26.5, 4.5], [4.5, -69., -26.5],
                               [-26.5, 4.5, -69.]])*1e-3
ACTUATOR_ROTATIONS = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                              [[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]],
                              [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]])
PLATFORM_ATTACHMENTS_M = np.array([[3.54, -11.5, 4.5], [4.5, 3.54, -11.5],
                                  [-11.5, 4.5, 3.54]])*1e-3
for _array in (ACTUATOR_ORIGINS_M, ACTUATOR_ROTATIONS, PLATFORM_ATTACHMENTS_M):
    _array.setflags(write=False)


@dataclass(frozen=True)
class ParallelGeometry:
    platform_attachments_m: tuple
    arm_length_m: float = ARM_LENGTH_M
    rotor_radius_m: float = ROTOR_RADIUS_M


FIRMWARE_GEOMETRY = ParallelGeometry(tuple(map(tuple, PLATFORM_ATTACHMENTS_M)))
# The saved v4 CAD ball-pair midpoints are 3.5 mm, not firmware's 3.54 mm.
# This profile uses measured geometric datums; it does not calibrate hardware.
CAD_GEOMETRY = ParallelGeometry(((.0035, -.0115, .0045), (.0045, .0035, -.0115),
                                 (-.0115, .0045, .0035)))


def _vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Parallel coordinates must be three finite values")
    return value


def rotor_attachments(angles_rad, geometry=FIRMWARE_GEOMETRY):
    """World ball-pair midpoints, relative to the neutral platform origin."""
    angle = _vector(angles_rad)-ROTOR_OFFSET_RAD
    local = geometry.rotor_radius_m*np.column_stack((np.cos(angle), -np.sin(angle), np.zeros(3)))
    return ACTUATOR_ORIGINS_M+np.einsum("ijk,ik->ij", ACTUATOR_ROTATIONS, local)


def forward(angles_rad, geometry=FIRMWARE_GEOMETRY):
    """Solve rod closure on the same greater-X branch as the upstream firmware."""
    centres = rotor_attachments(angles_rad, geometry)-geometry.platform_attachments_m
    delta = centres[1]-centres[0]
    d = np.linalg.norm(delta)
    if d <= np.finfo(float).eps*geometry.arm_length_m:
        raise ValueError("Parallel sphere centres are coincident")
    ex = delta/d
    v = centres[2]-centres[0]
    i = np.dot(v, ex)
    transverse = v-i*ex
    j = np.linalg.norm(transverse)
    if j <= np.finfo(float).eps*geometry.arm_length_m:
        raise ValueError("Parallel sphere centres are collinear")
    ey = transverse/j
    x = d/2
    y = (np.dot(v, v)-2*i*x)/(2*j)
    z2 = geometry.arm_length_m**2-x*x-y*y
    if z2 < -64*np.finfo(float).eps*geometry.arm_length_m**2:
        raise ValueError("Motor angles cannot close the parallel linkage")
    midpoint = centres[0]+x*ex+y*ey
    offset = np.sqrt(max(0., z2))*np.cross(ex, ey)
    first, second = midpoint+offset, midpoint-offset
    return first if first[0] > second[0] else second


def inverse(position_m, geometry=FIRMWARE_GEOMETRY):
    """Return clockwise motor radians for a fixed-orientation platform position."""
    displacement = _vector(position_m)+geometry.platform_attachments_m-ACTUATOR_ORIGINS_M
    local = np.einsum("ikj,ik->ij", ACTUATOR_ROTATIONS, displacement)
    planar = np.linalg.norm(local[:, :2], axis=1)
    if np.any(planar <= np.finfo(float).eps*geometry.arm_length_m):
        raise ValueError("Platform position leaves a motor angle undetermined")
    cosine = (np.sum(local*local, axis=1)+geometry.rotor_radius_m**2-geometry.arm_length_m**2)/(2*geometry.rotor_radius_m*planar)
    if np.any(np.abs(cosine) > 1+64*np.finfo(float).eps):
        raise ValueError("Platform position is outside the parallel workspace")
    phase = np.arctan2(local[:, 1], local[:, 0])
    half = np.arccos(np.clip(cosine, -1., 1.))
    first, second = phase+half, phase-half
    selected = np.where(np.cos(first) > np.cos(second), first, second)
    selected = np.arctan2(np.sin(selected), np.cos(selected))
    angles = ROTOR_OFFSET_RAD-selected
    if not np.allclose(forward(angles, geometry), position_m, rtol=0, atol=2e-12):
        raise ValueError("Platform position is on the unsupported assembly branch")
    return angles


def closure_residuals(angles_rad, position_m, geometry=FIRMWARE_GEOMETRY):
    """Signed rod-length errors in metres; no physics or state modification."""
    rods = _vector(position_m)+geometry.platform_attachments_m-rotor_attachments(angles_rad, geometry)
    return np.linalg.norm(rods, axis=1)-geometry.arm_length_m


def jacobian(angles_rad, geometry=FIRMWARE_GEOMETRY):
    """Platform metres per motor radian, from differentiated rod constraints."""
    angles = _vector(angles_rad)
    angle = angles-ROTOR_OFFSET_RAD
    rods = forward(angles, geometry)+geometry.platform_attachments_m-rotor_attachments(angles, geometry)
    local_velocity = geometry.rotor_radius_m*np.column_stack((-np.sin(angle), -np.cos(angle), np.zeros(3)))
    velocity = np.einsum("ijk,ik->ij", ACTUATOR_ROTATIONS, local_velocity)
    if np.linalg.cond(rods) > 1e10:
        raise ValueError("Parallel linkage is singular")
    return np.linalg.solve(rods, np.diag(np.einsum("ij,ij->i", rods, velocity)))
