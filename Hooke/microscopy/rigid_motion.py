"""Bounds on sampled rigid motion, including rotation about each body's origin.

A point at radius r moves at most r*theta under rotation through theta radians.
Combined with a translation bound, this gives a conservative surface-distance
bound. Observations cover recorded poses, not unobserved physical motion.
"""

import numpy as np


def rotation_angles(reference, rotations):
    cosine = (np.einsum("...ij,...ij->...", reference, rotations) - 1.) / 2.
    return np.arccos(np.clip(cosine, -1., 1.))


class RigidMotionBounds:
    def __init__(self, positions, rotations, segment_steps=1):
        self.positions, self.rotations = positions.copy(), rotations.copy()
        self.minimum, self.maximum = positions.copy(), positions.copy()
        self.angles = np.zeros(len(positions))
        self.segment_steps, self.steps = segment_steps, 0
        self.segments = []
        self.start_segment(positions, rotations)

    def start_segment(self, positions, rotations):
        self.start, self.rotation = positions.copy(), rotations.copy()
        self.lower, self.upper = positions.copy(), positions.copy()
        self.angle = np.zeros(len(positions))

    def observe(self, positions, rotations):
        np.minimum(self.minimum, positions, out=self.minimum)
        np.maximum(self.maximum, positions, out=self.maximum)
        np.maximum(self.angles, rotation_angles(self.rotations, rotations), out=self.angles)
        np.minimum(self.lower, positions, out=self.lower)
        np.maximum(self.upper, positions, out=self.upper)
        np.maximum(self.angle, rotation_angles(self.rotation, rotations), out=self.angle)
        self.steps += 1
        if self.steps % self.segment_steps == 0:
            self.finish_segment()
            self.start_segment(positions, rotations)

    def finish_segment(self):
        self.segments.append((self.start, self.rotation, self.lower, self.upper, self.angle))

    def arrays(self):
        segments = self.segments.copy()
        if self.steps % self.segment_steps:
            segments.append((self.start, self.rotation, self.lower, self.upper, self.angle))
        result = dict(rigid_body_positions=self.positions, rigid_body_rotations=self.rotations,
                      rigid_positions_min=self.minimum, rigid_positions_max=self.maximum,
                      rigid_angle_max=self.angles)
        for index, key in enumerate(("positions", "rotations", "min", "max", "angles")):
            result["rigid_segment_" + key] = np.array([segment[index] for segment in segments])
        return result


def body_displacement(start, lower, upper, angles, columns, radii):
    translation = np.linalg.norm(np.maximum(abs(lower - start), abs(upper - start)), axis=-1)
    return translation[..., columns] + radii * angles[..., columns]


def geom_pose(arrays, index, positions, rotations):
    column = arrays["rigid_geom_body_map"][index]
    rotation = rotations[column]
    return (rotation @ arrays["rigid_geom_local_rotations"][index],
            positions[column] + rotation @ arrays["rigid_geom_local_positions"][index])
