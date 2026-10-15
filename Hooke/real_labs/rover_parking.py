"""Bounded wheel-command feedback for parked inspection of a mobile base."""
import math
import numpy as np


def parking_twist(position, heading, linear_velocity, yaw_rate, anchor, anchor_heading):
    """Hold longitudinal position and heading through real wheel actuators.

    This supplies a small drive command, not a base constraint. The caller
    retains its positional drift, wheel-effort and contact checks. Lateral
    errors are not concealed; a nonholonomic base still has to pass that gate.
    """
    position=np.asarray(position,dtype=float)
    velocity=np.asarray(linear_velocity,dtype=float)
    anchor=np.asarray(anchor,dtype=float)
    if position.shape!=(3,) or velocity.shape!=(3,) or anchor.shape!=(3,):
        raise ValueError('Parking feedback requires three-dimensional position and velocity')
    if not np.isfinite(np.r_[position,velocity,anchor,heading,yaw_rate,anchor_heading]).all():
        raise ValueError('Parking feedback requires finite observed state')
    forward=np.array([math.cos(heading),math.sin(heading)])
    error=float(np.dot(anchor[:2]-position[:2],forward))
    speed=float(np.dot(velocity[:2],forward))
    heading_error=(anchor_heading-heading+math.pi)%(2*math.pi)-math.pi
    # Small steering corrections at rest scrub the contact patch. Keep the
    # wheels straight until an observed heading/rate disturbance is meaningful.
    angular_error = math.copysign(max(0., abs(heading_error) - .002), heading_error)
    angular_rate = math.copysign(max(0., abs(yaw_rate) - .01), yaw_rate)
    return (float(np.clip(6*error-.3*speed,-.02,.02)),
            float(np.clip(4*angular_error-.3*angular_rate,-.05,.05)))
