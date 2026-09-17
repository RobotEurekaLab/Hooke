"""Feedback wheel commands from observed rover pose, without pose integration."""

import math

import numpy as np

from surface.scene import WHEELS


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WheelNavigation:
    def __init__(self, model, mission):
        self.mission = mission
        self.body_id = model.body("rover").id
        self.actuators = np.array([model.actuator(name).id for name in WHEELS])
        self.state_indices = [int(model.joint(name).qposadr[0]) for name in WHEELS]
        self.target = np.asarray(mission.home, dtype=float)
        self.reverse = False
        self.parked = True
        self.park_heading = 0.0
        self.enabled = False

    def pose(self, data):
        rotation = data.xmat[self.body_id].reshape(3, 3)
        return data.xpos[self.body_id, :2].copy(), math.atan2(
            rotation[1, 0], rotation[0, 0]
        )

    def command(self, data):
        position, heading = self.pose(data)
        delta = self.target - position
        distance = np.linalg.norm(delta)
        if self.parked:
            direction = np.array([math.cos(heading), math.sin(heading)])
            speed = float(np.clip(2.0 * np.dot(delta, direction), -0.15, 0.15))
            desired = self.park_heading
        else:
            desired = math.atan2(delta[1], delta[0]) - (math.pi if self.reverse else 0)
            speed = min(self.mission.speed_limit_m_s, 0.9 * distance)
            speed *= -1 if self.reverse else 1
            speed *= max(0.0, math.cos(wrap_angle(desired - heading)))
        yaw_rate = float(np.clip(2.5 * wrap_angle(desired - heading), -0.8, 0.8))
        left = (
            speed - self.mission.half_track_m * yaw_rate
        ) / self.mission.wheel_radius_m
        right = (
            speed + self.mission.half_track_m * yaw_rate
        ) / self.mission.wheel_radius_m
        data.ctrl[self.actuators] = [left] * 3 + [right] * 3

    def park(self, data):
        self.target, self.park_heading = self.pose(data)
        self.parked = True
