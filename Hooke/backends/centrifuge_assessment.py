"""Independent mechanical evidence from observed rotor motion and interlocks."""

from dataclasses import dataclass
import math


@dataclass
class CentrifugeCycleSequence:
    loaded: bool = False
    started: bool = False
    unlocked: bool = False
    safety_violation: bool = False
    rotation_rad: float = 0.0
    hold_s: float = 0.0
    stopped_s: float = 0.0
    peak_rad_s: float = 0.0
    speed_rad_s: float = 0.0
    balance_ratio: float = 1.0
    locked: bool = False
    closed: bool = False
    seated_s: float = 0.0

    def update(self, dt, speed, balance, closed, locked, seated, program):
        self.speed_rad_s, self.balance_ratio = float(speed), float(balance)
        self.closed, self.locked = bool(closed), bool(locked)
        self.seated_s = self.seated_s + dt if seated else 0.0
        self.loaded |= self.seated_s >= 0.5
        moving = abs(speed) > program.safe_speed_rad_s
        if moving and self.loaded and closed and locked:
            self.started = True
        if self.started:
            self.peak_rad_s = max(self.peak_rad_s, abs(speed))
            self.rotation_rad += abs(speed) * dt
            nominal = program.rpm * math.pi / 30
            if abs(speed - nominal) <= nominal * 0.05 and closed and locked:
                self.hold_s += dt
            self.stopped_s = self.stopped_s + dt if not moving else 0.0
            if moving and (
                not closed or not locked or balance > program.balance_ratio_limit
            ):
                self.safety_violation = True
            if (
                not locked
                and closed
                and self.hold_s >= program.hold_seconds
                and self.stopped_s >= program.stopped_seconds
            ):
                self.unlocked = True

    def checks(self):
        return {
            "two_loads_seated_before_spin": self.loaded,
            "two_loads_retained_at_finish": self.seated_s >= 0.5,
            "actual_rotor_revolution": self.rotation_rad > 2 * math.pi,
            "actual_60rpm_hold_1s": self.hold_s >= 1.0,
            "measured_standstill_500ms": self.stopped_s >= 0.5,
            "balanced_load_below_5percent": self.balance_ratio <= 0.05,
            "lid_closed_at_finish": self.closed,
            "unlocked_after_standstill": self.unlocked and not self.locked,
            "no_moving_interlock_violation": not self.safety_violation,
        }
