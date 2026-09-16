"""Shared centrifuge programme driven by measured speed and safety inputs."""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class SpinProgram:
    rpm: float = 60.0
    hold_seconds: float = 1.0
    acceleration_rad_s2: float = 5.0
    torque_limit_nm: float = 0.1
    speed_gain: float = 0.08
    safe_speed_rad_s: float = 0.02
    stopped_seconds: float = 0.5
    balance_ratio_limit: float = 0.05

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value <= 0 for value in asdict(self).values()
        ):
            raise ValueError("Spin programme parameters must be finite and positive")
        if (
            self.safe_speed_rad_s >= self.rpm * math.pi / 30
            or self.balance_ratio_limit >= 1
        ):
            raise ValueError("Invalid safe speed or balance threshold")


class SpinController:
    """A finite-state programme; no position or speed state is overwritten."""

    def __init__(self, program=None):
        self.program = program or SpinProgram()
        self.state = "IDLE"
        self.target_rad_s = 0.0
        self.hold_s = self.stopped_s = self.rotation_rad = 0.0
        self.peak_rad_s = 0.0
        self.fault = None

    def start(self, speed, closed, locked, balance_ratio):
        if self.state != "IDLE":
            raise RuntimeError("Programme is not idle")
        self._validate(speed, balance_ratio)
        if not closed or not locked or balance_ratio > self.program.balance_ratio_limit:
            raise RuntimeError("Start requires a closed locked lid and balanced load")
        if abs(speed) > self.program.safe_speed_rad_s:
            raise RuntimeError("Start requires a stopped rotor")
        self.state = "ACCELERATING"

    @staticmethod
    def _validate(speed, balance_ratio):
        if (
            not math.isfinite(speed)
            or not math.isfinite(balance_ratio)
            or balance_ratio < 0
        ):
            raise ValueError("Invalid measured speed or balance")

    def stop(self):
        if self.state in ("ACCELERATING", "HOLDING"):
            self.state = "BRAKING"

    def update(self, dt, speed, closed, locked, balance_ratio):
        self._validate(speed, balance_ratio)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Expected a finite positive timestep")
        active = self.state in ("ACCELERATING", "HOLDING", "BRAKING")
        if active:
            reason = (
                "lid_open"
                if not closed
                else (
                    "lid_unlocked"
                    if not locked
                    else (
                        "unbalanced_load"
                        if balance_ratio > self.program.balance_ratio_limit
                        else None
                    )
                )
            )
            if reason and self.fault is None:
                self.fault, self.state = reason, "BRAKING"
            self.peak_rad_s = max(self.peak_rad_s, abs(speed))
            self.rotation_rad += abs(speed) * dt
        nominal = self.program.rpm * math.pi / 30
        requested = nominal if self.state in ("ACCELERATING", "HOLDING") else 0.0
        change = self.program.acceleration_rad_s2 * dt
        self.target_rad_s += max(-change, min(change, requested - self.target_rad_s))
        if self.state == "ACCELERATING" and abs(speed - nominal) <= nominal * 0.05:
            self.state = "HOLDING"
        if self.state == "HOLDING":
            if abs(speed - nominal) <= nominal * 0.05:
                self.hold_s += dt
            if self.hold_s >= self.program.hold_seconds:
                self.state = "BRAKING"
        if self.state == "BRAKING":
            self.stopped_s = (
                self.stopped_s + dt
                if abs(speed) <= self.program.safe_speed_rad_s
                else 0.0
            )
            if (
                self.stopped_s >= self.program.stopped_seconds
                and self.target_rad_s == 0
            ):
                self.state = "FAULT" if self.fault else "COMPLETE"
        if self.state in ("IDLE", "COMPLETE", "FAULT"):
            return 0.0
        torque = self.program.speed_gain * (self.target_rad_s - speed)
        return max(
            -self.program.torque_limit_nm, min(self.program.torque_limit_nm, torque)
        )

    def can_unlock(self, speed):
        return (
            self.state in ("IDLE", "COMPLETE", "FAULT")
            and math.isfinite(speed)
            and abs(speed) <= self.program.safe_speed_rad_s
            and (self.state == "IDLE" or self.stopped_s >= self.program.stopped_seconds)
        )

    def report(self):
        return dict(
            state=self.state,
            fault=self.fault,
            peak_rad_s=self.peak_rad_s,
            measured_rotation_rad=self.rotation_rad,
            measured_hold_s=self.hold_s,
            stopped_s=self.stopped_s,
            programme=asdict(self.program),
        )
