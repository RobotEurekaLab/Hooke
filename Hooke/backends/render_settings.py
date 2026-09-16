"""Shared image geometry and sampling; native light controls are explicit."""

from dataclasses import asdict, dataclass
import math
import os


@dataclass(frozen=True)
class RenderSettings:
    width: int = 640
    height: int = 480
    frames_per_second: float = 20.0
    native_sun_intensity: float = 1500.0
    native_ambient_intensity: float = 450.0
    native_color_pipeline: str = "physical"

    def __post_init__(self):
        if self.native_color_pipeline not in ("physical", "source_display"):
            raise ValueError("Native color pipeline must be physical or source_display")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or not 32 <= v <= 2048
            for v in (self.width, self.height)
        ):
            raise ValueError("Render dimensions must be integers in 32–2048")
        if (
            not math.isfinite(self.frames_per_second)
            or not 0 < self.frames_per_second <= 60
        ):
            raise ValueError("Render sampling must be in (0,60] FPS")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.native_sun_intensity, self.native_ambient_intensity)
        ):
            raise ValueError("Native light intensities must be finite and nonnegative")

    @classmethod
    def from_environment(cls):
        return cls(
            width=int(os.environ.get("HOOKE_RENDER_WIDTH", "640")),
            height=int(os.environ.get("HOOKE_RENDER_HEIGHT", "480")),
            frames_per_second=float(os.environ.get("HOOKE_RENDER_FPS", "20")),
            native_sun_intensity=float(
                os.environ.get("HOOKE_ISAAC_SUN_INTENSITY", "1500")
            ),
            native_ambient_intensity=float(
                os.environ.get("HOOKE_ISAAC_AMBIENT_INTENSITY", "450")
            ),
            native_color_pipeline=os.environ.get(
                "HOOKE_ISAAC_COLOR_PIPELINE", "physical"
            ),
        )

    def step_interval(self, dt):
        return max(1, round(1 / (self.frames_per_second * dt)))

    def report(self):
        return asdict(self)
