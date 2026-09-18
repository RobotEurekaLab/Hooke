"""Calibrated synthetic views driven by current sample, tool and focus state.

This demonstrator uses a Gaussian defocus response, not wave-optics microscopy.
Object coordinates are inputs to image formation, never controller observations.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from microscopy.scene import APPROACHES, PICK_TARGET, PUSH_TARGET, RADIUS, WELL


@dataclass(frozen=True)
class OpticalCalibration:
    field_width_m: float = .014
    pixels: int = 512
    defocus_scale_m: float = .00005

    @property
    def pixel_size_m(self):
        return self.field_width_m / self.pixels

    def pixel(self, xy):
        x, y = np.asarray(xy) / self.pixel_size_m
        return self.pixels / 2 + x, self.pixels / 2 - y

    def position(self, pixel):
        x, y = pixel
        return np.array([x - self.pixels / 2, self.pixels / 2 - y]) * self.pixel_size_m


class Microscope:
    scale_bar_m = .002

    def __init__(self, calibration=None):
        self.calibration = calibration or OpticalCalibration()

    def public_calibration(self):
        """Report the synthetic object-space mapping, separately from real optics."""
        c = self.calibration
        return dict(pixel_size_m=c.pixel_size_m, field_width_m=c.field_width_m,
                    pixels=c.pixels, scale_bar_m=self.scale_bar_m,
                    mapping="synthetic_object_space", physical_optics_calibrated=False)

    def render(self, state, *, annotate=True):
        c = self.calibration
        yy, xx = np.mgrid[:c.pixels, :c.pixels]
        light = 226 - 13 * ((xx - c.pixels / 2)**2 + (yy - c.pixels / 2)**2) / (c.pixels / 2)**2
        image = Image.fromarray(np.repeat(light.clip(0, 255).astype(np.uint8)[..., None], 3, axis=2))
        draw = ImageDraw.Draw(image)
        for i in range(-6, 7):
            p = c.pixel([i * .001, 0])[0]
            draw.line((p, 0, p, c.pixels), fill=(210, 212, 213))
            draw.line((0, p, c.pixels, p), fill=(210, 212, 213))
        def circle(xy, radius, **kwargs):
            x, y = c.pixel(xy)
            r = radius / c.pixel_size_m
            draw.ellipse((x-r, y-r, x+r, y+r), **kwargs)
        stage = np.asarray(state["stage_m"])
        for target in (PUSH_TARGET, PICK_TARGET):
            circle(target + stage, RADIUS * 1.15, outline=(152, 164, 168), width=2)
        circle(WELL + stage, .0012, fill=(201, 186, 214), outline=(97, 64, 116), width=5)
        fill = min(1.0, state["delivered_nl"] / 200)
        if fill:
            circle(WELL + stage, .00095 * np.sqrt(fill), fill=(167, 74, 183))
        for name, color in (("bead_push", (221, 128, 37)), ("bead_pick", (40, 100, 203))):
            circle(state["samples"][name][:2], RADIUS, fill=color, outline=tuple(v // 2 for v in color), width=2)
            xy = np.asarray(state["samples"][name][:2]) + [-.0002, .0002]
            circle(xy, .00013, fill=tuple(min(255, v + 50) for v in color))
        for tool in ("probe", "injector"):
            direction = APPROACHES[tool][:2]
            tip = np.asarray(state["tools"][tool][:2])
            start = tip + np.asarray(direction) * .009
            draw.line((*c.pixel(start), *c.pixel(tip)), fill=(57, 65, 72), width=4 if tool == "probe" else 2)
            circle(tip, .00010 if tool == "probe" else .00004, fill=(55, 65, 74))
        for sign, position in zip((1, -1), state["jaw_centres_m"]):
            centre = np.asarray(position[:2])
            start = centre + [.007, sign * .001]
            end = centre + [-.00065, 0]
            draw.line((*c.pixel(start), *c.pixel(end)), fill=(62, 73, 79), width=7)
        sigma = abs(state["focus_m"]) / c.defocus_scale_m
        image = image.filter(ImageFilter.GaussianBlur(sigma))
        # Stable small detector noise makes focus measurement reproducible.
        rng = np.random.default_rng(811)
        array = np.asarray(image).astype(float) + rng.normal(0, .55, (c.pixels, c.pixels, 1))
        image = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        if annotate:
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, c.pixels, 30), fill=(23, 33, 41))
            draw.text((12, 9), "LIVE MICRO VIEW  |  Gaussian defocus model", fill="white")
            length = self.scale_bar_m / c.pixel_size_m
            draw.rectangle((22, c.pixels-29, 22+length, c.pixels-25), fill=(20, 28, 34))
            draw.text((22, c.pixels-47), "2 mm", fill=(20, 28, 34))
            draw.text((c.pixels-148, c.pixels-22), f"t = {state['time_s']:.2f} s", fill=(20, 28, 34))
        return image

    def locate(self, image, sample):
        """Color-fiducial centroid using only the supplied image and calibration."""
        rgb = np.asarray(image, dtype=float)
        red, green, blue = rgb.transpose(2, 0, 1)
        if sample == "bead_push":
            mask = (red > 150) & (red > green * 1.25) & (green > blue * 1.4)
        elif sample == "bead_pick":
            mask = (blue > 140) & (blue > green * 1.35) & (green > red * 1.35)
        else:
            raise ValueError("Unknown sample fiducial")
        yy, xx = np.nonzero(mask)
        if len(xx) < 12:
            raise RuntimeError(f"{sample}: fiducial is not visible; check focus and occlusion")
        return self.calibration.position([xx.mean(), yy.mean()])

    @staticmethod
    def sharpness(image):
        gray = np.asarray(image, dtype=float).mean(axis=2)
        return float(np.mean(np.diff(gray, axis=0)**2) + np.mean(np.diff(gray, axis=1)**2))
