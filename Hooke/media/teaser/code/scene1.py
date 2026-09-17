"""Scene 1 - Robert Hooke. Painted portrait plate as the background, no candle animation."""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import *

BW, BH = 2304, 1296  # slightly larger than frame, for the slow push-in
_S = {}
PLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooke.jpg")


def _build():
    img = Image.open(PLATE).convert("RGB")
    # cover-fit the plate to the working canvas
    s = max(BW / img.width, BH / img.height)
    img = img.resize((int(img.width * s + 1), int(img.height * s + 1)), Image.LANCZOS)
    x0 = (img.width - BW) // 2
    y0 = int((img.height - BH) * 0.42)
    img = img.crop((x0, y0, x0 + BW, y0 + BH))
    a = to_arr(img)
    # gentle period grade: warm highlights, cool shadows, a touch more contrast
    a = np.clip(a, 0, 1) ** 1.06
    a = a * np.array([1.04, 0.99, 0.92], np.float32) + np.array([0.0, 0.004, 0.018], np.float32)
    _S["plate"] = a.astype(np.float32)
    _S["motes"] = rng(3).random((180, 5))
    _S["year"] = Text3D("1662.", make_font("timesbd.ttf", 300), "gold", tracking=0.06,
                        side=(0.35, 0.2, 0.07), side_back=(0.05, 0.025, 0.01))
    _S["name"] = Text3D("ROBERT HOOKE", make_font("timesbd.ttf", 230), "gold", tracking=0.12,
                        side=(0.32, 0.18, 0.06), side_back=(0.04, 0.02, 0.01), seed=3)


def render(t, dur=4.1):
    if not _S:
        _build()
    u = t / dur
    frame = zoom_crop(_S["plate"], 1.2 + 0.07 * ease_in_out(u), 0.5 + 0.02 * u, 0.5 - 0.01 * u)
    # soft lamp bloom breathing very slightly (no flicker, no candle animation)
    frame += radial(470, 330, 380, power=1.8)[..., None] * np.array([1.0, 0.82, 0.55], np.float32) * (0.1 + 0.01 * math.sin(t * 1.3))

    # dust in the lamp light
    layer = Image.new("L", (W, H))
    dd = ImageDraw.Draw(layer)
    for x0, y0, r, sp, ph in _S["motes"]:
        x = (x0 * W + t * (10 + 16 * sp) + 20 * math.sin(t * 0.6 + ph * 6)) % W
        y = (y0 * H + t * (4 + 7 * sp)) % H
        b = 0.25 + 0.75 * math.exp(-((x - 520) ** 2 + (y - 380) ** 2) / 260000)
        rad = 1 + r * 2.0
        dd.ellipse([x - rad, y - rad, x + rad, y + rad], fill=int(200 * b))
    frame += (np.asarray(layer.filter(ImageFilter.GaussianBlur(1.3)), np.float32) / 255)[..., None] \
        * np.array([1.0, 0.86, 0.62], np.float32) * 0.5

    # "1662." then his name, both in and out
    a1_in, a1_out = smooth((t - 0.15) / 0.45), smooth((t - 1.5) / 0.4)
    if a1_in > 0.01 and a1_out < 0.99:
        k = ease_out((t - 0.15) / 0.9)
        _S["year"].draw(frame, pos=(-430, -170, lerp(2400, 1150, k) - 90 * a1_out), rx=0.08,
                        ry=lerp(-0.5, -0.12, k), scale=0.62, depth=60, layers=14,
                        alpha=float(a1_in * (1 - a1_out)), sweep=lerp(-0.3, 1.4, (t - 0.3) / 1.2),
                        light=0.95, blur=max(0.0, 6 * (1 - k)), shadow=(30, 45, 20, 0.5))
    a2_in = smooth((t - 1.65) / 0.5)
    a2_out = smooth((t - (dur - 0.5)) / 0.5)
    if a2_in > 0.01:
        k = ease_out((t - 1.65) / 1.2)
        _S["name"].draw(frame, pos=(-150, 250, lerp(2100, 1130, k) - 40 * (t - 1.65)), rx=-0.04,
                        ry=lerp(0.35, 0.06, k), scale=0.6, depth=70, layers=16,
                        alpha=float(a2_in * (1 - a2_out)), sweep=lerp(-0.3, 1.4, (t - 1.9) / 1.6),
                        light=0.95, blur=max(0.0, 7 * (1 - k)), shadow=(25, 55, 24, 0.6))
    return frame
