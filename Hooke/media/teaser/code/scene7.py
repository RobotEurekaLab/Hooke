"""Scene 7 - NEVER CLOCK OUT. Macro: 8-channel liquid handler dispensing glowing reagents, nonstop."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *

_S = {}
SS = 2
PITCH = 150
P = 0.8  # seconds per dispense cycle
COLS = [(60, 240, 255), (255, 60, 200), (140, 255, 90), (255, 170, 40), (120, 130, 255)]


def _build():
    g = rng(71)
    f = ImageFont.truetype(FONTS + "consola.ttf", 30)
    wall = Image.new("RGB", (W, H * 2), (0, 0, 0))
    d = ImageDraw.Draw(wall)
    for r in range(0, H * 2, 38):
        for c in range(0, W, 240):
            if g.random() < 0.55:
                val = f"{g.random() * 9999:8.3f}"
                col = (40, 200, 255) if g.random() < 0.7 else (255, 60, 190)
                d.text((c + 10, r), val, font=f, fill=col)
    arr = to_arr(wall)
    _S["wall"] = blur_f(arr[::2, ::2], 3)  # half res, pre-defocused
    bok = np.zeros((H, W, 3), np.float32)
    for _ in range(40):
        x, y, r = g.random() * W, g.random() * H * 0.7, 20 + 60 * g.random()
        c = np.array(COLS[g.integers(len(COLS))], np.float32) / 255
        disc = radial(x, y, r, power=8)
        bok += disc[..., None] * c * 0.25
    _S["bokeh"] = bok
    _S["text"] = Text3D("NEVER CLOCK OUT.", make_font("bahnschrift.ttf", 190, "Bold"), "chrome", tracking=0.08,
                        side=(0.25, 0.04, 0.2), side_back=(0.03, 0.0, 0.03), seed=17)
    _S["kw"] = [Keyword("AROUND THE CLOCK"), Keyword("SEVEN DAYS A WEEK")]
    _S["hud"] = ImageFont.truetype(FONTS + "consola.ttf", 26)
    _S["label"] = ImageFont.truetype(FONTS + "bahnschrift.ttf", 44)


def tube_level(j, t):
    cycle = int(t // P)
    ph = (t % P) / P
    b = math.floor(j / 8)
    if b < cycle:
        return 1.0
    if b == cycle:
        return float(smooth((ph - 0.32) / 0.2))
    return 0.0


def draw_rack(d, gd, t, x_center, y_top, scale, shift, focus=True):
    tw, th = 88 * scale, 330 * scale
    j0 = int(math.floor((shift - 960) / (PITCH * scale))) - 2
    for j in range(j0, j0 + 20):
        x = x_center + j * PITCH * scale - shift
        if x < -120 or x > W + 120:
            continue
        lvl = tube_level(j, t) if focus else 1.0
        col = COLS[(math.floor(j / 8)) % len(COLS)] if focus else COLS[(j * 3) % len(COLS)]
        X0, X1 = (x - tw / 2) * SS, (x + tw / 2) * SS
        Y0, Y1 = y_top * SS, (y_top + th) * SS
        # glass body
        d.rounded_rectangle([X0, Y0, X1, Y1], radius=int(tw * SS / 2), fill=(14, 22, 30), outline=(120, 170, 190), width=int(3 * SS * scale))
        if lvl > 0.05:
            ly = y_top + th - (th * 0.72) * lvl
            d.rounded_rectangle([X0 + 6 * SS * scale, ly * SS, X1 - 6 * SS * scale, Y1 - 6 * SS * scale],
                                radius=int(tw * SS / 2.4), fill=tuple(int(c * 0.75) for c in col))
            gd.rounded_rectangle([x - tw / 2 + 8 * scale, ly, x + tw / 2 - 8 * scale, y_top + th - 8 * scale],
                                 radius=int(tw / 2.4), fill=col)
            # meniscus + bubbles
            gd.line([(x - tw / 2 + 10 * scale, ly), (x + tw / 2 - 10 * scale, ly)], fill=(255, 255, 255), width=max(1, int(3 * scale)))
            for k in range(3):
                by = y_top + th - 20 * scale - ((t * 140 + j * 37 + k * 70) % (th * 0.7 * lvl + 1))
                bx = x + (k - 1) * 18 * scale
                d.ellipse([(bx - 5 * scale) * SS, (by - 5 * scale) * SS, (bx + 5 * scale) * SS, (by + 5 * scale) * SS], fill=tuple(min(255, c + 90) for c in col))
        # rim + highlight stripe
        d.rectangle([X0 - 6 * SS * scale, Y0, X1 + 6 * SS * scale, Y0 + 18 * SS * scale], fill=(150, 170, 185))
        d.rectangle([X0 + tw * 0.18 * SS, Y0 + 30 * SS * scale, X0 + tw * 0.28 * SS, Y1 - 40 * SS * scale], fill=(170, 200, 215))


def render(t, dur=5.0):
    if not _S:
        _build()
    cycle = int(t // P)
    ph = (t % P) / P
    shake = 3 * math.exp(-((ph - 0.32) * 30) ** 2)
    camx = 40 * math.sin(t * 0.5) + shake * math.sin(t * 90)
    # background data wall + bokeh
    wall = np.roll(_S["wall"], -int(t * 60) % _S["wall"].shape[0], 0)[:H // 2]
    frame = resize_f(wall, W, H) * 0.22 + _S["bokeh"] * (0.8 + 0.2 * math.sin(t * 7))
    frame += np.array([0.0, 0.01, 0.03], np.float32)

    # back rack (defocused)
    img_b = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    gl_b = Image.new("RGB", (W, H), (0, 0, 0))
    shift_b = t * 90 + camx * 0.5
    draw_rack(ImageDraw.Draw(img_b), ImageDraw.Draw(gl_b), t, 200, 470, 0.62, shift_b, focus=False)
    back = to_arr(img_b.reduce(SS))
    gb = to_arr(gl_b)
    mask = back.max(-1, keepdims=True) > 0.01
    frame = np.where(mask, back, frame)
    frame += gb * 0.6
    frame = fast_blur(frame, 7)
    frame += fast_blur(gb, 20) * 0.9

    # 3D title sits between the rack and the pipetting head
    ts = 0.25
    if t > ts:
        a = ease_out((t - ts) / 0.8)
        _S["text"].draw(frame, pos=(-camx * 0.3, -40, lerp(2600, 1150, a) - 30 * (t - ts)), rx=0.05,
                        ry=lerp(-0.4, 0.0, a), scale=0.85, depth=70, layers=14,
                        alpha=float(smooth((t - ts) / 0.3) * (1 - smooth((t - 4.5) / 0.35))),
                        sweep=((t - ts) * 0.6) % 1.6 - 0.2, light=1.0, glow=0.5, glow_color=(1.0, 0.25, 0.8),
                        blur=max(0.0, 5 * (1 - a)))

    # front rack + head at 2x
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    gd = ImageDraw.Draw(glow)
    shift = PITCH * 8 * (cycle + smooth((ph - 0.72) / 0.28)) + camx
    x_center = 960 - 3.5 * PITCH
    draw_rack(d, gd, t, x_center, 620, 1.0, shift - 0)
    # pipetting head
    down = smooth(ph / 0.22) * (1 - smooth((ph - 0.45) / 0.2))
    hy = 95 + 155 * down
    hx0 = -80 + camx * 0.15
    hx1 = W + 80 + camx * 0.15
    d.rectangle([hx0 * SS, (hy - 150) * SS, hx1 * SS, (hy + 40) * SS], fill=(40, 46, 56))
    for k in range(6):
        d.rectangle([hx0 * SS, (hy - 150 + k * 6) * SS, hx1 * SS, (hy - 147 + k * 6) * SS], fill=(70 + k * 12, 78 + k * 12, 90 + k * 12))
    d.text(((x_center - PITCH * 0.6) * SS, (hy - 110) * SS), "AGS-1", font=ImageFont.truetype(FONTS + "bahnschrift.ttf", 44 * SS), fill=(190, 205, 220))
    gd.line([(hx0 + 10, hy + 30), (hx1 - 10, hy + 30)], fill=(60, 230, 255), width=4)
    for k in range(8):
        x = x_center + k * PITCH
        d.rectangle([(x - 26) * SS, (hy + 40) * SS, (x + 26) * SS, (hy + 150) * SS], fill=(150, 160, 172))
        d.rectangle([(x - 26) * SS, (hy + 40) * SS, (x - 14) * SS, (hy + 150) * SS], fill=(210, 220, 230))
        d.polygon([((x - 16) * SS, (hy + 150) * SS), ((x + 16) * SS, (hy + 150) * SS), (x * SS, (hy + 330) * SS)], fill=(225, 230, 235))
        # drop falling from the tip into the tube
        if 0.2 < ph < 0.42:
            u = (ph - 0.2) / 0.22
            dy = hy + 335 + u * 220
            col = COLS[cycle % len(COLS)]
            gd.ellipse([x - 9, dy - 14, x + 9, dy + 10], fill=col)
            d.ellipse([(x - 9) * SS, (dy - 14) * SS, (x + 9) * SS, (dy + 10) * SS], fill=col + (255,))
    front = np.asarray(img.reduce(SS), np.float32) / 255
    frame = frame * (1 - front[..., 3:]) + front[..., :3] * front[..., 3:]
    g = to_arr(glow)
    frame += g * 0.9 + fast_blur(g, 12) * 1.6 + fast_blur(g, 40) * 0.8
    # key light from above
    frame *= (0.6 + 0.7 * radial(960, 200, 900, power=1.5))[..., None]

    # HUD counters racing
    hud = Image.new("RGB", (W, H))
    hd = ImageDraw.Draw(hud)
    exp_n = 48213 + int(t * 37) + cycle * 8
    secs = 168 * 3600 + int(t * 1873)
    hd.text((70, 40), f"RUN {exp_n:,}   STATUS: CONTINUOUS", font=_S["hud"], fill=(90, 230, 255))
    hd.text((1400, 40), f"UPTIME {secs // 3600:03d}:{secs // 60 % 60:02d}:{secs % 60:02d}", font=_S["hud"], fill=(255, 90, 200))
    hd.text((1400, 1010), "24 / 7   ·   0 BREAKS", font=_S["hud"], fill=(90, 230, 255))
    frame += to_arr(hud) * 0.8
    _S["kw"][0].draw(frame, t, 1.85, 3.25, cy=250)
    _S["kw"][1].draw(frame, t, 3.25, 4.85, cy=250)
    return frame
