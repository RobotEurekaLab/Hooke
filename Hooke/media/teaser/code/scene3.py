"""Scene 3 - EXPERIMENTING WITHOUT END. Rapid macro montage of 17th-century instruments."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import *

_S = {}
BRASS = np.array([0.85, 0.6, 0.28], np.float32)
CUTS = [0.0, 0.5, 1.0, 1.51]
SHOTS = ["gears", "sparks", "clock"]


def gear_patch(R, teeth, seed, spokes=6):
    size = int(2 * R + 40)
    c = size / 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xs - c, ys - c
    r = np.sqrt(dx ** 2 + dy ** 2)
    th = np.arctan2(dy, dx)
    tooth_h = R * 0.09
    edge = R - tooth_h + tooth_h * np.clip(np.cos(teeth * th) * 2.5 + 0.5, 0, 1)
    body = np.clip(edge - r, 0, 1)
    rim_in = R - tooth_h - R * 0.12
    hub = R * 0.2
    ang = (th * spokes / (2 * math.pi)) % 1.0
    spoke = np.clip((R * 0.07 - np.abs(ang - 0.5) * 2 * math.pi * r / spokes) , 0, 1)
    solid = np.maximum(np.clip(r - rim_in, 0, 1), np.maximum(np.clip(hub - r, 0, 1), spoke))
    alpha = body * solid * np.clip(r - R * 0.05, 0, 1)
    n = fbm(size, size, seed, base=4, octaves=4)
    brushed = 0.5 + 0.5 * np.sin(r * 0.9 + n * 6)
    shade = 0.55 + 0.25 * brushed + 0.35 * np.cos(th + 2.3) * (r / R)
    bevel = np.exp(-((r - rim_in) / 6) ** 2) + np.exp(-((r - hub) / 5) ** 2)
    col = BRASS * shade[..., None] + bevel[..., None] * 0.35
    return np.concatenate([col, alpha[..., None]], -1).astype(np.float32)


def wood_bg(seed, tint=(0.18, 0.09, 0.04)):
    g = rng(seed)
    gr = resize_f(g.random((H // 3, 30), dtype=np.float32), W, H, Image.BICUBIC)
    low = fbm(W, H, seed + 1, base=3, octaves=4)
    return (np.array(tint, np.float32) * (0.5 + 0.6 * gr + 0.4 * low)[..., None]).astype(np.float32)


def _build():
    _S["g1"] = gear_patch(430, 40, 31)
    _S["g2"] = gear_patch(250, 23, 32, spokes=5)
    _S["g3"] = gear_patch(150, 14, 33, spokes=4)
    _S["wood"] = wood_bg(40)
    _S["night"] = (np.linspace(0.02, 0.06, H, dtype=np.float32)[:, None, None] * np.array([0.3, 0.45, 1.0])).repeat(W, 1)
    _S["stars"] = rng(41).random((160, 3))


def paste_rot(canvas, patch, cx, cy, angle, scale=1.0, blur_angles=0.0):
    """Composite a float RGBA patch rotated around its centre, optional motion blur."""
    img = Image.fromarray((clamp01(patch) * 255).astype(np.uint8), "RGBA")
    if scale != 1.0:
        img = img.resize((max(2, int(img.width * scale)), max(2, int(img.height * scale))), Image.BILINEAR)
    samples = [angle] if blur_angles == 0 else [angle - blur_angles, angle, angle + blur_angles]
    acc = None
    for a in samples:
        r = np.asarray(img.rotate(math.degrees(a), Image.BILINEAR), np.float32) / 255
        acc = r if acc is None else acc + r
    acc /= len(samples)
    x0, y0 = int(cx - img.width / 2), int(cy - img.height / 2)
    rgb = acc[..., :3] / np.maximum(acc[..., 3:], 1e-3)
    over(canvas, rgb, acc[..., 3], x0, y0)


def lit(frame, x, y, r, strength=1.6, col=(1.0, 0.7, 0.4), amb=0.15):
    L = amb + strength * radial(x, y, r, power=2)
    return frame * (L[..., None] * np.array(col, np.float32))


def shot_gears(u, v):
    f = _S["wood"] * 0.6
    spin = u * 2.2 + v
    paste_rot(f, _S["g1"], 700, 560, spin, 1.0, 0.05)
    paste_rot(f, _S["g2"], 700 + 430 + 250 - 45, 420, -spin * 40 / 23 + 0.07, 1.0, 0.08)
    paste_rot(f, _S["g3"], 1260, 900, spin * 40 / 14 + 0.2, 1.0, 0.12)
    f = lit(f, 1100, 350, 650, 1.9)
    return zoom_crop(f, 1.0 + 0.25 * u + 0.15 * v, 0.5 + 0.05 * v, 0.5, rot=0.1 * u - 0.05)


def shot_sparks(u, v):
    f = _S["wood"] * 0.35
    g = rng(int(v * 100) + 7)
    px, py = 900 + 200 * v, 640
    layer = Image.new("L", (W, H))
    d = ImageDraw.Draw(layer)
    T = u * 0.6
    for i in range(260):
        birth = g.random() * 0.45
        age = T - birth
        if age <= 0:
            continue
        ang = -math.pi / 2 + (g.random() - 0.5) * 2.6
        sp = 500 + 1600 * g.random()
        vx, vy = math.cos(ang) * sp, math.sin(ang) * sp
        x, y = px + vx * age, py + vy * age + 1400 * age * age
        x1, y1 = x - vx * 0.02, y - (vy + 2800 * age) * 0.02
        b = max(0.0, 1 - age / (0.25 + 0.4 * g.random()))
        d.line([(x1, y1), (x, y)], fill=int(255 * b), width=3)
    sp = np.asarray(layer, np.float32) / 255
    glow = fast_blur(sp, 10)
    flash = math.exp(-u * 5)
    f = lit(f, px, py, 380, 2.4 * (0.4 + flash), (1.0, 0.6, 0.25), 0.05)
    # flint and steel silhouettes
    im = to_img(f)
    dd = ImageDraw.Draw(im)
    dd.polygon([(px - 420, py + 40), (px - 20, py - 10), (px - 10, py + 30), (px - 400, py + 130)], fill=(30, 26, 24))
    dd.polygon([(px + 30, py - 40), (px + 380, py - 180), (px + 420, py - 120), (px + 40, py + 10)], fill=(55, 55, 60))
    f = to_arr(im)
    f += sp[..., None] * np.array([1.0, 0.85, 0.5]) * 1.4 + glow[..., None] * np.array([1.0, 0.45, 0.1]) * 2.5
    f += radial(px, py, 120, power=2)[..., None] * np.array([1.0, 0.8, 0.5]) * flash * 1.5
    return f


def shot_telescope(u, v):
    f = _S["night"].copy()
    lay = Image.new("L", (W, H))
    d = ImageDraw.Draw(lay)
    for x, y, s in _S["stars"]:
        r = 1 + 3 * s
        d.ellipse([x * W - r, y * H - r, x * W + r, y * H + r], fill=int(120 + 135 * s))
    f += fast_blur(np.asarray(lay, np.float32) / 255, 3)[..., None] * np.array([0.8, 0.9, 1.0]) * 0.8
    # long tube
    Lt, Rt = 3600, 120
    ys = np.linspace(-1, 1, 2 * Rt, dtype=np.float32)
    prof = np.sqrt(np.clip(1 - ys ** 2, 0, 1))
    shade = 0.25 + 0.75 * prof * (0.6 + 0.4 * (ys < -0.2)) + 0.6 * np.exp(-((ys + 0.45) / 0.08) ** 2)
    wood = np.array([0.35, 0.18, 0.08], np.float32)
    tube = np.ones((2 * Rt, Lt // 4, 3), np.float32) * wood * shade[:, None, None]
    xs = np.arange(Lt // 4)
    rings = ((xs % 220) < 22)[None, :, None]
    tube = np.where(rings, BRASS * shade[:, None, None] * 1.3, tube)
    alpha = np.ones((2 * Rt, Lt // 4, 1), np.float32)
    patch = np.concatenate([tube, alpha], -1)
    img = Image.fromarray((clamp01(patch) * 255).astype(np.uint8), "RGBA").resize((Lt, 2 * Rt))
    patch = np.asarray(img, np.float32) / 255
    cx = 960 - 900 * (u - 0.5) * (1 if v < 0.5 else -1)
    paste_rot(f, patch, cx, 560, math.radians(12 if v < 0.5 else -8))
    f = lit(f, 1400, 300, 900, 1.2, (0.9, 0.8, 0.7), 0.35)
    # lens glint + anamorphic streak
    gx, gy = cx + 1500 * math.cos(math.radians(12)) * (1 if v < 0.5 else -0.95), 560 - 300 * (1 if v < 0.5 else -0.2)
    xs2, ys2 = grid(W, H, 4)
    streak = np.exp(-((ys2 - gy) / 6) ** 2) * np.exp(-np.abs(xs2 - gx) / 500)
    f += resize_f(streak.astype(np.float32), W, H)[..., None] * np.array([0.5, 0.75, 1.0]) * 0.8
    f += radial(gx, gy, 40)[..., None] * np.array([0.9, 0.95, 1.0]) * 1.2
    return f


def shot_clock(u, v):
    f = _S["wood"] * 0.25
    cx, cy = 960 + 120 * v, 560
    osc = math.sin(u * 2 * math.pi * 2.2 + v * 3) * 2.0
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    R = 330
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(220, 160, 70, 255), width=34)
    for k in range(3):
        a = osc + k * 2 * math.pi / 3
        d.line([(cx, cy), (cx + (R - 10) * math.cos(a), cy + (R - 10) * math.sin(a))], fill=(200, 145, 60, 255), width=22)
    # hairspring spiral (Hooke's law)
    pts = []
    breathe = 1 + 0.12 * math.sin(osc)
    for i in range(700):
        th = i * 0.06
        rr = (14 + 5.2 * th) * breathe
        pts.append((cx + rr * math.cos(th + osc), cy + rr * math.sin(th + osc)))
    d.line(pts, fill=(235, 225, 210, 255), width=4)
    d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=(150, 20, 30, 255))
    arr = np.asarray(layer, np.float32) / 255
    over(f, arr[..., :3], arr[..., 3])
    paste_rot(f, _S["g3"], cx + 520, cy + 260, -math.floor(u * 12) * 0.22, 1.0)
    f = lit(f, cx - 300, cy - 400, 700, 2.0, (1.0, 0.75, 0.5), 0.12)
    return zoom_crop(f, 1.15 + 0.35 * u, 0.5 + 0.03 * v, 0.52)


def shot_pump(u, v):
    f = _S["wood"] * 0.3
    cx, cy = 820, 640
    im = to_img(f)
    d = ImageDraw.Draw(im)
    d.ellipse([cx - 420, cy + 200, cx + 420, cy + 300], fill=(150, 105, 45))
    d.rectangle([cx - 420, cy + 250, cx + 420, cy + 330], fill=(110, 75, 30))
    # pump barrel + piston
    px = cx + 620
    d.rectangle([px - 70, cy - 300, px + 70, cy + 320], fill=(160, 115, 50))
    for k in range(5):
        d.rectangle([px - 70 + k * 10, cy - 300, px - 60 + k * 10, cy + 320], fill=(190 - k * 8, 140 - k * 6, 60))
    rod = cy - 300 - 180 * (0.5 + 0.5 * math.sin(u * 20))
    d.rectangle([px - 12, rod, px + 12, cy - 300], fill=(200, 200, 205))
    d.rectangle([px - 120, rod - 20, px + 120, rod + 10], fill=(90, 60, 30))
    d.line([(cx + 380, cy + 260), (px - 70, cy + 200)], fill=(140, 100, 40), width=16)
    f = to_arr(im)
    # candle inside the receiver, starving of air
    life = max(0.0, 1 - u * 1.2)
    fl_h = 90 * life
    if fl_h > 3:
        f += radial(cx, cy + 60 - fl_h / 2, 30 + 30 * life, power=2)[..., None] * np.array([1.0, 0.6, 0.25]) * 1.6 * life
    im = to_img(f)
    d = ImageDraw.Draw(im)
    d.rectangle([cx - 22, cy + 70, cx + 22, cy + 220], fill=(210, 195, 165))
    f = to_arr(im)
    # glass dome: fresnel edge + highlights
    xs, ys = grid(W, H, 2)
    dx, dy = (xs - cx) / 380, (ys - (cy + 60)) / 480
    rr = np.sqrt(dx ** 2 + np.minimum(dy, 0) ** 2)
    inside = (rr < 1) & (dy < 0.3)
    fres = np.where(inside, rr ** 6 * 0.5, 0) + np.where(np.abs(rr - 1) < 0.012, 0.6, 0) * (dy < 0.3)
    hl = np.where(inside, np.exp(-((dx + 0.55) / 0.06) ** 2) * np.exp(-((dy + 0.3) / 0.35) ** 2), 0) * 0.8
    glass = resize_f((fres + hl).astype(np.float32), W, H)
    f += glass[..., None] * np.array([0.75, 0.85, 1.0])
    f = lit(f, cx - 200, cy - 500, 800, 1.6, (1.0, 0.8, 0.6), 0.25)
    return zoom_crop(f, 1.05 + 0.3 * u, 0.5, 0.5)


FUNCS = {"gears": shot_gears, "gears2": lambda u, v: shot_gears(u, 1.3), "sparks": shot_sparks,
         "telescope": shot_telescope, "clock": shot_clock, "pump": shot_pump}


def render(t, dur=1.5):
    if not _S:
        _build()
    i = max(k for k in range(len(CUTS) - 1) if CUTS[k] <= t)
    t0, t1 = CUTS[i], CUTS[i + 1]
    u = (t - t0) / (t1 - t0)
    frame = FUNCS[SHOTS[i]](u, (i * 0.37) % 1.0)
    # whip flash at each cut
    fl = math.exp(-(t - t0) * 14)
    frame = frame + fl * 0.6 * np.array([1.0, 0.85, 0.65], np.float32)
    return frame
