"""Scene 5 - 2026. Glitch out of the cell, fiber-optic data tunnel, reveal of an automated lab."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *
import scene4

_S = {}
CYAN = np.array([0.25, 0.9, 1.0], np.float32)
BLUE = np.array([0.15, 0.4, 1.0], np.float32)
MAG = np.array([1.0, 0.25, 0.8], np.float32)


def glyph_strip(seed, rows=60, size=26):
    g = rng(seed)
    f = ImageFont.truetype(FONTS + "consola.ttf", size)
    img = Image.new("L", (size, rows * size))
    d = ImageDraw.Draw(img)
    chars = "0123456789ABCDEF<>/{}[]#$%&*+=ATGC"
    for r in range(rows):
        d.text((2, r * size), chars[g.integers(len(chars))], font=f, fill=int(90 + 165 * g.random()))
    return np.asarray(img, np.float32) / 255


def _build():
    g = rng(61)
    _S["tunnel"] = g.random((1400, 5)).astype(np.float32)
    _S["strips"] = [glyph_strip(62 + i) for i in range(12)]
    _S["text"] = Text3D("2026.", make_font("bahnschrift.ttf", 380, "Bold"), "chrome", tracking=0.08,
                        side=(0.05, 0.25, 0.35), side_back=(0.0, 0.02, 0.05), seed=12)
    _S["kw"] = [Keyword("A NEW GOLDEN AGE"), Keyword("OF DISCOVERY")]
    _S["font_s"] = ImageFont.truetype(FONTS + "consola.ttf", 22)
    _S["font_m"] = ImageFont.truetype(FONTS + "bahnschrift.ttf", 30)


def glitch(frame, amount, seed):
    g = rng(seed)
    out = frame.copy()
    for _ in range(int(14 * amount)):
        y0 = g.integers(0, H - 40)
        h = g.integers(6, 90)
        dx = int(g.normal(0, 140 * amount))
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], dx, 1)
    s = int(24 * amount)
    out[..., 0] = np.roll(out[..., 0], s, 1)
    out[..., 2] = np.roll(out[..., 2], -s, 1)
    for _ in range(int(10 * amount)):
        x, y = g.integers(0, W - 200), g.integers(0, H - 60)
        out[y:y + g.integers(8, 60), x:x + g.integers(40, 300)] = CYAN * g.random() if g.random() < 0.5 else MAG * g.random()
    out[::3] *= 1 - 0.3 * amount
    return out


def tunnel(t, strength=1.0):
    frame = np.zeros((H, W, 3), np.float32) + np.array([0.0, 0.01, 0.03], np.float32)
    layer = Image.new("RGB", (W // 2, H // 2))
    d = ImageDraw.Draw(layer)
    cx, cy = W / 4, H / 4
    speed = 2.2
    for a, rr, z0, c, l in _S["tunnel"]:
        z = ((z0 * 6 + t * speed) % 6) + 0.15  # receding: z grows over time -> pulls back
        r = 40 + 900 * rr
        ang = a * 2 * math.pi
        x1, y1 = cx + math.cos(ang) * r / z, cy + math.sin(ang) * r / z * 0.8
        z2 = z + 0.08 + 0.4 * l
        x2, y2 = cx + math.cos(ang) * r / z2, cy + math.sin(ang) * r / z2 * 0.8
        fade = min(1.0, 3.0 / z) * strength
        col = CYAN if c < 0.6 else (BLUE if c < 0.9 else MAG)
        rgb = tuple(int(255 * min(1, v * fade)) for v in col)
        d.line([(x1, y1), (x2, y2)], fill=rgb, width=2 if z < 1 else 1)
    arr = resize_f(to_arr(layer), W, H)
    frame += arr * 1.6 + fast_blur(arr, 12) * 2.0
    # glyph rain on the tunnel walls
    for i, strip in enumerate(_S["strips"]):
        x = int((i + 0.5) * W / 12)
        if abs(x - W / 2) < 260:
            continue
        off = int((t * (300 + 40 * i) + i * 97) % strip.shape[0])
        col = np.roll(strip, off, 0)[:H]
        w = col.shape[1]
        edge = abs(x - W / 2) / (W / 2)
        add_patch(frame, col[..., None] * CYAN * 0.5 * edge * strength, x, 0)
    frame += radial(W / 2, H / 2, 160, power=2)[..., None] * np.array([0.6, 0.9, 1.0]) * 0.6 * strength
    return frame


def lab(t, fo):
    """Futuristic lab: screens wall, reflective floor, holograms."""
    frame = np.zeros((H, W, 3), np.float32) + np.array([0.005, 0.012, 0.03], np.float32)
    hor = 600
    img = to_img(frame)
    d = ImageDraw.Draw(img)
    par = t * 14
    screens = [(140 - par * 0.5, 190, 700, 520), (760 - par * 0.5, 160, 1220, 520), (1280 - par * 0.5, 190, 1840, 520)]
    for k, (x0, y0, x1, y1) in enumerate(screens):
        d.rectangle([x0, y0, x1, y1], fill=(4, 22, 34))
        d.rectangle([x0, y0, x1, y1], outline=(60, 200, 230), width=3)
    # animated charts
    for k, (x0, y0, x1, y1) in enumerate(screens):
        if k == 0:
            pts = []
            for i in range(80):
                u = i / 79
                if u > (t * 0.4 + 0.3) % 1.3:
                    break
                v = 0.5 + 0.3 * math.sin(u * 9 + k) * math.exp(-u) + 0.1 * math.sin(u * 40 + t * 3)
                pts.append((x0 + 30 + u * (x1 - x0 - 60), y1 - 40 - v * (y1 - y0 - 80)))
            if len(pts) > 1:
                d.line(pts, fill=(90, 240, 255), width=4)
            for gy in range(5):
                yy = y0 + 40 + gy * (y1 - y0 - 80) / 4
                d.line([(x0 + 30, yy), (x1 - 30, yy)], fill=(20, 70, 90), width=1)
        elif k == 1:
            for i in range(14):
                hgt = (0.3 + 0.6 * abs(math.sin(i * 1.7 + t * 2.3))) * (y1 - y0 - 90)
                bx = x0 + 30 + i * (x1 - x0 - 60) / 14
                d.rectangle([bx, y1 - 30 - hgt, bx + 20, y1 - 30], fill=(40, 170, 255) if i % 3 else (255, 70, 200))
            d.text((x0 + 30, y0 + 20), "THROUGHPUT  12,480 exp/h", font=_S["font_m"], fill=(150, 240, 255))
        else:
            cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2 + 10
            nodes = []
            for i in range(10):
                a = i * 0.63 + t * 0.6
                rr = 90 + 50 * math.sin(i * 2.1)
                nodes.append((cxm + rr * math.cos(a) * 1.6, cym + rr * math.sin(a) * 0.9))
            for i in range(10):
                for j in (i + 1, i + 3):
                    d.line([nodes[i], nodes[j % 10]], fill=(60, 180, 220), width=2)
            for (nx, ny) in nodes:
                d.ellipse([nx - 9, ny - 9, nx + 9, ny + 9], fill=(120, 255, 240))
    # floor grid
    for i in range(-20, 21):
        d.line([(W / 2 + i * 30, hor), (W / 2 + i * 260 - par * 2, H)], fill=(10, 60, 80), width=1)
    for j in range(12):
        z = ((j + t * 1.5) % 12) + 1
        yy = hor + 480 / z
        d.line([(0, yy), (W, yy)], fill=(10, 50, 70), width=1)
    # benches in silhouette
    for bx in (-80, 1180):
        d.polygon([(bx, 820), (bx + 800, 760), (bx + 800, 800), (bx, 880)], fill=(8, 16, 24))
        for k in range(4):
            ex = bx + 100 + k * 170
            d.rectangle([ex, 700 - k * 10, ex + 70, 790 - k * 12], fill=(12, 26, 38))
            d.line([(ex, 700 - k * 10), (ex + 70, 700 - k * 10)], fill=(80, 220, 255), width=2)
    frame = to_arr(img)
    # screen glow + floor reflection
    glow = fast_blur(frame, 30)
    frame += glow * 1.4
    refl = frame[hor - (H - hor):hor][::-1]
    refl = fast_blur(refl.copy(), 6)
    frame[hor:] += refl * np.linspace(0.45, 0.0, H - hor, dtype=np.float32)[:, None, None]
    # holographic panels
    for k, (px, py, pw, ph, ry) in enumerate([(300, 560, 360, 240, 0.5), (1450, 520, 400, 260, -0.55)]):
        pan = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pan)
        pd.rectangle([0, 0, pw - 1, ph - 1], fill=(40, 200, 255, 40), outline=(120, 240, 255, 220), width=3)
        for i in range(7):
            pd.text((16, 14 + i * 30), f"SAMPLE {1000 + int(t * 37) + i * 7:05d}  pH {6.8 + 0.1 * math.sin(i + t):.2f}",
                    font=_S["font_s"], fill=(170, 250, 255, 220))
        for yy in range(0, ph, 4):
            pd.line([(0, yy), (pw, yy)], fill=(0, 0, 0, 40))
        bob = 12 * math.sin(t * 1.3 + k)
        wq = pw * math.cos(ry)
        dz = pw * math.sin(ry) * 0.25
        quad = [(px - wq / 2 - par * 1.5, py - ph / 2 + bob - dz), (px + wq / 2 - par * 1.5, py - ph / 2 + bob + dz),
                (px + wq / 2 - par * 1.5, py + ph / 2 + bob - dz), (px - wq / 2 - par * 1.5, py + ph / 2 + bob + dz)]
        r = warp_to_quad(pan, quad)
        if r:
            im, x0, y0 = r
            a = np.asarray(im, np.float32) / 255
            flick = 0.85 + 0.15 * math.sin(t * 31 + k * 5)
            add_patch(frame, a[..., :3] * a[..., 3:] * 1.3 * flick, x0, y0)
    return frame


def render(t, dur=7.0):
    if not _S:
        _build()
    T_TUN, T_LAB = 0.45, 3.3
    if t < T_TUN:
        base = scene4.render(10.49)
        frame = glitch(base, 1.0 - 0.3 * t, int(t * 24) + 3)
        frame = frame * (1 - t / T_TUN) + tunnel(t) * (t / T_TUN)
        frame += math.exp(-t * 10) * 0.8
    elif t < T_LAB + 0.6:
        frame = tunnel(t)
        if t > T_LAB:
            k = (t - T_LAB) / 0.6
            labf = zoom_crop(lab(t, 0), 1.6 - 0.6 * ease_out(k), 0.5, 0.5)
            frame = frame * (1 - k) + labf * k + (1 - abs(k - 0.35) * 2.5) * 0.22 * (abs(k - 0.35) < 0.4)
    else:
        k = ease_out((t - T_LAB - 0.6) / 4.0)
        frame = zoom_crop(lab(t, 0), 1.0 + 0.06 * (1 - k) + 0.04 * (t - T_LAB) / 4, 0.5, 0.5)
    # "2026." materialises with a glitch as it is spoken
    ts = 1.3
    if t > ts:
        a = ease_out((t - ts) / 0.9)
        exit_k = smooth((t - 5.6) / 0.7)
        z = lerp(2400, 1000, a) - 700 * exit_k
        alpha = smooth((t - ts) / 0.25) * (1 - exit_k)
        if (t - ts) < 0.45 and int(t * 48) % 3 == 0:
            alpha *= 0.3
        _S["text"].draw(frame, pos=(0, -30, z), rx=0.06, ry=lerp(-0.45, 0.1, a) + 0.03 * math.sin(t), scale=1.0,
                        depth=80, layers=16, alpha=float(alpha), sweep=lerp(-0.3, 1.4, ((t - ts) % 3.0) / 1.6),
                        light=1.0, glow=0.55, glow_color=(0.2, 0.8, 1.0), blur=max(0.0, 6 * (1 - a)),
                        reflection=(0, 0.25) if t > T_LAB else None)
        if (t - ts) < 0.5:
            frame = glitch(frame, 0.5 * (1 - (t - ts) / 0.5), int(t * 24) + 99)
    _S["kw"][0].draw(frame, t, 2.55, 4.5, cy=820)
    _S["kw"][1].draw(frame, t, 4.5, 6.6, cy=820)
    return frame
