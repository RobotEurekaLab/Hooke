"""Scene 6 - HIS SUCCESSORS. An endless automated research facility, arms moving in perfect synchrony."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import *

_S = {}
SS = 2  # supersampling
F = 900.0
VPX, VPY = 960.0, 455.0
CAM_Y = 2.4
FOG = np.array([0.02, 0.07, 0.12])
CAPS = [(80, 240, 255), (90, 255, 170), (255, 80, 210), (120, 160, 255)]


def proj(x, y, z):
    return ((VPX + F * x / z) * SS, (VPY + F * (CAM_Y - y) / z) * SS)


def fogc(col, z):
    k = 1 - math.exp(-z / 38.0)
    return tuple(int(c * (1 - k) + f * 255 * k) for c, f in zip(col, FOG))


def _build():
    _S["text"] = Text3D("HIS SUCCESSORS.", make_font("bahnschrift.ttf", 210, "Bold"), "chrome", tracking=0.1,
                        side=(0.04, 0.2, 0.3), side_back=(0.0, 0.02, 0.05), seed=14)
    _S["kw"] = [Keyword("WE ARE BUILDING"), Keyword("COUNTLESS SCIENTISTS")]


def render(t, dur=5.6):
    if not _S:
        _build()
    img = Image.new("RGB", (W * SS, H * SS), (3, 9, 16))
    d = ImageDraw.Draw(img)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    # background gradient toward vanishing point
    spacing = 2.4
    off = (t * 4.2) % spacing
    th1 = -0.9 + 0.55 * math.sin(t * 2.6)
    th2 = 1.3 + 0.7 * math.sin(t * 2.6 + 1.1)
    # ceiling light strips & floor lines (glow layer, 1x)
    for x in (-1.6, 1.6, -5.2, 5.2):
        for y, w in ((11.0, 3), (0.0, 2)):
            p0 = (VPX + F * x / 1.0, VPY + F * (CAM_Y - y) / 1.0)
            gd.line([p0, (VPX, VPY)], fill=(40, 150, 200) if y > 0 else (20, 110, 150), width=w)
    items = []
    n_bays = 44
    for i in range(n_bays, -1, -1):
        z = i * spacing - off + 1.0
        if z < 0.6:
            continue
        for side in (-1, 1):
            for X, Y in ((3.0, 0.0), (7.2, 0.0), (7.2, 4.5), (11.5, 0.0), (11.5, 4.5), (11.5, 9.0)):
                items.append((z, side, X, Y, i))
    items.sort(key=lambda it: -it[0])
    for z, side, X, Y, i in items:
        x = side * X
        s = F * SS / z
        # bench
        top = [proj(x - 0.8, Y + 0.9, z), proj(x + 0.8, Y + 0.9, z), proj(x + 0.8, Y + 0.9, z + 2.0), proj(x - 0.8, Y + 0.9, z + 2.0)]
        d.polygon(top, fill=fogc((34, 44, 56), z))
        fx = x - side * 0.8
        face = [proj(fx, Y + 0.9, z), proj(fx, Y + 0.9, z + 2.0), proj(fx, Y, z + 2.0), proj(fx, Y, z)]
        d.polygon(face, fill=fogc((16, 22, 30), z))
        if Y > 0:  # mezzanine deck
            deck = [proj(x - 2.0, Y, z), proj(x + 2.0, Y, z), proj(x + 2.0, Y, z + spacing), proj(x - 2.0, Y, z + spacing)]
            d.polygon(deck, fill=fogc((12, 18, 26), z))
        e0, e1 = proj(fx, Y + 0.9, z), proj(fx, Y + 0.9, z + 2.0)
        gd.line([(e0[0] / SS, e0[1] / SS), (e1[0] / SS, e1[1] / SS)], fill=fogc((60, 220, 255), z * 0.6), width=max(1, int(3 * s / SS / 40)))
        # reaction capsule
        cx = x + side * 0.35
        c0, c1 = proj(cx - 0.13, Y + 1.65, z + 1.4), proj(cx + 0.13, Y + 0.9, z + 1.4)
        cc = CAPS[(i + int(X)) % len(CAPS)]
        d.rounded_rectangle([min(c0[0], c1[0]), c0[1], max(c0[0], c1[0]), c1[1]], radius=int(0.1 * s), fill=fogc((20, 30, 40), z))
        pulse = 0.6 + 0.4 * math.sin(t * 3 + i * 0.7)
        lv = Y + 0.95 + 0.55 * pulse
        g0, g1 = proj(cx - 0.1, lv, z + 1.4), proj(cx + 0.1, Y + 0.95, z + 1.4)
        gd.rectangle([min(g0[0], g1[0]) / SS, g0[1] / SS, max(g0[0], g1[0]) / SS, g1[1] / SS], fill=fogc(cc, z * 0.5))
        d.rectangle([min(g0[0], g1[0]), g0[1], max(g0[0], g1[0]), g1[1]], fill=fogc(tuple(int(c * 0.8) for c in cc), z * 0.5))
        # robot arm (all arms share one phase: perfect synchrony)
        bx, by = x - side * 0.1, Y + 0.9
        az = z + 0.6
        sh = (bx, by + 0.45)
        dirx = -side
        a1 = th1
        el = (sh[0] + dirx * 0.8 * math.cos(a1), sh[1] + 0.8 * math.sin(-a1))
        a2 = a1 + th2
        wr = (el[0] + dirx * 0.65 * math.cos(a2), el[1] + 0.65 * math.sin(-a2))
        metal = fogc((190, 200, 212), z)
        dark = fogc((70, 78, 90), z)
        p = lambda q: proj(q[0], q[1], az)
        d.polygon([proj(bx - 0.18, by, az), proj(bx + 0.18, by, az), proj(bx + 0.12, by + 0.45, az), proj(bx - 0.12, by + 0.45, az)], fill=dark)
        d.line([p(sh), p(el)], fill=metal, width=max(1, int(0.16 * s)))
        d.line([p(el), p(wr)], fill=metal, width=max(1, int(0.11 * s)))
        for jnt, r in ((sh, 0.12), (el, 0.09), (wr, 0.07)):
            q = p(jnt)
            rr = r * s
            d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=dark)
            gq = (q[0] / SS, q[1] / SS)
            gr = max(1.0, rr / SS * 0.55)
            gd.ellipse([gq[0] - gr, gq[1] - gr, gq[0] + gr, gq[1] + gr], outline=fogc((80, 230, 255), z * 0.7), width=max(1, int(gr / 3)))
        # gripper
        g_a = a2 + 0.5
        tip1 = (wr[0] + dirx * 0.2 * math.cos(g_a), wr[1] + 0.2 * math.sin(-g_a))
        tip2 = (wr[0] + dirx * 0.2 * math.cos(g_a - 1.0), wr[1] + 0.2 * math.sin(-(g_a - 1.0)))
        d.line([p(wr), p(tip1)], fill=metal, width=max(1, int(0.04 * s)))
        d.line([p(wr), p(tip2)], fill=metal, width=max(1, int(0.04 * s)))
    # scanning lasers across the aisle
    for k in range(6):
        z = (k * 9 + 5 - (t * 4.2) % 9)
        if z < 1:
            continue
        y = 1.2 + 0.9 * math.sin(t * 2 + k)
        a, b = proj(-3.0, y, z), proj(3.0, y, z)
        gd.line([(a[0] / SS, a[1] / SS), (b[0] / SS, b[1] / SS)], fill=fogc((60, 140, 255), z * 0.3), width=2)
    frame = to_arr(img.reduce(SS))
    g = to_arr(glow)
    frame += g * 1.2 + fast_blur(g, 14) * 2.2 + fast_blur(g, 40) * 1.5
    # atmospheric depth glow at the vanishing point
    frame += radial(VPX, VPY, 260, power=1.6)[..., None] * np.array([0.1, 0.35, 0.5], np.float32) * 0.9
    # glossy floor: reflect the upper half
    hor = int(VPY)
    refl = frame[:hor][::-1].copy()
    n = min(hor, H - hor)
    frame[hor:hor + n] += fast_blur(refl, 8)[:n] * np.linspace(0.3, 0.0, n, dtype=np.float32)[:, None, None]

    ts = 1.0
    if t > ts:
        a = ease_out((t - ts) / 1.4)
        z = lerp(3200, 1450, a) - 40 * (t - ts)
        _S["text"].draw(frame, pos=(0, -110, z), rx=0.04, ry=lerp(0.35, 0.0, a), scale=0.75, depth=70, layers=14,
                        alpha=float(smooth((t - ts) / 0.35) * (1 - smooth((t - 4.85) / 0.45))),
                        sweep=lerp(-0.3, 1.4, (t - ts - 0.8) / 2.0), light=1.0, glow=0.45, glow_color=(0.2, 0.75, 1.0),
                        blur=max(0.0, 5 * (1 - a)), reflection=(0, 0.3))
    _S["kw"][0].draw(frame, t, 0.1, 1.15, cy=830)
    _S["kw"][1].draw(frame, t, 2.95, 5.35, cy=830)
    return frame
