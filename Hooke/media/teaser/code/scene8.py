"""Scene 8 - ROBOT SCIENTISTS. Hero robot raises its arm; the code matrix becomes a galaxy. Plus end card."""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *

_S = {}
SS = 2
GX, GY = 1330, 470
CELL = 30


def _build():
    g = rng(81)
    cols, rows = W // CELL, (H - 240) // CELL
    f = ImageFont.truetype(FONTS + "consola.ttf", 24)
    glyph = Image.new("L", (W, H))
    d = ImageDraw.Draw(glyph)
    chars = "01ATGC{}<>/=+*#λΣΔ∂"
    for r in range(rows):
        for c in range(cols):
            d.text((c * CELL + 4, 120 + r * CELL), chars[g.integers(len(chars))], font=f, fill=int(120 + 135 * g.random()))
    _S["glyph"] = np.asarray(glyph, np.float32) / 255
    _S["cols"], _S["rows"] = cols, rows
    _S["speed"] = 6 + 10 * g.random(cols)
    _S["phase"] = g.random(cols) * rows
    # particles: start on glyph cells, end in a spiral galaxy
    n = cols * rows * 14
    cc = np.repeat(np.arange(cols * rows), 14)
    sx = (cc % cols) * CELL + CELL / 2 + g.normal(0, 6, n)
    sy = 120 + (cc // cols) * CELL + CELL / 2 + g.normal(0, 6, n)
    arm = g.integers(0, 3, n)
    r = np.abs(g.exponential(210, n)) + 12
    theta = arm * 2 * math.pi / 3 + np.log(r) * 2.4 + g.normal(0, 0.22, n)
    core = g.random(n) < 0.1
    r = np.where(core, np.abs(g.normal(0, 45, n)), r)
    theta = np.where(core, g.random(n) * 2 * math.pi, theta)
    _S["p"] = dict(sx=sx.astype(np.float32), sy=sy.astype(np.float32), r=r.astype(np.float32), th=theta.astype(np.float32),
                   delay=g.random(n).astype(np.float32), b=(0.4 + 0.6 * g.random(n)).astype(np.float32))
    rn = np.clip(r / 500, 0, 1)
    colr = np.stack([0.85 + 0.15 * (1 - rn), 0.7 * (1 - rn) + 0.35 * rn, 0.55 * (1 - rn) + 1.0 * rn], -1)
    mag = g.random(n) < 0.15
    colr[mag] = [1.0, 0.35, 0.85]
    _S["pcol"] = colr.astype(np.float32)
    neb = fbm(W // 4, H // 4, 82, base=4, octaves=5)
    neb2 = fbm(W // 4, H // 4, 83, base=6, octaves=5)
    rad = radial(GX / 4, GY / 4, 110, W // 4, H // 4, k=1, power=1.4)
    nebula = (neb[..., None] * np.array([0.25, 0.35, 1.0]) + neb2[..., None] * np.array([0.9, 0.2, 0.7]) * 0.6) * rad[..., None]
    _S["nebula"] = resize_f(nebula.astype(np.float32), W, H)
    st = np.zeros((H, W), np.float32)
    idx = g.integers(0, W * H, 1800)
    st.flat[idx] = g.random(1800) ** 3
    _S["stars"] = blur_f(st, 1) * 3
    _S["text"] = Text3D("ROBOT SCIENTISTS.", make_font("bahnschrift.ttf", 200, "Bold"), "chrome", tracking=0.07,
                        side=(0.04, 0.22, 0.32), side_back=(0.0, 0.02, 0.05), seed=19)
    _S["kw"] = [Keyword("ACCELERATING SCIENTIFIC DISCOVERY", size=66),
                Keyword("FOR OUR FUTURE!", size=76, color=(1.0, 0.95, 0.8), accent=(1.0, 0.55, 0.2))]


def robot_layers(t):
    """Returns (fill RGB float, mask float) of the humanoid robot at 1x."""
    img = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    m = Image.new("L", (W * SS, H * SS), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(m)
    S = lambda pts: [(x * SS, y * SS) for x, y in pts]

    def shape(kind, pts, col, **kw):
        getattr(d, kind)(S(pts) if kind == "polygon" else [c * SS for c in pts], fill=col, **kw)
        getattr(dm, kind)(S(pts) if kind == "polygon" else [c * SS for c in pts], fill=255, **kw)

    def limb(p0, p1, w, col):
        d.line(S([p0, p1]), fill=col, width=int(w * SS))
        dm.line(S([p0, p1]), fill=255, width=int(w * SS))
        for p in (p0, p1):
            shape("ellipse", [p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], col)

    body = (38, 42, 50)
    plate = (58, 64, 76)
    ox = -10 * math.sin(t * 0.3)
    # back arm (hanging)
    limb((500 + ox, 520), (470 + ox, 760), 70, (28, 31, 38))
    limb((470 + ox, 760), (480 + ox, 950), 58, (28, 31, 38))
    # torso
    shape("polygon", [(470 + ox, 470), (800 + ox, 470), (760 + ox, 700), (700 + ox, 1000), (540 + ox, 1000), (500 + ox, 700)], body)
    d.polygon(S([(560 + ox, 500), (760 + ox, 500), (730 + ox, 660), (590 + ox, 660)]), fill=plate)
    for k in range(4):
        d.line(S([(560 + ox, 700 + k * 45), (720 + ox, 700 + k * 45)]), fill=(24, 27, 33), width=6 * SS)
    # neck + head
    limb((640 + ox, 400), (640 + ox, 470), 60, (30, 33, 40))
    shape("rounded_rectangle", [555 + ox, 230, 735 + ox, 420], body, radius=70 * SS)
    d.rounded_rectangle([600 * SS + ox * SS, 280 * SS, 740 * SS + ox * SS, 340 * SS], radius=28 * SS, fill=(12, 14, 18))
    # raising arm
    k = ease_in_out((t - 0.4) / 3.6)
    sh = (770 + ox, 510)
    a1 = math.radians(lerp(100, -20, k))
    el = (sh[0] + 240 * math.cos(a1), sh[1] + 240 * math.sin(a1))
    a2 = a1 - math.radians(lerp(8, 10, k))
    wr = (el[0] + 220 * math.cos(a2), el[1] + 220 * math.sin(a2))
    limb(sh, el, 78, (44, 48, 57))
    limb(el, wr, 64, (44, 48, 57))
    shape("ellipse", [sh[0] - 62, sh[1] - 62, sh[0] + 62, sh[1] + 62], plate)
    # hand: palm + fingers opening toward the stars
    open_k = smooth((t - 3.0) / 1.2)
    for j in range(4):
        fa = a2 + math.radians(-22 + j * 15) * (0.4 + 0.6 * open_k)
        base = (wr[0] + 30 * math.cos(a2), wr[1] + 30 * math.sin(a2))
        mid = (base[0] + 55 * math.cos(fa), base[1] + 55 * math.sin(fa))
        tip = (mid[0] + 45 * math.cos(fa - 0.25 * (1 - open_k)), mid[1] + 45 * math.sin(fa - 0.25 * (1 - open_k)))
        limb(base, mid, 17, (60, 66, 78))
        limb(mid, tip, 14, (60, 66, 78))
    th_a = a2 + math.radians(60)
    limb((wr[0], wr[1]), (wr[0] + 60 * math.cos(th_a), wr[1] + 60 * math.sin(th_a)), 18, (60, 66, 78))
    shape("ellipse", [wr[0] - 42, wr[1] - 42, wr[0] + 42, wr[1] + 42], (50, 55, 65))
    fill = to_arr(img.reduce(SS))
    mask = np.asarray(m.reduce(SS), np.float32) / 255
    return fill, mask, (600 + ox, 280, 740 + ox, 340), wr


def god_rays(light, cx, cy, samples=14, decay=0.5):
    small = Image.fromarray(light[::4, ::4].astype(np.float32), "F")
    w, h = small.size
    c = (cx / 4, cy / 4)
    acc = np.zeros((h, w), np.float32)
    for i in range(samples):
        s = 1 + decay * i / samples
        co = (1 / s, 0, c[0] * (1 - 1 / s), 0, 1 / s, c[1] * (1 - 1 / s))
        acc += np.asarray(small.transform((w, h), Image.AFFINE, co, Image.BILINEAR))
    return resize_f(acc / samples, W, H)


def render(t, dur=7.0):
    if not _S:
        _build()
    frame = np.zeros((H, W, 3), np.float32) + np.array([0.0, 0.005, 0.015], np.float32)
    morph = smooth((t - 1.0) / 3.2)
    # matrix rain glyphs dissolving
    cols, rows = _S["cols"], _S["rows"]
    heads = (_S["phase"] + t * _S["speed"]) % (rows + 10)
    rr = np.arange(rows)[:, None]
    bright = np.exp(-np.clip(heads[None, :] - rr, 0, None) / 5.0) * (rr <= heads[None, :])
    bright_img = np.zeros((H, W), np.float32)
    bi = np.repeat(np.repeat(bright, CELL, 0), CELL, 1)
    bright_img[120:120 + bi.shape[0], :bi.shape[1]] = bi
    gl = _S["glyph"] * (0.15 + bright_img) * (1 - morph)
    frame += gl[..., None] * np.array([0.2, 1.0, 0.75], np.float32) * 0.9
    # particles
    P = _S["p"]
    mi = smooth((t - 1.0 - P["delay"] * 1.6) / 2.2)
    rot = 0.07 * t
    th = P["th"] + rot
    gx = GX + P["r"] * np.cos(th) * 1.25
    gy = GY + P["r"] * np.sin(th) * 0.5
    ca, sa = math.cos(-0.35), math.sin(-0.35)
    gx2 = GX + (gx - GX) * ca - (gy - GY) * sa
    gy2 = GY + (gx - GX) * sa + (gy - GY) * ca
    swirl = np.sin(mi * math.pi) * 120
    x = P["sx"] * (1 - mi) + gx2 * mi + swirl * np.cos(P["th"])
    y = P["sy"] * (1 - mi) + gy2 * mi + swirl * np.sin(P["th"])
    ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    idx = (y[ok].astype(int)) * W + x[ok].astype(int)
    acc = np.zeros((H * W, 3), np.float32)
    col = _S["pcol"][ok] * mi[ok, None] + np.array([0.3, 1.0, 0.8]) * (1 - mi[ok, None])
    np.add.at(acc, idx, col * P["b"][ok, None] * (0.25 + 0.75 * mi[ok, None]))
    acc = acc.reshape(H, W, 3)
    frame += acc * 0.9 + fast_blur(acc, 3) * 2.2 + fast_blur(acc, 24) * 2.5
    frame += _S["nebula"] * 0.9 * morph
    frame += _S["stars"][..., None] * morph * 0.8
    core = radial(GX, GY, 55, power=1.8)
    frame += core[..., None] * np.array([1.0, 0.85, 0.7]) * 0.35 * morph
    # sweeping lasers behind the robot
    lay = Image.new("RGB", (W, H))
    ld = ImageDraw.Draw(lay)
    for k, (c, sp) in enumerate([((60, 220, 255), 0.4), ((255, 60, 200), -0.33)]):
        a = math.radians(-25 + 40 * math.sin(t * sp + k))
        cx, cy = 300 + k * 1300, 900
        ld.line([(cx, cy), (cx + 2600 * math.cos(a - math.pi / 2 + 0.6 * (k * 2 - 1)), cy + 2600 * math.sin(a - math.pi / 2))], fill=c, width=3)
    la = to_arr(lay) * (1 - 0.6 * morph)
    frame += la * 0.5 + fast_blur(la, 16) * 0.8

    # robot
    fill, mask, visor, wr = robot_layers(t)
    light = (core * 3 * morph + 0.1) * (1 - mask)
    rays = god_rays(light, GX, GY)
    shade = (0.35 + 0.5 * radial(GX, GY, 900, power=1.2))[..., None]
    frame = frame * (1 - mask[..., None]) + fill * shade * mask[..., None]
    kk = 7
    rim_c = np.clip(mask - np.roll(mask, -kk, 1), 0, 1) + 0.6 * np.clip(mask - np.roll(mask, kk, 0), 0, 1)
    rim_m = np.clip(mask - np.roll(mask, kk, 1), 0, 1)
    rim_c = blur_f(rim_c, 2)
    rim_m = blur_f(rim_m, 2)
    frame += rim_c[..., None] * np.array([0.3, 0.9, 1.0]) * (0.8 + 0.8 * morph)
    frame += rim_m[..., None] * np.array([1.0, 0.25, 0.75]) * 0.9
    frame += (fast_blur(rim_c, 10)[..., None] * np.array([0.2, 0.7, 1.0]) + fast_blur(rim_m, 10)[..., None] * np.array([0.8, 0.1, 0.6])) * 0.8
    frame += rays[..., None] * np.array([0.6, 0.75, 1.0]) * 0.5
    # visor + chest core
    v = Image.new("L", (W, H))
    vd = ImageDraw.Draw(v)
    x0, y0, x1, y1 = visor
    blink = 0.8 + 0.2 * math.sin(t * 5)
    vd.rounded_rectangle([x0 + 20, y0 + 20, x1 - 10, y1 - 20], radius=10, fill=int(255 * blink))
    vd.ellipse([x1 - 40, y0 + 18, x1 - 16, y1 - 18], fill=255)
    va = np.asarray(v, np.float32) / 255
    frame += va[..., None] * np.array([0.5, 1.0, 1.0]) * 1.2 + fast_blur(va, 14)[..., None] * np.array([0.2, 0.8, 1.0]) * 2.0

    # title slam
    ts = 1.05
    if t > ts:
        a = ease_out((t - ts) / 0.45)
        z = lerp(380, 1180, a) + 25 * (t - ts)
        frame *= 1 - 0.25 * smooth((t - ts) / 0.5)
        _S["text"].draw(frame, pos=(0, 250, z), rx=-0.08, ry=lerp(0.25, 0.0, a), scale=0.95, depth=80, layers=16,
                        alpha=float(smooth((t - ts) / 0.12)), sweep=lerp(-0.3, 1.4, (t - ts - 0.3) / 1.6), light=1.05,
                        glow=0.6, glow_color=(0.2, 0.8, 1.0), blur=max(0.0, 8 * (1 - a)))
        fl = math.exp(-(t - ts) * 5)
        xs, ys = grid(W, H, 4)
        streak = np.exp(-((ys - 790) / 5) ** 2) * np.exp(-np.abs(xs - 960) / 700)
        frame += resize_f(streak.astype(np.float32), W, H)[..., None] * np.array([0.4, 0.8, 1.0]) * (0.5 + 1.5 * fl)
        frame += fl * 0.5
    _S["kw"][0].draw(frame, t, 2.2, 4.3, cy=268)
    _S["kw"][1].draw(frame, t, 4.35, 6.85, cy=268)
    return frame


def render_end(t, dur=4.4):
    """End card: AGS mark in gold, with a clearly legible descriptor."""
    if "logo" not in _S:
        src = Image.open(os.path.join(os.path.dirname(__file__), "..", "ags_robot.png")).convert("L")
        src = src.resize((int(src.width * 300 / src.height), 300), Image.LANCZOS)
        a_ = 1 - np.asarray(src, np.float32) / 255
        _S["logo"] = np.clip(a_ * 1.6, 0, 1)
        _S["f_big"] = make_font("bahnschrift.ttf", 130, "Bold")
        _S["f_small"] = make_font("bahnschrift.ttf", 44, "SemiBold")
    frame = np.zeros((H, W, 3), np.float32)
    fade = smooth((t - 0.35) / 0.9) * (1 - smooth((t - (dur - 0.8)) / 0.8))
    GOLD = np.array([1.0, 0.76, 0.26], np.float32)
    logo = _S["logo"]
    lh, lw = logo.shape
    add_patch(frame, logo[..., None] * np.array([0.98, 0.86, 0.6], np.float32) * 0.8 * fade,
              (W - lw) // 2, 215)
    # AGS in gold, with its own glow
    m_big = text_mask("AGS", _S["f_big"], 0.35)
    big = np.asarray(m_big, np.float32) / 255
    bx, by = (W - m_big.width) // 2, 530
    gl = resize_f(blur_f(resize_f(big, m_big.width // 2, m_big.height // 2), 8), m_big.width, m_big.height)
    add_patch(frame, gl[..., None] * GOLD * 0.8 * fade, bx, by)
    over(frame, GOLD * 1.05, np.clip(big, 0, 1) * fade, bx, by)
    sweep = np.exp(-((np.arange(m_big.width, dtype=np.float32) - lerp(-250, m_big.width + 250, (t - 0.7) / 2.2)) / 90) ** 2)
    add_patch(frame, (big * sweep[None, :])[..., None] * np.array([1.0, 0.97, 0.85], np.float32) * 0.9 * fade, bx, by)
    # descriptor: bigger, brighter, with a soft dark backing so it stays crisp
    m_small = text_mask("AUTONOMOUS  GENERALIST  SCIENTIST", _S["f_small"], 0.26)
    sm = np.asarray(m_small, np.float32) / 255
    sx, sy = (W - m_small.width) // 2, 752
    smg = resize_f(blur_f(resize_f(sm, m_small.width // 2, m_small.height // 2), 5), m_small.width, m_small.height)
    add_patch(frame, smg[..., None] * np.array([1.0, 0.85, 0.5], np.float32) * 0.35 * fade, sx, sy)
    over(frame, np.array([1.0, 0.97, 0.9], np.float32), np.clip(sm * 1.15, 0, 1) * fade, sx, sy)
    # thin gold rule under the wordmark
    rule_w = int(m_small.width * 1.04 * smooth((t - 0.5) / 1.0))
    if rule_w > 4:
        add_patch(frame, np.ones((3, rule_w, 3), np.float32) * GOLD * 0.55 * fade, (W - rule_w) // 2, 712)
    return frame
