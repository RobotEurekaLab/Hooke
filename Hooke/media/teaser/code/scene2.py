"""Scene 2 - THE QUOTA. Macro: quill signing the Curator's contract by candlelight."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from common import *

TW, TH = 2400, 1500
_S = {}
SIG_BOX = None


def _build():
    g = rng(21)
    n1 = fbm(TW, TH, 22, base=5, octaves=6)
    n2 = fbm(TW, TH, 23, base=30, octaves=3)
    fib = resize_f(g.random((TH // 2, TW // 12), dtype=np.float32), TW, TH, Image.BICUBIC)
    base = np.array([0.80, 0.67, 0.46], np.float32)
    col = base * (0.78 + 0.25 * n1 + 0.08 * fib)[..., None]
    stain = np.clip((n2 - 0.62) * 4, 0, 1)
    col *= (1 - 0.25 * stain)[..., None]
    col *= np.array([1.0, 0.96, 0.9], np.float32) ** (stain[..., None] * 3)
    # darkened, burnt edges
    xs, ys = grid(TW, TH, 1)
    e = np.minimum.reduce([xs, ys, TW - xs, TH - ys]) / 160.0 + (n1 - 0.5) * 0.6
    col *= (0.35 + 0.65 * smooth(e))[..., None]

    ink = Image.new("L", (TW, TH), 0)
    d = ImageDraw.Draw(ink)
    f_title = ImageFont.truetype(FONTS + "timesbi.ttf", 96)
    f_body = ImageFont.truetype(FONTS + "georgiai.ttf", 50)
    f_caps = ImageFont.truetype(FONTS + "timesbd.ttf", 54)
    f_sig = ImageFont.truetype(FONTS + "timesbi.ttf", 190)
    f_small = ImageFont.truetype(FONTS + "georgiai.ttf", 44)
    d.text((620, 120), "Articles of Agreement", font=f_title, fill=235)
    d.line([(620, 245), (1780, 245)], fill=160, width=3)
    lines = [
        "Whereas the ROYAL SOCIETY of London, for the improving",
        "of Natural Knowledge, doth appoint Mr. ROBERT HOOKE",
        "to the Office of  CURATOR OF EXPERIMENTS,",
        "it is Ordered that he shall, at every Weekly Meeting,",
        "furnish the Society with three or four considerable",
        "Experiments of his own devising, entirely new;",
        "and shall prosecute such others as the Society direct.",
    ]
    y = 300
    for ln in lines:
        d.text((330, y), ln, font=f_body, fill=215)
        y += 82
    d.text((330, 900), "Given this 12th day of November, Anno Domini 1662.", font=f_small, fill=200)
    d.line([(720, 1210), (1750, 1210)], fill=170, width=3)
    # margin sketch: a pump, a lens, ray lines
    d.ellipse([90, 420, 250, 580], outline=150, width=3)
    d.line([(170, 580), (170, 800)], fill=150, width=3)
    d.rectangle([130, 800, 210, 880], outline=150, width=3)
    for k in range(5):
        d.line([(40, 1000 + k * 20), (280, 1080 - k * 12)], fill=110, width=2)
    ink = ink.filter(ImageFilter.GaussianBlur(0.9))

    sig = Image.new("L", (TW, TH), 0)
    ImageDraw.Draw(sig).text((760, 990), "Robt. Hooke", font=f_sig, fill=255)
    sig = sig.filter(ImageFilter.GaussianBlur(0.8))
    sig_arr = np.asarray(sig, np.float32) / 255
    cols = np.where(sig_arr.max(0) > 0.3)[0]
    sx0, sx1 = int(cols[0]), int(cols[-1])
    # per-column ink centroid -> quill tip path
    ysg = np.arange(TH, dtype=np.float32)[:, None]
    wsum = sig_arr.sum(0) + 1e-3
    cy = (sig_arr * ysg).sum(0) / wsum
    cy[sig_arr.sum(0) < 0.5] = np.nan
    idx = np.arange(TW)
    good = ~np.isnan(cy)
    cy = np.interp(idx, idx[good], cy[good]).astype(np.float32)
    ink_arr = np.asarray(ink, np.float32) / 255
    inkcol = np.array([0.10, 0.07, 0.05], np.float32)
    col = col * (1 - 0.85 * ink_arr[..., None]) + inkcol * 0.85 * ink_arr[..., None]
    # wax seal
    sd = ImageDraw.Draw(seal_img := Image.new("RGBA", (TW, TH), (0, 0, 0, 0)))
    sx, sy = 2080, 1320
    for r, c in [(120, (70, 8, 8, 255)), (105, (130, 20, 18, 255)), (80, (95, 12, 12, 255)), (60, (150, 30, 25, 255))]:
        sd.ellipse([sx - r, sy - r, sx + r, sy + r], fill=c)
    sd.text((sx - 38, sy - 50), "RS", font=f_caps, fill=(70, 8, 8, 255))
    seal = np.asarray(seal_img.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    col = col * (1 - seal[..., 3:]) + seal[..., :3] * seal[..., 3:]
    alpha = np.clip(e * 3, 0, 1)
    _S.update(paper=col.astype(np.float32), sig=sig_arr, sx0=sx0, sx1=sx1, tip_y=cy, alpha=alpha)

    # background above the paper: dark room, candle bokeh
    bg = np.zeros((H, W, 3), np.float32)
    bg += np.linspace(0.05, 0.01, H, dtype=np.float32)[:, None, None] * np.array([0.6, 0.45, 0.35])
    _S["bg"] = bg
    _S["bokeh"] = [(160, 250, 110, 1.0), (520, 190, 70, 0.7), (1450, 230, 95, 0.9), (1780, 300, 130, 0.8),
                   (980, 170, 50, 0.5)]
    _S["text"] = [Text3D("CURATOR OF", make_font("timesbd.ttf", 190), "gold", tracking=0.12,
                         side=(0.3, 0.17, 0.06), side_back=(0.04, 0.02, 0.01), seed=4),
                  Text3D("EXPERIMENTS.", make_font("timesbd.ttf", 190), "gold", tracking=0.12,
                         side=(0.3, 0.17, 0.06), side_back=(0.04, 0.02, 0.01), seed=5)]


def _quad(t, dur):
    u = t / dur
    dx = -60 * u
    sc = 1.0 + 0.06 * u
    cx, cy = 960, 740
    base = [(-150, 260), (2070, 260), (3000, 1060), (-1080, 1060)]
    return [(cx + (x - cx) * sc + dx, cy + (y - cy) * sc) for x, y in base]


def _fwd(quad):
    return persp_coeffs(quad, [(0, 0), (TW, 0), (TW, TH), (0, TH)])


def _apply(co, x, y):
    a, b, c, d, e, f, g, h = co
    den = g * x + h * y + 1
    return (a * x + b * y + c) / den, (d * x + e * y + f) / den


def _bokeh_disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + yy ** 2) / r
    return (smooth((1 - d) * 6) * (0.75 + 0.25 * d ** 4)).astype(np.float32)


def render(t, dur=2.0):
    if not _S:
        _build()
    frame = _S["bg"].copy()
    for i, (bx, by, r, s) in enumerate(_S["bokeh"]):
        fl = 1 + 0.12 * math.sin(t * (9 + i) + i)
        disc = _bokeh_disc(r)
        add_patch(frame, disc[..., None] * np.array([1.0, 0.55, 0.22]) * 0.45 * s * fl, bx - r, by - r)

    # signature reveal
    p0, p1 = 0.1, 1.35
    prog = smooth((t - p0) / (p1 - p0)) if t < p1 else 1.0
    sx0, sx1 = _S["sx0"], _S["sx1"]
    rx = sx0 + (sx1 - sx0 + 40) * prog
    paper = _S["paper"]
    xs = np.arange(sx0 - 20, sx1 + 60)
    reveal = np.clip((rx - xs) / 10.0, 0, 1).astype(np.float32)
    y0, y1 = 950, 1250
    sl = _S["sig"][y0:y1, sx0 - 20:sx1 + 60] * reveal[None, :]
    wet = np.exp(-((xs - rx) / 90.0) ** 2).astype(np.float32)[None, :]
    region = paper[y0:y1, sx0 - 20:sx1 + 60].copy()
    inkc = np.array([0.05, 0.035, 0.03], np.float32)
    region = region * (1 - 0.95 * sl[..., None]) + inkc * 0.95 * sl[..., None] + (sl * wet)[..., None] * 0.18
    tex = paper.copy()
    tex[y0:y1, sx0 - 20:sx1 + 60] = region
    rgba = np.concatenate([clamp01(tex), _S["alpha"][..., None]], -1)
    img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    quad = _quad(t, dur)
    co_in = persp_coeffs([(0, 0), (TW, 0), (TW, TH), (0, TH)], quad)
    warped = np.asarray(img.transform((W, H), Image.PERSPECTIVE, co_in, Image.BICUBIC), np.float32) / 255

    # tip position on screen
    tip_tx = min(rx, sx1)
    tip_ty = _S["tip_y"][int(tip_tx)] + 18 * math.sin(t * 23) * (0 < prog < 1)
    co_fwd = _fwd(quad)
    tx, ty = _apply(co_fwd, tip_tx, tip_ty)
    # lighting pool on the paper around the signature
    pool = radial(tx + 60, ty - 40, 520, power=2.4)
    candle = radial(200, 150, 900, power=1.5)
    light = 0.08 + 1.35 * pool + 0.35 * candle
    lit = warped[..., :3] * (light[..., None] * np.array([1.0, 0.82, 0.62], np.float32))
    a = warped[..., 3:]
    frame = frame * (1 - a) + lit * a

    # ---- quill (hand held), drawn in screen space
    moving = 0 < prog < 1
    lift = 0 if moving else 60 * smooth((t - p1) / 0.8) + 60 * (1 - smooth((t - 1.2) / 1.0))
    tx2, ty2 = tx, ty - lift
    dirx, diry = 0.47, -0.88
    L = 900
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("L", (W, H), 0)
    dq = ImageDraw.Draw(layer)
    ds = ImageDraw.Draw(shadow)

    def pt(u, off=0.0):
        return (tx2 + dirx * L * u - diry * off, ty2 + diry * L * u + dirx * off)

    vane_l = [pt(u, -95 * math.sin(math.pi * (u - 0.3) / 0.7) ** 0.7) for u in np.linspace(0.3, 1.0, 20)]
    vane_r = [pt(u, 55 * math.sin(math.pi * (u - 0.3) / 0.7) ** 0.7) for u in np.linspace(1.0, 0.3, 20)]
    poly = vane_l + vane_r
    ds.polygon([(x - 45, y + 60) for x, y in poly], fill=110)
    ds.line([(x - 45, y + 60) for x, y in [pt(0.02), pt(1.0)]], fill=120, width=8)
    dq.polygon(poly, fill=(215, 205, 185, 255))
    for u in np.linspace(0.32, 0.98, 40):
        a0 = pt(u)
        dq.line([a0, pt(u + 0.05, -60 * math.sin(math.pi * (u - 0.3) / 0.7) ** 0.7)], fill=(150, 140, 125, 255), width=2)
    dq.line([pt(0.0), pt(1.02)], fill=(240, 232, 215, 255), width=7)
    dq.polygon([pt(0.0), pt(0.08, 7), pt(0.08, -7)], fill=(30, 25, 20, 255))
    sh = np.asarray(shadow.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255 * 0.5
    frame *= (1 - sh[..., None] * (a > 0.5))
    q = np.asarray(layer.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    qlight = (0.25 + 1.1 * radial(tx + 60, ty - 40, 700, power=2))[..., None] * np.array([1.0, 0.85, 0.66])
    frame = frame * (1 - q[..., 3:]) + q[..., :3] * qlight * q[..., 3:]

    # depth of field: focus at the writing line
    blurred = fast_blur(frame, 14)
    ys = np.arange(H, dtype=np.float32)
    m = smooth((np.abs(ys - ty) - 140) / 380)[:, None, None]
    frame = frame * (1 - m) + blurred * m

    # ---- 3D title: CURATOR OF EXPERIMENTS.
    for i, tx in enumerate(_S["text"]):
        ts = 0.1 + i * 0.22
        ai = ease_out((t - ts) / 0.7)
        alpha = smooth((t - ts) / 0.22) * (1 - smooth((t - (dur - 0.35)) / 0.35))
        if alpha <= 0.01:
            continue
        tx.draw(frame, pos=(-30, -250 + i * 250, lerp(760, 1150, ai)), rx=lerp(-0.5, -0.12, ai), ry=0.1,
                rz=-0.015, scale=0.82, depth=55, layers=13, alpha=float(alpha),
                sweep=lerp(-0.3, 1.4, (t - ts - 0.2) / 1.1), light=0.88,
                blur=max(0.0, 8 * (1 - ai)), shadow=(-25, 90, 26, 0.55))
    return frame
