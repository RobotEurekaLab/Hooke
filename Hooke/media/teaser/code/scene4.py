"""Scene 4 - CELL. Light on Hooke's microscope, push through the lens into cork cells."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import *

BW, BH = 2112, 1188
CW, CH = 2600, 1500
_S = {}
LENS = (1085, 705)  # objective tip in base coords


def cylinder_patch(length, r0, r1, base_col, bands=(), leather=False, seed=0):
    """Horizontal tapered cylinder with cylindrical shading. Returns float RGBA."""
    hmax = int(2 * max(r0, r1)) + 4
    xs = np.arange(length, dtype=np.float32)[None, :]
    ys = np.arange(hmax, dtype=np.float32)[:, None] - hmax / 2
    rad = r0 + (r1 - r0) * xs / length
    v = ys / rad
    alpha = np.clip((rad - np.abs(ys)) * 1.0, 0, 1)
    prof = np.sqrt(np.clip(1 - v ** 2, 0, 1))
    shade = 0.18 + 0.7 * prof * np.clip(0.8 - 0.5 * v, 0, 1.2) + 0.9 * np.exp(-((v + 0.5) / 0.12) ** 2)
    col = np.array(base_col, np.float32)[None, None, :] * shade[..., None]
    if leather:
        n = fbm(length, hmax, seed, base=10, octaves=4)
        col *= (0.7 + 0.5 * n)[..., None]
    for (b0, b1, bc) in bands:
        m = ((xs >= b0) & (xs < b1))[..., None]
        col = np.where(m, np.array(bc, np.float32) * shade[..., None] * 1.2, col)
    return np.concatenate([col, alpha[..., None]], -1).astype(np.float32)


def paste(base_img, patch, cx, cy, angle_deg):
    img = Image.fromarray((clamp01(patch) * 255).astype(np.uint8), "RGBA")
    img = img.rotate(angle_deg, Image.BICUBIC, expand=True)
    base_img.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


def voronoi_cells(w, h, sx, sy, seed):
    g = rng(seed)
    gw, gh = w // sx + 3, h // sy + 3
    jit = g.random((gh, gw, 2), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = (xs // sx).astype(int), (ys // sy).astype(int)
    d1 = np.full((h, w), 1e9, np.float32)
    d2 = np.full((h, w), 1e9, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = np.clip(gx + ox, 0, gw - 1)
            cy = np.clip(gy + oy, 0, gh - 1)
            px = (cx + 0.5 * (cy % 2) + 0.38 + 0.24 * jit[cy, cx, 0]) * sx
            py = (cy + 0.4 + 0.2 * jit[cy, cx, 1]) * sy
            d = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            d1 = np.where(closer, d, d1)
    return d1, d2


def _build():
    # ---------------- room + microscope
    g = rng(51)
    wall = fbm(BW, BH, 52, base=4, octaves=5)
    base = np.zeros((BH, BW, 3), np.float32) + np.array([0.12, 0.09, 0.07]) * (0.6 + 0.6 * wall)[..., None]
    tab_y = 880
    tg = resize_f(g.random((BH - tab_y, 50), dtype=np.float32), BW, BH - tab_y, Image.BICUBIC)
    base[tab_y:] = np.array([0.24, 0.13, 0.06]) * (0.5 + 0.6 * tg[..., None])
    img = to_img(base).convert("RGBA")
    d = ImageDraw.Draw(img)
    # window frame in far wall (top right)
    for k in range(3):
        d.rectangle([1500 + k * 170, 60, 1640 + k * 170, 420], fill=(40, 36, 34, 255))
    # turned wooden foot
    d.ellipse([850, 870, 1150, 930], fill=(60, 35, 18, 255))
    d.rectangle([850, 840, 1150, 900], fill=(80, 48, 24, 255))
    d.ellipse([850, 810, 1150, 870], fill=(110, 70, 35, 255))
    img = img
    brass = (0.85, 0.62, 0.3)
    pillar = cylinder_patch(520, 16, 16, brass)
    paste(img, pillar, 1000, 580, 90)
    # specimen arm + stage
    arm = cylinder_patch(150, 9, 9, brass)
    paste(img, arm, 1060, 780, 0)
    d = ImageDraw.Draw(img)
    d.ellipse([1095, 760, 1165, 790], fill=(150, 110, 60, 255))
    d.ellipse([1110, 758, 1150, 776], fill=(190, 150, 90, 255))
    # tube: angled body, tooled leather + gilt bands
    ang = math.degrees(math.atan2(-(LENS[1] - 190), LENS[0] - 870))  # pointing from eyepiece to lens
    body = cylinder_patch(560, 44, 62, (0.18, 0.1, 0.06), bands=[(0, 40, (0.9, 0.66, 0.3)), (180, 196, (0.9, 0.66, 0.3)),
                                                                  (360, 376, (0.9, 0.66, 0.3)), (520, 560, (0.9, 0.66, 0.3))],
                          leather=True, seed=53)
    mid = ((870 + LENS[0]) / 2, (190 + LENS[1]) / 2)
    # nose cone + eyepiece
    L_total = math.hypot(LENS[0] - 870, LENS[1] - 190)
    ux, uy = (LENS[0] - 870) / L_total, (LENS[1] - 190) / L_total
    paste(img, cylinder_patch(120, 26, 70, brass), 870 + ux * 20, 190 + uy * 20, ang + 180)
    paste(img, body, 870 + ux * 330, 190 + uy * 330, ang)
    paste(img, cylinder_patch(100, 60, 16, brass), LENS[0] - ux * 50, LENS[1] - uy * 50, ang)
    # clamp from pillar to tube
    paste(img, cylinder_patch(150, 12, 12, brass), 940, 390, 8)
    d = ImageDraw.Draw(img)
    d.ellipse([985, 370, 1025, 410], fill=(200, 150, 70, 255))
    # condenser globe + oil lamp
    gx, gy, gr = 1500, 600, 105
    paste(img, cylinder_patch(330, 7, 7, brass), gx, 740, 90)
    d = ImageDraw.Draw(img)
    d.ellipse([gx - 80, 890, gx + 80, 915], fill=(120, 85, 40, 255))
    d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(70, 80, 85, 255))
    d.ellipse([gx - gr + 10, gy - gr + 10, gx + gr - 10, gy + gr - 10], fill=(40, 50, 58, 255))
    d.ellipse([gx - 60, gy - 75, gx - 20, gy - 35], fill=(210, 220, 225, 255))
    d.rectangle([1700, 800, 1760, 900], fill=(90, 60, 30, 255))
    d.ellipse([1690, 780, 1770, 815], fill=(130, 95, 45, 255))
    alb = to_arr(img.convert("RGB"))
    q = 4
    sh = shaft(1750 / q, -250 / q, math.radians(126), 70 / q, 0.0009 * q, 1500 / q, BW // q, BH // q, k=1)
    lamp = radial(1730 / q, 760 / q, 50, BW // q, BH // q, k=1, power=2)
    glob = radial(gx / q, gy / q, 40, BW // q, BH // q, k=1, power=2)
    _S.update(alb=alb, shaft=sh, lamp=lamp, glob=glob, haze=fbm(BW // q, BH // q, 54, base=6, octaves=4),
              motes=rng(55).random((220, 5)))

    # ---------------- cork cells (Hooke's Micrographia, Observ. XVIII)
    d1, d2 = voronoi_cells(CW, CH, 120, 78, 56)
    wall_w = d2 - d1
    walls = np.exp(-(wall_w / 7.0) ** 2)
    interior = np.clip(d1 / 60.0, 0, 1)
    n = fbm(CW, CH, 57, base=6, octaves=5)
    back = radial(CW * 0.55, CH * 0.45, 900, CW, CH, k=4, power=1.6)
    cell = (np.array([0.35, 0.16, 0.05]) * (1 - interior)[..., None] * (0.6 + 0.6 * n)[..., None]
            + np.array([0.05, 0.02, 0.01]) * interior[..., None])
    cell += walls[..., None] * np.array([1.0, 0.78, 0.45]) * (0.55 + 0.6 * n)[..., None]
    # second, deeper layer showing through
    e1, e2 = voronoi_cells(CW // 2, CH // 2, 60, 39, 58)
    deep = resize_f(np.exp(-((e2 - e1) / 4.0) ** 2).astype(np.float32), CW, CH)
    cell += (deep * (interior > 0.3))[..., None] * np.array([0.5, 0.3, 0.12]) * 0.12
    cell *= (0.35 + 1.2 * back)[..., None]
    _S["cells"] = cell.astype(np.float32)
    _S["text"] = Text3D("CELL.", make_font("timesbd.ttf", 400), "amber", tracking=0.12,
                        side=(0.45, 0.22, 0.06), side_back=(0.08, 0.03, 0.01), seed=9)


def cells_view(t, blur_r):
    u = t
    scale = 1.25 + 0.04 * u
    v = zoom_crop(_S["cells"], scale, 0.5 + 0.01 * u, 0.5, rot=0.02 * u)
    if blur_r > 0.5:
        v = fast_blur(v, blur_r)
    return v


def render(t, dur=2.4):
    if not _S:
        _build()
    q = 4
    T_ZOOM, T_CELLS = 0.9, 1.45
    if t < T_CELLS:
        amb = np.array([0.2, 0.25, 0.35], np.float32) * 0.2
        L = np.zeros((BH // q, BW // q, 3), np.float32) + amb
        sh_col = np.array([1.0, 0.9, 0.75], np.float32)
        L += _S["shaft"][..., None] * sh_col * 3.6
        fl = 1 + 0.08 * math.sin(t * 13) + 0.05 * math.sin(t * 29)
        L += _S["lamp"][..., None] * np.array([1.0, 0.6, 0.25]) * 2.0 * fl
        L += _S["glob"][..., None] * np.array([1.0, 0.75, 0.4]) * 1.0 * fl
        lit = _S["alb"] * resize_f(L, BW, BH)
        vol = _S["shaft"] * (0.4 + 0.6 * np.roll(_S["haze"], int(t * 5), 1))
        lit += resize_f(vol, BW, BH)[..., None] * sh_col * 0.45
        lit += resize_f(_S["lamp"], BW, BH)[..., None] * np.array([1.0, 0.6, 0.25]) * 0.5 * fl
        # focused beam from globe to specimen
        cx, cy = LENS[0] / BW, LENS[1] / BH
        if t < T_ZOOM:
            s = 1.1 + 0.45 * ease_in_out(t / T_ZOOM)
            frame = zoom_crop(lit, s, lerp(0.53, cx, ease_in_out(t / T_ZOOM)), lerp(0.4, cy, ease_in_out(t / T_ZOOM)))
        else:
            k = (t - T_ZOOM) / (T_CELLS - T_ZOOM)
            s = 1.55 * math.exp(k * k * 3.2)
            acc = None
            for j in range(4):
                fr = zoom_crop(lit, s * (1 - 0.04 * j * k), cx, cy)
                acc = fr if acc is None else acc + fr
            frame = acc / 4
            # the lens opening swallows the frame
            r = 30 * math.exp(k * 6.2)
            xs, ys = grid(W, H, 2)
            dist = np.sqrt((xs - W / 2) ** 2 + (ys - H / 2) ** 2)
            m = resize_f(smooth((r - dist) / (r * 0.08 + 2)).astype(np.float32), W, H)
            ring = resize_f(np.exp(-((dist - r) / (r * 0.05 + 2)) ** 2).astype(np.float32), W, H)
            inner = cells_view(0, 18) * 0.8
            frame = frame * (1 - m[..., None]) + inner * m[..., None] + ring[..., None] * np.array([1.0, 0.8, 0.45]) * 0.8
            frame += (smooth((k - 0.85) / 0.15) * 0.45)
    else:
        k = t - T_CELLS
        blur_r = 16 * (1 - smooth(k / 0.35))
        frame = cells_view(k, blur_r)
        # eyepiece vignette opening up
        R = lerp(620, 1500, smooth(k / 0.6))
        xs, ys = grid(W, H, 2)
        dist = np.sqrt((xs - W / 2) ** 2 + (ys - H / 2) ** 2)
        vig = resize_f(smooth((R - dist) / 90).astype(np.float32), W, H)
        frame *= vig[..., None]
        frame += 0.45 * math.exp(-k * 6)
        # "CELL." rises out of the specimen as the word is spoken
        ts = 0.28
        if k > ts - 0.15:
            a = ease_out((k - ts) / 0.55)
            frame *= 1 - 0.35 * a
            _S["text"].draw(frame, pos=(0, -10, lerp(2000, 1020, a) + 40 * max(0, k - ts)), rx=lerp(0.9, 0.12, a),
                            ry=-0.05, scale=1.0, depth=90, layers=16, alpha=float(smooth((k - ts + 0.15) / 0.25)),
                            sweep=lerp(-0.3, 1.4, (k - ts - 0.1) / 0.8), light=1.05, blur=max(0.0, 10 * (1 - a)),
                            shadow=(30, 50, 26, 0.7), glow=0.35, glow_color=(1.0, 0.6, 0.2))
    return frame
