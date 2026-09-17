"""Shared rendering helpers: noise, blur, bloom, light, 3D extruded typography, post."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
FPS = 24
FONTS = "C:/Windows/Fonts/"

_rng_cache = {}


def rng(seed):
    return np.random.default_rng(seed)


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------- conversions
def to_img(arr):
    return Image.fromarray((clamp01(arr) * 255 + 0.5).astype(np.uint8))


def to_arr(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_f(arr, w, h, resample=Image.BILINEAR):
    """Resize a float array (HxW or HxWxC) without quantising."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize((w, h), resample))
    return np.stack([resize_f(arr[..., c], w, h, resample) for c in range(arr.shape[2])], -1)


def box_blur(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur_f(a, r):
    """Approx gaussian blur (3 box passes) for float arrays."""
    r = int(r)
    if r < 1:
        return a
    rr = max(1, int(r / 1.7))
    for _ in range(3):
        a = box_blur(a, rr, 0)
        a = box_blur(a, rr, 1)
    return a


def fast_blur(a, r):
    """Blur a full-res float image by working at reduced resolution."""
    h, w = a.shape[:2]
    if r < 3:
        return blur_f(a, r)
    k = 4 if r >= 12 else 2
    small = resize_f(a, w // k, h // k)
    small = blur_f(small, r / k)
    return resize_f(small, w, h)


def fbm(w, h, seed, base=8, octaves=6, persistence=0.55):
    g = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cw, ch = max(2, int(base * 2 ** o)), max(2, int(base * 2 ** o * h / w))
        n = g.random((ch, cw), dtype=np.float32)
        out += amp * resize_f(n, w, h, Image.BICUBIC)
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


_grid_cache = {}


def grid(w=W, h=H, k=1):
    key = (w, h, k)
    if key not in _grid_cache:
        ys, xs = np.mgrid[0:h:k, 0:w:k].astype(np.float32)
        _grid_cache[key] = (xs, ys)
    return _grid_cache[key]


def radial(cx, cy, radius, w=W, h=H, k=4, power=2.0):
    """Soft radial falloff computed at 1/k res and upscaled."""
    xs, ys = grid(w, h, k)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / radius
    m = 1.0 / (1.0 + d ** power)
    return resize_f(m, w, h)


def shaft(ox, oy, angle, width, spread, length, w=W, h=H, k=4):
    xs, ys = grid(w, h, k)
    dx, dy = math.cos(angle), math.sin(angle)
    along = (xs - ox) * dx + (ys - oy) * dy
    perp = np.abs((xs - ox) * dy - (ys - oy) * dx)
    wid = width * (1 + np.maximum(along, 0) * spread)
    m = np.exp(-(perp / wid) ** 2) * smooth(along / 80.0) * np.exp(-np.maximum(along, 0) / length)
    return resize_f(m.astype(np.float32), w, h)


def glow_layer(img_l_or_rgb, radius):
    """PIL image -> blurred float array (for glows)."""
    small = img_l_or_rgb.resize((img_l_or_rgb.width // 4, img_l_or_rgb.height // 4), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / 4))
    return to_arr(small.resize(img_l_or_rgb.size, Image.BILINEAR))


def over(canvas, rgb, alpha, x0=0, y0=0):
    """Alpha-composite patch (h,w,3)/(h,w) onto canvas at (x0,y0) in place, with clipping."""
    h, w = alpha.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas
    a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None]
    c = rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] if rgb.ndim == 3 else rgb
    region = canvas[cy0:cy1, cx0:cx1]
    region *= (1 - a)
    region += c * a
    return canvas


def add_patch(canvas, rgb, x0=0, y0=0):
    h, w = rgb.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x0 + w), min(canvas.shape[0], y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas
    canvas[cy0:cy1, cx0:cx1] += rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return canvas


def zoom_crop(arr, scale, cx=0.5, cy=0.5, rot=0.0):
    """Camera push: crop centre region of arr (any size) and resize to W x H."""
    h, w = arr.shape[:2]
    img = to_img(arr)
    cw, ch = w / scale, h / scale
    x0 = min(max(cx * w - cw / 2, 0), w - cw)
    y0 = min(max(cy * h - ch / 2, 0), h - ch)
    if rot:
        img = img.rotate(math.degrees(rot), Image.BILINEAR, center=(cx * w, cy * h))
    img = img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
    return to_arr(img)


# ---------------------------------------------------------------- perspective
def persp_coeffs(src, dst):
    """Coefficients for PIL PERSPECTIVE mapping output pts (dst) -> input pts (src)."""
    A, B = [], []
    for (x, y), (X, Y) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        B.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        B.append(Y)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def warp_to_quad(img, quad, pad=2):
    """Warp img onto the screen quad (tl,tr,br,bl). Returns (patch_img, x0, y0) or None."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = int(math.floor(min(xs))) - pad, int(math.floor(min(ys))) - pad
    x1, y1 = int(math.ceil(max(xs))) + pad, int(math.ceil(max(ys))) + pad
    # clip to screen generously
    x0c, y0c = max(x0, -200), max(y0, -200)
    x1c, y1c = min(x1, W + 200), min(y1, H + 200)
    if x1c - x0c < 2 or y1c - y0c < 2:
        return None
    w, h = img.size
    local = [(p[0] - x0c, p[1] - y0c) for p in quad]
    co = persp_coeffs([(0, 0), (w, 0), (w, h), (0, h)], local)
    out = img.transform((x1c - x0c, y1c - y0c), Image.PERSPECTIVE, co, Image.BILINEAR)
    return out, x0c, y0c


# ---------------------------------------------------------------- 3D typography
def make_font(name, size, variation=None):
    f = ImageFont.truetype(FONTS + name, size)
    if variation:
        f.set_variation_by_name(variation)
    return f


def text_mask(text, font, tracking=0.0, pad=40):
    """Render text with letter tracking into an L image."""
    size = font.size
    widths = []
    for ch in text:
        bb = font.getbbox(ch)
        widths.append(font.getlength(ch))
    total = sum(widths) + tracking * size * (len(text) - 1)
    asc, desc = font.getmetrics()
    img = Image.new("L", (int(total) + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for ch, wd in zip(text, widths):
        d.text((x, pad), ch, font=font, fill=255)
        x += wd + tracking * size
    bb = img.getbbox()
    return img.crop((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))


def face_texture(mask, style, seed=1):
    """RGB float texture for a text face."""
    w, h = mask.size
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    n = fbm(w, h, seed, base=6, octaves=5)
    brushed = resize_f(rng(seed + 5).random((h, max(4, w // 60)), dtype=np.float32), w, h, Image.BICUBIC)

    def stops(t, pts):
        t = np.clip(t, 0, 1)
        out = np.zeros(t.shape + (3,), np.float32)
        for i in range(len(pts) - 1):
            (p0, c0), (p1, c1) = pts[i], pts[i + 1]
            m = ((t >= p0) & (t <= p1))[..., None]
            u = ((t - p0) / (p1 - p0 + 1e-6))[..., None]
            out = np.where(m, np.array(c0) + (np.array(c1) - np.array(c0)) * u, out)
        return out

    if style == "gold":
        tex = stops(np.broadcast_to(ys, (h, w)) + 0.08 * (n - 0.5), [
            (0.0, (1.0, 0.86, 0.52)), (0.35, (0.82, 0.58, 0.24)), (0.52, (0.42, 0.25, 0.08)),
            (0.62, (0.72, 0.5, 0.2)), (1.0, (0.98, 0.8, 0.45))])
        tex *= (0.85 + 0.3 * brushed[..., None])
    elif style == "amber":
        tex = stops(np.broadcast_to(ys, (h, w)) + 0.1 * (n - 0.5), [
            (0.0, (1.0, 0.92, 0.7)), (0.5, (0.95, 0.62, 0.25)), (1.0, (0.6, 0.3, 0.08))])
    elif style == "chrome":
        tex = stops(np.broadcast_to(ys, (h, w)) + 0.04 * (n - 0.5), [
            (0.0, (0.92, 0.97, 1.0)), (0.42, (0.55, 0.65, 0.75)), (0.5, (0.08, 0.12, 0.18)),
            (0.58, (0.1, 0.35, 0.45)), (1.0, (0.5, 0.95, 1.0))])
        tex *= (0.9 + 0.2 * brushed[..., None])
    elif style == "neon":
        tex = stops(np.broadcast_to(ys, (h, w)), [
            (0.0, (0.85, 1.0, 1.0)), (0.5, (0.3, 0.95, 1.0)), (1.0, (0.1, 0.5, 1.0))])
    else:
        tex = np.ones((h, w, 3), np.float32)
    # bevel: bright rim at glyph edges
    m = np.asarray(mask, np.float32) / 255
    er = np.asarray(mask.filter(ImageFilter.MinFilter(5)), np.float32) / 255
    rim = np.clip(m - er, 0, 1)[..., None]
    tex = tex * (1 - 0.6 * rim) + rim * np.array([1.0, 0.97, 0.9], np.float32) * 0.9
    return tex.astype(np.float32)


class Text3D:
    def __init__(self, text, font, style="gold", tracking=0.08, side=(0.25, 0.14, 0.05),
                 side_back=(0.04, 0.02, 0.01), seed=1):
        self.mask = text_mask(text, font, tracking)
        self.tex = face_texture(self.mask, style, seed)
        self.style = style
        self.side = np.array(side, np.float32)
        self.side_back = np.array(side_back, np.float32)
        self.w, self.h = self.mask.size
        self.mask_arr = np.asarray(self.mask, np.float32) / 255

    def _project(self, pts, cx, cy, f):
        return [(cx + f * x / z, cy + f * y / z) for x, y, z in pts]

    def corners(self, pos, rx, ry, rz, scale, dz_local=0.0):
        w, h = self.w * scale, self.h * scale
        loc = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        out = []
        for x, y in loc:
            z = dz_local * scale
            # rotate z
            x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
            # rotate x
            y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
            # rotate y
            x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
            out.append((x + pos[0], y + pos[1], z + pos[2]))
        return out

    def draw(self, canvas, pos=(0, 0, 1000), rx=0.0, ry=0.0, rz=0.0, scale=1.0, depth=60,
             layers=14, alpha=1.0, sweep=None, emissive=0.0, light=1.0, blur=0.0,
             shadow=None, glow=0.0, glow_color=(0.3, 0.9, 1.0), f=1000.0, cx=W / 2, cy=H / 2,
             reflection=None):
        if alpha <= 0.002 or pos[2] <= 10:
            return canvas
        mask = self.mask
        if blur > 0.5:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        # soft contact shadow on the scene
        if shadow is not None:
            sx, sy, sblur, sop = shadow
            q = self._project(self.corners(pos, rx, ry, rz, scale), cx + sx, cy + sy, f)
            r = warp_to_quad(mask, q)
            if r:
                im, x0, y0 = r
                a = np.asarray(im.filter(ImageFilter.GaussianBlur(sblur)), np.float32) / 255 * sop * alpha
                over(canvas, np.zeros(a.shape + (3,), np.float32), a, x0, y0)
        # extrusion, back to front
        for k in range(layers, 0, -1):
            u = k / layers
            q = self._project(self.corners(pos, rx, ry, rz, scale, dz_local=depth * u), cx, cy, f)
            r = warp_to_quad(mask, q)
            if not r:
                continue
            im, x0, y0 = r
            a = np.asarray(im, np.float32) / 255 * alpha
            col = (self.side * (1 - u) + self.side_back * u) * light
            over(canvas, col, a, x0, y0)
        # face
        q = self._project(self.corners(pos, rx, ry, rz, scale), cx, cy, f)
        tex = self.tex
        if sweep is not None:
            xs = np.linspace(0, 1, self.w, dtype=np.float32)[None, :]
            ys = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
            band = np.exp(-((xs + ys * 0.35 - sweep) / 0.05) ** 2)
            tex = tex + band[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * 0.9
        rgba = np.concatenate([clamp01(tex * light), np.asarray(mask, np.float32)[..., None] / 255], -1)
        face_img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
        r = warp_to_quad(face_img, q)
        if r:
            im, x0, y0 = r
            fa = np.asarray(im, np.float32) / 255
            a = fa[..., 3] * alpha
            over(canvas, fa[..., :3], a, x0, y0)
            if glow > 0:
                gl = Image.fromarray((a * 255).astype(np.uint8), "L")
                pad = 80
                big = Image.new("L", (gl.width + 2 * pad, gl.height + 2 * pad))
                big.paste(gl, (pad, pad))
                g = np.asarray(big.resize((big.width // 4, big.height // 4)).filter(
                    ImageFilter.GaussianBlur(9)).resize(big.size, Image.BILINEAR), np.float32) / 255
                add_patch(canvas, g[..., None] * np.array(glow_color, np.float32) * glow, x0 - pad, y0 - pad)
        if reflection is not None:
            # mirrored, faded copy below the text (floor reflection)
            floor_y, op = reflection
            h_ = self.h * scale * f / pos[2]
            ql = self._project(self.corners(pos, rx, ry, rz, scale), cx, cy, f)
            base = max(ql[2][1], ql[3][1])
            mq = [(ql[3][0], base + 4), (ql[2][0], base + 4), (ql[1][0], base + 4 + h_ * 0.6),
                  (ql[0][0], base + 4 + h_ * 0.6)]
            r = warp_to_quad(face_img, mq)
            if r:
                im, x0, y0 = r
                fa = np.asarray(im.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
                fade = np.linspace(1, 0, fa.shape[0], dtype=np.float32)[:, None] ** 2
                over(canvas, fa[..., :3], fa[..., 3] * fade * op * alpha, x0, y0)
        return canvas


# ---------------------------------------------------------------- post
_grain = {}


def grain_frame(i):
    key = i % 6
    if key not in _grain:
        g = rng(900 + key).normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        _grain[key] = resize_f(g, W, H, Image.BILINEAR)
    return _grain[key]


_vig = None


def vignette():
    global _vig
    if _vig is None:
        xs, ys = grid(W, H, 4)
        d = ((xs - W / 2) / (W * 0.62)) ** 2 + ((ys - H / 2) / (H * 0.62)) ** 2
        _vig = resize_f((1 - 0.75 * smooth(d)).astype(np.float32), W, H)
    return _vig


def bloom(arr, thresh=0.72, strength=0.6, radius=18):
    small = arr[::4, ::4]
    b = np.maximum(small - thresh, 0)
    b = blur_f(b, radius / 4) + 0.5 * blur_f(b, radius / 1.5)
    return arr + resize_f(b, W, H) * strength


def post(arr, fi, part, grade=None, bloom_amt=0.6, grain_amt=None, letterbox=False, flicker=0.0,
         chroma=0.0, weave=True):
    t = fi / FPS
    if part == 1 and weave:
        dx = int(round(math.sin(fi * 1.7) * 0.8 + math.sin(fi * 0.31)))
        dy = int(round(math.sin(fi * 2.3) * 0.8))
        arr = np.roll(arr, (dy, dx), (0, 1))
    arr = bloom(arr, strength=bloom_amt)
    if chroma > 0:
        s = int(chroma)
        arr = arr.copy()
        arr[..., 0] = np.roll(arr[..., 0], s, 1)
        arr[..., 2] = np.roll(arr[..., 2], -s, 1)
    if grade is not None:
        lift, gamma, gain = grade
        arr = np.maximum(arr, 0) * np.array(gain, np.float32) + np.array(lift, np.float32)
        arr = np.power(np.maximum(arr, 0), 1.0 / np.array(gamma, np.float32))
    # filmic shoulder
    arr = 1 - np.exp(-arr * 1.35)
    arr = arr / (1 - math.exp(-1.35))
    arr *= vignette()[..., None]
    if flicker:
        arr *= 1 + flicker * (math.sin(t * 37) * 0.5 + math.sin(t * 23.3) * 0.5)
    ga = grain_amt if grain_amt is not None else (0.028 if part == 1 else 0.018)
    g = np.roll(grain_frame(fi), (fi * 131) % 400, 1)
    lum = arr.mean(-1, keepdims=True)
    arr = arr + g[..., None] * ga * (0.35 + 0.65 * (1 - np.abs(lum * 2 - 1)))
    if letterbox:
        bar = 138
        arr[:bar] = 0
        arr[H - bar:] = 0
    return clamp01(arr)


# ---------------------------------------------------------------- keyword overlays
class Keyword:
    """Flat motion-graphic keyword: wipes in, holds, drifts out. Used in the 2026 half."""

    def __init__(self, text, size=76, tracking=0.3, color=(0.88, 0.98, 1.0), accent=(0.25, 0.9, 1.0),
                 font="bahnschrift.ttf", variation="Bold"):
        self.mask = text_mask(text, make_font(font, size, variation), tracking, pad=12)
        self.arr = np.asarray(self.mask, np.float32) / 255
        self.color = np.array(color, np.float32)
        self.accent = np.array(accent, np.float32)
        h, w = self.arr.shape
        self.glow = resize_f(blur_f(resize_f(np.pad(self.arr, 24), w // 2 + 24, h // 2 + 24), 7), w + 48, h + 48)[24:24 + h, 24:24 + w]

    def draw(self, frame, t, t0, t1, cx=W / 2, cy=H / 2, fade=0.3, bar=True):
        a_in = smooth((t - t0) / 0.34)
        a_out = smooth((t - (t1 - fade)) / fade)
        alpha = float(a_in * (1 - a_out))
        if alpha <= 0.01:
            return frame
        h, w = self.arr.shape
        x0, y0 = int(cx - w / 2), int(cy - h / 2 + (1 - a_in) * 34 - a_out * 26)
        wipe = np.clip((a_in * (w + 160) - 80 - np.arange(w, dtype=np.float32)) / 90, 0, 1)[None, :]
        m = self.arr * wipe * alpha
        # dark plate so the words read over bright machinery
        over(frame, np.zeros(3, np.float32), np.clip(self.glow * 2.2, 0, 1) * wipe * 0.75 * alpha, x0, y0)
        add_patch(frame, self.glow[..., None] * wipe[..., None] * self.accent * 0.8 * alpha, x0, y0)
        over(frame, self.color * 1.15, np.clip(m, 0, 1), x0, y0)
        if bar:
            bw = int(w * a_in)
            by = y0 + h + 10
            if 0 <= by < H - 5 and bw > 4:
                strip = np.ones((5, bw, 3), np.float32) * self.accent * alpha
                add_patch(frame, strip, x0, by)
        return frame
