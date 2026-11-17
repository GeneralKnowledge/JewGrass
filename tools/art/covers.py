#!/usr/bin/env python3
"""Album covers for every act: square fractal artwork, made in code (numpy + Pillow, no GPU, no model).

    python3 tools/art/covers.py OUT_DIR [act ...]

Writes, for each act, the layout the take server reads from ~/band/artwork:
    OUT_DIR/<act>/album.jpg    1024 px cover with the act and album name
    OUT_DIR/<act>/artist.jpg   1024 px, the same artwork without lettering
    OUT_DIR/<act>/album-2048.png   full-size master
Every cover is deterministic: same code, same pixels. Same visual language as the lyric videos (glowing fractal
filaments on deep ink, soft bloom), but each act gets its own fractal family, framing and palette. All artwork and
names are original; nothing imitates a real artist, record or sleeve.
"""

import hashlib
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FONTS = os.path.join(REPO, "video", "assets", "fonts")
N = 2048  # master size; the 1024 covers are downsampled from it


# ---------------------------------------------------------------- helpers

def rng(seed, *salt):
    key = "|".join([str(seed)] + [str(s) for s in salt]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def font(name, size, variation=None):
    paths = {
        "oswald": os.path.join(FONTS, "Oswald-VF.ttf"),
        "baskerville": os.path.join(FONTS, "LibreBaskerville-VF.ttf"),
        "caveat": os.path.join(FONTS, "Caveat-VF.ttf"),
        "inter": "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf",
        "inter-thin": "/usr/share/fonts/opentype/inter/InterDisplay-ExtraLight.otf",
        "inter-light": "/usr/share/fonts/opentype/inter/InterDisplay-Light.otf",
        "mono": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
        "mono-light": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Light.ttf",
    }
    p = paths[name]
    if not os.path.exists(p):  # Inter / JetBrains Mono are system fonts; fall back to the repo's Oswald
        p = paths["oswald"]
    f = ImageFont.truetype(p, size)
    if variation:
        try:
            f.set_variation_by_name(variation)
        except Exception:
            pass
    return f


def lut(stops, n=1024):
    s = np.array(stops, np.float32) / 255.0
    xs = np.linspace(0, 1, len(s))
    q = np.linspace(0, 1, n)
    return np.stack([np.interp(q, xs, s[:, k]) for k in range(3)], 1).astype(np.float32)


def box_blur(a, r):
    """Separable box blur (edge-clamped); three passes approximate a Gaussian."""
    r = int(max(1, r))
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        p = np.pad(a, pad, mode="edge")
        c = np.cumsum(p, axis=ax, dtype=np.float64)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[ax]), axis=ax)
        lo = np.take(c, np.arange(0, c.shape[ax] - 2 * r - 1), axis=ax)
        a = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return a


def blur(a, r):
    for _ in range(3):
        a = box_blur(a, r)
    return a


def resize(a, size):
    """Float image resize through Pillow, channel by channel."""
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize(size, Image.BICUBIC))
    return np.stack([resize(a[..., k], size) for k in range(a.shape[2])], -1)


def grid(n=N, center=(0.0, 0.0), scale=1.5, rot=0.0, sx=1.0, sy=1.0):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    x = (x - n / 2) / (n / 2) * sx
    y = (y - n / 2) / (n / 2) * sy
    cr, sr = np.cos(rot), np.sin(rot)
    x, y = x * cr - y * sr, x * sr + y * cr
    return x * scale + center[0], y * scale + center[1]


def kaleido(x, y, k, offset=0.0):
    r = np.hypot(x, y)
    th = np.arctan2(y, x) + offset
    seg = 2 * np.pi / k
    th = np.abs((th % seg) - seg / 2)
    return r * np.cos(th), r * np.sin(th)


def escape(x, y, kind="julia", c=0j, iters=120, bail=4.0):
    """Smooth escape time, 0..1 (1 = never escaped), and the inside mask."""
    p = (x + 1j * y).astype(np.complex128).ravel()
    z = p.copy()
    cc = p.copy() if kind in ("mandel", "ship") else None
    out = np.zeros(z.size, np.float64)
    idx = np.arange(z.size)
    for n in range(iters):
        zz = z[idx]
        if kind == "ship":
            zz = (np.abs(zz.real) + 1j * np.abs(zz.imag)) ** 2 + cc[idx]
        elif kind == "mandel":
            zz = zz * zz + cc[idx]
        else:
            zz = zz * zz + c
        esc = np.abs(zz) > bail
        if esc.any():
            e = idx[esc]
            out[e] = n + 1 - np.log2(np.log(np.abs(zz[esc]) + 1e-12) + 1e-12)
        z[idx] = zz
        idx = idx[~esc]
        if not len(idx):
            break
    alive = np.zeros(z.size, bool)
    alive[idx] = True
    shape = x.shape
    return np.clip(out / iters, 0, 1).reshape(shape).astype(np.float32), alive.reshape(shape)


def glow(e, alive, pal, hue=0.0, cycles=2.2, gain=1.0, knee=0.4, ink=(0.03, 0.045, 0.05), bloom=0.6, gamma=0.85):
    """The video engine's look: dark far field, filaments glowing near the set's edge, deep ink inside, bloom."""
    v = (np.sqrt(e) * cycles + hue) % 1.0
    rgb = pal[(v * (len(pal) - 1)).astype(np.int32)]
    b = (0.07 + 0.93 * np.clip((e - 0.03) / knee, 0, 1) ** gamma) * gain
    b = np.where(alive, 0.0, b).astype(np.float32)
    lit = rgb * b[..., None]
    lit[alive] = np.array(ink, np.float32)
    n = lit.shape[0]
    small = resize(lit, (n // 4, n // 4))
    lit = lit + bloom * resize(blur(small, 3), (n, n))
    return lit


def vignette(img, strength=0.5, power=2.2):
    n = img.shape[0]
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - n / 2, y - n / 2) / (n / 2 * 1.414)
    return img * (1 - strength * r ** power)[..., None]


def scrim(img, top=0.0, bottom=0.0, h=0.42):
    """Darken the top and/or bottom edge so lettering reads."""
    n = img.shape[0]
    t = np.linspace(0, 1, n, dtype=np.float32)
    m = np.ones(n, np.float32)
    if bottom:
        m *= 1 - bottom * np.clip((t - (1 - h)) / h, 0, 1) ** 1.6
    if top:
        m *= 1 - top * np.clip((h - t) / h, 0, 1) ** 1.6
    return img * m[:, None, None]


def grain(img, seed, amount=0.02):
    g = rng(seed, "grain").normal(0, 1, img.shape[:2]).astype(np.float32)
    return img + amount * g[..., None]


def to_image(img):
    return Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.0) * 255 + 0.5).astype(np.uint8), "RGB")


def spaced(draw, xy, text, fnt, fill, tracking=0.0, anchor="l"):
    """Draw text with letter spacing (tracking in ems). anchor l/c/r on x; y is the baseline-ish top."""
    x, y = xy
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    extra = tracking * fnt.size
    total = sum(widths) + extra * (len(text) - 1)
    if anchor == "c":
        x -= total / 2
    elif anchor == "r":
        x -= total
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += w + extra
    return total


def fit(draw, text, name, max_w, start, tracking=0.0, variation=None):
    size = start
    while size > 10:
        f = font(name, size, variation)
        w = sum(draw.textlength(ch, font=f) for ch in text) + tracking * size * (len(text) - 1)
        if w <= max_w:
            return f
        size -= 4
    return font(name, size, variation)


# ---------------------------------------------------------------- the acts
# Each returns (artwork float RGB N x N, letter(draw) function).

HOUSE = [(24, 21, 22), (126, 70, 150), (40, 130, 140), (236, 228, 200), (246, 196, 104),
         (170, 60, 120), (60, 40, 90), (24, 21, 22)]  # the lyric videos' inks


def natural_capital():
    # the Deed to the Glen v3 pattern Reuben picked: 8-fold Julia from the main cardioid's edge, house inks
    a = rng(1707, "psych").uniform(0, 6.28) + 190.275 * 0.035 * 1.3
    w = np.exp(1j * a)
    c = complex((w / 2 - w * w / 4) * (1.0 + 0.02 * np.sin(190.275 * 0.07) + 0.03 * 0.8282))
    x, y = grid(scale=0.95 - 0.12 * 0.7474, rot=0.35)
    x, y = x + 0.0, y + 0.0
    x, y = kaleido(x, y, 8)
    e, alive = escape(x, y, "julia", c, iters=90)
    img = glow(e, alive, lut(HOUSE), hue=0.315, gain=1.3)
    img = scrim(vignette(img, 0.45), bottom=0.92, h=0.40)

    def letter(d, n):
        f = fit(d, "NATURAL CAPITAL", "baskerville", n * 0.84, int(n * 0.085), 0.12, "Bold")
        spaced(d, (n / 2, n * 0.80), "NATURAL CAPITAL", f, (246, 214, 140), 0.12, "c")
        f2 = font("baskerville", int(n * 0.026), "Regular")
        spaced(d, (n / 2, n * 0.80 + f.size * 1.35), "FOLLOW THE MONEY", f2, (236, 228, 200), 0.38, "c")
    return img, letter


def book_of_names():
    # a rose window: 12-fold Julia in gold leaf, crimson and bone, set in black like a page
    x, y = grid(scale=1.05)
    r = np.hypot(x, y)
    kx, ky = kaleido(x, y, 12, offset=np.pi / 12)
    e, alive = escape(kx * 1.25, ky * 1.25, "julia", complex(-0.7269, 0.1889), iters=140)
    pal = lut([(8, 6, 6), (70, 12, 16), (150, 30, 30), (205, 160, 70), (245, 228, 185), (190, 140, 60), (40, 8, 10), (8, 6, 6)])
    img = glow(e, alive, pal, hue=0.08, cycles=1.6, gain=1.25, ink=(0.02, 0.015, 0.012))
    win = np.clip((0.80 - r) / 0.012, 0, 1)
    ring = np.exp(-((r - 0.815) / 0.006) ** 2) + 0.5 * np.exp(-((r - 0.845) / 0.003) ** 2)
    gold = np.array([0.86, 0.68, 0.32], np.float32)
    img = img * win[..., None] + ring[..., None] * gold
    paper = np.array([0.035, 0.028, 0.024], np.float32)
    img = img + (1 - win)[..., None] * paper

    def letter(d, n):
        f = fit(d, "BOOK OF NAMES", "baskerville", n * 0.7, int(n * 0.06), 0.22, "Bold")
        spaced(d, (n / 2, n * 0.035), "BOOK OF NAMES", f, (226, 196, 130), 0.22, "c")
        # the redacted line under the title: a name list that exists and then doesn't
        f2 = font("baskerville", int(n * 0.024), "Regular")
        y = n * 0.935
        spaced(d, (n / 2, y), "NATURAL CAPITAL  ·  PART II", f2, (200, 180, 150), 0.3, "c")
        bars = [(0.30, 0.15), (0.47, 0.08), (0.57, 0.13)]
        for bx, bw in bars:
            d.rectangle([n * bx, n * 0.900, n * (bx + bw), n * 0.912], fill=(0, 0, 0))
            d.rectangle([n * bx, n * 0.900, n * (bx + bw), n * 0.9015], fill=(40, 30, 26))
    return img, letter, 0  # no scrim


def after_hours():
    # four voices, one blend: a 4-fold Julia (the rabbit's curls) in late-night neon, soft and wide
    x, y = grid(scale=0.62, rot=0.2)
    x, y = kaleido(x, y, 4, offset=np.pi / 4)
    e, alive = escape(x + 0.05, y + 0.02, "julia", complex(-0.1225, 0.7449), iters=110)
    pal = lut([(18, 8, 40), (255, 80, 160), (150, 90, 255), (70, 215, 255), (255, 205, 235), (120, 40, 200), (18, 8, 40)])
    img = glow(e, alive, pal, hue=0.55, cycles=1.4, gain=1.5, knee=0.25, ink=(0.05, 0.02, 0.10), bloom=1.1)
    img = scrim(vignette(img, 0.4), top=0.7, h=0.3)

    def letter(d, n):
        f = fit(d, "AFTER HOURS", "inter-thin", n * 0.86, int(n * 0.11), 0.28)
        spaced(d, (n / 2, n * 0.06), "AFTER HOURS", f, (255, 236, 248), 0.28, "c")
    return img, letter


def mr_sidequest():
    # ordinary life as a side quest: a bright Mandelbrot spiral, quantised into big pixels like a game map
    x, y = grid(center=(-0.7436, 0.1318), scale=0.0042, rot=0.6)
    e, alive = escape(x, y, "mandel", iters=420)
    pal = lut([(30, 40, 150), (0, 190, 175), (250, 245, 225), (255, 210, 0), (255, 95, 75), (30, 40, 150)])
    v = (np.sqrt(e) * 3.0 + 0.1) % 1.0
    img = pal[(v * 1023).astype(np.int32)]
    img[alive] = np.array([0.09, 0.08, 0.2], np.float32)
    # pixelate: 96 x 96 blocks
    b = N // 96
    small = img[b // 2::b, b // 2::b][:96, :96]
    img = np.repeat(np.repeat(small, b, 0), b, 1)
    img = np.pad(img, ((0, N - img.shape[0]), (0, N - img.shape[1]), (0, 0)), mode="edge")
    img = scrim(img, bottom=0.88, h=0.40)

    def letter(d, n):
        f = fit(d, "MR. SIDEQUEST", "inter", n * 0.84, int(n * 0.11), 0.0)
        y = n * 0.80
        spaced(d, (n / 2 + n * 0.008, y + n * 0.008), "MR. SIDEQUEST", f, (30, 40, 150), 0.0, "c")
        spaced(d, (n / 2, y), "MR. SIDEQUEST", f, (255, 214, 0), 0.0, "c")
        # quest marker over the title
        cx, cy, s = n * 0.5, n * 0.70, n * 0.04
        d.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=(255, 214, 0), outline=(30, 40, 150), width=max(2, n // 256))
        fm = font("inter", int(s * 1.3))
        d.text((cx, cy), "!", font=fm, fill=(30, 40, 150), anchor="mm")
    return img, letter


def nightshift_fm():
    # a station you can't quite tune: Julia filaments in violet and red, swept by slow horizontal bands
    x, y = grid(scale=1.2, rot=-0.15, sx=1.0, sy=1.0)
    e, alive = escape(x, y, "julia", complex(-0.8, 0.156), iters=140)
    pal = lut([(8, 3, 14), (70, 0, 95), (215, 20, 95), (255, 120, 60), (255, 230, 245), (40, 0, 70), (8, 3, 14)])
    img = glow(e, alive, pal, hue=0.12, cycles=1.8, gain=1.25, ink=(0.03, 0.0, 0.05), bloom=0.9)
    t = np.linspace(0, 1, N, dtype=np.float32)
    bands = 0.82 + 0.18 * np.sin(t * 2 * np.pi * 260) ** 2  # fine scanlines
    sweep = 1 + 0.25 * np.exp(-((t - 0.38) / 0.02) ** 2)  # one brighter tuning band
    img = img * (bands * sweep)[:, None, None]
    img = scrim(vignette(img, 0.5), bottom=0.85, h=0.36)

    def letter(d, n):
        f = fit(d, "NIGHTSHIFT FM", "mono", n * 0.84, int(n * 0.085), 0.04)
        spaced(d, (n * 0.08, n * 0.70), "NIGHTSHIFT FM", f, (255, 230, 245), 0.04)
        f2 = font("mono-light", int(n * 0.03))
        spaced(d, (n * 0.08, n * 0.70 + f.size * 1.25), "PHANTOM FREQUENCY", f2, (255, 120, 140), 0.25)
        # tuning dial
        y0, x0, x1 = n * 0.90, n * 0.08, n * 0.92
        d.line([(x0, y0), (x1, y0)], fill=(150, 110, 160), width=max(1, n // 512))
        fs = font("mono-light", int(n * 0.016))
        for i, mhz in enumerate(range(88, 109)):
            x = x0 + (x1 - x0) * i / 20
            tall = mhz % 2 == 0
            d.line([(x, y0 - (n * 0.018 if tall else n * 0.009)), (x, y0)], fill=(150, 110, 160), width=max(1, n // 512))
            if mhz % 4 == 0:
                d.text((x, y0 + n * 0.012), str(mhz), font=fs, fill=(150, 110, 160), anchor="mt")
        nx = x0 + (x1 - x0) * (103.3 - 88) / 20
        d.line([(nx, y0 - n * 0.04), (nx, y0 + n * 0.008)], fill=(255, 40, 80), width=max(2, n // 256))
    return img, letter


def iron_canticle():
    # candles for a dead sun: Burning Ship spires in ember and blood under a black sun with a thin corona
    x, y = grid(center=(-1.7615, -0.0285), scale=0.048, rot=0.0)
    e, alive = escape(x, y, "ship", iters=260)
    pal = lut([(4, 2, 2), (55, 6, 6), (150, 25, 10), (240, 100, 25), (255, 210, 140), (255, 250, 225), (90, 12, 8), (4, 2, 2)])
    img = glow(e, alive, pal, hue=0.0, cycles=1.5, gain=1.35, knee=0.3, ink=(0.012, 0.004, 0.004), bloom=0.8)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32) / N
    r = np.hypot(xx - 0.5, yy - 0.30)
    R = 0.135
    disc = np.clip((r - R) / 0.002, 0, 1)
    corona = np.exp(-np.clip(r - R, 0, None) / 0.018) * (r > R)
    halo = np.exp(-np.clip(r - R, 0, None) / 0.09) * (r > R)
    img = img * disc[..., None] + corona[..., None] * np.array([1.0, 0.55, 0.2], np.float32) * 0.9 \
        + halo[..., None] * np.array([0.5, 0.08, 0.03], np.float32) * 0.5
    img = img * disc[..., None] + (1 - disc)[..., None] * np.array([0.006, 0.003, 0.003], np.float32)
    img = scrim(vignette(img, 0.55), bottom=0.8, h=0.3)

    def letter(d, n):
        f = fit(d, "IRON CANTICLE", "baskerville", n * 0.82, int(n * 0.075), 0.3, "Bold")
        spaced(d, (n / 2, n * 0.82), "IRON CANTICLE", f, (240, 200, 150), 0.3, "c")
        f2 = font("baskerville", int(n * 0.026), "Italic")
        d.text((n / 2, n * 0.82 + f.size * 1.45), "Candles for a Dead Sun", font=f2, fill=(220, 120, 70), anchor="mt")
    return img, letter


def last_carriage():
    # the overnight north: a Julia stretched sideways into motion streaks, cold steel, one seam of first light
    x, y = grid(scale=1.3, sx=0.32, sy=1.0, rot=np.pi / 2)
    e, alive = escape(x, y, "julia", complex(-0.75, 0.11), iters=160)
    pal = lut([(3, 6, 12), (18, 40, 68), (60, 110, 140), (160, 195, 212), (40, 80, 110), (3, 6, 12)])
    img = glow(e, alive, pal, hue=0.2, cycles=1.6, gain=1.2, ink=(0.01, 0.015, 0.025), bloom=0.7)
    t = np.linspace(0, 1, N, dtype=np.float32)
    seam = np.exp(-((t - 0.64) / 0.0025) ** 2) + 0.35 * np.exp(-((t - 0.64) / 0.03) ** 2)
    fade = np.linspace(0.2, 1.0, N, dtype=np.float32) ** 1.5  # light grows left to right, towards morning
    img = img + seam[:, None, None] * fade[None, :, None] * np.array([1.0, 0.72, 0.45], np.float32) * 0.9
    img = scrim(vignette(img, 0.5), top=0.9, h=0.34)

    def letter(d, n):
        f = fit(d, "LAST CARRIAGE", "oswald", n * 0.84, int(n * 0.12), 0.08, "Light")
        spaced(d, (n * 0.08, n * 0.05), "LAST CARRIAGE", f, (210, 228, 236), 0.08)
        f2 = font("oswald", int(n * 0.03), "Regular")
        spaced(d, (n * 0.08, n * 0.05 + f.size * 1.35), "OVERNIGHT NORTH", f2, (240, 190, 140), 0.4)
    return img, letter


def lowbridge():
    # concrete and one streetlight: a grey Mandelbrot filament field with a single band of sodium orange
    x, y = grid(center=(-0.7453, 0.1127), scale=0.0065, rot=1.1)
    e, alive = escape(x, y, "mandel", iters=500)
    grey = lut([(6, 6, 7), (40, 40, 44), (110, 110, 116), (190, 190, 192), (70, 70, 74), (6, 6, 7)])
    img = glow(e, alive, grey, hue=0.0, cycles=2.0, gain=1.25, knee=0.3, ink=(0.01, 0.01, 0.012), bloom=0.5)
    v = (np.sqrt(e) * 2.0) % 1.0
    band = np.exp(-((v - 0.62) / 0.04) ** 2) * (~alive) * np.clip((e - 0.05) / 0.3, 0, 1)
    img = img * (1 - 0.8 * band[..., None]) + band[..., None] * np.array([1.0, 0.55, 0.12], np.float32) * 1.1
    img = 0.15 * img.mean(-1, keepdims=True) + 0.85 * img
    img = scrim(vignette(img, 0.55), bottom=0.85, h=0.34)

    def letter(d, n):
        f = fit(d, "LOWBRIDGE", "oswald", n * 0.84, int(n * 0.2), 0.02, "Bold")
        spaced(d, (n * 0.08, n * 0.70), "LOWBRIDGE", f, (232, 232, 230), 0.02)
        f2 = font("oswald", int(n * 0.03), "Regular")
        spaced(d, (n * 0.08 + n * 0.004, n * 0.70 + f.size * 1.32), "UNDER THE BRIDGE", f2, (255, 145, 35), 0.4)
    return img, letter


ACTS = {
    "natural-capital": natural_capital,
    "book-of-names": book_of_names,
    "after-hours": after_hours,
    "mr-sidequest": mr_sidequest,
    "nightshift-fm": nightshift_fm,
    "iron-canticle": iron_canticle,
    "last-carriage": last_carriage,
    "lowbridge": lowbridge,
}


def render(act, out):
    res = ACTS[act]()
    img, letter = res[0], res[1]
    img = grain(img, act, 0.012)
    art = to_image(img)
    os.makedirs(os.path.join(out, act), exist_ok=True)
    art.resize((1024, 1024), Image.LANCZOS).save(os.path.join(out, act, "artist.jpg"), quality=92)
    cover = art.copy()
    letter(ImageDraw.Draw(cover), N)
    cover.save(os.path.join(out, act, "album-2048.png"))
    cover.resize((1024, 1024), Image.LANCZOS).save(os.path.join(out, act, "album.jpg"), quality=92)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    for act in sys.argv[2:] or ACTS:
        render(act, out)
        print(act, "done", flush=True)


if __name__ == "__main__":
    main()
