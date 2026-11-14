"""Plates: the painted backgrounds, split into depth layers for parallax.

A plate is a list of RGBA layers (float32, premultiplied alpha) at the plate
size (larger than the frame, so the camera can move), each with a depth
0 (sky, never moves) .. 1 (nearest).

Two sources:
  * procedural "placeholder" plates drawn here in the house style, used for
    tests and as layouts for img2img;
  * generated plates (SD 1.5 on the laptop) + a depth map, split into layers
    by `split_by_depth`.

Recurring locations are "master plates" with states: the same hill, seeded
the same way, with things added (a fence, a development) in later states.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw

from . import palette as P
from .noise import fbm1d, fbm2d, rng


class Layer:
    def __init__(self, rgba, depth, name):
        self.rgba = rgba  # float32 HxWx4 premultiplied
        self.depth = depth
        self.name = name


def _premul(rgb, alpha):
    a = alpha[..., None].astype(np.float32)
    return np.concatenate([rgb.astype(np.float32) * a, a], axis=2)


def _solid(h, w, rgb):
    return np.broadcast_to(np.array(P.f(rgb), np.float32), (h, w, 3)).copy()


def _ridge_mask(h, w, y_base, amp, seed, octaves=6, base=3, roughness=0.0):
    """Mask (1 below the ridge line) with a slightly torn, printed edge."""
    line = fbm1d(w, seed, octaves=octaves, base=base)
    ys = (y_base - amp * line) * h
    yy = np.arange(h, dtype=np.float32)[:, None]
    m = (yy >= ys[None, :]).astype(np.float32)
    if roughness:
        tear = fbm2d(h, w, seed + 7, octaves=3, base=40)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        m = np.clip((m - 0.5 + (tear - 0.5) * roughness) * 6 + 0.5, 0, 1)
    else:
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
    return m, ys


def _hatching(h, w, spacing=7, angle=-35, width=1.6, seed=0):
    """Woodcut-style hatching: 1 on lines, 0 between. Wobbly, not ruled."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = np.deg2rad(angle)
    wob = (fbm2d(h // 8 + 1, w // 8 + 1, seed + 11, octaves=2, base=3) - 0.5) * 6
    wob = cv2.resize(wob, (w, h))
    d = (xx * np.cos(th) + yy * np.sin(th) + wob) % spacing
    return np.clip(1 - np.abs(d - spacing / 2) / width, 0, 1).astype(np.float32)


def _grade_rgb(rgb, shade):
    """Darken by shade (0..1) towards peat, keeping the ink look."""
    peat = np.array(P.f(P.PEAT), np.float32)
    return rgb * (1 - shade[..., None]) + peat * shade[..., None]


def hill_plate(w, h, seed=1707, state="bare", tod="dawn", layout="hill", props=()):
    """The recurring hill. state: bare | fenced. tod: dawn | dusk | night.
    layout: hill (the glen) | sealoch (the hill pushed back, wide water).
    props: names from engine/props.py painted into the layers."""
    layers = []
    # Sky
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    if tod == "dawn":
        top, bottom = np.array(P.f(P.mix(P.FOG, P.HEATHER, 0.25))), np.array(P.f(P.mix(P.BONE, (240, 200, 160), 0.35)))
        sun = (0.68, 0.36, P.f((250, 236, 205)))
    elif tod == "dusk":
        top, bottom = np.array(P.f(P.mix(P.PEAT, P.HEATHER, 0.55))), np.array(P.f(P.mix(P.HEATHER, P.WARNING_RED, 0.18)))
        sun = (0.30, 0.42, P.f((222, 160, 130)))
    else:  # night
        top, bottom = np.array(P.f(P.mix(P.PEAT, P.LOCH, 0.15))), np.array(P.f(P.mix(P.PEAT, P.HEATHER, 0.45)))
        sun = (0.78, 0.16, P.f(P.mix(P.BONE, P.FOG, 0.2)))
    sky = top * (1 - t) + bottom * t
    sky = np.broadcast_to(sky, (h, w, 3)).copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx / w - sun[0]) * w / h, yy / h - sun[1])
    disk = np.clip(((0.065 if tod != "night" else 0.03) - d) * 300, 0, 1)
    glow = np.exp(-(d / (0.25 if tod != "night" else 0.1)) ** 2) * 0.35
    sky = sky * (1 - disk[..., None]) + np.array(sun[2]) * disk[..., None]
    sky = np.clip(sky + glow[..., None] * np.array(sun[2]), 0, 1)
    # mottled sky (print texture)
    sky *= (0.94 + 0.06 * fbm2d(h, w, seed + 1, octaves=4, base=6))[..., None]
    if tod == "night":
        g = rng(seed, "stars")
        for _ in range(260):
            x, y = int(g.uniform(0, w)), int(g.uniform(0, h * 0.5))
            cv2.circle(sky, (x, y), 1, P.f(P.BONE) if g.random() > 0.3 else P.f(P.FOG), -1, cv2.LINE_AA)
    layers.append(Layer(_premul(sky, np.ones((h, w), np.float32)), 0.0, "sky"))

    # Far ridge: heather, washed out by distance
    m, far_ridge = _ridge_mask(h, w, 0.56, 0.20, seed + 2, base=4)
    far = _solid(h, w, P.mix(P.HEATHER, P.FOG, 0.45))
    far *= (0.9 + 0.1 * fbm2d(h, w, seed + 3, base=8))[..., None]
    layers.append(Layer(_premul(far, m), 0.15, "far"))

    # Loch: a flat band of teal with pale streaks
    lm = np.zeros((h, w), np.float32)
    loch_top, loch_bot = (0.56, 1.0) if layout == "sealoch" else (0.60, 0.70)
    lm[int(h * loch_top):int(h * loch_bot)] = 1
    lm = cv2.GaussianBlur(lm, (0, 0), 1.0)
    loch = _solid(h, w, P.LOCH)
    streak = (fbm2d(h, w, seed + 4, octaves=3, base=60) > 0.72).astype(np.float32) * 0.25
    loch = loch + streak[..., None] * np.array(P.f(P.BONE)) * 0.6
    layers.append(Layer(_premul(np.clip(loch, 0, 1), lm), 0.3, "loch"))

    # Mid: the hill itself, moss with woodcut hatching in its shadow side
    if layout == "sealoch":  # the hill pushed back to a headland on the right
        m, ys = _ridge_mask(h, w, 0.80, 0.36, seed + 5, base=2, octaves=5, roughness=0.6)
        fade = np.clip((np.arange(w) / w - 0.52) * 6, 0, 1)
        ys = ys * fade + h * 1.2 * (1 - fade)
        m = m * fade[None, :]
    else:
        m, ys = _ridge_mask(h, w, 0.80, 0.36, seed + 5, base=2, octaves=5, roughness=0.6)
    mid = _solid(h, w, P.mix(P.MOSS, P.LOCH, 0.35))
    shade = np.clip((xx / w - 0.30) * 1.6, 0, 1) * 0.7 + 0.25 * fbm2d(h, w, seed + 6, base=5)
    hatch = _hatching(h, w, spacing=11, width=2.2, seed=seed)
    mid = _grade_rgb(mid, np.clip(shade * (0.25 + 0.75 * hatch), 0, 0.9))
    layers.append(Layer(_premul(mid, m), 0.5, "mid"))
    hill_ridge = ys

    # Near: peat-black foreground bank with grass tufts
    m, nys = _ridge_mask(h, w, 0.95, 0.12, seed + 8, base=5, roughness=0.8)
    img = Image.fromarray((m * 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    g = rng(seed, "tufts")
    for _ in range(int(w / 6)):
        x = float(g.uniform(0, w))
        base_y = float(nys[int(min(max(x, 0), w - 1))]) + 4
        for _ in range(int(g.integers(2, 6))):
            ln = float(g.uniform(12, 46)) * h / 1200
            ang = float(g.uniform(-0.5, 0.5))
            dr.line([(x, base_y), (x + np.sin(ang) * ln, base_y - np.cos(ang) * ln)], fill=255, width=2)
    m = np.asarray(img, np.float32) / 255
    near = _solid(h, w, P.PEAT)
    near += (fbm2d(h, w, seed + 9, base=30) * 0.05)[..., None]
    near_layer = _premul(near, m)

    if state in ("fenced",):
        near_layer = _over(near_layer, fence_rgba(w, h, nys, seed))
    layers.append(Layer(near_layer, 0.85, "near"))

    plate = {"layers": layers, "w": w, "h": h, "hill_ridge": hill_ridge, "near_ridge": nys,
             "far_ridge": far_ridge, "loch_y": h * (loch_top + min(loch_bot, 0.8)) / 2,
             "seed": seed, "state": state, "tod": tod, "anchors": []}
    if props:
        from .props import paint_props
        plate["anchors"] = paint_props(plate, props, seed)
    return plate


def _over(dst, src):
    a = src[..., 3:4]
    return src + dst * (1 - a)


def fence_rgba(w, h, ridge, seed):
    """A new post-and-wire fence slicing across the foreground."""
    s = 2
    img = Image.new("L", (w * s, h * s), 0)
    dr = ImageDraw.Draw(img)
    g = rng(seed, "fence")
    posts = []
    x = -40.0
    while x < w + 80:
        gy = float(ridge[int(min(max(x, 0), w - 1))])
        top = gy - h * 0.20 - float(g.uniform(-6, 6))
        lean = float(g.uniform(-4, 4))
        posts.append((x, gy + 10, x + lean, top))
        x += w * 0.075 + float(g.uniform(-10, 10))
    for (x0, y0, x1, y1) in posts:
        dr.line([(x0 * s, y0 * s), (x1 * s, y1 * s)], fill=255, width=int(h * 0.009 * s))
    for k in range(3):
        pts = [((x1) * s, (y1 + (k + 0.3) * h * 0.05 + 4) * s) for (_, _, x1, y1) in posts]
        dr.line(pts, fill=255, width=int(2 * s))
    a = np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255
    rgb = _solid(h, w, P.PEAT)
    return _premul(rgb, a)


def split_by_depth(rgb, depth, cuts=(0.25, 0.5, 0.75), depths=(0.0, 0.2, 0.5, 0.85)):
    """Split a generated image into parallax layers using a depth map.

    rgb: float32 HxWx3 0..1. depth: float32 HxW, 0 far .. 1 near (normalised).
    Each layer is the image where depth falls in its band, with soft edges and
    holes behind nearer layers filled by inpainting so movement doesn't reveal
    black gaps.
    """
    h, w = depth.shape
    edges = (-1.0,) + tuple(cuts) + (2.0,)
    layers = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        band = (depth >= lo).astype(np.float32) if i > 0 else np.ones((h, w), np.float32)
        band = cv2.GaussianBlur(band, (0, 0), 1.5)
        src = rgb
        if i < len(edges) - 2:
            nearer = (depth >= edges[i + 1]).astype(np.uint8) * 255
            nearer = cv2.dilate(nearer, np.ones((9, 9), np.uint8))
            src = cv2.inpaint((rgb * 255).astype(np.uint8), nearer, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
        layers.append(Layer(_premul(src, band), depths[i], f"d{i}"))
    return {"layers": layers, "w": w, "h": h}
