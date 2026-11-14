"""Props and actors for the Glen world, drawn in code in the house style.

Props are static and painted into a plate layer once (lodge, crofts, school,
plantation, cages, turbine towers, chalets, chapel). Each prop records anchors
(plate coordinates + layer) so per-frame *actors* can animate on top of it
(flames, turbine blades, the ship, the coach, the school sign, stamps, the
Laird).

Everything is flat screen-print silhouette: peat ink, bone paper, one or two
lit colours. Red is only ever money/ownership.
"""
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from . import palette as P
from . import typography as T
from .noise import rng

LAMP = (246, 196, 104)   # lit windows
FLAME = (238, 122, 44)
FLAME_CORE = (252, 214, 120)
SS = 2  # supersampling for one-time prop painting


# ---------------------------------------------------------------- painting
class Painter:
    """Collects shapes per colour for one plate layer, then composites them."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.masks = {}

    def draw(self, rgb):
        if rgb not in self.masks:
            self.masks[rgb] = Image.new("L", (self.w * SS, self.h * SS), 0)
        return ImageDraw.Draw(self.masks[rgb])

    def apply(self, rgba):
        for rgb, m in self.masks.items():
            a = np.asarray(m.resize((self.w, self.h), Image.LANCZOS), np.float32) / 255
            col = np.array(P.f(rgb), np.float32)
            src = np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)
            rgba[...] = src + rgba * (1 - a[..., None])
        return rgba


def S(pts):
    return [(x * SS, y * SS) for x, y in pts]


def _ground(plate, layer, x):
    ridge = plate[{"mid": "hill_ridge", "far": "far_ridge", "near": "near_ridge"}[layer]]
    return float(ridge[int(np.clip(x, 0, plate["w"] - 1))])


# ---------------------------------------------------------------- static props
def lodge(pt, plate, x, s):
    """A baronial shooting lodge: turrets, crow-stepped gables, lit windows."""
    y = _ground(plate, "mid", x) + 0.035 * plate["h"]
    u = plate["h"] / 100 * s
    ink = pt.draw(P.PEAT)
    body = [(x - 9 * u, y), (x - 9 * u, y - 9 * u), (x + 9 * u, y - 9 * u), (x + 9 * u, y)]
    ink.polygon(S(body), fill=255)
    # crow-stepped gable
    for k in range(4):
        ink.rectangle(S([(x - 4 * u + k * u, y - 9 * u - (k + 1) * 1.2 * u), (x + 4 * u - k * u, y - 9 * u)]), fill=255)
    # turrets with conical roofs
    for tx in (x - 9 * u, x + 9 * u):
        ink.rectangle(S([(tx - 2 * u, y - 13 * u), (tx + 2 * u, y)]), fill=255)
        ink.polygon(S([(tx - 2.6 * u, y - 13 * u), (tx, y - 18.5 * u), (tx + 2.6 * u, y - 13 * u)]), fill=255)
    lit = pt.draw(LAMP)
    for r in range(2):
        for c in range(4):
            wx, wy = x - 6 * u + c * 4 * u, y - 7.5 * u + r * 3.6 * u
            lit.rectangle(S([(wx, wy), (wx + 1.3 * u, wy + 2 * u)]), fill=255)
    return [("lodge-window", (x + 6 * u, y - 4 * u), "mid")]


def crofts(pt, plate, xs, s):
    """Small thatched blackhouses strung along the hillside."""
    out = []
    ink = pt.draw(P.PEAT)
    for x in xs:
        y = _ground(plate, "mid", x) + 0.05 * plate["h"]
        u = plate["h"] / 100 * s
        ink.rounded_rectangle(S([(x - 4 * u, y - 2.6 * u), (x + 4 * u, y)]), radius=int(0.6 * u * SS), fill=255)
        ink.chord(S([(x - 4.8 * u, y - 6 * u), (x + 4.8 * u, y - 0.6 * u)]), 180, 360, fill=255)
        ink.rectangle(S([(x - 0.6 * u, y - 2 * u), (x + 0.6 * u, y)]), fill=0)  # door gap
        out.append(("croft", (x, y - 4 * u), "mid"))
    return out


def plantation(pt, plate, x0, x1, s):
    """Carbon forest: trees planted in a perfect grid, blocks of identical cones."""
    ink = pt.draw(P.mix(P.PEAT, P.MOSS, 0.35))
    u = plate["h"] / 100 * s
    for x in np.arange(x0, x1, 2.4 * u):
        g = _ground(plate, "mid", x)
        for r in range(6):
            y = g + (4 + r * 3.2) * u
            ink.polygon(S([(x - 1 * u, y), (x, y - 3.2 * u), (x + 1 * u, y)]), fill=255)
    return [("plantation", ((x0 + x1) / 2, _ground(plate, "mid", (x0 + x1) / 2)), "mid")]


def turbine_towers(pt, plate, xs, s):
    ink = pt.draw(P.mix(P.BONE, P.FOG, 0.5))
    out = []
    for x in xs:
        y = _ground(plate, "far", x) + 0.01 * plate["h"]
        u = plate["h"] / 100 * s
        ink.polygon(S([(x - 0.35 * u, y), (x - 0.15 * u, y - 14 * u), (x + 0.15 * u, y - 14 * u), (x + 0.35 * u, y)]), fill=255)
        out.append(("turbine", (x, y - 14 * u, u), "far"))
    return out


def salmon_cages(pt, plate, xs, s):
    ink = pt.draw(P.mix(P.PEAT, P.LOCH, 0.3))
    y0 = plate["loch_y"]
    out = []
    for x in xs:
        u = plate["h"] / 100 * s
        for k in range(3):
            cx = x + k * 6.5 * u
            ink.ellipse(S([(cx - 3 * u, y0 - 0.9 * u), (cx + 3 * u, y0 + 0.9 * u)]), outline=255, width=int(0.45 * u * SS))
        out.append(("cages", (x + 6.5 * u, y0), "loch"))
    return out


def school(pt, plate, x, s):
    """The village school with its bell-cote. Its sign is an actor (it changes)."""
    y = _ground(plate, "mid", x) + 0.07 * plate["h"]
    u = plate["h"] / 100 * s
    ink = pt.draw(P.PEAT)
    ink.polygon(S([(x - 10 * u, y), (x - 10 * u, y - 7 * u), (x, y - 12 * u), (x + 10 * u, y - 7 * u), (x + 10 * u, y)]), fill=255)
    ink.rectangle(S([(x - 1.2 * u, y - 15 * u), (x + 1.2 * u, y - 11 * u)]), fill=255)
    ink.polygon(S([(x - 1.8 * u, y - 15 * u), (x, y - 17 * u), (x + 1.8 * u, y - 15 * u)]), fill=255)
    lit = pt.draw(LAMP)
    for c in (-7, -3, 3, 7):
        lit.rectangle(S([(x + c * u - 0.8 * u, y - 5 * u), (x + c * u + 0.8 * u, y - 2.4 * u)]), fill=255)
    return [("school-sign", (x, y - 8.2 * u, u), "mid")]


def chalets(pt, plate, x0, x1, s, seed):
    """The development: identical luxury chalets in rows, every window lit."""
    g = rng(seed, "chalets")
    u = plate["h"] / 100 * s
    ink = pt.draw(P.PEAT)
    lit = pt.draw(LAMP)
    for row in range(2):
        for x in np.arange(x0 + row * 4 * u, x1, 9 * u):
            y = _ground(plate, "mid", x) + (6 + row * 8) * u
            ink.polygon(S([(x - 3 * u, y), (x - 3 * u, y - 3 * u), (x, y - 5.5 * u), (x + 3 * u, y - 3 * u), (x + 3 * u, y)]), fill=255)
            lit.polygon(S([(x - 1.6 * u, y - 0.4 * u), (x - 1.6 * u, y - 2.6 * u), (x, y - 4 * u), (x + 1.6 * u, y - 2.6 * u), (x + 1.6 * u, y - 0.4 * u)]), fill=255)
    # tower crane
    cx = x1 - 3 * u
    cy = _ground(plate, "mid", cx) + 4 * u
    ink.rectangle(S([(cx - 0.4 * u, cy - 26 * u), (cx + 0.4 * u, cy)]), fill=255)
    ink.rectangle(S([(cx - 16 * u, cy - 26 * u), (cx + 5 * u, cy - 25.3 * u)]), fill=255)
    ink.line(S([(cx - 12 * u, cy - 25.3 * u), (cx - 12 * u, cy - 17 * u)]), fill=255, width=int(0.2 * u * SS))
    return [("development", ((x0 + x1) / 2, _ground(plate, "mid", (x0 + x1) / 2)), "mid")]


def kirk(pt, plate, x, s):
    """The parish kirk with its clock tower (the Laird owns the clock)."""
    y = _ground(plate, "mid", x) + 0.05 * plate["h"]
    u = plate["h"] / 100 * s
    ink = pt.draw(P.PEAT)
    ink.rectangle(S([(x - 7 * u, y - 6 * u), (x + 5 * u, y)]), fill=255)
    ink.polygon(S([(x - 7.6 * u, y - 6 * u), (x - 1 * u, y - 10 * u), (x + 5.6 * u, y - 6 * u)]), fill=255)
    ink.rectangle(S([(x + 3 * u, y - 16 * u), (x + 7 * u, y)]), fill=255)
    ink.polygon(S([(x + 2.6 * u, y - 16 * u), (x + 5 * u, y - 22 * u), (x + 7.4 * u, y - 16 * u)]), fill=255)
    face = pt.draw(P.BONE)
    face.ellipse(S([(x + 3.8 * u, y - 14.6 * u), (x + 6.2 * u, y - 12.2 * u)]), fill=255)
    lit = pt.draw(LAMP)
    for c in (-5, -2.4, 0.2):
        lit.chord(S([(x + c * u, y - 4.8 * u), (x + (c + 1.2) * u, y - 1.2 * u)]), 180, 360, fill=255)
        lit.rectangle(S([(x + c * u, y - 3 * u), (x + (c + 1.2) * u, y - 1.2 * u)]), fill=255)
    return [("kirk-clock", (x + 5 * u, y - 13.4 * u, u), "mid")]


PROPS = {"lodge", "crofts", "plantation", "turbines", "cages", "school", "chalets", "kirk", "fence"}


def paint_props(plate, props, seed):
    """Paint static props into the plate's layers. Returns anchors."""
    w, h = plate["w"], plate["h"]
    by_layer = {L.name: L for L in plate["layers"]}
    painters = {}

    def pt_for(name):
        if name not in painters:
            painters[name] = Painter(w, h)
        return painters[name]

    anchors = []
    for p in props:
        name, _, arg = p.partition(":")
        if name == "lodge":
            anchors += lodge(pt_for("mid"), plate, w * float(arg or 0.30), 1.0)
        elif name == "crofts":
            anchors += crofts(pt_for("mid"), plate, [w * f for f in (0.42, 0.53, 0.61, 0.70)], 0.9)
        elif name == "plantation":
            anchors += plantation(pt_for("mid"), plate, w * 0.55, w * 0.92, 1.0)
        elif name == "turbines":
            anchors += turbine_towers(pt_for("far"), plate, [w * f for f in (0.18, 0.27, 0.36, 0.72, 0.83)], 1.0)
        elif name == "cages":
            anchors += salmon_cages(pt_for("loch"), plate, [w * 0.06, w * 0.24], 2.2)
        elif name == "school":
            anchors += school(pt_for("mid"), plate, w * float(arg or 0.62), 1.2)
        elif name == "chalets":
            anchors += chalets(pt_for("mid"), plate, w * 0.30, w * 0.90, 1.0, seed)
        elif name == "kirk":
            anchors += kirk(pt_for("mid"), plate, w * float(arg or 0.12), 1.1)
        elif name == "fence":
            pass  # painted by plates.hill_plate state
        else:
            raise ValueError(f"unknown prop {p!r}")
    for lname, pt in painters.items():
        pt.apply(by_layer[lname].rgba)
    return anchors


# ---------------------------------------------------------------- actors (per frame)
def flames(frame, pts_frame, scale, t, amount, seed=0):
    """Flickering roof fires at each anchor (already in frame coords)."""
    if amount <= 0:
        return
    over = frame.copy()
    for k, (x, y) in enumerate(pts_frame):
        g = rng(seed, "flame", k)
        ph = g.uniform(0, 6.28)
        for layer, col, sz in ((0, FLAME, 1.0), (1, FLAME_CORE, 0.55)):
            hgt = scale * 7 * sz * (0.8 + 0.2 * np.sin(t * 11 + ph) + 0.15 * np.sin(t * 17.3 + ph * 2))
            wid = scale * 4.2 * sz
            pts = []
            for j in range(9):
                a = j / 8
                xx = x - wid / 2 + a * wid
                tip = np.sin(a * np.pi) ** 0.7
                yy = y - hgt * tip * (0.7 + 0.3 * np.sin(t * 9 + j * 1.9 + ph))
                pts.append((xx, yy))
            pts.append((x + wid / 2, y + scale * 1.2))
            pts.append((x - wid / 2, y + scale * 1.2))
            cv2.fillPoly(over, [np.round(np.array(pts) * 4).astype(np.int32)], P.f(col), cv2.LINE_AA, shift=2)
    # glow
    a = min(amount, 1.0)
    frame[...] = frame * (1 - a) + over * a


def turbine_blades(frame, x, y, u, angle, rgb=None):
    col = P.f(rgb or P.mix(P.BONE, P.FOG, 0.5))
    for k in range(3):
        th = angle + k * 2 * np.pi / 3
        tip = (x + np.cos(th) * 7 * u, y + np.sin(th) * 7 * u)
        side = (x + np.cos(th + 0.12) * 1.4 * u, y + np.sin(th + 0.12) * 1.4 * u)
        cv2.fillPoly(frame, [np.round(np.array([(x, y), side, tip]) * 4).astype(np.int32)], col, cv2.LINE_AA, shift=2)
    cv2.circle(frame, (int(x * 4), int(y * 4)), int(0.6 * u * 4), col, -1, cv2.LINE_AA, shift=2)


@lru_cache(maxsize=4)
def ship_sprite(height):
    """An emigrant sailing ship, side on, as a dark silhouette."""
    s = 4
    H = height * s
    W = int(H * 1.4)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    u = H / 100
    d.polygon([(8 * u, 72 * u), (132 * u, 72 * u), (118 * u, 88 * u), (20 * u, 88 * u)], fill=255)
    for mx, mh in ((45, 64), (88, 70)):
        d.rectangle([mx * u, (72 - mh) * u, (mx + 1.6) * u, 72 * u], fill=255)
        for k in range(3):
            top = (72 - mh + 6 + k * 19) * u
            d.polygon([((mx - 16 + k * 2) * u, top), ((mx + 18 - k * 2) * u, top),
                       ((mx + 16 - k * 2) * u, top + 15 * u), ((mx - 14 + k * 2) * u, top + 15 * u)], fill=255)
    d.line([(118 * u, 72 * u), (138 * u, 40 * u)], fill=255, width=int(1.4 * u))
    a = np.asarray(img.resize((W // s, H // s), Image.LANCZOS), np.float32) / 255
    col = np.array(P.f(P.mix(P.PEAT, P.FOG, 0.25)), np.float32)
    return np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)


@lru_cache(maxsize=8)
def coach_sprite(height):
    """A tour coach, flat side elevation."""
    s = 4
    H = height * s
    W = int(H * 3.0)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    u = H / 100
    d.rounded_rectangle([4 * u, 10 * u, 296 * u, 82 * u], radius=int(10 * u), fill=255)
    for k in range(7):
        d.rectangle([(26 + k * 38) * u, 20 * u, (54 + k * 38) * u, 46 * u], fill=0)
    for wx in (60, 236):
        d.ellipse([(wx - 14) * u, 70 * u, (wx + 14) * u, 98 * u], fill=255)
    a = np.asarray(img.resize((W // s, H // s), Image.LANCZOS), np.float32) / 255
    col = np.array(P.f(P.PEAT), np.float32)
    return np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)


@lru_cache(maxsize=32)
def stamp_sprite(text, height, rgb=P.WARNING_RED, seed=0):
    """A rubber-stamp impression: boxed condensed caps, uneven ink."""
    f = T.font("stencil", height)
    l, t, r, b = f.getbbox(text)
    pad = int(height * 0.35)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    d.text((pad - l, pad - t), text, font=f, fill=255)
    bw = max(3, height // 10)
    d.rectangle([bw, bw, w - bw, h - bw], outline=255, width=bw)
    a = np.asarray(img, np.float32) / 255
    g = rng(seed, "stamp", text)
    ink = cv2.resize(g.random((h // 6 + 2, w // 6 + 2)).astype(np.float32), (w, h))
    speck = g.random((h, w)) > 0.06
    a = a * np.clip(0.55 + ink * 0.6, 0, 1) * speck
    col = np.array(P.f(rgb), np.float32)
    return np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)


@lru_cache(maxsize=4)
def laird_sprite(height, pose="toast"):
    """The Laird as a political-cartoon cut-out: tall, tweed, chin up, dram
    raised. Pasted on with a paper border so he never belongs to the land."""
    s = 3
    H = height * s
    W = int(H * 0.62)
    m = Image.new("L", (W, H), 0)
    tweed = Image.new("L", (W, H), 0)
    skin = Image.new("L", (W, H), 0)
    glass = Image.new("L", (W, H), 0)
    u = H / 100
    dm, dt, ds, dg = (ImageDraw.Draw(x) for x in (m, tweed, skin, glass))
    cx = W / 2
    # legs (tartan trews), shoes
    dm.polygon([(cx - 7 * u, 62 * u), (cx - 1 * u, 62 * u), (cx - 2 * u, 95 * u), (cx - 7 * u, 95 * u)], fill=255)
    dm.polygon([(cx + 1 * u, 62 * u), (cx + 7 * u, 62 * u), (cx + 7 * u, 95 * u), (cx + 2 * u, 95 * u)], fill=255)
    dm.ellipse([cx - 9 * u, 93 * u, cx - 1 * u, 98 * u], fill=255)
    dm.ellipse([cx + 1 * u, 93 * u, cx + 10 * u, 98 * u], fill=255)
    # tweed jacket: barrel chest
    dt.polygon([(cx - 11 * u, 26 * u), (cx + 11 * u, 26 * u), (cx + 13 * u, 64 * u), (cx - 13 * u, 64 * u)], fill=255)
    # lowered arm
    dt.polygon([(cx - 11 * u, 27 * u), (cx - 16 * u, 50 * u), (cx - 12 * u, 52 * u), (cx - 8 * u, 32 * u)], fill=255)
    # raised arm with the dram
    dt.polygon([(cx + 9 * u, 28 * u), (cx + 20 * u, 18 * u), (cx + 23 * u, 21 * u), (cx + 12 * u, 34 * u)], fill=255)
    ds.ellipse([cx + 19 * u, 13 * u, cx + 25 * u, 19 * u], fill=255)  # hand
    dg.polygon([(cx + 20 * u, 4 * u), (cx + 27 * u, 4 * u), (cx + 26 * u, 13 * u), (cx + 21 * u, 13 * u)], fill=255)
    # head: long face, chin up, big nose, flat cap
    ds.ellipse([cx - 6 * u, 7 * u, cx + 6 * u, 25 * u], fill=255)
    ds.polygon([(cx + 4 * u, 13 * u), (cx + 10 * u, 17 * u), (cx + 4 * u, 19 * u)], fill=255)
    dm.polygon([(cx - 7.5 * u, 10 * u), (cx + 9 * u, 9 * u), (cx + 5 * u, 5 * u), (cx - 5 * u, 5 * u)], fill=255)
    dm.rectangle([cx - 4 * u, 25 * u, cx + 4 * u, 27 * u], fill=255)  # collar/tie shadow
    # bristling moustache
    dm.polygon([(cx + 1 * u, 19.5 * u), (cx + 8 * u, 19 * u), (cx + 4 * u, 21.5 * u)], fill=255)
    # eye: a smug slit
    dm.line([(cx + 1 * u, 13.5 * u), (cx + 4.5 * u, 13 * u)], fill=255, width=int(0.8 * u))

    def arr(img):
        return np.asarray(img.resize((W // s, H // s), Image.LANCZOS), np.float32) / 255

    a_ink, a_tw, a_sk, a_gl = arr(m), arr(tweed), arr(skin), arr(glass)
    h, w = a_ink.shape
    # tweed texture: fine herringbone of two browns
    yy, xx = np.mgrid[0:h, 0:w]
    herring = ((((xx // 3) + (yy // 3) * ((xx // 12) % 2 * 2 - 1)) % 4) < 2).astype(np.float32)
    tw_col = np.array(P.f((122, 96, 66)), np.float32) * (0.85 + 0.15 * herring[..., None])
    rgb = np.zeros((h, w, 3), np.float32)
    rgb = rgb * (1 - a_tw[..., None]) + tw_col * a_tw[..., None]
    rgb = rgb * (1 - a_sk[..., None]) + np.array(P.f((222, 160, 140)), np.float32) * a_sk[..., None]
    rgb = rgb * (1 - a_gl[..., None]) + np.array(P.f((214, 150, 60)), np.float32) * a_gl[..., None]
    rgb = rgb * (1 - a_ink[..., None]) + np.array(P.f(P.PEAT), np.float32) * a_ink[..., None]
    alpha = np.clip(a_ink + a_tw + a_sk + a_gl, 0, 1)
    # paper cut-out border + soft drop shadow
    border = cv2.dilate((alpha > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    border = cv2.GaussianBlur(border, (0, 0), 0.8)
    paper = np.array(P.f(P.BONE), np.float32)
    rgb = rgb * alpha[..., None] + paper * (border - alpha).clip(0, 1)[..., None]
    alpha = np.maximum(alpha, border)
    out = np.concatenate([rgb, alpha[..., None]], axis=2)
    return out
