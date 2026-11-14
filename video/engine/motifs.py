"""Recurring motifs, drawn in code so they look identical in every video.

Defined here: the crow (watches from the edge; never changes) and the red
boundary line (warning red = ownership; creeps across the land).
More (the deed, the fund logo, "due Friday" stamps) are listed in
video/motifs/natural-capital.yaml and get added as the songs need them.
"""
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw

from . import palette as P
from .noise import fbm1d


@lru_cache(maxsize=8)
def crow_sprite(height, pose="perched"):
    """Premultiplied RGBA crow silhouette, `height` px tall."""
    s = 4
    H = height * s
    W = int(H * 1.25)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    u = H / 100.0
    if pose == "perched":
        # body (leaning forward), head, beak, tail, legs
        d.ellipse([30 * u, 30 * u, 78 * u, 72 * u], fill=255)
        d.polygon([(34 * u, 50 * u), (60 * u, 58 * u), (14 * u, 92 * u), (8 * u, 86 * u)], fill=255)  # tail
        d.ellipse([62 * u, 18 * u, 86 * u, 42 * u], fill=255)  # head
        d.polygon([(82 * u, 24 * u), (108 * u, 33 * u), (83 * u, 38 * u)], fill=255)  # beak
        d.polygon([(40 * u, 40 * u), (72 * u, 46 * u), (64 * u, 66 * u), (34 * u, 62 * u)], fill=255)  # wing
        d.line([(52 * u, 70 * u), (50 * u, 92 * u)], fill=255, width=int(2.4 * u))
        d.line([(62 * u, 70 * u), (64 * u, 92 * u)], fill=255, width=int(2.4 * u))
        d.line([(44 * u, 92 * u), (70 * u, 92 * u)], fill=255, width=int(2 * u))
        # ragged feather tips
        for k in range(5):
            x = 10 * u + k * 3 * u
            d.polygon([(x, 86 * u), (x + 5 * u, 88 * u), (x - 2 * u, 97 * u)], fill=255)
    else:  # flying
        d.ellipse([40 * u, 40 * u, 80 * u, 58 * u], fill=255)
        d.ellipse([74 * u, 38 * u, 92 * u, 54 * u], fill=255)
        d.polygon([(90 * u, 43 * u), (106 * u, 47 * u), (90 * u, 50 * u)], fill=255)
        d.polygon([(48 * u, 46 * u), (30 * u, 8 * u), (70 * u, 44 * u)], fill=255)
        d.polygon([(52 * u, 50 * u), (20 * u, 70 * u), (66 * u, 54 * u)], fill=255)
        d.polygon([(42 * u, 50 * u), (18 * u, 46 * u), (16 * u, 60 * u)], fill=255)
    a = np.asarray(img.resize((W // s, H // s), Image.LANCZOS), np.float32) / 255
    col = np.array(P.f(P.PEAT), np.float32)
    return np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)


def boundary_path(plate, seed=0, drop=0.10):
    """Points (plate coords) for a boundary line running over the hill,
    a little below the ridge, surveyed-straight in places."""
    w, h = plate["w"], plate["h"]
    ridge = plate["hill_ridge"]
    xs = np.linspace(-20, w + 20, 160)
    ys = np.interp(xs, np.arange(w), ridge) + h * drop
    wob = (fbm1d(len(xs), seed + 31, octaves=3, base=4) - 0.5) * h * 0.03
    # surveyors straighten it: blend towards straight segments
    seg = np.interp(xs, xs[::20], ys[::20])
    ys = 0.55 * seg + 0.45 * ys + wob
    return np.stack([xs, ys], axis=1)


def draw_path(frame, pts, progress, width, rgb=P.WARNING_RED, opacity=1.0):
    """Draw the first `progress` (0..1) of a polyline onto a float frame."""
    if progress <= 0 or len(pts) < 2:
        return
    seglen = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    L = cum[-1] * min(progress, 1.0)
    k = int(np.searchsorted(cum, L))
    sub = pts[:max(k, 1)]
    if k < len(pts):
        t = (L - cum[k - 1]) / max(seglen[k - 1], 1e-6) if k > 0 else 0
        tip = pts[k - 1] + (pts[k] - pts[k - 1]) * t if k > 0 else pts[0]
        sub = np.vstack([sub, tip])
    col = tuple(float(c) * opacity for c in P.f(rgb))
    overlay = frame.copy()
    cv2.polylines(overlay, [np.round(sub * 4).astype(np.int32)], False, col, int(width), cv2.LINE_AA, shift=2)
    if opacity < 1:
        frame[...] = frame * (1 - opacity) + overlay * opacity
    else:
        frame[...] = overlay
