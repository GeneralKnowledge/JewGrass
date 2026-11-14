"""Lyrics as part of the scene.

Text modes (docs/video/video-engine.md section 7):
  chorus-sign   stencilled onto a board fixed to the scene (moves with parallax)
  spoken-title  documentary lower-third, typed in word by word, serif
  slam          one phrase fills the screen
  quiet         small, integrated, for verses
  voicemail     handwritten (Ewan)

Word sprites are rendered once and cached, so per-frame cost is compositing.
"""
import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from . import palette as P
from .noise import fbm2d

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
FONTS = {
    "stencil": ("Oswald-VF.ttf", b"Bold"),
    "serif": ("LibreBaskerville-VF.ttf", None),
    "hand": ("Caveat-VF.ttf", b"SemiBold"),
}


@lru_cache(maxsize=64)
def font(kind, size):
    path, var = FONTS[kind]
    f = ImageFont.truetype(os.path.join(FONT_DIR, path), size)
    if var:
        try:
            f.set_variation_by_name(var)
        except Exception:
            pass
    return f


def _premul_from_l(alpha_u8, rgb):
    a = alpha_u8.astype(np.float32) / 255
    col = np.array(P.f(rgb), np.float32)
    return np.concatenate([col[None, None, :] * a[..., None], a[..., None]], axis=2)


@lru_cache(maxsize=512)
def word_sprite(text, kind, size, rgb, distress=0.0, seed=0):
    """RGBA premultiplied sprite for one word. distress 0..1 eats the ink."""
    f = font(kind, size)
    l, _, r, _ = f.getbbox(text)
    asc, desc = f.getmetrics()
    pad = size // 6
    # every sprite shares the font's ascent/descent box, so words sit on one baseline
    w, h = r - l + 2 * pad, asc + desc + 2 * pad
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad - l, pad), text, font=f, fill=255)
    a = np.asarray(img, np.float32) / 255
    if distress > 0:
        n = fbm2d(h, w, seed + len(text) * 13, octaves=4, base=max(3, w // 30))
        a = a * np.clip((n - distress * 0.6) * 4, 0, 1)
    return _premul_from_l((a * 255).astype(np.uint8), rgb), (l, 0)


def layout_words(words, kind, size, max_width, gap_ratio=0.28):
    """Greedy line-wrap. Returns list of (word, x, y) relative to a block,
    plus block (w, h)."""
    f = font(kind, size)
    space = int(size * gap_ratio)
    lines, cur, cur_w = [], [], 0
    for wd in words:
        ww = f.getbbox(wd)[2] - f.getbbox(wd)[0]
        if cur and cur_w + space + ww > max_width:
            lines.append((cur, cur_w))
            cur, cur_w = [], 0
        cur.append((wd, ww))
        cur_w += (space if cur_w else 0) + ww
    if cur:
        lines.append((cur, cur_w))
    asc, desc = f.getmetrics()
    line_h = int((asc + desc) * 0.92)
    out, bw = [], max(lw for _, lw in lines) if lines else 0
    for i, (ws, lw) in enumerate(lines):
        x = (bw - lw) // 2
        for wd, ww in ws:
            out.append((wd, x, i * line_h))
            x += ww + space
    return out, (bw, line_h * len(lines))


def paste(dst, sprite, x, y, opacity=1.0):
    """Premultiplied 'over' of sprite onto dst at integer x, y (clipped)."""
    H, W = dst.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    d = dst[y0:y1, x0:x1]
    d[...] = s + d * (1 - s[..., 3:4])


def sign_board(w, h, seed=0):
    """A rough painted board (bone paint on boards) to stencil the chorus on."""
    a = np.ones((h, w), np.float32)
    rough = fbm2d(h, w, seed + 5, octaves=4, base=8)
    rgb = np.array(P.f(P.BONE), np.float32)[None, None, :] * (0.82 + 0.18 * rough[..., None])
    # board seams
    for k in range(1, 3):
        y = int(h * k / 3)
        rgb[max(y - 1, 0):y + 1] *= 0.6
    edge = np.zeros((h, w), np.float32)
    edge[3:-3, 3:-3] = 1
    a = cv2.GaussianBlur(edge, (0, 0), 1.0) * a
    return np.concatenate([rgb * a[..., None], a[..., None]], axis=2)
