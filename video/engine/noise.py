"""Seeded noise. Deterministic: same seed, same array, on any machine."""
import hashlib

import numpy as np
import cv2


def rng(seed, *salt):
    """A numpy Generator derived from a seed plus any salt (strings/ints)."""
    key = "|".join([str(seed)] + [str(s) for s in salt]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def fbm2d(h, w, seed, octaves=5, base=4, persistence=0.5):
    """Fractal value noise in [0,1], shape (h, w), float32."""
    g = rng(seed, "fbm", h, w)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    cells = base
    for _ in range(octaves):
        grid = g.random((cells + 1, int(cells * w / h) + 2)).astype(np.float32)
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        total += amp
        amp *= persistence
        cells *= 2
    out /= total
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-6)


def fbm1d(n, seed, octaves=6, base=3, persistence=0.55):
    """Fractal 1-D noise in [0,1], length n (for ridge lines)."""
    g = rng(seed, "fbm1d", n)
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    amp, total, cells = 1.0, 0.0, base
    for _ in range(octaves):
        pts = g.random(cells + 1)
        xs = np.linspace(0, 1, cells + 1)
        # cosine interpolation
        idx = np.clip((x * cells).astype(int), 0, cells - 1)
        t = x * cells - idx
        t = (1 - np.cos(t * np.pi)) / 2
        out += amp * (pts[idx] * (1 - t) + pts[idx + 1] * t)
        total += amp
        amp *= persistence
        cells *= 2
    out /= total
    return (out - out.min()) / max(out.max() - out.min(), 1e-9)
