"""Atmosphere and film finish. All textures are precomputed from seeds; the
per-frame work is scrolling, blending and channel shifts."""
import numpy as np
import cv2

from . import palette as P
from .noise import fbm2d, rng


class Atmosphere:
    def __init__(self, W, H, seed):
        self.W, self.H = W, H
        lw, lh = W // 4, H // 4
        # two fog sheets at quarter resolution, twice as wide as the frame so
        # they can scroll; blended at low res and upscaled per frame (fog is soft)
        self.lw, self.lh = lw, lh
        self.fog = []
        for k in range(2):
            n = fbm2d(lh, lw * 2, seed + 100 + k, octaves=5, base=3)
            self.fog.append((np.clip((n - 0.35) * 1.8, 0, 1) ** 1.3).astype(np.float32))
        yy = np.linspace(0, 1, lh, dtype=np.float32)[:, None]
        self.ground_mist = np.clip((yy - 0.35) / 0.45, 0, 1) ** 1.5  # hugs the low ground
        # grain: 6 frames at half res, cycled
        g = rng(seed, "grain")
        self.grain = [cv2.resize(g.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H),
                                 interpolation=cv2.INTER_LINEAR) for _ in range(6)]
        paper = fbm2d(H // 2, W // 2, seed + 300, octaves=6, base=20)
        self.paper = cv2.resize(paper, (W, H))[..., None]
        self.paper_mul = 1 - 0.08 + 0.08 * (0.85 + 0.3 * self.paper)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx / W - 0.5) * 1.2, (yy / H - 0.5))
        self.vignette = (1 - np.clip((r - 0.35) * 1.1, 0, 1) ** 1.6 * 0.75)[..., None]

    def fog_layer(self, k, t, density, speed=18.0, tint=P.FOG):
        """Premultiplied RGBA fog sheet k at time t."""
        x = (t * speed * (1 + 0.6 * k) / 4) % self.lw
        i0, frac = int(x), x - int(x)
        sheet = self.fog[k][:, i0:i0 + self.lw + 1]
        sheet = sheet[:, :-1] * (1 - frac) + sheet[:, 1:] * frac  # sub-pixel scroll, no stepping
        a = np.clip(sheet * self.ground_mist * density, 0, 0.92)
        col = np.array(P.f(tint), np.float32)
        small = np.concatenate([col[None, None] * a[..., None], a[..., None]], axis=2)
        return cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

    def finish(self, img, frame_idx, grain=0.022, misreg=1):
        """Paper texture, ink misregistration, vignette, grain. img float32 HxWx3."""
        if not hasattr(self, "_mul3"):
            self._mul3 = cv2.merge([(self.paper_mul * self.vignette)[..., 0]] * 3)
        out = cv2.multiply(img, self._mul3)
        if misreg:
            out[:, misreg:, 0] = out[:, :-misreg, 0].copy()   # red plate slips right
            out[:-misreg, :, 2] = out[misreg:, :, 2].copy()   # blue plate slips up
        g = self.grain[(frame_idx // 3) % 6] * grain  # grain changes at 10 fps: filmic, and far cheaper to encode
        cv2.add(out, cv2.merge([g, g, g]), dst=out)
        return np.clip(out, 0, 1, out=out)


class Particles:
    """Drifting seeds/embers. Positions are closed-form in t (no state), so
    any frame can be rendered alone."""

    def __init__(self, n, W, H, seed):
        g = rng(seed, "particles", n)
        self.x0 = g.uniform(0, W, n)
        self.y0 = g.uniform(0, H, n)
        self.vx = g.uniform(8, 30, n)
        self.vy = g.uniform(-14, -3, n)
        self.ph = g.uniform(0, 2 * np.pi, n)
        self.r = g.uniform(1.0, 2.6, n)
        self.W, self.H = W, H

    def draw(self, frame, t, amount, rgb=P.BONE):
        if amount <= 0.01:
            return
        n = int(len(self.x0) * min(amount, 1))
        x = (self.x0[:n] + self.vx[:n] * t + 6 * np.sin(t * 0.9 + self.ph[:n])) % self.W
        y = (self.y0[:n] + self.vy[:n] * t + 4 * np.cos(t * 1.3 + self.ph[:n])) % self.H
        col = tuple(float(c) for c in P.f(rgb))
        over = frame.copy()
        for xi, yi, ri in zip(x, y, self.r[:n]):
            cv2.circle(over, (int(xi * 4), int(yi * 4)), int(ri * 4), col, -1, cv2.LINE_AA, shift=2)
        a = 0.55 * min(amount, 1)
        frame[...] = frame * (1 - a) + over * a
