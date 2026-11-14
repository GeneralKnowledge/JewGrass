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


class Psych:
    """The strange layer: fractals in the sky, heat-haze warps, ink slips on
    the drums. Closed-form in time and the precomputed music curves, so it is
    as deterministic as everything else.

    Shot parameters (`psych:` in the storyboard), all 0..1 unless noted:
      sky      blend a Julia-set fractal into the sky (behind the land)
      overlay  screen the fractal over the whole frame (the glen dissolves)
      kaleido  mirror segments (int, 0 = off) for chorus/breakdown patterns
      warp     heat-haze displacement of the picture, driven by the bass
      chroma   ink plates slipping apart on drum hits
    """

    # the house inks, pushed hotter: the glen on a bad trip, not a rainbow
    INKS = [P.PEAT, (126, 70, 150), (40, 130, 140), (236, 228, 200), (246, 196, 104),
            (170, 60, 120), (60, 40, 90), P.PEAT]

    def __init__(self, W, H, seed, analysis):
        self.W, self.H = W, H
        self.lw, self.lh = (320, int(320 * H / W)) if W >= H else (int(320 * W / H), 320)
        y, x = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        s = 2.0 / min(self.lw, self.lh)
        self.zx = (x - self.lw / 2) * s
        self.zy = (y - self.lh / 2) * s
        # palette LUT through the house inks
        stops = np.array([P.f(c) for c in self.INKS], np.float32)
        xs = np.linspace(0, 1, len(stops))
        q = np.linspace(0, 1, 512)
        self.lut = np.stack([np.interp(q, xs, stops[:, k]) for k in range(3)], 1).astype(np.float32)
        hits = np.array(analysis["hits"]["drums"], np.float32)
        self.hit_phase = np.cumsum(hits) / analysis["fps"]  # colour turns with every hit
        self.bass = np.array(analysis["env"]["bass"], np.float32)
        self.energy = np.array(analysis["macro"]["energy"], np.float32)
        self.hits = hits
        g = rng(seed, "psych")
        self.c_phase = g.uniform(0, 6.28)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.base_x, self.base_y = xx, yy
        self.wy = np.sin(yy / H * 6.0 * np.pi).astype(np.float32)
        self.wx = np.sin(xx / W * 4.0 * np.pi).astype(np.float32)

    def _i(self, i):
        return min(i, len(self.bass) - 1)

    def fractal(self, t, i, kaleido=0, iters=28):
        """RGB float32 (W x H) fractal frame at time t."""
        k = self._i(i)
        zx, zy = self.zx, self.zy
        rot = t * 0.05
        zoom = 0.85 - 0.15 * self.energy[k] - 0.05 * np.sin(t * 0.21)
        cr, sr = np.cos(rot), np.sin(rot)
        x = (zx * cr - zy * sr) * zoom
        y = (zx * sr + zy * cr) * zoom
        if kaleido:
            r = np.hypot(x, y)
            th = np.arctan2(y, x)
            seg = 2 * np.pi / kaleido
            th = np.abs((th % seg) - seg / 2)
            x, y = r * np.cos(th), r * np.sin(th)
        # the Julia constant wanders slowly; the bass leans on it
        a = self.c_phase + t * 0.07
        c = complex(0.7885 * np.cos(a), 0.7885 * np.sin(a)) * (1 + 0.04 * self.bass[k])
        z = x + 1j * y
        out = np.zeros(z.shape, np.float32)
        alive = np.ones(z.shape, bool)
        for n in range(iters):
            z[alive] = z[alive] * z[alive] + c
            esc = alive & (np.abs(z) > 4)
            out[esc] = n + 1 - np.log2(np.log(np.abs(z[esc]) + 1e-9) + 1e-9)
            alive &= ~esc
        out[alive] = iters
        v = (np.sqrt(out / iters) * 2.5 + 0.06 * t + 0.12 * self.hit_phase[k]) % 1.0
        rgb = self.lut[(v * 511).astype(np.int32)]
        return cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

    def warp(self, img, t, i, amount):
        if amount <= 0:
            return img
        k = self._i(i)
        amp = amount * (3 + 16 * self.bass[k])
        mx = self.base_x + amp * self.wy * np.float32(np.sin(t * 1.3))
        my = self.base_y + amp * 0.6 * self.wx * np.float32(np.cos(t * 0.9))
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def chroma(self, img, i, amount):
        k = self._i(i)
        s = int(round(amount * 10 * max(self.hits[k] - 0.5, 0) * 2))
        if s <= 0:
            return img
        out = img.copy()
        out[:, s:, 0] = img[:, :-s, 0]
        out[:, :-s, 2] = img[:, s:, 2]
        return out
