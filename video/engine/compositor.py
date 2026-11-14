"""Stage 4: the deterministic compositor.

frame(i) = f(plan, analysis, assets, seed, i). No state carries between
frames: anything that depends on history (flash schedule, budgets) is
precomputed for the whole song in __init__, so chunks can render in parallel
and any frame can be re-rendered alone.

Back to front: sky, far, fog A, loch, mid (+ red line), fog B, near (+ fence,
crow, chorus sign), particles, screen text, grade, flashes, film finish.
"""
import numpy as np
import cv2

cv2.setNumThreads(1)  # parallelism comes from render workers, not OpenCV

from . import palette as P
from . import plates as PL
from . import motifs as M
from . import typography as T
from . import props as PR
from .fx import Atmosphere, Particles, Psych

PLATE_SCALE = 1.25  # plates are bigger than the frame so the camera can move

DEFAULT_BUDGET = {"intro": 0.2, "verse": 0.3, "chorus": 0.7, "bridge": 0.1,
                  "breakdown": 0.1, "outro": 0.15}

# The strange layer, by section type (a shot's own `psych:` overrides it).
PSYCH_DEFAULT = {"intro": {"sky": 0.15}, "verse": {"sky": 0.35, "warp": 0.3},
                 "chorus": {"sky": 0.85, "overlay": 0.12, "kaleido": 6, "warp": 0.6, "chroma": 0.6},
                 "bridge": {"sky": 0.25}, "breakdown": {"sky": 1.0, "overlay": 0.5, "kaleido": 8, "warp": 1.0, "chroma": 1.0},
                 "outro": {"sky": 0.1}}

WEATHER = {"mist-low": 0.5, "fog-heavy": 0.75, "clear": 0.0}


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


class Compositor:
    def __init__(self, plan, analysis, W=1920, H=1080):
        self.plan, self.a = plan, analysis
        self.W, self.H = W, H
        self.U = min(H, W * 0.5625)  # sprite/text size unit (portrait-safe)
        self.text_on = True  # thumbnails switch the lyrics off
        self.fps = analysis["fps"]
        self.seed = int(plan["meta"].get("seed", 1707))
        self.atm = Atmosphere(W, H, self.seed)
        self.psych = Psych(W, H, self.seed, analysis)
        self.particles = Particles(140, W, H, self.seed)
        self.plates = {}
        for sh in plan["shots"]:
            if sh.get("plate", "fractal") != "fractal":
                self._plate(sh["plate"])
        self.flashes = self._flash_schedule()
        self.shot_at = np.zeros(analysis["frames"], np.int32)
        for k, sh in enumerate(plan["shots"]):
            i0, i1 = int(round(sh["t0"] * self.fps)), int(round(sh["t1"] * self.fps))
            self.shot_at[i0:i1] = k
        self.board_cache = {}

    # ---------- assets ----------
    def _plate(self, key):
        """'hill-dawn@fenced' -> procedural hill, dawn light, fenced state.
        A generated plate (assets/plates/<key>.png + .depth.png) wins if present."""
        if key in self.plates:
            return self.plates[key]
        pw, ph = int(self.W * PLATE_SCALE), int(self.H * PLATE_SCALE)
        body, *props = key.split("+")
        name, _, state = body.partition("@")
        base, _, tod = name.partition("-")
        if base in ("hill", "sealoch"):
            plate = PL.hill_plate(pw, ph, seed=self.seed, state=state or "bare", tod=tod or "dawn",
                                  layout=base, props=tuple(props))
        else:
            raise ValueError(f"no plate source for {key!r}")
        # keep layers as uint8 (premultiplied): a quarter of the memory of float32,
        # which matters with several workers on an 8 GB laptop
        for L in plate["layers"]:
            L.rgba = (np.clip(L.rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
        plate["boundary"] = M.boundary_path(plate, self.seed)
        self.plates[key] = plate
        return plate

    def _flash_schedule(self):
        """Frames that flash: big drum hits in shots that allow it, at most
        one every 0.4 s (under 3 per second, photosensitivity rule)."""
        hits = np.array(self.a["hits"]["drums"])
        out = np.zeros(len(hits), np.float32)
        last = -1e9
        for sh in self.plan["shots"]:
            if "flash" not in (sh.get("react") or {}).get("drums", ""):
                continue
            i0, i1 = int(sh["t0"] * self.fps), min(int(sh["t1"] * self.fps), len(hits))
            for i in range(i0, i1):
                if hits[i] > 0.9 and (i - last) / self.fps >= 0.4 and hits[i] >= hits[max(i - 1, 0)]:
                    for k in range(4):
                        if i + k < len(out):
                            out[i + k] = max(out[i + k], 0.10 * (1 - k / 4))
                    last = i
        return out

    # ---------- helpers ----------
    def _budget(self, sh):
        if "budget" in (sh.get("react") or {}):
            return float(sh["react"]["budget"])
        sec = sh.get("at", "verse")
        for k, v in DEFAULT_BUDGET.items():
            if sec.startswith(k):
                return v
        return 0.3

    def _psych(self, sh):
        if "psych" in sh:
            return sh["psych"] or {}
        sec = sh.get("at", "verse")
        for k, v in PSYCH_DEFAULT.items():
            if sec.startswith(k):
                return v
        return {}

    def _camera(self, sh, p, t, i, budget):
        cam = sh.get("camera") or {}
        move, speed = cam.get("move", "hold"), float(cam.get("speed", 1.0))
        e = ease(p) if cam.get("ease", True) else p
        zoom, px, py = 1.0, 0.0, 0.0
        m = 0.07 * self.W * speed
        if move == "push-in":
            zoom = 1.0 + 0.16 * speed * e
        elif move == "pull-out":
            zoom = 1.0 + 0.16 * speed * (1 - e)
        elif move == "drift-left":
            px = m * (1 - 2 * e)
        elif move == "drift-right":
            px = -m * (1 - 2 * e)
        elif move == "rise":
            py = 0.05 * self.H * speed * (1 - 2 * e)
        if "target" in cam:  # drift the centre towards a point (plate fractions) as we move
            tx, ty = cam["target"]
            px += (tx - 0.5) * self.W * PLATE_SCALE * 0.6 * e
            py += (ty - 0.5) * self.H * PLATE_SCALE * 0.6 * e
        # breathing so nothing is ever fully frozen
        px += 3 * np.sin(t * 0.37)
        py += 2 * np.sin(t * 0.23 + 1)
        # bass: low, slow sway; big 808 hits shudder the ground (near layers)
        react = sh.get("react") or {}
        shake = 0.0
        if react.get("bass"):
            hb = self.a["hits"]["bass"][i] if i < len(self.a["hits"]["bass"]) else 0
            shake = budget * (0.5 * self.a["env"]["bass"][i] + 1.5 * max(hb - 0.6, 0)) * 7
        return zoom, px, py, float(cam.get("parallax", 1.0)), shake

    def _layer_matrix(self, plate, depth, zoom, px, py, parallax, shake, i):
        pw, ph = plate["w"], plate["h"]
        k = parallax * depth
        z = 1 + (zoom - 1) * (0.25 + 0.75 * depth)
        sx = shake * depth * np.sin(i * 1.7)
        sy = shake * depth * np.cos(i * 2.3)
        cx = pw / 2 + px * (0.1 + 0.9 * k) + sx
        cy = ph / 2 + py * (0.1 + 0.9 * k) + sy
        return np.array([[z, 0, self.W / 2 - z * cx], [0, z, self.H / 2 - z * cy]], np.float32)

    def _over(self, dst, src_rgba):
        """Premultiplied 'over', in place. OpenCV ops: ~5x faster than numpy here."""
        a = src_rgba[..., 3]
        inv = cv2.merge([1.0 - a] * 3)
        cv2.multiply(dst, inv, dst=dst)
        cv2.add(dst, src_rgba[..., :3], dst=dst)

    def _warp(self, rgba, Mx, replicate=False):
        out = cv2.warpAffine(rgba, Mx, (self.W, self.H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT)
        if out.dtype == np.uint8:
            out = out.astype(np.float32) * (1.0 / 255)
        return out

    def _active_line(self, sh, t, linger=2.0):
        """The lyric line showing at time t within this shot, and how many of
        its words have been sung (with a per-word age for animation)."""
        lines = [ln for ln in self.plan["lines"] if ln.get("words")
                 and sh["t0"] - 0.3 <= ln["words"][0][0] < sh["t1"]]
        cur = None
        for j, ln in enumerate(lines):
            start = ln["words"][0][0] - 0.12
            nxt = lines[j + 1]["words"][0][0] - 0.12 if j + 1 < len(lines) else 1e9
            end = min(nxt, ln["words"][-1][0] + linger)
            if start <= t < end:
                cur = (ln, end)
        if not cur:
            return None, [], 0.0
        ln, end = cur
        ages = [t - wt + 0.12 for wt, _ in ln["words"]]
        fade = min(1.0, max(0.0, (end - t) / 0.4))
        return ln, ages, fade

    # ---------- text modes ----------
    def _chorus_board(self, sh, t, plate, Mnear):
        txt = sh.get("text") or {}
        ln, ages, fade = self._active_line(sh, t, linger=3.0)
        bw, bh = int(self.W * 0.42), int(self.H * 0.24)
        key = ("board", bw, bh)
        if key not in self.board_cache:
            self.board_cache[key] = T.sign_board(bw, bh, self.seed)
        board = self.board_cache[key].copy()
        if ln:
            words = [w.upper() for _, w in ln["words"]]
            # fit: the largest size whose wrapped block fits inside the board
            size = int(bh * 0.40)
            while True:
                lay, (lw, lh) = T.layout_words(words, "stencil", size, int(bw * 0.88))
                if (lw <= bw * 0.9 and lh <= bh * 0.86) or size <= 12:
                    break
                size = int(size * 0.92)
            pad = size // 6  # sprites carry this much padding on every side
            ox, oy = (bw - lw) // 2 - pad, (bh - lh) // 2 - pad
            for (word, x, y), age in zip(lay, ages):
                if age <= 0:
                    continue
                # distress weathers the paint but must never make the chorus unreadable
                spr, _ = T.word_sprite(word, "stencil", size, P.PEAT, min(float(txt.get("distress", 0.15)), 1.0) * 0.55, self.seed)
                op = min(age / 0.08, 1.0)  # slapped on fast, like a stencil
                T.paste(board, spr, ox + x, oy + y - int(4 * (1 - op)), op * fade)
        # place the board on the fence line (plate coords), then camera-transform
        anchor = txt.get("anchor", [0.30, None])
        ax = anchor[0] * plate["w"]
        ridge = plate["near_ridge"]
        ay = (anchor[1] * plate["h"]) if anchor[1] is not None else \
            float(ridge[int(min(ax, plate["w"] - 1))]) - plate["h"] * 0.20 - bh * 0.15
        tilt = np.deg2rad(float(txt.get("tilt", -2.0)))
        c, s = np.cos(tilt), np.sin(tilt)
        Mb = np.array([[c, -s, ax - bw / 2], [s, c, ay - bh / 2]], np.float32)  # board -> plate
        Mf = np.vstack([Mnear, [0, 0, 1]]) @ np.vstack([Mb, [0, 0, 1]])
        return self._warp(board, Mf[:2].astype(np.float32))

    def _lower_third(self, img, sh, t, kind="serif", rgb=P.BONE, size_ratio=0.042, where="left"):
        ln, ages, fade = self._active_line(sh, t)
        if not ln:
            return
        size = int(self.U * size_ratio)
        words = [w for _, w in ln["words"]]
        lay, (lw, lh) = T.layout_words(words, kind, size, int(self.W * 0.62))
        x0 = int(self.W * 0.08) if where == "left" else (self.W - lw) // 2
        y0 = int(self.H * 0.80) - lh
        pad = size // 6
        # quiet band behind the text for legibility
        band = img[max(y0 - size // 2, 0):y0 + lh + size // 2]
        band *= (1 - 0.35 * fade)
        for (word, x, y), age in zip(lay, ages):
            if age <= 0:
                continue
            spr, _ = T.word_sprite(word, kind, size, rgb)
            op = min(age / 0.25, 1.0) * fade
            self._paste_rgb(img, spr, x0 + x - pad, y0 + y - pad, op)
        if where == "left":
            rule_y = y0 + lh + size // 3
            img[rule_y:rule_y + 2, x0:x0 + int(lw * min(1, fade))] = P.f(rgb)

    # ---------- the lyric-video mode ----------
    def _lyric_lines(self):
        """All lines with their display windows, computed once: a line appears
        a little before its first word (so it can be read ahead) and leaves when
        the next one arrives, or a while after its last word if music follows."""
        if hasattr(self, "_ll"):
            return self._ll
        lines = [ln for ln in self.plan["lines"] if ln.get("words")]
        out = []
        # a line comes in a little before its first word, but never before the
        # previous line's last word has had its moment
        starts = []
        for j, ln in enumerate(lines):
            first = ln["words"][0][0]
            st = first - 0.45
            if j:
                st = min(max(st, lines[j - 1]["words"][-1][0] + 0.2), first - 0.05)
            starts.append(st)
        for j, ln in enumerate(lines):
            start = starts[j]
            nxt = starts[j + 1] if j + 1 < len(lines) else 1e9
            last = ln["words"][-1][0]
            dur_last = max(0.35, min(1.2, (last - ln["words"][0][0]) / max(len(ln["words"]) - 1, 1) * 1.5))
            end = min(nxt, last + dur_last + 1.6)
            out.append({"line": ln, "start": start, "end": end, "next": lines[j + 1] if j + 1 < len(lines) else None})
        self._ll = out
        return out

    @staticmethod
    def _soft_shadow(spr, blur):
        a = spr[..., 3]
        sh = cv2.GaussianBlur(a, (0, 0), blur) * 0.85
        out = np.zeros_like(spr)
        out[..., 3] = sh
        return out

    def _set_line(self, img, words, cx, cy, size, max_w, states, base_op, kind="stencil", upper=True, idle=0.32):
        """Draw a line centred at (cx, cy). states[k] in 0..1 = how 'sung' word k is."""
        ws = [w.upper() if upper else w for w in words]
        lay, (lw, lh) = T.layout_words(ws, kind, size, max_w)
        # balance two-line wraps: prefer splitting near the middle
        if lh > size * 1.5 and len(ws) > 2:
            for frac in (0.55, 0.6, 0.65, 0.7, 0.75):
                lay2, (lw2, lh2) = T.layout_words(ws, kind, size, int(max_w * frac))
                if lh2 <= lh:
                    lay, lw, lh = lay2, lw2, lh2
                    break
        pad = size // 6
        x0, y0 = int(cx - lw / 2) - pad, int(cy - lh / 2) - pad
        key = ("shadow", kind, size)
        for (word, x, y), st in zip(lay, states):
            spr, _ = T.word_sprite(word, kind, size, P.BONE)
            sk = (word, kind, size)
            if sk not in self.board_cache:
                self.board_cache[sk] = self._soft_shadow(spr, size / 10)
            self._paste_rgb(img, self.board_cache[sk], x0 + x + size // 25, y0 + y + size // 18,
                            base_op * (0.45 + 0.45 * st))
        for (word, x, y), st in zip(lay, states):
            # unsung: dim bone; being sung: warm lamp gold; sung: full bone
            if st >= 1:
                spr, _ = T.word_sprite(word, kind, size, P.BONE)
                self._paste_rgb(img, spr, x0 + x, y0 + y, base_op)
            else:
                dim, _ = T.word_sprite(word, kind, size, P.BONE)
                self._paste_rgb(img, dim, x0 + x, y0 + y, base_op * (idle + 0.2 * st))
                if st > 0:
                    hot, _ = T.word_sprite(word, kind, size, (246, 196, 104))
                    self._paste_rgb(img, hot, x0 + x, y0 + y - int(3 * st), base_op * min(1, st * 2.5))
        return lh

    def _scrim(self, img, cy, half, strength):
        """A soft dark band behind the words: the land stays visible, the words read."""
        if strength <= 0.01:
            return
        if not hasattr(self, "_scrim_cache") or self._scrim_cache[0] != (cy, half):
            y = (np.arange(self.H, dtype=np.float32) - cy) / half
            prof = np.exp(-0.5 * (y * y) ** 1.5)
            self._scrim_cache = ((cy, half), prof[:, None, None])
        img *= 1 - strength * self._scrim_cache[1]

    def _lyric(self, img, t, txt):
        size = int(self.U * float(txt.get("size", 0.085)))
        max_w = int(self.W * (0.80 if self.W >= self.H else 0.86))
        cy = self.H * float(txt.get("y", 0.62 if self.W >= self.H else 0.62))
        live = []
        for L in self._lyric_lines():
            if not (L["start"] <= t < L["end"] + 0.2):
                continue
            fin = ease((t - L["start"]) / 0.25)
            fout = 1 - ease((t - L["end"]) / 0.2)
            live.append((L, fin, fout))
        # the band fades in with the first line of a passage and out after the last
        near = [L for L in self._lyric_lines() if L["start"] - 0.8 <= t < L["end"] + 1.2]
        if near:
            L = near[0]
            sc = min(ease((t - (L["start"] - 0.8)) / 0.8), 1 - ease((t - (L["end"] + 0.4)) / 0.8))
            if len(near) > 1:
                sc = 1.0
            self._scrim(img, int(cy + size * 0.35), size * 2.1, 0.55 * sc)
        for L, fin, fout in live:
            ln = L["line"]
            words = [w for _, w in ln["words"]]
            times = [wt for wt, _ in ln["words"]]
            # each word turns gold as it starts, then settles to bone as the next begins
            states = []
            for k, wt in enumerate(times):
                nxt = times[k + 1] if k + 1 < len(times) else wt + 0.5
                if t < wt:
                    states.append(0.0)
                elif t < nxt:
                    states.append(min(1.0, 0.2 + (t - wt) / 0.08) * 0.99)
                else:
                    states.append(1.0)
            op = fin * fout
            lift = (1 - fin) * size * 0.25 - (1 - fout) * size * 0.25
            lh = self._set_line(img, words, self.W / 2, cy + lift, size, max_w, states, op, idle=0.6)
            # the next line, smaller, so the eye can read ahead
            nx = L["next"]
            if nx and nx["words"][0][0] - t < 6:
                nwords = [w for _, w in nx["words"]]
                self._set_line(img, nwords, self.W / 2, cy + lh / 2 + size * 0.85 + lift, int(size * 0.52), max_w,
                               [0.0] * len(nwords), op * 0.9, idle=0.62)

    def _slam(self, img, sh, t):
        ln, ages, fade = self._active_line(sh, t, linger=1.0)
        if not ln:
            return
        size = int(self.U * 0.17)
        words = [w.upper() for _, w in ln["words"]]
        while True:
            lay, (lw, lh) = T.layout_words(words, "stencil", size, int(self.W * 0.86))
            if (lw <= self.W * 0.9 and lh <= self.H * 0.8) or size <= 16:
                break
            size = int(size * 0.92)
        pad = size // 6
        x0, y0 = (self.W - lw) // 2 - pad, (self.H - lh) // 2 - pad
        for (word, x, y), age in zip(lay, ages):
            if age <= 0:
                continue
            op = min(age / 0.06, 1) * fade
            shadow, _ = T.word_sprite(word, "stencil", size, P.PEAT)
            self._paste_rgb(img, shadow, x0 + x + size // 18, y0 + y + size // 18, op * 0.85)
            spr, _ = T.word_sprite(word, "stencil", size, P.BONE, 0.1, self.seed)
            self._paste_rgb(img, spr, x0 + x, y0 + y, op)

    def _paste_rgb(self, img, spr, x, y, op):
        h, w = spr.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, self.W), min(y + h, self.H)
        if x1 <= x0 or y1 <= y0:
            return
        s = spr[y0 - y:y1 - y, x0 - x:x1 - x] * op
        d = img[y0:y1, x0:x1]
        d[...] = s[..., :3] + d * (1 - s[..., 3:4])

    # ---------- the frame ----------
    def frame(self, i):
        t = i / self.fps
        k = int(self.shot_at[min(i, len(self.shot_at) - 1)])
        sh = self.plan["shots"][k]
        img = self._body(i, sh)
        # dissolve: the previous shot keeps running underneath for a moment
        if sh.get("transition") == "dissolve" and k > 0:
            dur = float(sh.get("dissolve_s", 1.6))
            a = ease((t - sh["t0"]) / dur)
            if a < 1:
                img = img * a + self._body(i, self.plan["shots"][k - 1]) * (1 - a)
        if sh.get("plate", "fractal") == "fractal":
            # words go on once, after any dissolve (not once per blended shot)
            txt = sh.get("text") or {}
            if self.text_on and txt.get("mode") == "lyric":
                self._lyric(img, t, txt)
        if sh.get("title"):
            self._title(img, sh, t)
        img = self.atm.finish(img, i)
        return (img * 255 + 0.5).astype(np.uint8)

    def _title(self, img, sh, t):
        """A song title as the song arrives: small kicker over a big serif title."""
        ti = sh["title"]
        hold = float(ti.get("hold", 5.0))
        age = t - sh["t0"] - float(ti.get("delay", 0.4))
        if age < 0 or age > hold + 1.0:
            return
        op = ease(age / 0.7) * (1 - ease((age - hold) / 1.0))
        cy = self.H * float(ti.get("y", 0.34))
        size = int(self.U * float(ti.get("size", 0.095)))
        words = str(ti["text"]).split()
        lay, (lw, lh) = T.layout_words(words, "serif", size, int(self.W * 0.8))
        pad = size // 6
        x0, y0 = int((self.W - lw) / 2) - pad, int(cy - lh / 2) - pad + int((1 - ease(age / 0.9)) * size * 0.2)
        self._scrim(img, int(cy), lh * 1.3 + size, 0.35 * op)
        for word, x, y in lay:
            spr, _ = T.word_sprite(word, "serif", size, P.BONE)
            sk = ("t-shadow", word, size)
            if sk not in self.board_cache:
                self.board_cache[sk] = self._soft_shadow(spr, size / 10)
            self._paste_rgb(img, self.board_cache[sk], x0 + x + size // 25, y0 + y + size // 18, op * 0.7)
            self._paste_rgb(img, spr, x0 + x, y0 + y, op)
        if ti.get("kicker"):
            ks = int(size * 0.32)
            kw = str(ti["kicker"]).upper().split()
            klay, (kw_, kh) = T.layout_words(kw, "stencil", ks, int(self.W * 0.8))
            kx0, ky0 = int((self.W - kw_) / 2) - ks // 6, y0 - kh - int(size * 0.25)
            for word, x, y in klay:
                spr, _ = T.word_sprite(word, "stencil", ks, (246, 196, 104))
                self._paste_rgb(img, spr, kx0 + x, ky0 + y, op * 0.9)

    def _body(self, i, sh):
        if sh.get("plate", "fractal") == "fractal":
            return self._fractal_body(i, sh)
        return self._landscape_body(i, sh)

    def _fractal_body(self, i, sh):
        """Fractal-only look: the Julia field fills the frame, then the words."""
        t = i / self.fps
        ps = self._psych(sh)
        img = self.psych.field(t, i, ps)
        g = sh.get("grade") or {}
        img *= float(g.get("exposure", 1.0))
        if sh is self.plan["shots"][-1] and sh.get("fade_out", True):
            img *= 1 - ease((t - (sh["t1"] - 1.5)) / 1.5)
        if sh is self.plan["shots"][0]:
            img *= ease(t / 1.0)
        return img

    def _landscape_body(self, i, sh):
        t = i / self.fps
        p = (t - sh["t0"]) / max(sh["t1"] - sh["t0"], 1e-6)
        budget = self._budget(sh)
        plate = self.plates[sh["plate"]]
        zoom, px, py, par, shake = self._camera(sh, p, t, i, budget)
        weather = sh.get("weather") or []
        fog_d = max([WEATHER.get(w, 0) for w in weather] + [0])
        fog_d *= 1 + 0.25 * budget * self.a["env"]["bass"][min(i, self.a["frames"] - 1)]

        ps = self._psych(sh)
        fr = None
        img = np.zeros((self.H, self.W, 3), np.float32)
        Ms = {}
        for L in plate["layers"]:
            Mx = self._layer_matrix(plate, L.depth, zoom, px, py, par, shake, i)
            Ms[L.name] = Mx
            if L.name == "sky":  # opaque: no blend needed
                img[...] = self._warp(L.rgba, Mx, replicate=True)[..., :3]
                if ps.get("sky"):
                    k = ps["sky"] * (0.65 + 0.35 * self.a["macro"]["energy"][min(i, self.a["frames"] - 1)])
                    if fr is None:
                        fr, fm = self.psych.fractal(t, i, int(ps.get("kaleido", 0)))
                    w_ = fm * k
                    img[...] = img * (1 - w_) + fr * w_
            else:
                self._over(img, self._warp(L.rgba, Mx))
            for act in (sh.get("actors") or []):
                if self._actor_layer(act) == L.name:
                    self._actor(img, act, Mx, plate, sh, p, t, i)
            if L.name == "far" and fog_d:
                self._over(img, self.atm.fog_layer(0, t, fog_d * 0.8))
            if L.name == "mid":
                rl = (sh.get("motifs") or {}).get("red-line")
                if rl:
                    a0, a1 = rl.get("from", 0.0), rl.get("to", 1.0)
                    prog = a0 + (a1 - a0) * ease(p)
                    pts = plate["boundary"]
                    Mx3 = np.vstack([Mx, [0, 0, 1]])
                    fp = (np.hstack([pts, np.ones((len(pts), 1))]) @ Mx3.T)[:, :2]
                    M.draw_path(img, fp, prog, max(3, self.H // 220))
                if fog_d:
                    self._over(img, self.atm.fog_layer(1, t, fog_d))

        Mnear = Ms.get("near")
        motifs = sh.get("motifs") or {}
        crow = motifs.get("crow")
        if crow and Mnear is not None:
            self._draw_crow(img, crow, plate, Mnear, p, t)
        txt = sh.get("text") or {}
        mode = txt.get("mode", "none") if self.text_on else "none"
        if mode == "chorus-sign" and Mnear is not None:
            self._over(img, self._chorus_board(sh, t, plate, Mnear))

        # the strange layer over the land, before the text so lyrics stay legible
        if ps.get("overlay"):
            if fr is None:
                fr, fm = self.psych.fractal(t, i, int(ps.get("kaleido", 0)))
            k = ps["overlay"] * (0.3 + 0.7 * self.a["env"]["bass"][min(i, self.a["frames"] - 1)])
            img = 1 - (1 - img) * (1 - fr * k)  # screen blend
        if ps.get("warp"):
            img = self.psych.warp(img, t, i, ps["warp"])

        for act in (sh.get("actors") or []):
            if self._actor_layer(act) == "screen":
                self._actor(img, act, None, plate, sh, p, t, i)

        # fine particles follow the fiddle/banjo/guitars
        if "seeds" in weather or "embers" in weather:
            amt = budget * 1.4 * self.a["env"]["other"][min(i, self.a["frames"] - 1)] + 0.15
            self.particles.draw(img, t, amt, P.BONE if "seeds" in weather else (230, 150, 90))

        # grade the picture before the words so the words keep their own brightness
        g = sh.get("grade") or {}
        if g:
            img *= float(g.get("exposure", 1.0))
            warm = float(g.get("warm", 0.0))
            img[..., 0] *= 1 + warm * 0.08
            img[..., 2] *= 1 - warm * 0.08
        if mode == "spoken-title":
            self._lower_third(img, sh, t, "serif", P.BONE, 0.040, "left")
        elif mode == "quiet":
            self._lower_third(img, sh, t, "serif", P.BONE, 0.034, "center")
        elif mode == "voicemail":
            self._lower_third(img, sh, t, "hand", P.BONE, 0.055, "center")
        elif mode == "slam":
            self._slam(img, sh, t)
        elif mode == "lyric":
            self._lyric(img, t, txt)


        if ps.get("chroma"):
            img = self.psych.chroma(img, i, ps["chroma"])

        fl = self.flashes[i] if i < len(self.flashes) else 0
        if fl:
            img += fl

        # transitions
        tr = sh.get("transition")
        if tr == "fog-in":
            k = 1 - ease((t - sh["t0"]) / 0.8)
            if k > 0:
                img = img * (1 - k) + np.array(P.f(P.FOG), np.float32) * k
        if sh is self.plan["shots"][-1] and sh.get("fade_out", True):
            k = ease((t - (sh["t1"] - 1.5)) / 1.5)
            img *= 1 - k
        if sh is self.plan["shots"][0]:
            img *= ease(t / 1.0)

        return img

    # ---------- actors ----------
    ACTOR_LAYERS = {"flames": "mid", "blades": "far", "ship": "loch", "coach": "mid",
                    "sign": "mid", "stamp": "screen", "laird": "screen"}

    def _actor_layer(self, act):
        return act.get("layer", self.ACTOR_LAYERS.get(act["type"], "screen"))

    @staticmethod
    def _tx(Mx, x, y):
        return (Mx[0, 0] * x + Mx[0, 1] * y + Mx[0, 2], Mx[1, 0] * x + Mx[1, 1] * y + Mx[1, 2])

    def _when(self, act, sh, key="at", default=0.0):
        """Actor start time: shot fraction (at: 0.4) or a sung word (word: Friday)."""
        if "word" in act:
            for ln in self.plan["lines"]:
                for wt, w in ln.get("words", []):
                    if sh["t0"] <= wt < sh["t1"] and w.strip(".,!?'\"").lower() == act["word"].lower():
                        return wt
        f = act.get(key, default)
        return sh["t0"] + f * (sh["t1"] - sh["t0"])

    def _paste_rotated(self, img, spr, cx, cy, scale, rot_deg, op=1.0):
        h, w = spr.shape[:2]
        R = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, scale)
        bw, bh = int(w * scale * 1.3) + 4, int(h * scale * 1.6) + 4
        R[0, 2] += bw / 2 - w / 2
        R[1, 2] += bh / 2 - h / 2
        out = cv2.warpAffine(spr, R, (bw, bh), flags=cv2.INTER_LINEAR)
        self._paste_rgb(img, out, int(cx - bw / 2), int(cy - bh / 2), op)

    def _actor(self, img, act, Mx, plate, sh, p, t, i):
        kind = act["type"]
        if "span" in act:
            a0, a1 = act["span"]
            if not (a0 <= p <= a1):
                return
            p = (p - a0) / max(a1 - a0, 1e-6)
        anchors = plate.get("anchors", [])
        z = Mx[0, 0] if Mx is not None else 1.0
        if kind == "flames":
            t0 = self._when(act, sh, default=0.0)
            amt = min(max((t - t0) / 1.5, 0), 1)
            pts = [self._tx(Mx, *a[1][:2]) for a in anchors if a[0] == "croft"]
            PR.flames(img, pts, min(plate["h"], plate["w"] * 0.5625) / 100 * 1.4 * z, t, amt, self.seed)
        elif kind == "blades":
            s0, s1 = act.get("spin", [1.0, 1.0])
            # angle = integral of a speed that eases from s0 to s1 across the shot
            dur = sh["t1"] - sh["t0"]
            q = min(max(p, 0), 1)
            ang = 1.6 * dur * (s0 * q + (s1 - s0) * q * q / 2) + sh["t0"] * 1.6 * s0
            for k, a in enumerate(x for x in anchors if x[0] == "turbine"):
                x, y = self._tx(Mx, a[1][0], a[1][1])
                PR.turbine_blades(img, x, y, a[1][2] * z, ang + k * 0.9)
        elif kind == "ship":
            x0, x1 = act.get("x", [0.1, 0.4])
            xp = plate["w"] * (x0 + (x1 - x0) * p)
            x, y = self._tx(Mx, xp, plate["loch_y"])
            spr = PR.ship_sprite(int(self.U * act.get("size", 0.16)))
            self._paste_rgb(img, spr, int(x - spr.shape[1] / 2), int(y - spr.shape[0] * 0.86), 1.0)
        elif kind == "coach":
            x0, x1 = act.get("x", [1.1, 0.5])
            xp = plate["w"] * (x0 + (x1 - x0) * ease(p * 1.4))
            yp = plate["h"] * act.get("y", 0.82)  # the shore road along the foot of the hill
            x, y = self._tx(Mx, xp, yp)
            spr = PR.coach_sprite(int(self.U * 0.07))
            self._paste_rgb(img, spr, int(x - spr.shape[1] / 2), int(y - spr.shape[0] * 0.9), 1.0)
            if act.get("tag") and p > 0.45:
                tag = PR.stamp_sprite(act["tag"], int(self.U * 0.05), seed=self.seed)
                k = min((p - 0.45) / 0.03, 1)
                self._paste_rotated(img, tag, x, y - spr.shape[0] * 1.5, 1.4 - 0.4 * k, -6, k)
        elif kind == "sign":
            texts = act.get("texts", ["GLEN SCHOOL"])
            swap = self._when({"word": act["swap_word"]} if "swap_word" in act else {"at": act.get("swap", 0.5)}, sh)
            a = next((x for x in anchors if x[0] == "school-sign"), None)
            if a is None:
                return
            cur = texts[0] if t < swap or len(texts) == 1 else texts[1]
            size = max(10, int(a[1][2] * z * 1.5 * self.H / plate["h"] * 1.25))
            spr, _ = T.word_sprite(cur, "serif", size, P.PEAT)
            h, w = spr.shape[:2]
            board = np.zeros((h + 6, w + 12, 4), np.float32)
            board[...] = list(P.f(P.BONE)) + [1.0]
            T.paste(board, spr, 6, 3)
            x, y = self._tx(Mx, a[1][0], a[1][1])
            flip = abs(t - swap) < 0.15  # the sign is swapped with a jolt
            self._paste_rgb(img, board, int(x - board.shape[1] / 2), int(y - board.shape[0] / 2 + (4 if flip else 0)), 1.0)
        elif kind == "stamp":
            t0 = self._when(act, sh)
            if t < t0:
                return
            k = min((t - t0) / 0.09, 1)  # slammed on in 3 frames
            spr = PR.stamp_sprite(act["text"], int(self.U * act.get("size", 0.07)), seed=self.seed)
            x, y = act.get("pos", [0.5, 0.5])
            self._paste_rotated(img, spr, x * self.W, y * self.H, 1.5 - 0.5 * k, act.get("rot", -7), k)
        elif kind == "laird":
            tin = sh["t0"] + act.get("in", 0.0) * (sh["t1"] - sh["t0"])
            tout = sh["t0"] + act.get("out", 1.0) * (sh["t1"] - sh["t0"])
            if not (tin <= t <= tout + 0.6):
                return
            k = ease((t - tin) / 0.6) * (1 - ease((t - tout) / 0.6))
            spr = PR.laird_sprite(int(self.U * act.get("size", 0.62)))
            side = act.get("side", "right")
            h, w = spr.shape[:2]
            bob = 4 * self.a["env"]["bass"][min(i, self.a["frames"] - 1)] * self._budget(sh)
            if side == "right":
                x = int(self.W * 0.80 - w / 2 + (1 - k) * self.W * 0.35)
            else:
                x = int(self.W * 0.20 - w / 2 - (1 - k) * self.W * 0.35)
                spr = spr[:, ::-1].copy()
            y = int(self.H * 0.97 - h + bob)
            # a soft cast shadow so he reads as pasted on
            sh_spr = np.zeros_like(spr)
            sh_spr[..., 3] = cv2.GaussianBlur(spr[..., 3], (0, 0), 6) * 0.45
            self._paste_rgb(img, sh_spr, x + 10, y + 8, 1.0)
            self._paste_rgb(img, spr, x, y, 1.0)

    def _draw_crow(self, img, crow, plate, Mnear, p, t):
        pose = crow if isinstance(crow, str) else crow.get("pose", "perched-left")
        side = 0.12 if pose.endswith("left") else 0.86
        x = side * plate["w"]
        y = float(plate["near_ridge"][int(x)])
        hpx = int(self.U * 0.10)
        if pose == "leaves" and p > 0.55:
            q = (p - 0.55) / 0.45
            spr = M.crow_sprite(hpx, "flying")
            fx, fy = (np.array([x, y - hpx], np.float32) @ Mnear[:, :2].T + Mnear[:, 2])
            fx -= q * self.W * 0.5
            fy -= q * self.H * 0.6 + np.sin(t * 9) * 6
            self._paste_rgb(img, spr, int(fx), int(fy), 1.0)
            return
        spr = M.crow_sprite(hpx, "perched")
        if pose.endswith("right"):
            spr = spr[:, ::-1].copy()
        fx, fy = (np.array([x, y], np.float32) @ Mnear[:, :2].T + Mnear[:, 2])
        z = Mnear[0, 0]
        if abs(z - 1) > 0.01:
            spr = cv2.resize(spr, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
        self._paste_rgb(img, spr, int(fx - spr.shape[1] / 2), int(fy - spr.shape[0] * 0.95), 1.0)
