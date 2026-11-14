#!/usr/bin/env python3
"""Render the motif library to short audition WAVs (numpy only, no installs).

    python3 tools/design/motif_audition.py docs/song-design/motifs.yaml OUT_DIR

These are sketches so we can hear and judge the tunes, not production sounds.
Writes one WAV per motif plus motifs-all.wav with a gap between them.
"""
import math
import os
import re
import sys
import wave

import numpy as np
import yaml

SR = 44100
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def freq(p):
    m = re.fullmatch(r"([A-G])([#b]?)(-?\d)", p)
    if not m:
        raise ValueError(f"bad pitch {p!r}")
    n = NOTE[m[1]] + {"#": 1, "b": -1, "": 0}[m[2]] + 12 * (int(m[3]) + 1)
    return 440.0 * 2 ** ((n - 69) / 12)


def env(n, a=0.01, r=0.08):
    e = np.ones(n)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def tone(f, dur, voice):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if voice == "pluck":  # Karplus-Strong string
        p = max(2, int(SR / f))
        buf = np.random.default_rng(int(f)).uniform(-1, 1, p)
        out = np.empty(n)
        for i in range(n):
            out[i] = buf[i % p]
            buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
        return out * env(n, 0.002, 0.05)
    if voice == "pipe":  # bright reedy tone with a little vibrato
        vib = 1 + 0.004 * np.sin(2 * math.pi * 5.5 * t)
        ph = 2 * math.pi * f * np.cumsum(vib) / SR
        out = sum(np.sin(k * ph) / k ** 0.8 for k in range(1, 12) if k % 2 or k < 4)
        return 0.35 * out * env(n, 0.015, 0.03)
    if voice == "bell":  # simple FM bell
        mod = 3.5 * np.exp(-t * 4) * np.sin(2 * math.pi * f * 2.76 * t)
        return 0.6 * np.sin(2 * math.pi * f * t + mod) * np.exp(-t * 2.2) * env(n, 0.002, 0.05)
    if voice == "keys":  # soft electric-piano-ish
        out = np.sin(2 * math.pi * f * t) + 0.25 * np.sin(4 * math.pi * f * t) * np.exp(-t * 6)
        return 0.5 * out * np.exp(-t * 1.5) * env(n, 0.005, 0.1)
    # pad
    out = sum(np.sin(2 * math.pi * f * d * t) for d in (0.997, 1.0, 1.003))
    return 0.25 * out * env(n, 0.4, 0.6)


def render(m):
    beat = 60.0 / m["bpm"]
    voice = m.get("voice", "keys")
    lines = []
    for key in ("notes", "chords"):
        if key not in m:
            continue
        v = "pad" if key == "chords" and voice in ("pad",) else ("keys" if key == "chords" else voice)
        parts, pos = [], 0.0
        for item in m[key]:
            pitch, length = item.rsplit(":", 1)
            dur = float(length) * beat
            if pitch != "R":
                for p in pitch.split("+"):
                    parts.append((pos, tone(freq(p), dur * (1.15 if v == "pluck" else 1.0), v)))
            pos += dur
        buf = np.zeros(int((pos + 1.0) * SR))
        for start, sig in parts:
            i = int(start * SR)
            buf[i:i + len(sig)] += sig[: len(buf) - i]
        lines.append(buf)
    n = max(len(b) for b in lines)
    out = np.zeros(n)
    for b in lines:
        out[: len(b)] += b * (0.6 if len(lines) > 1 else 1.0)
    if m.get("drone"):
        t = np.arange(n) / SR
        out += 0.12 * tone(freq(m["drone"]), n / SR, "pipe")[:n] + 0.08 * np.sin(2 * math.pi * freq(m["drone"]) / 2 * t)
    return out


def write(path, x):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.8
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    lib = yaml.safe_load(open(src))["motifs"]
    allparts = []
    for slug, m in lib.items():
        x = render(m)
        x = np.concatenate([x, x]) if len(x) < 4 * SR else x  # play short motifs twice
        write(os.path.join(out_dir, f"motif-{slug}.wav"), x)
        allparts += [x / (np.max(np.abs(x)) + 1e-9), np.zeros(SR)]
        print(f"motif-{slug}.wav  {len(x) / SR:.1f} s  ({m['name']})")
    write(os.path.join(out_dir, "motifs-all.wav"), np.concatenate(allparts))


if __name__ == "__main__":
    main()
