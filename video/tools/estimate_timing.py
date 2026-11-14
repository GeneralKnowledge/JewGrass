"""Before the real audio arrives: a rough timing.yaml + a guide track from the
beat grid and lyrics alone, so a song's storyboard can be built and previewed.

  python tools/estimate_timing.py beats.json lyrics.txt OUT_DIR [--intro-bars 8] [--outro-at 233.6]

Every lyric line gets bars in proportion to its syllables (spoken lines are
slower); the timing is a guess, to be replaced by tools/align_lyrics.py on
the real vocal stem. The guide track is a click on the grid (kick on the
downbeat, hats on beats), so engine.analysis has something to read.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from align_lyrics import parse_lyrics, syllables, bars_from_beats  # noqa: E402

SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beats")
    ap.add_argument("lyrics")
    ap.add_argument("out_dir")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--intro-bars", type=int, default=8)
    ap.add_argument("--outro-at", type=float, default=None, help="time the spoken outro starts (s)")
    a = ap.parse_args()

    raw = json.load(open(a.beats))
    duration = a.duration or (raw["beats"][-1] + 2.5)
    beats, downs, tempo = bars_from_beats(a.beats, duration)
    secs = parse_lyrics(a.lyrics)
    nbars = int(np.searchsorted(downs, duration))
    bar_len = float(np.median(np.diff(downs)))

    # fixed: intro, and the outro if we know where it is
    outro_bar = int(np.searchsorted(downs, a.outro_at) - 1) if a.outro_at else nbars - 3
    middle = secs[1:-1] if secs[-1]["name"].startswith("outro") else secs[1:]
    weights = []
    for s in middle:
        syl = sum(syllables(w) for ln in s["lines"] for w in ln.split())
        weights.append(syl + 6)  # a little instrumental air per section
    avail = outro_bar - a.intro_bars
    alloc = np.maximum(2, np.round(np.array(weights) / sum(weights) * avail)).astype(int)
    alloc[-1] += avail - alloc.sum()

    starts = [0, a.intro_bars]
    for n in alloc[:-1]:
        starts.append(starts[-1] + int(n))
    if secs[-1]["name"].startswith("outro"):
        starts.append(outro_bar)
    sec_out, lines = [], []
    for si, s in enumerate(secs):
        b0 = starts[si]
        b1 = starts[si + 1] if si + 1 < len(secs) else nbars
        sec_out.append({"name": s["name"], "start_bar": b0, "bars": b1 - b0})
        t0, t1 = downs[b0] + 0.05, downs[min(b1, len(downs) - 1)] - 0.3
        sy = [[syllables(w) for w in ln.split()] for ln in s["lines"]]
        total = sum(sum(x) for x in sy) + 2 * len(sy)
        per = (t1 - t0) / max(total, 1)
        t = t0
        for ln, ws in zip(s["lines"], sy):
            words = []
            for w, k in zip(ln.split(), ws):
                words.append([round(float(t), 3), w])
                t += k * per
            t += 2 * per
            lines.append({"section": s["name"], "text": ln, "words": words})

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "timing.yaml"), "w") as fh:
        yaml.safe_dump({"note": "ESTIMATED from the beat grid only (no audio yet); replace with tools/align_lyrics.py",
                        "tempo": round(float(tempo), 2), "sections": sec_out, "lines": lines},
                       fh, sort_keys=False, allow_unicode=True, width=200)
    json.dump({"tempo": float(tempo), "beats": [float(x) for x in beats],
               "downbeats": [float(x) for x in downs if 0 <= x < duration]},
              open(os.path.join(a.out_dir, "grid.json"), "w"))

    # guide track
    y = np.zeros(int(duration * SR), np.float32)
    g = np.random.default_rng(3)
    k = np.arange(int(0.25 * SR)) / SR
    kick = (np.sin(2 * np.pi * np.cumsum(50 + 80 * np.exp(-k * 30)) / SR) * np.exp(-k * 8)).astype(np.float32)
    hat = (g.normal(0, 1, int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 90) * 0.15).astype(np.float32)
    dset = set(np.round(downs, 2))
    for b in beats:
        i = int(b * SR)
        sig = kick * 0.8 if round(b, 2) in dset else hat
        y[i:i + len(sig)] += sig[: max(0, len(y) - i)]
    sf.write(os.path.join(a.out_dir, "master.wav"), np.clip(y, -1, 1), SR)
    for s in sec_out:
        print(f"  {s['name']:<12} bar {s['start_bar']:>3} ({s['bars']} bars)")


if __name__ == "__main__":
    main()
